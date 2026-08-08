from typing import List

import mlflow
import numpy as np
import pylab as pl
from lightning.pytorch.callbacks import Callback
import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
import complexNN.nn as cnn
import lightning as L
import torchvision
from PIL import Image
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchvision import transforms

from beams import plot_u, x_meshgrid, y_meshgrid


class ConvBlock(L.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            cnn.cConv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            cnn.cBatchNorm2d(out_channels),
            cnn.cRelu(),
            cnn.cDropout(),
            cnn.cConv2d(out_channels, out_channels, kernel_size, padding=1, bias=False),  # 不共享卷积核，各通道独立
            cnn.cBatchNorm2d(out_channels),
            cnn.cRelu(),
        )

    def forward(self, x):
        return self.conv(x)


class cTransposeConv(L.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2, padding=0,
                 bias: bool = False, dilation=1, groups: int = 1):
        super().__init__()
        assert in_channels % groups == 0, "In_channels should be an integer multiple of groups."
        assert out_channels % groups == 0, "Out_channels should be an integer multiple of groups."

        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.randn((in_channels, out_channels // groups, *kernel_size), dtype=torch.cfloat))

        self.bias = nn.Parameter(torch.randn((out_channels,), dtype=torch.cfloat)) if bias else None

    def forward(self, x):
        if not x.dtype == torch.cfloat:
            x = torch.complex(x, torch.zeros_like(x))
        return torch.nn.functional.conv_transpose2d(x,
                                                    self.weight, self.bias, self.stride, self.padding, 0, self.dilation,
                                                    self.groups)


class Bottleneck(L.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            cnn.cConv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            cnn.cBatchNorm2d(out_channels),
            cnn.cRelu(),
            cnn.cDropout(),
            cnn.cConv2d(out_channels, out_channels, kernel_size, padding=1, bias=False),
            cnn.cBatchNorm2d(out_channels),
            cnn.cRelu(),
            cnn.cDropout(),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(L.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, up_sample_rate):
        super().__init__()
        self.conv = nn.Sequential(
            cTransposeConv(in_channels, out_channels, kernel_size, stride=up_sample_rate, padding=0, bias=False),
            cnn.cBatchNorm2d(out_channels),
            cnn.cRelu(),
            cnn.cDropout(),
        )

    def forward(self, x):
        return self.conv(x)


class OutConv(L.LightningModule):
    def __init__(self, in_channels, kernel_size, categories):
        super().__init__()
        self.conv = nn.Sequential(
            cnn.cConv2d(in_channels, out_channels=categories, kernel_size=kernel_size, padding=1, bias=False),  # 这里没有分组
            # 新增softmax
            # cnn.cBatchNorm2d(categories),
            # cnn.cSoftmax(dim=1)
        )

    def forward(self, x):
        return self.conv(x).real


def tv_loss(y_pred):
    """
    计算 Total Variation Loss (正则化项)
    强制预测图在空间上连续，减少孤立噪点
    pred: (Batch, Channel, Height, Width)
    """
    horizon_diff = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    vertic_diff = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

    tv_loss = torch.sum(horizon_diff) + torch.sum(vertic_diff)
    loss_scaled = tv_loss / (y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3])
    return loss_scaled


def dice_loss(y_pred, label, smooth=1e-5):
    """
    计算 Dice Loss
    pred: 经过 sigmoid 后的概率图
    target: 真实标签
    """
    y_pred = y_pred.view(-1)
    label = label.view(-1)

    intersection = (y_pred * label).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + label.sum() + smooth)
    return 1 - dice


class ComplexUNet(L.LightningModule):
    def __init__(self, in_channels, categories, channels: List[int], lr, tv_weight, dice_weight, weight_decay, *args,
                 **kwargs):
        super(ComplexUNet, self).__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.categories = categories
        self.channels = channels
        self.encoder_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.lr = lr
        self.tv_weight = tv_weight
        self.dice_weight = dice_weight
        self.weight_decay = weight_decay
        self.val_acc_metric = MulticlassAccuracy(num_classes=categories, average="macro")
        self.val_iou_metric = MulticlassJaccardIndex(num_classes=categories, average="macro")
        for i in range(len(self.channels) - 1):
            self.encoder_blocks.append(ConvBlock(in_channels, out_channels=self.channels[i], kernel_size=3))
            in_channels = self.channels[i]  # 下一层的输入通道
            # --------------down block----------------
            self.down_blocks.append(cnn.cAvgPool2d(2))
            # self.down_blocks.append(cnn.cMaxPool2d(2))

        self.bottleneck = ConvBlock(self.channels[-2], self.channels[-1], kernel_size=3)
        for i in reversed(range(len(self.channels) - 1)):
            self.up_blocks.append(UpConv(self.channels[i + 1], self.channels[i], 2, up_sample_rate=2))
            # ------------------shortcut 在这里---------------
            # self.channels[i] * 2: 启用shortcut
            self.decoder_blocks.append(ConvBlock(self.channels[i] * 2, self.channels[i], kernel_size=3))

        self.out_conv = OutConv(self.channels[0], kernel_size=3, categories=self.categories)
        # --- [优化] 预定义颜色表并注册为 Buffer ---
        # 这样它会自动跟随模型移动到 GPU/CPU，且不会作为参数被优化
        palette = torch.tensor([
            [0, 0, 0],  # Class 0: Black (背景)
            [0, 255, 255],  # Class 1: Cyan
            # [255, 255, 0],  # Class 2: Yellow
            # [0, 0, 255],  # Class 3: Blue
            # 调换了颜色
            [0, 0, 255],  # Class 3: Blue
            [255, 255, 0],  # Class 2: Yellow
            [255, 0, 0],  # Class 4: Red
            [0, 255, 0],  # Class 5: Green
            [255, 0, 255]  # Class 6: Magenta (新增，品红) <--- 加上这一行
        ], dtype=torch.uint8)
        self.register_buffer("palette", palette)

    def decode_seg_map(self, mask_indices):
        """
        mask_indices: [B, H, W]
        Returns: [B, 3, H, W] Float Tensor (0~1)
        """
        # 直接使用 self.palette，无需重复创建
        rgb = self.palette[mask_indices]  # [B, H, W, 3]
        rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
        return rgb

    def total_loss(self, y_pred_logits, label_onehot):
        """
        模型使用的损失函数
        :param y_pred_logits:
        :param label_onehot:
        :return: loss,_dice_loss,_tv_loss
        """
        ce_loss = F.cross_entropy(y_pred_logits, label_onehot)
        y_pred_softmax = F.softmax(y_pred_logits, dim=1)
        _tv_loss = tv_loss(y_pred_softmax)
        _dice_loss = dice_loss(y_pred_softmax, label_onehot)
        loss = ce_loss + self.tv_weight * _tv_loss + self.dice_weight * _dice_loss
        self.log_dict({"train_loss": loss, "train_dice_loss": _dice_loss, "train_tv_loss": _tv_loss})
        return loss, _dice_loss, _tv_loss

    def forward(self, x):
        shorcut = []
        for conv, down in zip(self.encoder_blocks, self.down_blocks):
            x = conv(x)
            # -------------shortcut-----------
            shorcut.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, conv in zip(self.up_blocks, self.decoder_blocks):
            x = up(x)
            # ------------shortcut------------
            x = torch.cat([x, shorcut.pop()], dim=1)
            x = conv(x)
        y = self.out_conv(x)
        return y

    def training_step(self, batch, batch_idx):
        img, label_onehot = batch
        y_pred_logits = self.forward(img)
        loss, _dice_loss, _tv_loss = self.total_loss(y_pred_logits, label_onehot)
        self.log_dict({"train_loss": loss, "train_dice_loss": _dice_loss, "train_tv_loss": _tv_loss})
        return loss

    def validation_step(self, batch, batch_idx):
        img, label_onehot = batch
        # img shape: [B, 2, H, W], dtype: torch.complex64 / complex128
        y_pred_logits = self.forward(img)
        loss, _dice_loss, _tv_loss = self.total_loss(y_pred_logits, label_onehot)

        y_pred = torch.argmax(y_pred_logits, dim=1)
        label = torch.argmax(label_onehot, dim=1)

        self.val_acc_metric(y_pred, label)
        self.val_iou_metric(y_pred, label)
        self.log_dict({"val_loss": loss, "val_dice_loss": _dice_loss, "val_tv_loss": _tv_loss})
        # --- 图像记录逻辑 (专门针对双通道复数) ---
        if batch_idx == 0:
            num_samples = min(20, img.shape[0] // 2)

            # 1. 获取样本 [B, 2, H, W]
            indices = torch.linspace(0, img.shape[0] - 1, num_samples).long()
            raw_complex_samples = img[indices]

            # 2. 分离通道并取模 (Magnitude) -> [B, H, W]
            # torch.abs() 对复数会自动计算 sqrt(real^2 + imag^2)
            mag_ch1 = raw_complex_samples[:, 0, :, :].abs()
            mag_ch2 = raw_complex_samples[:, 1, :, :].abs()

            # 3. 归一化函数 (将任意范围的值映射到 0-1 以便可视化)
            def normalize_to_01(tensor_bhw):
                # 针对每个样本单独归一化，避免某个样本太亮导致其他全黑
                # reshape 为 [B, -1] 计算 min/max
                flat = tensor_bhw.flatten(1)
                min_v = flat.min(dim=1, keepdim=True)[0].unsqueeze(2)
                max_v = flat.max(dim=1, keepdim=True)[0].unsqueeze(2)
                # 避免除以 0
                return (tensor_bhw - min_v) / (max_v - min_v + 1e-8)

            mag_ch1_norm = normalize_to_01(mag_ch1)
            mag_ch2_norm = normalize_to_01(mag_ch2)

            # 4. 维度调整: [B, H, W] -> [B, 1, H, W] -> [B, 3, H, W] (Gray to RGB)
            # 必须转为 3 通道才能和后面的彩色 Mask 拼接
            raw_vis1 = mag_ch1_norm.unsqueeze(1).repeat(1, 3, 1, 1)
            raw_vis2 = mag_ch2_norm.unsqueeze(1).repeat(1, 3, 1, 1)

            # 5. 解码预测结果和标签
            pred_colored = self.decode_seg_map(y_pred[:num_samples])
            target_colored = self.decode_seg_map(label[:num_samples])

            # 6. 拼接: [Raw1模, Raw2模, 预测, 标签] (并在宽度方向拼接 dim=3)
            # 最终展示顺序：输入通道1 | 输入通道2 | 预测结果 | 真实标签
            comparison = torch.cat([raw_vis1, raw_vis2, pred_colored, target_colored], dim=3)

            # 7. 制作网格
            grid_tensor = torchvision.utils.make_grid(comparison, nrow=1, padding=2)

            # 8. MLflow 记录
            ndarr = grid_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            client = self.logger.experiment
            run_id = self.logger.run_id
            client.log_image(run_id, ndarr, f"epoch_{self.current_epoch}_complex_vis.png")

    def on_validation_epoch_end(self):
        acc = self.val_acc_metric.compute()
        iou = self.val_iou_metric.compute()
        self.log_dict({"val_acc": acc, "val_mIoU": iou})

        self.val_acc_metric.reset()
        self.val_iou_metric.reset()

    def predict_step(self, batch, batch_idx):
        img, label_onehot = batch
        y_pred_logits = self.forward(img)
        y_pred = torch.argmax(y_pred_logits, dim=1)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
