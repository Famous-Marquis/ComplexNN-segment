from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset
from Model import ComplexUNet, AutoRegisterModelCallback


@dataclass
class Config:
    tags: Dict = field(default_factory=lambda:
                    {"description": "加入正则化",
                    "dataset":      "PolSF v1"})
    run_name: str = "加Dice, 减层数, 权重正则化: 1e-3 [" + datetime.now().strftime("%m%d-%H:%M")+"]"
    threshold: float = 0.50
    metric_name: str = "val_mIoU"
    model_reg_name: str = "UNet_candidate"
    data_inch: int = 2
    data_catgory: int = 7
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    batch_size: int = 32
    lr: float = 10e-4
    weight_decay: float = 1e-3
    dice_weight: float = 1.
    tv_weight: float = 0.0
    max_epochs: int = 30
    accelerator: str = "gpu"
    devices: int = 1
    train_img_path: str = "./PolSF/SF-RISAT/train_img_stack.npy"
    train_label_path: str = "./PolSF/SF-RISAT/train_label_stack.npy"
    test_img_path: str = "./PolSF/SF-RISAT/test_img_stack.npy"
    test_label_path: str = "./PolSF/SF-RISAT/test_label_stack.npy"


def load_data(img_data_path, label_data_path):
    img_data = np.load(img_data_path)  # N,H,W,C
    label_data = np.load(label_data_path)
    img_tensor = torch.from_numpy(img_data).type(torch.cfloat).permute(0, 3, 1, 2)  # ->N,C,H,W
    label_tensor = torch.from_numpy(label_data).long()
    label_onehot = torch.nn.functional.one_hot(label_tensor, config.data_catgory).type(torch.float32)
    label_onehot = label_onehot.permute(0, 3, 1, 2)  # ->N,C,H,W
    return TensorDataset(img_tensor, label_onehot)

if __name__ == '__main__':
    config = Config()
    # load data
    train_dataset = load_data(config.train_img_path, config.train_label_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = load_data(config.test_img_path, config.test_label_path)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # define logger
    mllogger = MLFlowLogger(experiment_name="UNet_experiment", run_name=config.run_name, tracking_uri="file:./mlruns",
                            log_model=True, tags=config.tags)
    auto_reg_callback = AutoRegisterModelCallback(threshold=config.threshold, metric_name=config.metric_name,
                                                  model_reg_name=config.model_reg_name)
    checkpoint_callback = ModelCheckpoint(filename='epoch_{epoch:02d}_step_{step:04d}',auto_insert_metric_name=False)
    trainer = Trainer(max_epochs=config.max_epochs, devices=1, accelerator=config.accelerator, logger=mllogger,
                      callbacks=[auto_reg_callback, checkpoint_callback])
    complex_unet = ComplexUNet(in_channels=config.data_inch, categories=config.data_catgory, channels=config.channels,
                               lr=config.lr, batch_size=config.batch_size, max_epochs=config.max_epochs,
                               dice_weight=config.dice_weight, tv_weight=config.tv_weight, weight_decay=config.weight_decay)
    trainer.fit(complex_unet, train_loader, val_loader, )

    # 模型在训练集与验证集上分别评估测试
    complex_unet.eval()
    print("训练集性能:")
    trainer.validate(complex_unet,train_loader)
    print("测试集性能")
    trainer.validate(complex_unet,val_loader,)