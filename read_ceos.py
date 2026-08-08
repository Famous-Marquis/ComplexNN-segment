from typing import Literal

from PIL import Image
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

def read_risat_ceos(file_path):
    """
    读取 RISAT/CEOS 格式的 dat 文件并转换为复数 numpy 数组
    """
    # 打开数据集
    ds = gdal.Open(file_path)

    if ds is None:
        print(f"无法打开文件: {file_path}")
        return None

    # 获取图像尺寸
    width = ds.RasterXSize
    height = ds.RasterYSize
    print(f"图像尺寸: 宽={width}, 高={height}, 波段数={ds.RasterCount}")

    # CEOS SLC 数据通常是一个包含复数数据的波段
    # 或者有时会被 GDAL 解析为两个波段 (Band 1=I, Band 2=Q)

    # 尝试读取第一波段
    band = ds.GetRasterBand(1)

    # 获取数据类型名称
    datatype_name = gdal.GetDataTypeName(band.DataType)
    print(f"GDAL识别的数据类型: {datatype_name}")

    # 读取所有数据
    data = band.ReadAsArray()

    # --- 关键转换步骤 ---
    # 情况 1: GDAL 直接识别为复数 (CInt16, CFloat32)
    if 'Complex' in datatype_name:
        print("直接读取为复数数据...")
        complex_data = data  # 已经是复数了

    # 情况 2: 某些老格式可能把 I 和 Q 分开存 (极少见，但在 RISAT 上要小心)
    # 如果读出来不是复数，但你有两个波段，可能需要手动组合
    else:
        # 这种情况比较少见，通常 GDAL 对 CEOS SLC 支持很好
        # 如果遇到这种情况，通常需要: data = band1 + 1j * band2
        print("警告：未直接识别为复数类型，可能需要手动合并 I/Q 通道")
        complex_data = data

    return complex_data

def extract_complex_data(file_path, target_file, polar: Literal["RH", "RV"]):
    # 2. 读取数据
    print(f"正在读取{polar}数据...")
    slc_rh = read_risat_ceos(file_path_rh)

    if slc_rh is not None:
        print("\n转换成功！")
        print("数据类型:", slc_rh.dtype)
        print("数据形状:", slc_rh.shape)
        print("第一个像素值:", slc_rh[0, 0])

        # 3. 简单的可视化检查 (显示幅度图)
        # 取对数幅度以便看得清
        amplitude = 20 * np.log10(np.abs(slc_rh) + 1e-6)

        # 下采样显示 (为了快一点，只显示 1/10 大小)
        plt.imshow(amplitude[10648:4257:-1, 2486:7414,], cmap='gray')
        plt.title(f"{polar} Amplitude (Log scale)")
        plt.colorbar()
        plt.show()

        slc_clamp = slc_rh[4257:10648, 2486:7414,]
        print(slc_clamp.shape)
        print(slc_clamp.dtype)
        np.save(target_file, slc_clamp)
        return slc_clamp

def preprocess_complex_data(data):
    """
    输入: data (H, W, C) 复数数据
    输出: norm_data (H, W, C) 实数数据 (实部虚部分开，或保持复数根据网络需求)
    """

    # 1. 分离实部和虚部 (因为我们不能对复数直接比较大小)
    real_part = data.real
    imag_part = data.imag

    # 定义一个内部函数来处理单个分量
    def clip_and_normalize(arr):
        # --- 第一步：计算统计量 ---
        mean = np.mean(arr)
        std = np.std(arr)

        # --- 第二步：截断 (Clipping) ---
        # 这一步非常重要！我们只保留 ±3倍标准差内的数据
        # 超过 3sigma 的值通常是强散射点，虽然有物理意义，
        # 但对神经网络的权重更新非常不友好。
        upper_limit = mean + 3 * std
        lower_limit = mean - 3 * std

        # 将小于下限的设为下限，大于上限的设为上限
        arr_clipped = np.clip(arr, lower_limit, upper_limit)

        # --- 第三步：标准化 (Z-Score) ---
        # 结果会分布在 -3 到 +3 之间，非常适合神经网络输入
        arr_norm = (arr_clipped - mean) / (std + 1e-8)  # 加个小值防除零

        return arr_norm

    # 分别处理
    print("正在处理实部...")
    norm_real = clip_and_normalize(real_part)
    print("正在处理虚部...")
    norm_imag = clip_and_normalize(imag_part)

    # --- 第四步：组合 ---
    # 如果你的网络接受复数输入 (CV-CNN)，就组合回去
    complex_norm = norm_real + 1j * norm_imag
    print("预处理形状：",complex_norm.shape)
    # 如果你的网络是普通实数网络 (ResNet, UNet)，通常把它们堆叠成不同通道
    # 变成 (H, W, 2*C)
    # output_data = np.stack([norm_real, norm_imag], axis=-1)

    # 注意：如果原图是多通道（比如 HH, HV），你需要对每个通道分别做，或者一起堆叠
    return complex_norm


import os
from skimage.transform import resize


def make_dataset(data_img, label_img, patch_size=128, stride=64, save_dir='./SF-RISAT',N=3496,mode='train'):
    """
    data_img:  原始数据 (H, W, C)
    label_img: 标签数据 (H, W)
    mode:      'train' 会进行数据增强, 'test' 仅切片
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    H, W,C = data_img.shape
    count = 0
    img_stack=np.zeros((N,patch_size,patch_size,C),dtype=np.complex64)
    label_stack=np.zeros((N,patch_size,patch_size),dtype=np.int64)
    # 遍历图像
    for r in range(0, H - patch_size + 1, stride):
        for C in range(0, W - patch_size + 1, stride):

            # 1. 切片
            patch_data = data_img[r: r + patch_size, C: C + patch_size]
            patch_label = label_img[r: r + patch_size, C: C + patch_size]

            # --- 过滤无效样本 (可选) ---
            # 如果这一块全是 0 (未标记区域)，通常可以丢弃，不参与训练
            # if np.all(patch_label == 0):
            #     continue

            # 2. 保存原始切片
            # 命名格式: mode_序号.npy
            img_stack[count,:,:,:] = patch_data
            label_stack[count,:,:] = patch_label
            count += 1

            # --- 3. 数据增强 (仅限训练集) ---
            if mode == 'train':
                # 数据和标签必须做完全相同的变换！！！

                # A. 旋转 90度
                aug_data_1 = np.rot90(patch_data, 1)
                aug_label_1 = np.rot90(patch_label, 1)
                img_stack[count, :, :, :] = aug_data_1
                label_stack[count, :, :] = aug_label_1


                # B. 旋转 180度
                aug_data_2 = np.rot90(patch_data, 2)
                aug_label_2 = np.rot90(patch_label, 2)
                img_stack[count, :, :, :] = aug_data_2
                label_stack[count, :, :] = aug_label_2
                count += 1
                # C. 旋转 270度
                aug_data_3 = np.rot90(patch_data, 3)
                aug_label_3 = np.rot90(patch_label, 3)
                img_stack[count, :, :, :] = aug_data_3
                label_stack[count, :, :] = aug_label_3
                count += 1
                # D. 水平翻转 (Flip)
                aug_data_4 = np.fliplr(patch_data)
                aug_label_4 = np.fliplr(patch_label)
                img_stack[count, :, :, :] = aug_data_4
                label_stack[count, :, :] = aug_label_4
                count += 1
                # E. 缩放 (Scaling) - 稍微复杂一点
                # 注意：标签必须用 'nearest' (最近邻) 插值，防止出现不存在的小数类别
                # 缩放通常不保存为文件，而是在训练读取时实时做（On-the-fly），
                # 但如果硬要存，可以随机 Crop 一个中心区域再放大回来。
                # 这里为了简单，暂不存缩放版本，旋转翻转通常足够了。
    # np.save(os.path.join(save_dir, mode+'_img_stack.npy'), img_stack)
    # np.save(os.path.join(save_dir, mode+'_label_stack.npy'), label_stack)
    print(f"{mode}集处理完毕，共生成 {count} 个样本（含增强）。")
    return img_stack, label_stack
if __name__ == '__main__':

    # --- 使用示例 ---

    # 1. 设置文件路径 (注意路径中的斜杠)
    # 假设你在 D:/下载-D/SAN_FRANCISCO_RISAT/SAN_FRANCISCO_RISAT/163791211/scene_RH/
    # 请修改为你实际的绝对路径
    file_path_rh = "../Dataset/SAN_FRANCISCO_RISAT/SAN_FRANCISCO_RISAT/163791211/scene_RH/dat_01.001"
    file_path_rv="../Dataset/SAN_FRANCISCO_RISAT/SAN_FRANCISCO_RISAT/163791211/scene_RV/dat_01.001"
    target_file_rh="./SF-RISAT/RH.npy"
    target_file_rv="./SF-RISAT/RV.npy"
    slc_rh=extract_complex_data(file_path_rh, target_file_rh, polar="RH")
    slc_rv=extract_complex_data(file_path_rv, target_file_rv, polar="RV")
    slc_rv_rh=np.stack((slc_rv,slc_rh),axis=-1)
    # slc: [H,W,C:rv,rh] np.complex64
    complex_slc=preprocess_complex_data(slc_rv_rh)
    np.save("./SF-RISAT/original_data.npy", complex_slc)

    # --- 主程序逻辑 ---

    # 假设 loaded_data 是你的复数数据 (4928, 6391, C)
    # 假设 loaded_label 是你的标签图 (4928, 6391)
    # 这里的 loaded_data 需要你自己用之前的 gdal 代码读进来并拼合成 numpy
    loaded_label=Image.open("./SF-RISAT/SF-RISAT-label2d.png")
    loaded_label=np.array(loaded_label)
    loaded_color_label=Image.open("./SF-RISAT/SF-RISAT-label3d.png")
    loaded_color_label=np.array(loaded_color_label)
    # 1. 物理分割 (Spatial Split) - 前 80% 行做训练，后 20% 做测试
    assert loaded_label.shape[:2]==complex_slc.shape[:2]

    train_data_1 = complex_slc[128:3192, :]
    train_label_1 = loaded_label[128:3192, :]

    train_data_2 = complex_slc[3192+128:, :]
    train_label_2 = loaded_label[3192+128:, :]

    test_data_1 = complex_slc[3192:3192+128, :]
    test_label_1 = loaded_label[3192:3192+128, :]

    test_data_2 = complex_slc[0:128, :]
    test_label_2 = loaded_label[0:128, :]

    print("训练区域大小:", train_data_1.shape)
    print("测试区域大小:", test_data_1.shape)

    # 2. 生成训练集 (步长较小=重叠，开启增强)
    # 128 大小，64步长 => 50% 重叠
    train_img_stack1,train_label_stack1=make_dataset(train_data_1, train_label_1, patch_size=128, stride=64, mode='train1',N=3496)
    train_img_stack2,train_label_stack2=make_dataset(train_data_2, train_label_2, patch_size=128, stride=64, mode='train2',N=3496)
    train_img_stack=np.concatenate((train_img_stack1,train_img_stack2),axis=0)
    train_label_stack=np.concatenate((train_label_stack1,train_label_stack2),axis=0)
    # 3. 生成测试集 (步长=大小=无重叠，关闭增强)
    test_img_stack1,test_label_stack1=make_dataset(test_data_1, test_label_1, patch_size=128, stride=128, mode='test1',N=38)
    test_img_stack2,test_label_stack2=make_dataset(test_data_2, test_label_2, patch_size=128, stride=128, mode='test2',N=38)
    test_img_stack=np.concatenate((test_img_stack1,test_img_stack2),axis=0)
    test_label_stack=np.concatenate((test_label_stack1,test_label_stack2),axis=0)

    np.save("./SF-RISAT/train_img_stack.npy", train_img_stack)
    np.save("./SF-RISAT/train_label_stack.npy", train_label_stack)

    np.save("./SF-RISAT/test_img_stack.npy", test_img_stack)
    np.save("./SF-RISAT/test_label_stack.npy", test_label_stack)
