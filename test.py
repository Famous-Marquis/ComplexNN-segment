
if __name__ == '__main__':
    # test Pytorch cuda cudnn
    import torch
    x = torch.rand(5, 3)
    print(x)

    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前GPU设备: {torch.cuda.current_device()}")
        print(f"GPU设备名称: {torch.cuda.get_device_name()}")
        print(f"CUDA设备数量: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"cuDNN是否启用: {torch.backends.cudnn.enabled}")