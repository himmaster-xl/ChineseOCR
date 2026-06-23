"""数据增强 — 训练集在线增强 + 验证集归一化。

手写汉字的类内差异大（不同作者书写风格不同），
适当的数据增强可以提高模型的泛化能力。

增强策略说明:
- RandomRotation: 容忍不同程度的手写倾斜
- RandomAffine(translate): 容忍字符位置偏移
- RandomAffine(scale): 容忍字符大小变化

归一化: (pixel/255 - 0.5) / 0.5 将像素值映射到 [-1, 1]
（灰度图像的常见预处理方式，IS = IN = 0.5）
"""

from torchvision import transforms


def get_train_transforms(image_size: int = 112):
    """获取训练集的数据增强管道。

    包含随机旋转、平移、缩放以增加数据多样性。
    所有操作针对手写汉字特点设计了保守的范围，
    避免过度增强导致字形失真。

    Args:
        image_size: 输出图像的边长（像素）

    Returns:
        torchvision.transforms.Compose 组合变换
    """
    return transforms.Compose([
        # numpy ndarray -> PIL Image（torchvision transforms 的输入要求）
        transforms.ToPILImage(),
        # 随机旋转 ±5° — 模拟手写倾斜
        transforms.RandomRotation(degrees=5),
        # 随机平移 ±5% + 缩放 95%~105% — 模拟位置和大小差异
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        # 统一尺寸
        transforms.Resize((image_size, image_size)),
        # PIL -> Tensor (0~1 浮点数)
        transforms.ToTensor(),
        # 标准化到 [-1, 1]
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_val_transforms(image_size: int = 112):
    """获取验证集/测试集的预处理管道（无随机增强）。

    Args:
        image_size: 输出图像的边长（像素）

    Returns:
        torchvision.transforms.Compose 组合变换
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
