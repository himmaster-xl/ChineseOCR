"""数据模块测试"""
import numpy as np
import torch
import pytest
import sys
sys.path.insert(0, 'src')

from data.transforms import get_train_transforms, get_val_transforms


class TestTransforms:
    """数据增强管道测试"""

    def test_train_transforms_output_shape(self):
        """训练增强输出 (1, 112, 112) 的标准化张量"""
        transform = get_train_transforms(image_size=112)
        # 模拟 HDF5 读出的 uint8 图像 (112, 112)
        img = np.random.randint(0, 255, (112, 112), dtype=np.uint8)

        out = transform(img)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 112, 112), f"期望 (1, 112, 112)，实际 {out.shape}"

    def test_val_transforms_deterministic(self):
        """验证集增强无随机性（仅 resize + normalize）"""
        transform = get_val_transforms(image_size=112)
        img = np.random.randint(0, 255, (112, 112), dtype=np.uint8)

        out1 = transform(img)
        out2 = transform(img)

        assert torch.allclose(out1, out2), "验证集变换应确定性输出"

    def test_normalize_range(self):
        """标准化后值应在合理范围内（均值 0.5，标准差 0.5）"""
        transform = get_val_transforms(image_size=112)
        img = np.ones((112, 112), dtype=np.uint8) * 128  # 中灰

        out = transform(img)

        # mean=0.5, std=0.5 -> 128 -> (128/255 - 0.5) / 0.5 ≈ 0.0039
        assert out.mean().abs() < 0.1
