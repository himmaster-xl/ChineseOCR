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


import h5py, tempfile, os
from data.dataset import HDF5Dataset


class TestHDF5Dataset:
    """HDF5 数据集类测试"""

    @pytest.fixture
    def sample_hdf5(self):
        """创建包含模拟数据的临时 HDF5 文件"""
        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        tmp.close()

        with h5py.File(tmp.name, 'w') as f:
            # 训练集: 100 张 112x112 的 uint8 图像
            f.create_dataset('train/images', data=np.random.randint(
                0, 255, (100, 112, 112), dtype=np.uint8
            ))
            f.create_dataset('train/labels', data=np.random.randint(
                0, 3755, (100,), dtype=np.int64
            ))
            # 验证集: 20 张
            f.create_dataset('val/images', data=np.random.randint(
                0, 255, (20, 112, 112), dtype=np.uint8
            ))
            f.create_dataset('val/labels', data=np.random.randint(
                0, 3755, (20,), dtype=np.int64
            ))

        yield tmp.name
        os.unlink(tmp.name)

    def test_length(self, sample_hdf5):
        """len(dataset) 应返回样本数"""
        ds = HDF5Dataset(sample_hdf5, split='train')
        assert len(ds) == 100

    def test_getitem_output(self, sample_hdf5):
        """__getitem__ 返回 (Tensor, int)"""
        ds = HDF5Dataset(sample_hdf5, split='train', transform=get_val_transforms())
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, (int, np.integer))
        assert img.shape == (1, 112, 112)

    def test_different_splits(self, sample_hdf5):
        """不同 split 返回不同数据"""
        train_ds = HDF5Dataset(sample_hdf5, split='train')
        val_ds = HDF5Dataset(sample_hdf5, split='val')
        assert len(train_ds) == 100
        assert len(val_ds) == 20

    def test_close(self, sample_hdf5):
        """close() 方法正常关闭 HDF5 文件"""
        ds = HDF5Dataset(sample_hdf5, split='train')
        ds.close()
        # 关闭后 file.id 应无效
        assert not ds.file.id
