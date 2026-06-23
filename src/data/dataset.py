"""HDF5 数据集 — 封装预处理后的 HDF5 高效读取。

推荐使用无压缩 HDF5 (hwdb_fast.h5)，随机访问极快。
gzip 压缩版 (hwdb.h5) 随机读很慢，已弃用。
"""

import h5py
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """从 HDF5 文件读取手写汉字数据的 PyTorch Dataset。

    支持训练/验证/测试三种 split。
    配合无压缩 HDF5 时随机访问近乎瞬时。

    Args:
        hdf5_path: HDF5 文件路径
        split: 'train' / 'val' / 'test'
        transform: torchvision 变换管道
    """

    def __init__(self, hdf5_path: str, split: str = "train", transform=None):
        self.hdf5_path = hdf5_path
        self.split = split
        self.transform = transform
        self._open()

    def _open(self):
        self.file = h5py.File(self.hdf5_path, "r")
        self.images = self.file[f"{self.split}/images"]
        self.labels = self.file[f"{self.split}/labels"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("file", None)
        state.pop("images", None)
        state.pop("labels", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def close(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()
