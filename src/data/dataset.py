"""HDF5 数据集 — 封装 CASIA-HWDB1.1 预处理为 HDF5 后的高效读取。

为什么用 HDF5 而不是直接从 PNG 文件夹加载：
1. 240 万个小文件直接从磁盘读取极慢（IO 瓶颈）
2. HDF5 支持分块存储和压缩，加载速度快 10-100 倍
3. 不需要额外的标签文件映射，一张 HDF5 文件包含全部数据

HDF5 内部结构:
    /train/images      (N_train, 112, 112) uint8
    /train/labels      (N_train,)          int64
    /val/images        (N_val, 112, 112)   uint8
    /val/labels        (N_val,)            int64
    /test/images       (N_test, 112, 112)  uint8
    /test/labels       (N_test,)           int64
"""

import h5py
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """从 HDF5 文件读取手写汉字数据集的 PyTorch Dataset。

    支持训练/验证/测试三种 split，每个 split 有独立的 images 和 labels。
    读取时按需加载（通过 h5py 的延迟读取机制），不占满内存。

    Args:
        hdf5_path: HDF5 文件的完整路径
        split: 数据子集名称，'train' / 'val' / 'test'
        transform: torchvision 变换管道，None 则返回原始 uint8 numpy
    """

    def __init__(self, hdf5_path: str, split: str = "train", transform=None):
        # 保存参数以便 pickle 后重建
        self.hdf5_path = hdf5_path
        self.split = split
        self.transform = transform

        # 打开文件但不立即读取所有数据（h5py 默认延迟加载）
        self._open()

    def _open(self):
        """打开 HDF5 文件并获取数据集引用（支持 pickle 后重新打开）。"""
        self.file = h5py.File(self.hdf5_path, "r")
        self.images = self.file[f"{self.split}/images"]  # shape: (N, H, W)
        self.labels = self.file[f"{self.split}/labels"]  # shape: (N,)

    def __getstate__(self):
        """Pickle 时：排除不可序列化的 h5py 对象。"""
        state = self.__dict__.copy()
        del state["file"]
        del state["images"]
        del state["labels"]
        return state

    def __setstate__(self, state):
        """Unpickle 时：恢复状态并重新打开 HDF5 文件。"""
        self.__dict__.update(state)
        self._open()

    def __len__(self) -> int:
        """返回数据集中的样本总数。"""
        return len(self.labels)

    def __getitem__(self, idx: int):
        """根据索引获取单个样本。

        HDF5 支持按索引切片读取，只从磁盘加载需要的部分。

        Args:
            idx: 样本索引 (0 <= idx < len(dataset))

        Returns:
            (image, label) 元组:
                - image: torch.Tensor 或 numpy.ndarray
                - label: int
        """
        # h5py 支持整数索引，直接读取单张图像
        img = self.images[idx]  # (H, W) numpy uint8
        label = int(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def close(self):
        """关闭 HDF5 文件句柄，释放资源。"""
        self.file.close()
