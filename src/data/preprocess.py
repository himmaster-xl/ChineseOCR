"""数据预处理 — 将 CASIA-HWDB1.1 的 GNT 格式转换为训练友好的 HDF5。

CASIA-HWDB1.1 GNT 文件格式:
    [样本数量(4B, uint32)]
    样本1: [样本大小(4B)] [标签(2B GB2312)] [宽度(2B)] [高度(2B)] [位图(宽*高 字节)]
    样本2: ...

HDF5 输出格式:
    /train/images     (N, 112, 112) uint8
    /train/labels     (N,) int64 (0~3754)
    /val/images       (M, 112, 112) uint8
    /val/labels       (M,) int64
    /test/images      (K, 112, 112) uint8
    /test/labels      (K,) int64

用法:
    python -m src.data.preprocess --raw_dir data/raw --output data/processed/hwdb.h5
"""

import struct
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image


def build_label_map(gnt_files: list) -> dict:
    """扫描所有 GNT 文件，构建「GB2312编码 -> 类别索引」的映射表。

    Args:
        gnt_files: GNT 文件路径列表

    Returns:
        dict: {gb2312_tag_code: class_index}，按 GB2312 编码排序编号
    """
    tag_set = set()
    for gnt_path in tqdm(gnt_files, desc="扫描标签"):
        with open(gnt_path, "rb") as f:
            num_samples = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_samples):
                size = struct.unpack("<I", f.read(4))[0]
                tag_code = f.read(2)  # GB2312 双字节编码
                # 跳过宽高 + 位图
                f.read(2 + 2 + size - 2)
                tag_set.add(tag_code)

    # 按 GB2312 编码排序编号
    sorted_tags = sorted(tag_set)
    return {tag: idx for idx, tag in enumerate(sorted_tags)}


def parse_gnt_file(gnt_path: str) -> tuple:
    """解析单个 GNT 文件，返回 (图像列表, 标签字符串列表)。

    Args:
        gnt_path: GNT 文件路径

    Returns:
        images: numpy uint8 数组列表，每个形状为 (H, W)
        labels: GB2312 解码后的汉字字符串列表
    """
    images = []
    labels = []

    with open(gnt_path, "rb") as f:
        num_samples = struct.unpack("<I", f.read(4))[0]

        for _ in range(num_samples):
            # 读取样本头
            sample_size = struct.unpack("<I", f.read(4))[0]
            tag_code = f.read(2)
            width = struct.unpack("<H", f.read(2))[0]
            height = struct.unpack("<H", f.read(2))[0]

            # 读取位图数据
            bitmap_data = f.read(width * height)

            # 将 1D 位图重塑为 2D 图像
            img = np.frombuffer(bitmap_data, dtype=np.uint8).reshape(height, width)

            # GB2312 解码
            label = tag_code.decode("gb2312-80", errors="replace")

            images.append(img)
            labels.append(label)

    return images, labels


def create_hdf5(
    images: list,
    label_map: dict,
    class_indices: list,
    output_path: str,
    split: str = "train",
    resize_to: int = 112,
):
    """将图像和标签写入 HDF5 文件。

    所有图像统一 resize 到 (resize_to, resize_to) 并转为 uint8。
    标签转为 class_indices 中的整数索引。

    Args:
        images: numpy 图像数组列表
        label_map: GB2312 编码 -> 类别索引映射（保留参数，兼容旧调用）
        class_indices: 每张图对应的类别索引列表
        output_path: 输出 HDF5 文件路径
        split: 数据集子集名称
        resize_to: 统一缩放的尺寸
    """
    # 预处理所有图像到统一尺寸
    processed = []
    for img in tqdm(images, desc=f"Resize {split}"):
        # PIL 处理 resize 比 numpy 快
        pil_img = Image.fromarray(img, mode="L")
        pil_img = pil_img.resize((resize_to, resize_to), Image.BILINEAR)
        processed.append(np.array(pil_img, dtype=np.uint8))

    images_array = np.stack(processed, axis=0)  # (N, H, W)
    labels_array = np.array(class_indices, dtype=np.int64)

    # 写入 HDF5（追加模式，覆盖已有 split）
    with h5py.File(output_path, "a") as f:
        if split in f:
            del f[split]
        grp = f.create_group(split)
        grp.create_dataset(
            "images", data=images_array, compression="gzip", compression_opts=4
        )
        grp.create_dataset("labels", data=labels_array)

    print(f"已写入 {split}: {len(processed)} 样本 -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CASIA-HWDB1.1 GNT -> HDF5 预处理")
    parser.add_argument("--raw_dir", type=str, required=True, help="原始 GNT 文件目录")
    parser.add_argument("--output", type=str, required=True, help="输出 HDF5 文件路径")
    parser.add_argument("--image_size", type=int, default=112, help="统一缩放尺寸")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    gnt_files = list(raw_dir.rglob("*.gnt"))
    if not gnt_files:
        print(f"警告: 在 {raw_dir} 中未找到 .gnt 文件")
        return
    print(f"找到 {len(gnt_files)} 个 GNT 文件")

    # 第一步：扫描所有标签构建映射
    label_map = build_label_map(gnt_files)
    print(f"共 {len(label_map)} 个不重复汉字")

    # 第二步：解析所有文件并收集数据（按作者分组用于划分）
    all_images, all_indices, all_authors = [], [], []
    for gnt_path in tqdm(gnt_files, desc="解析 GNT"):
        # 从文件名提取作者编号（CASIA 命名惯例: xxx-xxx.gnt）
        author_id = (
            int(gnt_path.stem.split("-")[0])
            if "-" in gnt_path.stem
            else 0
        )

        images, labels = parse_gnt_file(str(gnt_path))
        for img, lbl in zip(images, labels):
            all_images.append(img)
            all_indices.append(label_map[lbl])
            all_authors.append(author_id)

    # 第三步：按作者划分训练集/验证集/测试集（80/10/10）
    unique_authors = list(set(all_authors))
    np.random.seed(42)
    np.random.shuffle(unique_authors)

    n_authors = len(unique_authors)
    n_train_auth = int(n_authors * 0.8)
    n_val_auth = int(n_authors * 0.1)

    train_authors = set(unique_authors[:n_train_auth])
    val_authors = set(unique_authors[n_train_auth:n_train_auth + n_val_auth])
    test_authors = set(unique_authors[n_train_auth + n_val_auth:])

    # 第四步：写入 HDF5
    splits = {"train": train_authors, "val": val_authors, "test": test_authors}
    for split_name, author_set in splits.items():
        split_images, split_indices = [], []
        for img, idx, author in zip(all_images, all_indices, all_authors):
            if author in author_set:
                split_images.append(img)
                split_indices.append(idx)

        create_hdf5(
            split_images, label_map, split_indices,
            args.output, split=split_name, resize_to=args.image_size,
        )

    # 保存标签映射（GB2312 编码 -> 汉字字符串）到 HDF5 属性
    idx_to_label = {idx: tag for tag, idx in label_map.items()}
    sorted_tags = [idx_to_label[i] for i in range(len(label_map))]
    # GB2312 双字节编码解码为汉字字符串
    char_list = [tag.decode("gb2312-80", errors="replace") for tag in sorted_tags]

    with h5py.File(args.output, "a") as f:
        f.attrs["num_classes"] = len(char_list)
        dt = h5py.string_dtype(encoding="utf-8")
        f.attrs.create("labels", char_list, dtype=dt)

    print(f"预处理完成: {args.output}")
    print(f"训练作者: {sum(1 for a in all_authors if a in train_authors)}")
    print(f"验证作者: {sum(1 for a in all_authors if a in val_authors)}")
    print(f"测试作者: {sum(1 for a in all_authors if a in test_authors)}")


if __name__ == "__main__":
    main()
