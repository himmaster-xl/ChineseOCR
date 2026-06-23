"""数据预处理 — 将 CASIA GNT 格式转换为训练友好的 HDF5。

GNT 文件格式（无文件头，直接为样本序列）:
    样本1: [样本大小(4B, 含自身)] [标签(2B, uint16 LE)] [宽度(2B)] [高度(2B)] [位图(w*h 字节)]
    样本2: ...
    读到 EOF 为止

标签编码:
    - 中文汉字: 标签值 = uint16 LE 读取的 GB2312 双字节（如 0xC4B5 → '的'）
    - ASCII符号: 标签值 < 256，高字节为 0x00（如 0x0021 → '!'）

HDF5 输出格式:
    /train/images     (N, 112, 112) uint8
    /train/labels     (N,) int64 (0 ~ num_classes-1)
    /val/images       ...
    /val/labels       ...
    /test/images      ...
    /test/labels      ...
    .attrs/labels     汉字字符串列表（按 class_index 排列）
    .attrs/num_classes  总类别数

用法:
    python -m src.data.preprocess --raw_dir data/raw --output data/processed/hwdb.h5
"""

import struct
import argparse
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image


def tag_to_char(tag_value: int) -> str:
    """将 GNT 文件中的 uint16 标签值解码为可显示的字符。

    Args:
        tag_value: 从文件读取的 uint16 LE 标签值

    Returns:
        解码后的字符（中文汉字或 ASCII 符号）
    """
    tag_bytes = struct.pack("<H", tag_value)  # 还原为文件中的两个字节

    # 高字节为 0x00 表示 ASCII 范围字符
    if tag_bytes[1] == 0x00:
        return chr(tag_bytes[0]) if 32 <= tag_bytes[0] <= 126 else f"[{tag_value}]"

    # 尝试 GB2312 / GBK 解码
    try:
        return tag_bytes.decode("gb2312-80")
    except (UnicodeDecodeError, LookupError):
        pass
    try:
        return tag_bytes.decode("gbk", errors="replace")
    except (UnicodeDecodeError, LookupError):
        pass

    return f"[{tag_value}]"


def build_label_map(gnt_files: list) -> dict:
    """扫描所有 GNT 文件，构建 {tag_value: class_index} 的映射表。

    Args:
        gnt_files: GNT 文件路径列表

    Returns:
        dict: {tag_value: class_index}，按 tag_value 升序编号
    """
    tag_set = set()
    for gnt_path in tqdm(gnt_files, desc="扫描标签"):
        with open(gnt_path, "rb") as f:
            data = f.read()

        offset = 0
        while offset < len(data):
            sample_size = struct.unpack_from("<I", data, offset)[0]
            # sample_size 包含自身 4 字节，最小合法样本: 4+2+2+2+1=11
            if sample_size < 11 or offset + sample_size > len(data):
                break
            tag_value = struct.unpack_from("<H", data, offset + 4)[0]
            tag_set.add(tag_value)
            offset += sample_size

    # 按 tag_value 升序编号
    sorted_tags = sorted(tag_set)
    return {tag: idx for idx, tag in enumerate(sorted_tags)}


def parse_gnt_file(gnt_path: str) -> tuple:
    """解析单个 GNT 文件。

    Args:
        gnt_path: GNT 文件路径

    Returns:
        (images: list[np.ndarray], class_indices: list[int])
    """
    images = []
    class_indices = []

    with open(gnt_path, "rb") as f:
        data = f.read()

    offset = 0
    while offset < len(data):
        sample_size = struct.unpack_from("<I", data, offset)[0]
        if sample_size < 11 or offset + sample_size > len(data):
            break

        tag_value = struct.unpack_from("<H", data, offset + 4)[0]
        width = struct.unpack_from("<H", data, offset + 6)[0]
        height = struct.unpack_from("<H", data, offset + 8)[0]

        # 验证: sample_size = 4(s_size) + 2(tag) + 2(w) + 2(h) + w*h
        expected = 10 + width * height
        if sample_size != expected:
            # 跳过损坏样本
            offset += 1
            continue

        bitmap_data = data[offset + 10 : offset + sample_size]
        try:
            img = np.frombuffer(bitmap_data, dtype=np.uint8).reshape(height, width)
        except ValueError:
            offset += sample_size
            continue

        images.append(img)
        class_indices.append(tag_value)  # 先存原始 tag，后续再映射
        offset += sample_size

    return images, class_indices


def create_hdf5(
    images: list,
    class_indices: list,
    output_path: str,
    split: str = "train",
    resize_to: int = 112,
):
    """将图像和已映射的类别索引写入 HDF5 文件。

    所有图像统一 resize 到 (resize_to, resize_to)。

    Args:
        images: numpy 图像数组列表
        class_indices: 每张图对应的类别索引 (0 ~ num_classes-1)
        output_path: 输出 HDF5 文件路径
        split: 数据集子集名称
        resize_to: 统一缩放的尺寸
    """
    processed = []
    for img in tqdm(images, desc=f"Resize {split}"):
        pil_img = Image.fromarray(img, mode="L")
        pil_img = pil_img.resize((resize_to, resize_to), Image.BILINEAR)
        processed.append(np.array(pil_img, dtype=np.uint8))

    images_array = np.stack(processed, axis=0)
    labels_array = np.array(class_indices, dtype=np.int64)

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
    parser = argparse.ArgumentParser(description="CASIA GNT -> HDF5 预处理")
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
    print(f"共 {len(label_map)} 个不重复类别")

    # 第二步：解析所有文件并收集数据
    all_images, all_indices, all_authors = [], [], []

    for gnt_path in tqdm(gnt_files, desc="解析 GNT"):
        author_id = (
            int(gnt_path.stem.split("-")[0])
            if "-" in gnt_path.stem
            else hash(gnt_path.stem) % 10000
        )

        images, raw_tags = parse_gnt_file(str(gnt_path))
        for img, tag_val in zip(images, raw_tags):
            all_images.append(img)
            all_indices.append(label_map[tag_val])
            all_authors.append(author_id)

    print(f"总样本数: {len(all_images):,}")

    # 第三步：按作者划分（80/10/10）
    unique_authors = list(set(all_authors))
    np.random.seed(42)
    np.random.shuffle(unique_authors)

    n = len(unique_authors)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_authors = set(unique_authors[:n_train])
    val_authors = set(unique_authors[n_train : n_train + n_val])
    test_authors = set(unique_authors[n_train + n_val :])

    # 第四步：写入 HDF5
    for split_name, author_set in [
        ("train", train_authors),
        ("val", val_authors),
        ("test", test_authors),
    ]:
        split_images, split_indices = [], []
        for img, idx, author in zip(all_images, all_indices, all_authors):
            if author in author_set:
                split_images.append(img)
                split_indices.append(idx)

        create_hdf5(
            split_images, split_indices,
            args.output, split=split_name, resize_to=args.image_size,
        )

    # 第五步：保存标签映射到 HDF5 属性
    idx_to_tag = {idx: tag for tag, idx in label_map.items()}
    sorted_tags = [idx_to_tag[i] for i in range(len(label_map))]
    char_list = [tag_to_char(t) for t in sorted_tags]

    with h5py.File(args.output, "a") as f:
        f.attrs["num_classes"] = len(char_list)
        dt = h5py.string_dtype(encoding="utf-8")
        f.attrs.create("labels", char_list, dtype=dt)

    total_train = sum(1 for a in all_authors if a in train_authors)
    total_val = sum(1 for a in all_authors if a in val_authors)
    total_test = sum(1 for a in all_authors if a in test_authors)

    print(f"\n预处理完成: {args.output}")
    print(f"  类别数: {len(label_map)}")
    print(f"  总样本: {len(all_images):,}")
    print(f"  训练:   {sum(1 for a in all_authors if a in train_authors)} 作者")
    print(f"  验证:   {sum(1 for a in all_authors if a in val_authors)} 作者")
    print(f"  测试:   {sum(1 for a in all_authors if a in test_authors)} 作者")


if __name__ == "__main__":
    main()
