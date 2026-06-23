"""重建 HDF5 — 去掉 gzip 压缩，随机读取快 100+ 倍。

用大块批量读取规避 gzip 每张解压的开销。
无压缩 HDF5 随机访问是纯磁盘寻道，几乎瞬时。

用法:
    python src/data/rebuild_hdf5.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import h5py
import numpy as np
from tqdm import tqdm

SRC = "data/processed/hwdb.h5"
DST = "data/processed/hwdb_fast.h5"
CHUNK = 5000  # 批量读取块大小

print(f"源: {SRC} (gzip 压缩)")
print(f"目标: {DST} (无压缩)")

with h5py.File(SRC, "r") as src:
    for split in ["train", "val", "test"]:
        n_src = len(src[f"{split}/images"])
        print(f"\n  {split}: {n_src:,} 张")

        # 创建无压缩目标数据集
        with h5py.File(DST, "a") as dst:
            if split in dst:
                del dst[split]
            grp = dst.create_group(split)
            ds_img = grp.create_dataset(
                "images", shape=(n_src, 112, 112), dtype=np.uint8
                # 无 compression 参数 → 不压缩
            )
            ds_lbl = grp.create_dataset(
                "labels", shape=(n_src,), dtype=np.int64
            )

            # 批量读取 + 批量写入
            for start in tqdm(range(0, n_src, CHUNK), desc=f"Copy {split}"):
                end = min(start + CHUNK, n_src)
                ds_img[start:end] = src[f"{split}/images"][start:end]
                ds_lbl[start:end] = src[f"{split}/labels"][start:end]

    # 复制属性
    print("\n复制标签属性...")
    with h5py.File(DST, "a") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v

# 测试速度
print(f"\n随机读取速度测试 ({DST}):")
with h5py.File(DST, "r") as f:
    imgs = f["train/images"]
    idx = np.random.choice(len(imgs), 128, replace=False).tolist()

    import time
    t0 = time.time()
    for i in idx:
        _ = imgs[i]
    t = time.time() - t0
    print(f"  随机 128 张: {t:.2f}s", end="")
    if t < 1:
        print("  ✅ 极快!")
    elif t < 5:
        print("  ✅ 可接受")
    else:
        print("  ❌ 仍太慢")
