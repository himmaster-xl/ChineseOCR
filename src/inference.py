"""命令行推理 — 对单张手写汉字图片进行识别。

用法:
    python -m src.inference \
        --image my_handwriting.png \
        --checkpoint outputs/checkpoints/best.pt \
        --config configs/vit_small_hwdb.yaml

输出: Top-5 预测汉字及对应置信度。
"""

import argparse
import sys
from pathlib import Path

# 支持 PyCharm 一键运行：将项目根目录加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from PIL import Image

from src.config import Config
from src.data.transforms import get_val_transforms
from src.model.vit import VisionTransformer


def load_label_list(hdf5_path: str) -> list:
    """从 HDF5 属性中加载汉字标签列表。

    预处理时已将 GB2312 编码->汉字的映射保存到 HDF5 根组属性中，
    这里直接读取即可获取按类别索引排列的汉字字符串。

    Args:
        hdf5_path: HDF5 文件路径

    Returns:
        汉字字符串列表，索引对应类别 ID
    """
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        if 'labels' in f.attrs:
            return list(f.attrs['labels'])
    raise FileNotFoundError(
        f"HDF5 文件中未找到 labels 属性，请重新运行预处理脚本: {hdf5_path}"
    )


def predict(
    model, image_path: str, label_list: list,
    device: torch.device, top_k: int = 5,
) -> list:
    """对单张图片进行推理。

    Args:
        model: 已加载权重的 ViT 模型
        image_path: 输入图片路径
        label_list: 类别标签列表
        device: 计算设备
        top_k: 返回前 k 个预测结果

    Returns:
        [(汉字, 置信度), ...] 列表，按置信度降序排列
    """
    # 加载图片并预处理
    img = Image.open(image_path).convert("L")
    img = img.resize((112, 112), Image.BILINEAR)
    img_array = np.array(img, dtype=np.uint8)

    transform = get_val_transforms(112)
    tensor = transform(img_array).unsqueeze(0).to(device)  # (1, 1, 112, 112)

    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    # 取 Top-K
    topk_probs, topk_indices = probs.topk(top_k, dim=1)
    results = [
        (label_list[idx], prob.item())
        for idx, prob in zip(topk_indices[0], topk_probs[0])
    ]
    return results


def main():
    parser = argparse.ArgumentParser(description="单张手写汉字识别")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--config", type=str, default="configs/vit_small_hwdb.yaml")
    parser.add_argument("--top_k", type=int, default=5, help="显示 Top-K 结果")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = VisionTransformer(
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        num_classes=cfg.model.num_classes,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"已加载模型 (epoch {checkpoint['epoch']})")

    # 加载标签列表
    label_list = load_label_list(cfg.data.hdf5_path)

    # 预测
    results = predict(model, args.image, label_list, device, args.top_k)

    print(f"\n{'='*40}")
    print(f"图片: {args.image}")
    print(f"识别结果 (Top-{args.top_k}):")
    print("-" * 40)
    for rank, (char, conf) in enumerate(results, 1):
        bar = "#" * int(conf * 20)
        print(f"  {rank}. {char}  [{bar}] {conf:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
