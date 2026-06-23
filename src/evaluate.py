"""评估脚本 — 加载训练好的 checkpoint 在测试集上计算最终指标。

用法:
    python -m src.evaluate \
        --config configs/vit_small_hwdb.yaml \
        --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

# 支持 PyCharm 一键运行：将项目根目录加入搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data.dataset import HDF5Dataset
from src.data.transforms import get_val_transforms
from src.model.vit import VisionTransformer


@torch.no_grad()
def evaluate(model, dataloader, device):
    """在测试集上评估模型，输出详细的 Top-1/Top-5 准确率。

    Args:
        model: 已加载权重的 ViT 模型
        dataloader: 测试数据加载器
        device: 计算设备

    Returns:
        (top1_acc, top5_acc, total_samples)
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    pbar = tqdm(dataloader, desc="Evaluating")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        _, pred_top1 = logits.topk(1, dim=1)
        correct_top1 += (pred_top1.squeeze() == labels).sum().item()

        _, pred_top5 = logits.topk(5, dim=1)
        correct_top5 += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)
        pbar.set_postfix({
            "top1": f"{correct_top1/total:.4f}",
            "top5": f"{correct_top5/total:.4f}",
        })

    return correct_top1 / total, correct_top5 / total, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT on test set")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    args = parser.parse_args()

    # 解析相对路径（PyCharm 工作目录可能不是项目根目录）
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = Config.from_yaml(str(config_path))

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    hdf5_path = Path(cfg.data.hdf5_path)
    if not hdf5_path.is_absolute():
        hdf5_path = PROJECT_ROOT / hdf5_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载测试数据
    test_ds = HDF5Dataset(
        str(hdf5_path), split="test",
        transform=get_val_transforms(cfg.model.image_size),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True,
    )
    print(f"测试样本: {len(test_ds):,}")

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

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"已加载 checkpoint: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")

    # 评估
    top1, top5, total = evaluate(model, test_loader, device)
    print(f"\n{'='*40}")
    print(f"测试集结果: {total:,} 样本")
    print(f"Top-1 准确率: {top1:.4f} ({top1*100:.2f}%)")
    print(f"Top-5 准确率: {top5:.4f} ({top5*100:.2f}%)")
    print(f"{'='*40}")

    test_ds.close()


if __name__ == "__main__":
    main()
