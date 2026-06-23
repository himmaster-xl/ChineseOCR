"""快速验证脚本 — 用 1/20 数据快速跑通全流程。

用法:
    python src/train_quick.py

只取 5% 数据，训练 5 个 epoch，验证整个 pipeline 没问题后
再用 src/train.py 正式训练。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from src.config import Config
from src.utils import set_seed, save_checkpoint
from src.data.dataset import HDF5Dataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.model.vit import VisionTransformer


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """快速验证，只算 Top-1"""
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(dataloader, desc="Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        with autocast("cuda"):
            logits = model(images)
        _, preds = logits.topk(1, dim=1)
        correct += (preds.squeeze() == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Quick train on 1/20 data")
    parser.add_argument("--config", type=str, default="configs/vit_small_hwdb.yaml")
    parser.add_argument("--fraction", type=float, default=0.05, help="数据比例 (默认 1/20)")
    parser.add_argument("--epochs", type=int, default=5, help="快速训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="小 batch 防 OOM")
    args = parser.parse_args()

    # 解析配置路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = Config.from_yaml(str(config_path))

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"模式: 快速验证 — {args.fraction:.0%} 数据, {args.epochs} epochs")

    # ── 加载完整数据集 ──
    hdf5_path = Path(cfg.data.hdf5_path)
    if not hdf5_path.is_absolute():
        hdf5_path = PROJECT_ROOT / hdf5_path
    hdf5_path = str(hdf5_path)

    full_train = HDF5Dataset(hdf5_path, split="train",
                             transform=get_train_transforms(cfg.model.image_size))
    full_val   = HDF5Dataset(hdf5_path, split="val",
                             transform=get_val_transforms(cfg.model.image_size))

    # ── 随机采样 1/20 ──
    np.random.seed(42)
    n_train = int(len(full_train) * args.fraction)
    n_val   = int(len(full_val)   * args.fraction)
    train_idx = np.random.choice(len(full_train), n_train, replace=False)
    val_idx   = np.random.choice(len(full_val),   n_val,   replace=False)

    train_ds = Subset(full_train, train_idx)
    val_ds   = Subset(full_val,   val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"训练样本: {len(train_ds):,} / {len(full_train):,} ({args.fraction:.0%})")
    print(f"验证样本: {len(val_ds):,}   / {len(full_val):,}   ({args.fraction:.0%})")

    # ── 构建模型 ──
    model = VisionTransformer(
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        num_classes=cfg.model.num_classes,
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                   weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    scaler = GradScaler("cuda")

    # ── 训练 ──
    print(f"\n{'='*50}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        # 验证
        acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Top-1: {acc:.4f}")

    # ── 保存 ──
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "quick.pt"
    save_checkpoint(model, optimizer, args.epochs, total_loss, str(save_path))
    print(f"\n{'='*50}")
    print(f"快速验证完成！模型已保存: {save_path}")
    print(f"最终验证 Top-1: {acc:.4f} ({acc*100:.1f}%)")
    print(f"如果流程正常，正式训练用: python src/train.py")

    full_train.close()
    full_val.close()


if __name__ == "__main__":
    main()
