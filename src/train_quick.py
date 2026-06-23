"""快速验证脚本 — 小数据量跑通全流程。

用法:
    python src/train_quick.py

默认 10%% 数据 (~8万张)，5 epochs，约 25 分钟完成 (ResNet)。
跑通后再用 src/train.py 正式训练。

调整:
    python src/train_quick.py --fraction 0.05 --epochs 3   (更快，~10分钟)
    python src/train_quick.py --fraction 0.2  --epochs 5   (更准)
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
from src.model.resnet import ResNet


@torch.no_grad()
def validate(model, dataloader, device):
    """快速验证，只算 Top-1"""
    model.eval()
    correct, total = 0, 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with autocast("cuda"):
            logits = model(images)
        _, preds = logits.topk(1, dim=1)
        correct += (preds.squeeze() == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Quick pipeline validation")
    parser.add_argument("--config", type=str, default="configs/vit_small_hwdb.yaml")
    parser.add_argument("--fraction", type=float, default=0.1, help="数据比例 (默认 10%%)")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = Config.from_yaml(str(config_path))

    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"快速验证: {args.fraction:.0%} 数据 x {args.epochs} epochs, batch={args.batch_size}")
    print(f"预计时间: ~25 分钟 (ResNet)")

    hdf5_path = Path(cfg.data.hdf5_path)
    if not hdf5_path.is_absolute():
        hdf5_path = PROJECT_ROOT / hdf5_path
    hdf5_path = str(hdf5_path)

    full_train = HDF5Dataset(hdf5_path, split="train",
                             transform=get_train_transforms(cfg.model.image_size))
    full_val   = HDF5Dataset(hdf5_path, split="val",
                             transform=get_val_transforms(cfg.model.image_size))

    # 随机采样子集
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

    print(f"训练样本: {len(train_ds)} / 验证样本: {len(val_ds)}")

    model_type = getattr(cfg.model, 'type', 'vit')
    if model_type == 'resnet':
        model = ResNet(
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
        ).to(device)
    else:
        model = VisionTransformer(
            image_size=cfg.model.image_size,
            patch_size=cfg.model.patch_size,
            in_channels=cfg.model.in_channels,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            num_classes=cfg.model.num_classes,
        ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,} ({model_type})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                   weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    scaler = GradScaler("cuda")

    print(f"{'='*50}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
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

        acc = validate(model, val_loader, device)
        print(f"  -> Train Loss: {total_loss/len(train_loader):.4f} | Val Top-1: {acc:.4f}")

    save_path = PROJECT_ROOT / "outputs" / "checkpoints" / "quick.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, args.epochs, total_loss, str(save_path))
    print(f"\n{'='*50}")
    print(f"[OK] 快速验证通过！模型: {save_path}")
    print(f"正式训练: python src/train.py")

    full_train.close()
    full_val.close()


if __name__ == "__main__":
    main()
