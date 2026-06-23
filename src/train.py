"""训练脚本 — 完整的训练循环，支持 AMP 混合精度、梯度累积、早停。

用法:
    python -m src.train --config configs/vit_small_hwdb.yaml

训练过程在 RTX 4060 (8GB VRAM) 上经 AMP 优化后，
batch_size=128 时显存占用约 5-6GB，留有余量。
"""

import argparse
import sys
from pathlib import Path

# 支持 PyCharm 一键运行：将项目根目录加入搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.config import Config
from src.utils import set_seed, EarlyStopping, save_checkpoint
from src.data.dataset import HDF5Dataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.model.vit import VisionTransformer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
) -> float:
    """执行一个训练 epoch。

    Args:
        model: ViT 模型（训练模式）
        dataloader: 训练数据加载器
        optimizer: AdamW 优化器
        criterion: 损失函数（CrossEntropyLoss with label smoothing）
        scaler: AMP 梯度缩放器
        device: 计算设备
        grad_accum_steps: 梯度累积步数

    Returns:
        本 epoch 的平均训练损失
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # AMP 自动混合精度前向传播
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps

        # AMP 反向传播
        scaler.scale(loss).backward()

        # 每 grad_accum_steps 步更新一次参数
        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """在验证集上评估模型。

    Args:
        model: ViT 模型（评估模式）
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备

    Returns:
        (平均损失, top-1 准确率, top-5 准确率)
    """
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        total += labels.size(0)

        # Top-1: 最高 logit 对应的类别
        _, pred_top1 = logits.topk(1, dim=1)
        correct_top1 += (pred_top1.squeeze() == labels).sum().item()

        # Top-5: 前 5 个最高 logit 中是否包含正确类别
        _, pred_top5 = logits.topk(5, dim=1)
        correct_top5 += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

    return (
        total_loss / len(dataloader),
        correct_top1 / total,
        correct_top5 / total,
    )


def main():
    parser = argparse.ArgumentParser(description="Train ViT on CASIA-HWDB1.1")
    parser.add_argument(
        "--config", type=str, default="configs/vit_small_hwdb.yaml",
        help="YAML 配置文件路径",
    )
    args = parser.parse_args()

    # ---- 加载配置 ----
    # 解析相对路径（PyCharm 运行目录可能不是项目根目录）
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = Config.from_yaml(str(config_path))
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 构建数据加载器 ----
    # HDF5 路径也支持相对路径解析
    hdf5_path = Path(cfg.data.hdf5_path)
    if not hdf5_path.is_absolute():
        hdf5_path = PROJECT_ROOT / hdf5_path
    hdf5_path = str(hdf5_path)

    train_ds = HDF5Dataset(
        hdf5_path, split="train",
        transform=get_train_transforms(cfg.model.image_size),
    )
    val_ds = HDF5Dataset(
        hdf5_path, split="val",
        transform=get_val_transforms(cfg.model.image_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True,
    )
    print(f"训练样本: {len(train_ds):,}, 验证样本: {len(val_ds):,}")

    # ---- 构建模型 ----
    model = VisionTransformer(
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数总量: {total_params:,}")

    # ---- 优化器 & 损失函数 & 调度器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # Label Smoothing: 将 hard label [0,0,1,0] 平滑为 [eps, eps, 1-eps, eps]
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

    # Cosine Warmup: 前 warmup_epochs 线性增加，然后 cosine 衰减
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.train.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs - cfg.train.warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.train.warmup_epochs],
    )

    # ---- 训练循环 ----
    scaler = GradScaler('cuda')
    stopper = EarlyStopping(patience=10, mode="max")
    best_acc = 0.0
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n开始训练 — {cfg.train.epochs} epochs, batch_size={cfg.train.batch_size}")
    print("=" * 60)

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, cfg.train.grad_accum_steps,
        )

        val_loss, top1, top5 = validate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{cfg.train.epochs} | "
            f"LR: {lr_now:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Top-1: {top1:.4f} | "
            f"Top-5: {top5:.4f}"
        )

        if top1 > best_acc:
            best_acc = top1
            save_checkpoint(model, optimizer, epoch, val_loss,
                            checkpoint_dir / "best.pt")
            print(f"  -> 保存最佳模型 (Top-1: {best_acc:.4f})")

        save_checkpoint(model, optimizer, epoch, val_loss,
                        checkpoint_dir / "last.pt")

        if stopper(top1):
            print(f"早停触发于 epoch {epoch}，最佳 Top-1: {best_acc:.4f}")
            break

        scheduler.step()

    print(f"\n训练完成！最佳 Top-1: {best_acc:.4f}")
    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
