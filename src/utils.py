"""工具函数 — 随机种子、早停、Checkpoint 管理。

提供训练过程中通用的辅助功能，所有函数无业务耦合，可直接复用。
"""

import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """固定所有随机数生成器的种子，确保实验可复现。

    同时固定 Python random、NumPy、PyTorch CPU 和 CUDA 的种子。
    CUDA 确定性算法会略微降低性能，但对学习项目影响可忽略。

    Args:
        seed: 随机种子值，默认为 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 确定性模式：确保相同输入产生相同输出（轻微性能损失）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """早停机制 — 监控验证指标，连续 patience 次未改善即触发停止。

    用法:
        stopper = EarlyStopping(patience=10, mode='max')
        for epoch in range(epochs):
            val_acc = validate()
            if stopper(val_acc):
                print("早停触发！")
                break

    Attributes:
        patience: 容忍的未改善次数
        mode: 'max' 表示越大越好（如准确率），'min' 表示越小越好（如损失）
        best_score: 当前最佳分数
        counter: 连续未改善计数
    """

    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def __call__(self, score: float) -> bool:
        """检查是否应触发早停。

        Args:
            score: 本轮验证指标（准确率或损失）

        Returns:
            True 表示应停止训练，False 表示继续
        """
        # 判断是否有改善
        improved = (
            score > self.best_score
            if self.mode == "max"
            else score < self.best_score
        )

        if improved:
            self.best_score = score
            self.counter = 0
            return False  # 继续训练
        else:
            self.counter += 1
            # 连续 patience 次无改善 → 停止
            return self.counter >= self.patience


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    """保存模型和优化器状态的完整 checkpoint。

    Args:
        model: 当前模型实例
        optimizer: 当前优化器实例
        epoch: 当前 epoch 编号
        loss: 当前验证损失
        path: 保存路径（如 outputs/checkpoints/best.pt）
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, path)


if __name__ == "__main__":
    import tempfile, os

    # 测试 set_seed — 两次随机结果应相同
    set_seed(42)
    a = torch.randn(3, 3)
    set_seed(42)
    b = torch.randn(3, 3)
    assert torch.allclose(a, b), "set_seed: 可重复性验证失败"
    print("[OK] set_seed works")

    # 测试 EarlyStopping — 连续下降应触发停止
    es = EarlyStopping(patience=3, mode='max')
    assert not es(0.9)  # 首次，best_score=0.9
    assert not es(0.8)  # 未改善, counter=1
    assert not es(0.85) # 未改善, counter=2
    assert es(0.8)      # 连续 3 次未改善 → 应停止
    print("[OK] EarlyStopping works")

    # 测试 save_checkpoint — 文件存在且可加载
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    tmp.close()
    save_checkpoint(model, opt, epoch=5, loss=0.5, path=tmp.name)
    ckpt = torch.load(tmp.name, map_location='cpu')
    assert ckpt['epoch'] == 5
    assert abs(ckpt['loss'] - 0.5) < 1e-6
    print("[OK] save_checkpoint works")
    os.unlink(tmp.name)
