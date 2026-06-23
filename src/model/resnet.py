"""ResNet CNN — 手写汉字识别专用。

针对 112×112 灰度图设计：
- 相比 ViT 训练快 5-10 倍，每步 ~0.2s
- 卷积天然擅长捕捉笔画/部首等局部特征
- 3 个 stage，逐步降采样 112→56→28→14→7

架构:
    Conv(1→64, 7×7) → MaxPool
    → Stage1(64→128)×2 → Stage2(128→256)×2 → Stage3(256→512)×2
    → GlobalAvgPool → Linear(512→3490)
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet 基本残差块: Conv3×3 → BN → ReLU → Conv3×3 → BN → +残差 → ReLU"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接：如果维度或空间尺寸变了，用 1×1 卷积对齐
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    """轻量 ResNet，专为 112×112 灰度手写汉字分类设计。

    Args:
        in_channels: 输入通道 (1=灰度)
        num_classes: 分类数 (3490)
        layers: 每 stage 的 block 数，默认 [2,2,2] (类似 ResNet-14)
        base_channels: 第一层通道数，每 stage 翻倍
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3490,
        layers: list = None,
        base_channels: int = 64,
    ):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2]  # 3 个 stage，共 12 层卷积 + 1 输入层

        # ── 输入层: 112 → 56 ──
        self.conv1 = nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)  # 56 → 28

        # ── Stage 1: 28×28, 64→128 channels ──
        self.stage1 = self._make_stage(base_channels, base_channels * 2, layers[0], stride=1)

        # ── Stage 2: 28→14, 128→256 channels ──
        self.stage2 = self._make_stage(base_channels * 2, base_channels * 4, layers[1], stride=2)

        # ── Stage 3: 14→7, 256→512 channels ──
        self.stage3 = self._make_stage(base_channels * 4, base_channels * 8, layers[2], stride=2)

        # ── 输出 ──
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # → (B, 512, 1, 1)
        self.fc = nn.Linear(base_channels * 8, num_classes)

        # 初始化
        self._init_weights()

    def _make_stage(self, in_ch: int, out_ch: int, blocks: int, stride: int):
        """构建一个 stage：第一个 block 可能降采样，后续 block 保持尺寸。"""
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量 (B, 512)。"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        return x.flatten(1)  # (B, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 → (B, num_classes) logits。"""
        return self.fc(self.forward_features(x))
