"""Patch Embedding — 将图像分割为固定大小 patch 并线性投影到嵌入空间。

ViT 的第一步：将原始图像转换为 Transformer 可处理的 token 序列。
使用步长=patch_size 的 Conv2d 一次性完成「分块 + 线性投影」，比手动切分高效。
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """将输入图像分割为不重叠的 patch 并线性投影到指定的嵌入维度。

    公式（隐式执行）:
        patches = unfold(image, patch_size)
        tokens = patches @ W^T

    Args:
        in_channels: 输入通道数（灰度=1, RGB=3）
        patch_size: 每个 patch 的边长（像素）
        hidden_dim: 投影后的嵌入维度（同时也是 Transformer 的 d_model）
    """

    def __init__(self, in_channels: int = 1, patch_size: int = 4, hidden_dim: int = 384):
        super().__init__()
        self.patch_size = patch_size

        # Conv2d(kernel=patch_size, stride=patch_size) 等价于：
        #   1. 将图像切成不重叠的 patch_size × patch_size 块
        #   2. 每个块展平后用线性变换投影到 hidden_dim
        self.proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, C, H, W) 输入图像

        Returns:
            (B, N_patches, hidden_dim) patch token 序列
        """
        # (B, C, H, W) -> (B, hidden_dim, H/patch, W/patch)
        x = self.proj(x)
        # 展平空间维度 -> (B, hidden_dim, N_patches)
        x = x.flatten(2)
        # 转换为 Transformer 期望的 (B, N_patches, hidden_dim)
        x = x.transpose(1, 2)
        return x
