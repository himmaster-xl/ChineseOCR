"""Transformer Encoder Layer — 自注意力 + 前馈网络的残差组合。

每个 Encoder 层包含两个子层（Pre-LN 架构，ViT 标准）：
1. LayerNorm -> Multi-Head Self-Attention -> 残差连接
2. LayerNorm -> MLP -> 残差连接

Pre-LN（先归一化再计算）比 Post-LN 训练更稳定，不需要长期 warmup。
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .mlp import MLP


class TransformerEncoder(nn.Module):
    """单个 Transformer Encoder 层（Pre-LN 架构）。

    子层1: x = x + Attention(LayerNorm(x))
    子层2: x = x + MLP(LayerNorm(x))

    Args:
        hidden_dim: 隐藏维度（d_model）
        num_heads: 注意力头数
        mlp_ratio: MLP 中间层扩展倍数
        dropout: Dropout 概率
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 子层1：自注意力
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)

        # 子层2：前馈网络
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 — Pre-LN 残差结构。

        Args:
            x: (B, N, D) 输入 token 序列

        Returns:
            (B, N, D) 编码后的 token 序列
        """
        # 子层1: 自注意力 + 残差
        # LayerNorm 放在注意力之前（Pre-LN），梯度流更稳定
        x = x + self.attn(self.norm1(x))
        # 子层2: 前馈网络 + 残差
        x = x + self.mlp(self.norm2(x))
        return x
