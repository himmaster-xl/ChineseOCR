"""Transformer Encoder — 堆叠 N 个 Encoder 层构成完整编码器。

将输入 token 序列通过多层自注意力+前馈网络逐层变换，
最终输出全局上下文感知的表示。最后加一层 LayerNorm 统一输出。
"""

import torch
import torch.nn as nn

from .encoder import TransformerEncoder


class Transformer(nn.Module):
    """由 N 个相同结构的 Encoder 层堆叠而成的 Transformer 编码器。

    Args:
        num_layers: Encoder 层数（ViT-Small = 12）
        hidden_dim: 隐藏维度
        num_heads: 注意力头数
        mlp_ratio: MLP 扩展比
        dropout: Dropout 概率
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ModuleList 确保每层参数被正确注册到优化器中
        self.layers = nn.ModuleList([
            TransformerEncoder(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        # 最终 LayerNorm：统一输出分布，有利于分类头
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """逐层编码输入序列。

        Args:
            x: (B, N, D) token 序列（已含 [CLS] token 和位置编码）

        Returns:
            (B, N, D) 编码后的 token 序列
        """
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
