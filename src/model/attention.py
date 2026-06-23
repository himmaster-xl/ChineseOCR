"""Multi-Head Self-Attention — Transformer 的核心机制。

让每个 token 关注所有其他 token，根据内容相似度加权聚合信息。
「多头」意味着将特征分成多组独立计算注意力，最后拼接——
类比为让 token 从多个不同的「视角」观察上下文。
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头自注意力（Self-Attention）。

    计算公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    其中 Q、K、V 由输入 x 经一次联合线性投影后拆分得到（效率优化）。

    Args:
        hidden_dim: 输入/输出维度（d_model）
        num_heads: 注意力头数（hidden_dim 必须能被 num_heads 整除）
        dropout: Attention 权重上的 Dropout 概率
    """

    def __init__(
        self, hidden_dim: int = 384, num_heads: int = 6, dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"
        )

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 每个头的维度
        self.scale = self.head_dim ** -0.5  # 缩放因子 1/sqrt(d_k)，防止点积过大导致 softmax 梯度消失

        # 将 Q、K、V 的三个投影矩阵合并为一个 Linear 层，减少 GPU 内核调用
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)  # 多头拼接后的输出投影
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, N, D) 输入序列

        Returns:
            (B, N, D) 注意力聚合后的特征
        """
        B, N, D = x.shape

        # 1. QKV 投影 + 拆分为多头
        # (B, N, 3*D) -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # -> (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 计算 Scaled Dot-Product Attention
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # softmax 沿最后一个维度（key 维度），让每个 query 对所有 key 的权重和为 1
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 3. 加权聚合 Value
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = attn @ v
        # 合并多头 -> (B, N, D)
        x = x.transpose(1, 2).reshape(B, N, D)

        # 4. 输出投影
        x = self.proj(x)
        x = self.dropout(x)
        return x
