"""MLP — Transformer 中的前馈网络。

每个 Transformer 层在自注意力之后接一个两层的全连接网络。
第一层扩展到 hidden_dim * mlp_ratio，第二层还原回 hidden_dim。
使用 GELU 激活函数（比 ReLU 更平滑，在 ViT 中表现更好）。
"""

import torch.nn as nn


class MLP(nn.Module):
    """两层全连接前馈网络，GELU 激活 + Dropout 正则化。

    结构: Linear -> GELU -> Dropout -> Linear -> Dropout

    Args:
        hidden_dim: 隐藏层维度（即 Transformer 的 d_model）
        mlp_ratio: 中间层的扩展倍数（默认 4，即 384 -> 1536 -> 384）
        dropout: Dropout 概率
    """

    def __init__(self, hidden_dim: int = 384, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()

        inner_dim = hidden_dim * mlp_ratio  # 384 * 4 = 1536

        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.act = nn.GELU()  # Gaussian Error Linear Unit — ViT 标准激活
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播。

        Args:
            x: (B, N, D) 输入特征

        Returns:
            (B, N, D) 输出特征，shape 不变
        """
        x = self.fc1(x)       # (B, N, inner_dim)
        x = self.act(x)       # GELU 非线性
        x = self.dropout(x)   # 训练时随机失活，防止过拟合
        x = self.fc2(x)       # (B, N, D)
        x = self.dropout(x)
        return x
