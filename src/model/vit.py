"""Vision Transformer (ViT) — 完整模型。

将原始 ViT 论文（Dosovitskiy et al., 2021）适配到手写汉字识别场景：
- 灰度单通道输入（原论文为 RGB 三通道）
- 3755 类中文汉字分类（原论文为 ImageNet 1000 类）
- ViT-Small 规模（12 层、384 维、6 头），适合 8GB 显存

组件组装顺序:
    图像 -> PatchEmbedding -> [CLS] + Position Encoding
         -> Transformer -> [CLS] token -> Classification Head -> 3755 类
"""

import torch
import torch.nn as nn

from .patch_embed import PatchEmbedding
from .transformer import Transformer


class VisionTransformer(nn.Module):
    """Vision Transformer 完整模型。

    将输入图像分割为 patch，经 Transformer 编码后
    使用 [CLS] token 的最终表示进行分类。

    Args:
        image_size: 输入图像尺寸（正方形，灰度图）
        patch_size: 每个 patch 的边长
        in_channels: 输入通道数（1 = 灰度）
        hidden_dim: Transformer 的隐藏维度
        num_layers: Encoder 层数
        num_heads: 注意力头数
        mlp_ratio: MLP 中间层扩展比
        num_classes: 分类类别数（3755 个汉字）
        dropout: Dropout 概率
    """

    def __init__(
        self,
        image_size: int = 112,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_ratio: int = 4,
        num_classes: int = 3755,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 计算 patch 数量
        num_patches = (image_size // patch_size) ** 2  # (112/4)^2 = 784

        # ---- 组件 1: Patch Embedding ----
        self.patch_embed = PatchEmbedding(in_channels, patch_size, hidden_dim)

        # ---- 组件 2: 可学习的 [CLS] Token ----
        # [CLS] 是 classification token 的缩写
        # 它不来自图像，而是模型自行学习的一个「汇总」向量
        # 经过 Transformer 后，[CLS] 聚合了全图信息，用于最终分类
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # ---- 组件 3: 可学习的位置编码 ----
        # 由于自注意力没有内置的序列顺序概念，
        # 必须显式地告诉每个 token 「它在图中的位置」
        # 这里使用可学习的位置编码（每个位置一个向量）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)

        # ---- 组件 4: Transformer 编码器 ----
        self.transformer = Transformer(
            num_layers, hidden_dim, num_heads, mlp_ratio, dropout
        )

        # ---- 组件 5: 分类头 ----
        # 取 [CLS] token 的最终表示，映射到 num_classes 维
        self.head = nn.Linear(hidden_dim, num_classes)

        # 参数初始化（Kaiming 初始化 + 截断正态）
        self._init_weights()

    def _init_weights(self):
        """使用截断正态分布初始化可学习参数。

        位置编码和 [CLS] token 使用较小的标准差 (0.02)，
        避免初始化值太大导致训练初期不稳定。
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module):
        """对 Linear 和 LayerNorm 进行标准初始化。

        Linear: 截断正态，偏置为零
        LayerNorm: 权重为 1，偏置为零
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取图像特征（[CLS] token 的最终表示）。

        此方法可用于需要图像特征嵌入的场景，
        例如 t-SNE 可视化、相似度检索等。

        Args:
            x: (B, C, H, W) 输入图像

        Returns:
            (B, hidden_dim) [CLS] token 特征向量
        """
        B = x.shape[0]

        # Step 1: 图像 -> patch tokens
        x = self.patch_embed(x)  # (B, 784, 384)

        # Step 2: 在前面拼接 [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 384)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 785, 384)

        # Step 3: 加上位置编码（学习得到的空间位置信息）
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Step 4: 通过 Transformer 编码
        x = self.transformer(x)  # (B, 785, 384)

        # Step 5: 取出 [CLS] token（序列的第一个位置）
        return x[:, 0]  # (B, 384)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 — 输出 3755 类的 logits。

        Args:
            x: (B, 1, 112, 112) 灰度图像张量

        Returns:
            (B, 3755) 类别 logits（未经 softmax）
        """
        features = self.forward_features(x)  # (B, 384)
        return self.head(features)  # (B, 3755)
