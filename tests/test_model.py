"""模型各模块的单元测试 — 验证输入输出 shape 和前向传播无报错"""
import torch
import pytest
import sys
sys.path.insert(0, 'src')

from model.patch_embed import PatchEmbedding


class TestPatchEmbedding:
    """PatchEmbedding 模块测试"""

    def test_output_shape(self):
        """验证输出 shape: (B, C, H, W) -> (B, (H/patch)*(W/patch), hidden_dim)"""
        pe = PatchEmbedding(in_channels=1, patch_size=4, hidden_dim=384)
        x = torch.randn(2, 1, 112, 112)  # 灰度图，batch=2

        out = pe(x)

        # 112/4 = 28, 28*28 = 784 patches
        assert out.shape == (2, 784, 384), f"期望 (2, 784, 384)，实际 {out.shape}"

    def test_different_input_size(self):
        """验证不同输入尺寸也能正确计算"""
        pe = PatchEmbedding(in_channels=3, patch_size=8, hidden_dim=512)
        x = torch.randn(1, 3, 224, 224)

        out = pe(x)

        # 224/8 = 28, 28*28 = 784 patches
        assert out.shape == (1, 784, 512), f"期望 (1, 784, 512)，实际 {out.shape}"

    def test_gradient_flow(self):
        """验证梯度能正常回传"""
        pe = PatchEmbedding()
        x = torch.randn(2, 1, 112, 112, requires_grad=False)
        out = pe(x)
        loss = out.sum()
        loss.backward()

        # 检查投影卷积权重有梯度
        assert pe.proj.weight.grad is not None, "参数应有梯度"
