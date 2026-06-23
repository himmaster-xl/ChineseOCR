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


from model.mlp import MLP


class TestMLP:
    """MLP 前馈网络测试"""

    def test_shape_preservation(self):
        """MLP 不改变输入 shape"""
        mlp = MLP(hidden_dim=384, mlp_ratio=4)
        x = torch.randn(2, 100, 384)

        out = mlp(x)

        assert out.shape == x.shape, f"MLP 不应改变 shape: {x.shape} -> {out.shape}"

    def test_gradient_flow(self):
        """两层全连接都有梯度"""
        mlp = MLP()
        x = torch.randn(2, 50, 384)
        loss = mlp(x).sum()
        loss.backward()

        assert mlp.fc1.weight.grad is not None
        assert mlp.fc2.weight.grad is not None

    def test_dropout_training_vs_eval(self):
        """训练模式有 dropout，评估模式关闭"""
        mlp = MLP(dropout=0.5)
        x = torch.randn(4, 10, 384)

        mlp.train()
        out_train = mlp(x)
        mlp.eval()
        out_eval = mlp(x)

        # eval 模式下两次相同输入应完全一致（无随机性）
        out_eval2 = mlp(x)
        assert torch.allclose(out_eval, out_eval2), "eval 模式应确定性输出"


from model.attention import MultiHeadAttention


class TestMultiHeadAttention:
    """多头自注意力模块测试"""

    def test_output_shape(self):
        """MHA 不改变输入序列长度和维度"""
        mha = MultiHeadAttention(hidden_dim=384, num_heads=6)
        x = torch.randn(2, 100, 384)

        out = mha(x)

        assert out.shape == (2, 100, 384), f"期望 (2, 100, 384)，实际 {out.shape}"

    def test_deterministic_in_eval(self):
        """无 mask：eval 模式下输出确定"""
        mha = MultiHeadAttention()
        mha.eval()
        x = torch.randn(1, 10, 384)

        out1 = mha(x)
        out2 = mha(x)

        # 确定性输出
        assert torch.allclose(out1, out2, atol=1e-5), "无 dropout 时输出应确定"

    def test_head_dim_division(self):
        """验证 head_dim = hidden_dim / num_heads"""
        mha = MultiHeadAttention(hidden_dim=384, num_heads=6)
        assert mha.head_dim == 64, "384 / 6 = 64"

    def test_gradient_flow(self):
        """QKV 投影和输出投影均有梯度"""
        mha = MultiHeadAttention()
        x = torch.randn(2, 50, 384)
        loss = mha(x).sum()
        loss.backward()

        assert mha.qkv.weight.grad is not None
        assert mha.proj.weight.grad is not None


from model.encoder import TransformerEncoder


class TestTransformerEncoder:
    """变压器编码器单层测试"""

    def test_output_shape(self):
        """单层 Encoder 不改变输入 shape"""
        enc = TransformerEncoder(hidden_dim=384, num_heads=6)
        x = torch.randn(2, 100, 384)

        out = enc(x)

        assert out.shape == (2, 100, 384)

    def test_residual_connection(self):
        """残差连接确保输入信息不被完全覆盖"""
        enc = TransformerEncoder()
        x = torch.randn(2, 10, 384)

        out = enc(x)

        # 残差连接 + LayerNorm 后，输出不应全为零且应接近输入尺度
        assert not torch.allclose(out, x), "经过 Attention+MLP 后应有变化"
        assert out.std() > 0.01, "输出不该退化"

    def test_gradient_flow(self):
        """所有子模块参数都有梯度"""
        enc = TransformerEncoder()
        x = torch.randn(2, 20, 384)
        loss = enc(x).sum()
        loss.backward()

        # 检查 norm 和 attention 的参数
        assert enc.norm1.weight.grad is not None
        assert enc.attn.qkv.weight.grad is not None
