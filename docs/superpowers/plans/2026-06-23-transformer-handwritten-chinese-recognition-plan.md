# Transformer 手写汉字识别 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 ViT-Small 的 3755 类离线手写汉字识别系统（训练 + 评估 + Gradio Demo）

**Architecture:** 纯 PyTorch 从头实现 ViT，模块化拆分为 PatchEmbedding → MultiHeadAttention → MLP → TransformerEncoder → Transformer → VisionTransformer，数据端 CASIA-HWDB1.1 经 GNT→HDF5 预处理后供训练使用

**Tech Stack:** Python 3.10, PyTorch 2.10+cu126, HDF5, Gradio, YAML

## Global Constraints

- 所有代码模块划分清晰、单一职责，文件控制在 200 行以内
- 每个函数/类有详细 docstring 注释
- 遵循 TDD：先写测试 → 确认失败 → 实现 → 确认通过
- 环境：conda `paddle`，CUDA 12.6，RTX 4060 8GB
- DRY, YAGNI, 仅覆盖 3755 类分类

---

### Task 1: 项目脚手架 — .gitignore / requirements.txt / 目录结构

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/model/__init__.py`
- Create: `configs/vit_small_hwdb.yaml`
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: 无
- Produces: 项目目录骨架，供所有后续任务使用

- [ ] **Step 1: 创建 .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/

# 数据（原始数据不提交）
data/raw/
data/processed/*.h5

# 训练输出
outputs/

# IDE
.vscode/
.idea/

# 环境
venv/
.conda/

# Jupyter
.ipynb_checkpoints/

# OS
Thumbs.db
.DS_Store
```

- [ ] **Step 2: 创建 requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
h5py>=3.8.0
pyyaml>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
gradio>=3.30.0
pillow>=9.5.0
```

- [ ] **Step 3: 创建所有 `__init__.py` 和核心 YAML 配置**

`configs/vit_small_hwdb.yaml`:
```yaml
# Model
model:
  image_size: 112
  patch_size: 4
  in_channels: 1
  hidden_dim: 384
  num_layers: 12
  num_heads: 6
  mlp_ratio: 4
  num_classes: 3755
  dropout: 0.1

# Training
train:
  batch_size: 128
  epochs: 100
  lr: 0.0003
  weight_decay: 0.05
  warmup_epochs: 10
  label_smoothing: 0.1
  grad_accum_steps: 1

# Data
data:
  hdf5_path: "data/processed/hwdb.h5"
  num_workers: 4
```

- [ ] **Step 4: 创建目录结构**

```bash
mkdir -p src/data src/model configs tests demo outputs/checkpoints outputs/logs data/raw data/processed
```

- [ ] **Step 5: 验证：列出所有文件并初始化版本提交**

```bash
ls -R src/ configs/ tests/
git add -A && git commit -m "feat: scaffold project structure"
```

---

### Task 2: 配置系统 — Config 类

**Files:**
- Create: `src/config.py`

**Interfaces:**
- Consumes: 无
- Produces: `Config.from_yaml(path: str) -> Config` — 返回包含所有配置字段的 dataclass 实例
- Produces: `Config` 字段: `image_size: int, patch_size: int, in_channels: int, hidden_dim: int, num_layers: int, num_heads: int, mlp_ratio: int, num_classes: int, dropout: float, batch_size: int, epochs: int, lr: float, weight_decay: float, warmup_epochs: int, label_smoothing: float, grad_accum_steps: int, hdf5_path: str, num_workers: int`

- [ ] **Step 1: 写配置类的测试**

`src/config.py` 文件尾添加 `__main__` 自测块（简单验证即可）：

```python
# 在文件末尾
if __name__ == "__main__":
    import tempfile, os
    yaml_content = """
    model:
      image_size: 112
      patch_size: 4
      in_channels: 1
      hidden_dim: 384
      num_layers: 12
      num_heads: 6
      mlp_ratio: 4
      num_classes: 3755
      dropout: 0.1
    train:
      batch_size: 128
      epochs: 100
      lr: 0.0003
      weight_decay: 0.05
      warmup_epochs: 10
      label_smoothing: 0.1
      grad_accum_steps: 1
    data:
      hdf5_path: "data/processed/test.h5"
      num_workers: 4
    """
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    tmp.write(yaml_content)
    tmp.close()
    
    cfg = Config.from_yaml(tmp.name)
    assert cfg.model.hidden_dim == 384
    assert cfg.train.lr == 0.0003
    assert cfg.data.hdf5_path == "data/processed/test.h5"
    print("All config tests passed!")
    os.unlink(tmp.name)
```

- [ ] **Step 2: 运行自测验证失败**

```bash
conda run -n paddle python src/config.py
```
期望: `NameError: name 'Config' is not defined`

- [ ] **Step 3: 实现 Config 类**

```python
"""配置系统 — 从 YAML 文件解析并结构化存储所有超参数。

使用方式:
    cfg = Config.from_yaml("configs/vit_small_hwdb.yaml")
    print(cfg.model.hidden_dim)  # 384
"""

from dataclasses import dataclass
import yaml


@dataclass
class ModelConfig:
    """模型架构超参数"""
    image_size: int
    patch_size: int
    in_channels: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_ratio: int
    num_classes: int
    dropout: float


@dataclass
class TrainConfig:
    """训练过程超参数"""
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    label_smoothing: float
    grad_accum_steps: int


@dataclass
class DataConfig:
    """数据路径与加载超参数"""
    hdf5_path: str
    num_workers: int


@dataclass
class Config:
    """总的配置容器，组合模型、训练、数据三个子配置。

    使用 YAML 文件实例化:
        cfg = Config.from_yaml("path/to/config.yaml")
    """
    model: ModelConfig
    train: TrainConfig
    data: DataConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 配置文件加载所有配置。

        Args:
            path: YAML 配置文件路径

        Returns:
            Config 实例，包含 model / train / data 三个子配置
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw["train"]),
            data=DataConfig(**raw["data"]),
        )
```

- [ ] **Step 4: 运行自测验证通过**

```bash
conda run -n paddle python src/config.py
```
期望: `All config tests passed!`

- [ ] **Step 5: 提交**

```bash
git add src/config.py && git commit -m "feat: add config system with YAML parsing"
```

---

### Task 3: 工具函数 — set_seed / EarlyStopping / save_checkpoint

**Files:**
- Create: `src/utils.py`

**Interfaces:**
- Consumes: 无
- Produces: `set_seed(seed: int = 42) -> None`
- Produces: `EarlyStopping(patience: int = 10, mode: str = 'max')` — `__call__(score: float) -> bool` 返回 True 时停止
- Produces: `save_checkpoint(model, optimizer, epoch: int, loss: float, path: str) -> None`

- [ ] **Step 1: 写测试**

```python
# 文件末尾自测
if __name__ == "__main__":
    import torch, tempfile, os
    
    # 测试 set_seed — 两次随机结果应相同
    set_seed(42)
    a = torch.randn(3, 3)
    set_seed(42)
    b = torch.randn(3, 3)
    assert torch.allclose(a, b), "set_seed: 可重复性验证失败"
    print("✓ set_seed works")

    # 测试 EarlyStopping — 连续下降应触发停止
    es = EarlyStopping(patience=3, mode='max')
    assert not es(0.9)  # 首次，best_score=0.9
    assert not es(0.8)  # 未改善, counter=1
    assert not es(0.85) # 未改善, counter=2
    assert es(0.8)      # 连续 3 次未改善 → 应停止
    print("✓ EarlyStopping works")

    # 测试 save_checkpoint — 文件存在且可加载
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    tmp.close()
    save_checkpoint(model, opt, epoch=5, loss=0.5, path=tmp.name)
    ckpt = torch.load(tmp.name, map_location='cpu')
    assert ckpt['epoch'] == 5
    assert abs(ckpt['loss'] - 0.5) < 1e-6
    print("✓ save_checkpoint works")
    os.unlink(tmp.name)
```

- [ ] **Step 2: 运行自测验证失败**

```bash
conda run -n paddle python src/utils.py
```
期望: `NameError: name 'set_seed' is not defined`

- [ ] **Step 3: 实现工具函数**

```python
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
```

- [ ] **Step 4: 运行自测验证通过**

```bash
conda run -n paddle python src/utils.py
```
期望: 三行 ✓ 输出

- [ ] **Step 5: 提交**

```bash
git add src/utils.py && git commit -m "feat: add utility functions — seed, early stopping, checkpoint"
```

---

### Task 4: Patch Embedding 模块

**Files:**
- Create: `tests/test_model.py`
- Create: `src/model/patch_embed.py`

**Interfaces:**
- Consumes: 无
- Produces: `PatchEmbedding(in_channels=1, patch_size=4, hidden_dim=384)` — 输入 `(B, C, H, W)` 返回 `(B, N_patches, hidden_dim)`

- [ ] **Step 1: 写测试**

`tests/test_model.py`:
```python
"""模型各模块的单元测试 — 验证输入输出 shape 和前向传播无报错"""
import torch
import pytest
import sys
sys.path.insert(0, 'src')

from model.patch_embed import PatchEmbedding


class TestPatchEmbedding:
    """PatchEmbedding 模块测试"""

    def test_output_shape(self):
        """验证输出 shape: (B, C, H, W) → (B, (H/patch)*(W/patch), hidden_dim)"""
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
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n paddle python -m pytest tests/test_model.py -v
```
期望: `ModuleNotFoundError: No module named 'model.patch_embed'`

- [ ] **Step 3: 实现 PatchEmbedding**

```python
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
        # (B, C, H, W) → (B, hidden_dim, H/patch, W/patch)
        x = self.proj(x)
        # 展平空间维度 → (B, hidden_dim, N_patches)
        x = x.flatten(2)
        # 转换为 Transformer 期望的 (B, N_patches, hidden_dim)
        x = x.transpose(1, 2)
        return x
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n paddle python -m pytest tests/test_model.py -v
```
期望: 3 tests passed

- [ ] **Step 5: 提交**

```bash
git add src/model/patch_embed.py tests/test_model.py && git commit -m "feat: implement PatchEmbedding module"
```

---

### Task 5: MLP 前馈网络

**Files:**
- Modify: `tests/test_model.py` (追加 TestMLP 类)
- Create: `src/model/mlp.py`

**Interfaces:**
- Consumes: 无
- Produces: `MLP(hidden_dim=384, mlp_ratio=4, dropout=0.1)` — 输入 `(B, N, D)` 返回 `(B, N, D)`

- [ ] **Step 1: 追加测试**

在 `tests/test_model.py` 末尾追加：

```python
from model.mlp import MLP


class TestMLP:
    """MLP 前馈网络测试"""

    def test_shape_preservation(self):
        """MLP 不改变输入 shape"""
        mlp = MLP(hidden_dim=384, mlp_ratio=4)
        x = torch.randn(2, 100, 384)

        out = mlp(x)

        assert out.shape == x.shape, f"MLP 不应改变 shape: {x.shape} → {out.shape}"

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
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n paddle python -m pytest tests/test_model.py::TestMLP -v
```
期望: `ModuleNotFoundError: No module named 'model.mlp'`

- [ ] **Step 3: 实现 MLP**

```python
"""MLP — Transformer 中的前馈网络。

每个 Transformer 层在自注意力之后接一个两层的全连接网络。
第一层扩展到 hidden_dim * mlp_ratio，第二层还原回 hidden_dim。
使用 GELU 激活函数（比 ReLU 更平滑，在 ViT 中表现更好）。
"""

import torch.nn as nn


class MLP(nn.Module):
    """两层全连接前馈网络，GELU 激活 + Dropout 正则化。

    结构: Linear → GELU → Dropout → Linear → Dropout

    Args:
        hidden_dim: 隐藏层维度（即 Transformer 的 d_model）
        mlp_ratio: 中间层的扩展倍数（默认 4，即 384→1536→384）
        dropout: Dropout 概率
    """

    def __init__(self, hidden_dim: int = 384, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()

        inner_dim = hidden_dim * mlp_ratio  # 384 * 4 = 1536

        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.act = nn.GELU()  # Gaussian Error Linear Unit — ViT 标准激活
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n paddle python -m pytest tests/test_model.py::TestMLP -v
```
期望: 3 tests passed

- [ ] **Step 5: 提交**

```bash
git add src/model/mlp.py tests/test_model.py && git commit -m "feat: implement MLP feed-forward network"
```

---

### Task 6: Multi-Head Self-Attention

**Files:**
- Modify: `tests/test_model.py` (追加 TestMultiHeadAttention 类)
- Create: `src/model/attention.py`

**Interfaces:**
- Consumes: 无
- Produces: `MultiHeadAttention(hidden_dim=384, num_heads=6, dropout=0.1)` — 输入 `(B, N, D)` 返回 `(B, N, D)`

- [ ] **Step 1: 追加测试**

```python
from model.attention import MultiHeadAttention


class TestMultiHeadAttention:
    """多头自注意力模块测试"""

    def test_output_shape(self):
        """MHA 不改变输入序列长度和维度"""
        mha = MultiHeadAttention(hidden_dim=384, num_heads=6)
        x = torch.randn(2, 100, 384)

        out = mha(x)

        assert out.shape == (2, 100, 384), f"期望 (2, 100, 384)，实际 {out.shape}"

    def test_causal_is_not_applied(self):
        """无 mask：每个 token 能看到所有其他 token（包括未来 token）"""
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
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n paddle python -m pytest tests/test_model.py::TestMultiHeadAttention -v
```

- [ ] **Step 3: 实现 MultiHeadAttention**

```python
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
        Attention(Q, K, V) = softmax(QK^T / √d_k) · V

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
        self.scale = self.head_dim ** -0.5  # 缩放因子 1/√d_k，防止点积过大导致 softmax 梯度消失

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
        # (B, N, 3*D) → (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # → (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 计算 Scaled Dot-Product Attention
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) → (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # softmax 沿最后一个维度（key 维度），让每个 query 对所有 key 的权重和为 1
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 3. 加权聚合 Value
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) → (B, num_heads, N, head_dim)
        x = attn @ v
        # 合并多头 → (B, N, D)
        x = x.transpose(1, 2).reshape(B, N, D)

        # 4. 输出投影
        x = self.proj(x)
        x = self.dropout(x)
        return x
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n paddle python -m pytest tests/test_model.py::TestMultiHeadAttention -v
```
期望: 4 tests passed

- [ ] **Step 5: 提交**

```bash
git add src/model/attention.py tests/test_model.py && git commit -m "feat: implement Multi-Head Self-Attention"
```

---

### Task 7: Transformer Encoder 层

**Files:**
- Modify: `tests/test_model.py` (追加 TestTransformerEncoder 类)
- Create: `src/model/encoder.py`

**Interfaces:**
- Consumes: `MultiHeadAttention` (Task 6), `MLP` (Task 5)
- Produces: `TransformerEncoder(hidden_dim=384, num_heads=6, mlp_ratio=4, dropout=0.1)` — 输入 `(B, N, D)` 返回 `(B, N, D)`

- [ ] **Step 1: 追加测试**

```python
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
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n paddle python -m pytest tests/test_model.py::TestTransformerEncoder -v
```

- [ ] **Step 3: 实现 TransformerEncoder**

```python
"""Transformer Encoder Layer — 自注意力 + 前馈网络的残差组合。

每个 Encoder 层包含两个子层（Pre-LN 架构，ViT 标准）：
1. LayerNorm → Multi-Head Self-Attention → 残差连接
2. LayerNorm → MLP → 残差连接

Pre-LN（先归一化再计算）比 Post-LN 训练更稳定，不需要学习率 warmup。
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n paddle python -m pytest tests/test_model.py::TestTransformerEncoder -v
```
期望: 3 tests passed

- [ ] **Step 5: 提交**

```bash
git add src/model/encoder.py tests/test_model.py && git commit -m "feat: implement Transformer Encoder layer"
```

---

### Task 8: Transformer（堆叠 Encoder）

**Files:**
- Modify: `tests/test_model.py` (追加 TestTransformer 类)
- Create: `src/model/transformer.py`

**Interfaces:**
- Consumes: `TransformerEncoder` (Task 7)
- Produces: `Transformer(num_layers=12, hidden_dim=384, num_heads=6, mlp_ratio=4, dropout=0.1)` — 输入 `(B, N, D)` 返回 `(B, N, D)`

- [ ] **Step 1: 追加测试**

```python
from model.transformer import Transformer


class TestTransformer:
    """堆叠 Transformer 测试"""

    def test_output_shape(self):
        """12 层 Transformer 不改变 shape"""
        transformer = Transformer(num_layers=12, hidden_dim=384, num_heads=6)
        x = torch.randn(2, 785, 384)  # cls_token + 784 patches

        out = transformer(x)

        assert out.shape == (2, 785, 384)

    def test_final_layernorm(self):
        """最后一层 LayerNorm 确保输出标准化"""
        transformer = Transformer(num_layers=2)  # 2层便于测试
        x = torch.randn(1, 10, 384)
        out = transformer(x)

        # LayerNorm 后均值接近 0，方差接近 1
        assert out.mean().abs() < 0.5

    def test_gradient_flows_through_all_layers(self):
        """梯度能通过所有 12 层回传"""
        transformer = Transformer(num_layers=3, hidden_dim=64, num_heads=2)
        x = torch.randn(1, 10, 64)
        loss = transformer(x).sum()
        loss.backward()

        grad_norms = [
            p.grad.norm().item()
            for p in transformer.parameters()
            if p.grad is not None
        ]
        # 所有参数的梯度都应非零
        assert all(g > 0 for g in grad_norms), "所有层都应有梯度"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n paddle python -m pytest tests/test_model.py::TestTransformer -v
```

- [ ] **Step 3: 实现 Transformer**

```python
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n paddle python -m pytest tests/test_model.py::TestTransformer -v
```
期望: 3 tests passed

- [ ] **Step 5: 提交**

```bash
git add src/model/transformer.py tests/test_model.py && git commit -m "feat: implement stacked Transformer Encoder"
```

---

### Task 9: Vision Transformer 完整模型

**Files:**
- Modify: `tests/test_model.py` (追加 TestVisionTransformer 类)
- Create: `src/model/vit.py`

**Interfaces:**
- Consumes: `PatchEmbedding` (Task 4), `Transformer` (Task 8)
- Produces: `VisionTransformer(image_size=112, patch_size=4, in_channels=1, hidden_dim=384, num_layers=12, num_heads=6, mlp_ratio=4, num_classes=3755, dropout=0.1)`
  - 输入 `(B, 1, 112, 112)` 返回 `(B, 3755)`
  - 额外暴露 `forward_features(x)` 方法返回 [CLS] 特征向量 `(B, 384)`

- [ ] **Step 1: 追加测试**

```python
from model.vit import VisionTransformer


class TestVisionTransformer:
    """完整 ViT 模型测试"""

    @pytest.fixture
    def model(self):
        """创建 ViT-Small 实例用于测试"""
        return VisionTransformer(
            image_size=112, patch_size=4, in_channels=1,
            hidden_dim=384, num_layers=12, num_heads=6,
            num_classes=3755,
        )

    def test_output_shape(self, model):
        """输入 (B, 1, 112, 112) → 输出 (B, 3755)"""
        x = torch.randn(2, 1, 112, 112)
        out = model(x)
        assert out.shape == (2, 3755), f"期望 (2, 3755)，实际 {out.shape}"

    def test_forward_features(self, model):
        """forward_features 返回 [CLS] token (B, 384)"""
        x = torch.randn(2, 1, 112, 112)
        features = model.forward_features(x)
        assert features.shape == (2, 384), f"期望 (2, 384)，实际 {features.shape}"

    def test_cls_token_is_learned(self, model):
        """cls_token 是可学习参数"""
        assert model.cls_token.requires_grad, "cls_token 应可学习"
        assert model.cls_token.shape == (1, 1, 384)

    def test_pos_embed_shape(self, model):
        """位置编码覆盖所有 patch + [CLS]"""
        expected_num = (112 // 4) ** 2 + 1  # 784 + 1 = 785
        assert model.pos_embed.shape == (1, expected_num, 384)

    def test_gradient_flow_full(self, model):
        """端到端梯度回传正常"""
        x = torch.randn(1, 1, 112, 112)
        loss = model(x).sum()
        loss.backward()

        # 检查分类头的梯度
        assert model.head.weight.grad is not None
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n paddle python -m pytest tests/test_model.py::TestVisionTransformer -v
```

- [ ] **Step 3: 实现 ViT**

```python
"""Vision Transformer (ViT) — 完整模型。

将原始 ViT 论文（Dosovitskiy et al., 2021）适配到手写汉字识别场景：
- 灰度单通道输入（原论文为 RGB 三通道）
- 3755 类中文汉字分类（原论文为 ImageNet 1000 类）
- ViT-Small 规模（12 层、384 维、6 头），适合 8GB 显存

组件组装顺序:
    图像 → PatchEmbedding → [CLS] + Position Encoding
         → Transformer → [CLS] token → Classification Head → 3755 类
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

        # Step 1: 图像 → patch tokens
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
```

- [ ] **Step 4: 运行全部模型测试**

```bash
conda run -n paddle python -m pytest tests/test_model.py -v
```
期望: `18 passed` （累计全部 18 个测试）

- [ ] **Step 5: 验证设备兼容性**

```bash
conda run -n paddle python -c "
import torch
from src.model.vit import VisionTransformer
model = VisionTransformer()
x = torch.randn(4, 1, 112, 112)
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    out = model(x)
    print(f'GPU output shape: {out.shape}')
    print(f'GPU VRAM used: {torch.cuda.memory_allocated() // 1024**2} MB')
else:
    out = model(x)
    print(f'CPU output shape: {out.shape}')
model.eval()
with torch.no_grad():
    out = model(x[:1])
    print(f'Inference shape: {out.shape}')
"
```

- [ ] **Step 6: 提交**

```bash
git add src/model/vit.py tests/test_model.py && git commit -m "feat: implement complete Vision Transformer model"
```

---

### Task 10: 数据增强模块

**Files:**
- Create: `tests/test_dataset.py`
- Create: `src/data/transforms.py`

**Interfaces:**
- Consumes: 无
- Produces: `get_train_transforms(image_size=112) -> transforms.Compose`
- Produces: `get_val_transforms(image_size=112) -> transforms.Compose`

- [ ] **Step 1: 写测试**

`tests/test_dataset.py`:
```python
"""数据模块测试"""
import numpy as np
import torch
import sys
sys.path.insert(0, 'src')

from data.transforms import get_train_transforms, get_val_transforms


class TestTransforms:
    """数据增强管道测试"""

    def test_train_transforms_output_shape(self):
        """训练增强输出 (1, 112, 112) 的标准化张量"""
        transform = get_train_transforms(image_size=112)
        # 模拟 HDF5 读出的 uint8 图像 (112, 112)
        img = np.random.randint(0, 255, (112, 112), dtype=np.uint8)

        out = transform(img)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 112, 112), f"期望 (1, 112, 112)，实际 {out.shape}"

    def test_val_transforms_deterministic(self):
        """验证集增强无随机性（仅 resize + normalize）"""
        transform = get_val_transforms(image_size=112)
        img = np.random.randint(0, 255, (112, 112), dtype=np.uint8)

        out1 = transform(img)
        out2 = transform(img)

        assert torch.allclose(out1, out2), "验证集变换应确定性输出"

    def test_normalize_range(self):
        """标准化后值应在合理范围内（均值 0.5，标准差 0.5）"""
        transform = get_val_transforms(image_size=112)
        img = np.ones((112, 112), dtype=np.uint8) * 128  # 中灰

        out = transform(img)

        # mean=0.5, std=0.5 → 128 → (128/255 - 0.5) / 0.5 ≈ 0.0039
        assert out.mean().abs() < 0.1
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n paddle python -m pytest tests/test_dataset.py -v
```

- [ ] **Step 3: 实现 transforms.py**

```python
"""数据增强 — 训练集在线增强 + 验证集归一化。

手写汉字的类内差异大（不同作者书写风格不同），
适当的数据增强可以提高模型的泛化能力。

增强策略说明:
- RandomRotation: 容忍不同程度的手写倾斜
- RandomAffine(translate): 容忍字符位置偏移
- RandomAffine(scale): 容忍字符大小变化

归一化: (pixel/255 - 0.5) / 0.5 将像素值映射到 [-1, 1]
（灰度图像的常见预处理方式）
"""

from torchvision import transforms


def get_train_transforms(image_size: int = 112):
    """获取训练集的数据增强管道。

    包含随机旋转、平移、缩放以增加数据多样性。
    所有操作针对手写汉字特点设计了保守的范围，
    避免过度增强导致字形失真。

    Args:
        image_size: 输出图像的边长（像素）

    Returns:
        torchvision.transforms.Compose 组合变换
    """
    return transforms.Compose([
        # numpy ndarray → PIL Image（torchvision transforms 的输入要求）
        transforms.ToPILImage(),
        # 随机旋转 ±5° — 模拟手写倾斜
        transforms.RandomRotation(degrees=5),
        # 随机平移 ±5% + 缩放 95%~105% — 模拟位置和大小差异
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        # 统一尺寸
        transforms.Resize((image_size, image_size)),
        # PIL → Tensor (0~1 浮点数)
        transforms.ToTensor(),
        # 标准化到 [-1, 1]
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_val_transforms(image_size: int = 112):
    """获取验证集/测试集的预处理管道（无随机增强）。

    Args:
        image_size: 输出图像的边长（像素）

    Returns:
        torchvision.transforms.Compose 组合变换
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n paddle python -m pytest tests/test_dataset.py -v
```
期望: 3 tests passed

- [ ] **Step 5: 提交**

```bash
git add src/data/transforms.py tests/test_dataset.py && git commit -m "feat: add data augmentation transforms"
```

---

### Task 11: HDF5 Dataset 类

**Files:**
- Modify: `tests/test_dataset.py` (追加 TestHDF5Dataset 类)
- Create: `src/data/dataset.py`

**Interfaces:**
- Consumes: `get_train_transforms`, `get_val_transforms` (Task 10)
- Produces: `HDF5Dataset(hdf5_path, split='train', transform=None)` — 实现 `__len__` 和 `__getitem__`，返回 `(image, label)`

- [ ] **Step 1: 追加测试**

```python
import h5py, tempfile, os
from data.dataset import HDF5Dataset


class TestHDF5Dataset:
    """HDF5 数据集类测试"""

    @pytest.fixture
    def sample_hdf5(self):
        """创建包含模拟数据的临时 HDF5 文件"""
        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        tmp.close()

        with h5py.File(tmp.name, 'w') as f:
            # 训练集: 100 张 112x112 的 uint8 图像
            f.create_dataset('train/images', data=np.random.randint(
                0, 255, (100, 112, 112), dtype=np.uint8
            ))
            f.create_dataset('train/labels', data=np.random.randint(
                0, 3755, (100,), dtype=np.int64
            ))
            # 验证集: 20 张
            f.create_dataset('val/images', data=np.random.randint(
                0, 255, (20, 112, 112), dtype=np.uint8
            ))
            f.create_dataset('val/labels', data=np.random.randint(
                0, 3755, (20,), dtype=np.int64
            ))

        yield tmp.name
        os.unlink(tmp.name)

    def test_length(self, sample_hdf5):
        """len(dataset) 应返回样本数"""
        ds = HDF5Dataset(sample_hdf5, split='train')
        assert len(ds) == 100

    def test_getitem_output(self, sample_hdf5):
        """__getitem__ 返回 (Tensor, int)"""
        ds = HDF5Dataset(sample_hdf5, split='train', transform=get_val_transforms())
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, (int, np.integer))
        assert img.shape == (1, 112, 112)

    def test_different_splits(self, sample_hdf5):
        """不同 split 返回不同数据"""
        train_ds = HDF5Dataset(sample_hdf5, split='train')
        val_ds = HDF5Dataset(sample_hdf5, split='val')
        assert len(train_ds) == 100
        assert len(val_ds) == 20

    def test_close(self, sample_hdf5):
        """close() 方法正常关闭 HDF5 文件"""
        ds = HDF5Dataset(sample_hdf5, split='train')
        ds.close()
        # 关闭后 file.id 应无效
        assert not ds.file.id
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n paddle python -m pytest tests/test_dataset.py::TestHDF5Dataset -v
```

- [ ] **Step 3: 实现 HDF5Dataset**

```python
"""HDF5 数据集 — 封装 CASIA-HWDB1.1 预处理为 HDF5 后的高效读取。

为什么用 HDF5 而不是直接从 PNG 文件夹加载：
1. 240 万个小文件直接从磁盘读取极慢（IO 瓶颈）
2. HDF5 支持分块存储和压缩，加载速度快 10-100 倍
3. 不需要额外的标签文件映射，一张 HDF5 文件包含全部数据

HDF5 内部结构:
    /train/images      (N_train, 112, 112) uint8
    /train/labels      (N_train,)          int64
    /val/images        (N_val, 112, 112)   uint8
    /val/labels        (N_val,)            int64
    /test/images       (N_test, 112, 112)  uint8
    /test/labels       (N_test,)           int64
"""

import h5py
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """从 HDF5 文件读取手写汉字数据集的 PyTorch Dataset。

    支持训练/验证/测试三种 split，每个 split 有独立的 images 和 labels。
    读取时按需加载（通过 h5py 的延迟读取机制），不占满内存。

    Args:
        hdf5_path: HDF5 文件的完整路径
        split: 数据子集名称，'train' / 'val' / 'test'
        transform: torchvision 变换管道，None 则返回原始 uint8 numpy
    """

    def __init__(self, hdf5_path: str, split: str = "train", transform=None):
        # 打开文件但不立即读取所有数据（h5py 默认延迟加载）
        self.file = h5py.File(hdf5_path, "r")
        self.images = self.file[f"{split}/images"]  # shape: (N, H, W)
        self.labels = self.file[f"{split}/labels"]  # shape: (N,)
        self.transform = transform

    def __len__(self) -> int:
        """返回数据集中的样本总数。"""
        return len(self.labels)

    def __getitem__(self, idx: int):
        """根据索引获取单个样本。

        HDF5 支持按索引切片读取，只从磁盘加载需要的部分。

        Args:
            idx: 样本索引 (0 ≤ idx < len(dataset))

        Returns:
            (image, label) 元组:
                - image: torch.Tensor 或 numpy.ndarray
                - label: int
        """
        # h5py 支持整数索引，直接读取单张图像
        img = self.images[idx]  # (H, W) numpy uint8
        label = int(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def close(self):
        """关闭 HDF5 文件句柄，释放资源。"""
        self.file.close()
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n paddle python -m pytest tests/test_dataset.py -v
```
期望: 7 tests passed（含之前的 transforms 3 个）

- [ ] **Step 5: 提交**

```bash
git add src/data/dataset.py tests/test_dataset.py && git commit -m "feat: implement HDF5 dataset loader"
```

---

### Task 12: 数据预处理脚本 — GNT → HDF5

**Files:**
- Create: `src/data/preprocess.py`
- Create: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: 无（独立脚本）
- Produces (CLI): `python -m src.data.preprocess --raw_dir data/raw --output data/processed/hwdb.h5`

- [ ] **Step 1: 写轻量测试**

`tests/test_preprocess.py`:
```python
"""预处理模块测试"""
import struct, tempfile, os, numpy as np, h5py, sys
sys.path.insert(0, 'src')

from data.preprocess import parse_gnt_file, create_hdf5


class TestGNTParser:
    """GNT 二进制格式解析测试"""

    def test_parse_single_sample(self):
        """解析一个最小 GNT 样本"""
        # 构造一个虚拟 GNT 样本
        label = "啊"  # GB2312 编码为 \xb0\xa1
        tag_code = label.encode("gb2312-80")
        # 模拟一张 2x2 的灰度图
        bitmap = bytes([0, 255, 128, 64])  # 4 字节
        width, height = 2, 2

        # GNT 格式: [样本大小(4B)] + [标签(2B)] + [宽(2B)] + [高(2B)] + [位图数据]
        sample_data = struct.pack("<IH2B", 8, tag_code, width, height) + bitmap
        # GNT 文件格式: [样本数(4B)] + 样本1 + 样本2 + ...
        file_data = struct.pack("<I", 1) + sample_data

        tmp = tempfile.NamedTemporaryFile(suffix='.gnt', delete=False)
        tmp.write(file_data)
        tmp.close()

        images, labels = parse_gnt_file(tmp.name)
        os.unlink(tmp.name)

        assert len(images) == 1
        assert labels[0] == "啊"
        assert images[0].shape == (2, 2)

    def test_create_hdf5(self):
        """HDF5 创建和结构验证"""
        images = [np.random.randint(0, 255, (112, 112), dtype=np.uint8) for _ in range(10)]
        labels = [f"char_{i}" for i in range(10)]

        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        tmp.close()

        create_hdf5(images, labels, tmp.name, split='train', resize_to=112)

        with h5py.File(tmp.name, 'r') as f:
            assert 'train/images' in f
            assert f['train/images'].shape == (10, 112, 112)
            assert len(f['train/labels']) == 10

        os.unlink(tmp.name)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda run -n paddle python -m pytest tests/test_preprocess.py -v
```

- [ ] **Step 3: 实现 preprocess.py**

```python
"""数据预处理 — 将 CASIA-HWDB1.1 的 GNT 格式转换为训练友好的 HDF5。

CASIA-HWDB1.1 GNT 文件格式:
    [样本数量(4B, uint32)]
    样本1: [样本大小(4B)] [标签(2B GB2312)] [宽度(2B)] [高度(2B)] [位图(宽*高 字节)]
    样本2: ...

HDF5 输出格式:
    /train/images     (N, 112, 112) uint8
    /train/labels     (N,) int64 (0~3754)
    /val/images       (M, 112, 112) uint8
    /val/labels       (M,) int64
    /test/images      (K, 112, 112) uint8
    /test/labels      (K,) int64

用法:
    python -m src.data.preprocess --raw_dir data/raw --output data/processed/hwdb.h5
"""

import struct
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image


# GB2312 一级字库 3755 汉字 → 整数索引映射
# 通过 CASIA 提供的标签映射文件构建（运行时从数据中动态构建）
def build_label_map(gnt_files: list) -> dict:
    """扫描所有 GNT 文件，构建「GB2312编码→类别索引」的映射表。

    Args:
        gnt_files: GNT 文件路径列表

    Returns:
        dict: {gb2312_tag_code: class_index}，按首次出现顺序编号
    """
    tag_set = set()
    for gnt_path in tqdm(gnt_files, desc="扫描标签"):
        with open(gnt_path, "rb") as f:
            num_samples = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_samples):
                size = struct.unpack("<I", f.read(4))[0]
                tag_code = f.read(2)  # GB2312 双字节编码
                # 跳过宽高 + 位图
                f.read(2 + 2 + size - 2)
                tag_set.add(tag_code)

    # 按 GB2312 编码排序编号
    sorted_tags = sorted(tag_set)
    return {tag: idx for idx, tag in enumerate(sorted_tags)}


def parse_gnt_file(gnt_path: str) -> tuple[list[np.ndarray], list[str]]:
    """解析单个 GNT 文件，返回 (图像列表, 标签字符串列表)。

    Args:
        gnt_path: GNT 文件路径

    Returns:
        images: numpy uint8 数组列表，每个形状为 (H, W)
        labels: GB2312 解码后的汉字字符串列表
    """
    images = []
    labels = []

    with open(gnt_path, "rb") as f:
        num_samples = struct.unpack("<I", f.read(4))[0]

        for _ in range(num_samples):
            # 读取样本头
            sample_size = struct.unpack("<I", f.read(4))[0]
            tag_code = f.read(2)
            width = struct.unpack("<H", f.read(2))[0]
            height = struct.unpack("<H", f.read(2))[0]

            # 读取位图数据
            bitmap_data = f.read(width * height)

            # 将 1D 位图重塑为 2D 图像
            img = np.frombuffer(bitmap_data, dtype=np.uint8).reshape(height, width)

            # GB2312 解码
            label = tag_code.decode("gb2312-80", errors="replace")

            images.append(img)
            labels.append(label)

    return images, labels


def create_hdf5(
    images: list[np.ndarray],
    labels: list[str],
    label_map: dict,
    class_indices: list[int],
    output_path: str,
    split: str = "train",
    resize_to: int = 112,
):
    """将图像和标签写入 HDF5 文件。

    所有图像统一 resize 到 (resize_to, resize_to) 并转为 uint8。
    标签转为 class_indices 中的整数索引。

    Args:
        images: numpy 图像数组列表
        labels: GB2312 标签字符串列表
        label_map: GB2312 编码 → 类别索引的映射
        class_indices: 保存时使用的类别索引（应与 images 一一对应，由调用方提前映射）
        output_path: 输出 HDF5 文件路径
        split: 数据集子集名称
        resize_to: 统一缩放的尺寸
    """
    # 预处理所有图像到统一尺寸
    processed = []
    for img in tqdm(images, desc=f"Resize {split}"):
        # PIL 处理 resize 比 numpy 快
        pil_img = Image.fromarray(img, mode="L")
        pil_img = pil_img.resize((resize_to, resize_to), Image.BILINEAR)
        processed.append(np.array(pil_img, dtype=np.uint8))

    images_array = np.stack(processed, axis=0)  # (N, H, W)
    labels_array = np.array(class_indices, dtype=np.int64)

    # 写入 HDF5（追加模式，不覆盖已有 split）
    with h5py.File(output_path, "a") as f:
        if split in f:
            del f[split]  # 覆盖已有 split
        grp = f.create_group(split)
        grp.create_dataset("images", data=images_array, compression="gzip", compression_opts=4)
        grp.create_dataset("labels", data=labels_array)

    print(f"已写入 {split}: {len(processed)} 样本 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CASIA-HWDB1.1 GNT → HDF5 预处理")
    parser.add_argument("--raw_dir", type=str, required=True, help="原始 GNT 文件目录")
    parser.add_argument("--output", type=str, required=True, help="输出 HDF5 文件路径")
    parser.add_argument("--image_size", type=int, default=112, help="统一缩放尺寸")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    gnt_files = list(raw_dir.rglob("*.gnt"))
    if not gnt_files:
        print(f"警告: 在 {raw_dir} 中未找到 .gnt 文件")
        return
    print(f"找到 {len(gnt_files)} 个 GNT 文件")

    # 第一步：扫描所有标签构建映射
    label_map = build_label_map(gnt_files)
    print(f"共 {len(label_map)} 个不重复汉字")

    # 第二步：解析所有文件并收集数据（按作者分组用于划分）
    all_images, all_indices, all_authors = [], [], []
    for gnt_path in tqdm(gnt_files, desc="解析 GNT"):
        # 从文件名提取作者编号（CASIA 命名惯例: xxx-xxx.gnt）
        author_id = int(gnt_path.stem.split("-")[0]) if "-" in gnt_path.stem else 0

        images, labels = parse_gnt_file(str(gnt_path))
        for img, lbl in zip(images, labels):
            all_images.append(img)
            all_indices.append(label_map[lbl])
            all_authors.append(author_id)

    # 第三步：按作者划分训练集/验证集/测试集（80/10/10）
    unique_authors = list(set(all_authors))
    np.random.seed(42)
    np.random.shuffle(unique_authors)

    n_authors = len(unique_authors)
    n_train_auth = int(n_authors * 0.8)
    n_val_auth = int(n_authors * 0.1)

    train_authors = set(unique_authors[:n_train_auth])
    val_authors = set(unique_authors[n_train_auth:n_train_auth + n_val_auth])
    test_authors = set(unique_authors[n_train_auth + n_val_auth:])

    # 第四步：写入 HDF5
    splits = {"train": train_authors, "val": val_authors, "test": test_authors}
    for split_name, author_set in splits.items():
        split_images, split_indices = [], []
        for img, idx, author in zip(all_images, all_indices, all_authors):
            if author in author_set:
                split_images.append(img)
                split_indices.append(idx)

        create_hdf5(split_images, [], label_map, split_indices,
                    args.output, split=split_name, resize_to=args.image_size)

    # 保存标签映射（GB2312 编码 → 汉字字符串）到 HDF5 属性
    # 构建反向映射: {class_index: (gb_tag, char_str)}
    idx_to_label = {idx: tag for tag, idx in label_map.items()}
    sorted_labels = [idx_to_label[i] for i in range(len(label_map))]
    # GB2312 双字节编码解码为汉字字符串
    char_list = [tag.decode("gb2312-80", errors="replace") for tag in sorted_labels]

    with h5py.File(args.output, "a") as f:
        f.attrs["num_classes"] = len(char_list)
        # 存储为可变长字符串数组
        dt = h5py.string_dtype(encoding="utf-8")
        f.attrs.create("labels", char_list, dtype=dt)

    print(f"预处理完成: {args.output}")
    print(f"训练: {sum(1 for a in all_authors if a in train_authors)} 作者")
    print(f"验证: {sum(1 for a in all_authors if a in val_authors)} 作者")
    print(f"测试: {sum(1 for a in all_authors if a in test_authors)} 作者")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试验证通过**

```bash
conda run -n paddle python -m pytest tests/test_preprocess.py -v
```
期望: 2 tests passed

- [ ] **Step 5: 提交**

```bash
git add src/data/preprocess.py tests/test_preprocess.py && git commit -m "feat: add GNT to HDF5 preprocessing pipeline"
```

---

### Task 13: 训练脚本

**Files:**
- Create: `src/train.py`

**Interfaces:**
- Consumes: `Config` (Task 2), `set_seed`/`EarlyStopping`/`save_checkpoint` (Task 3), `HDF5Dataset` (Task 11), `get_train_transforms`/`get_val_transforms` (Task 10), `VisionTransformer` (Task 9)
- Produces: 命令行接口 `python -m src.train --config configs/vit_small_hwdb.yaml`

- [ ] **Step 1: 实现训练脚本（无独立测试 — 通过手动运行验证）**

```python
"""训练脚本 — 完整的训练循环，支持 AMP 混合精度、梯度累积、早停。

用法:
    python -m src.train --config configs/vit_small_hwdb.yaml

训练过程在 RTX 4060 (8GB VRAM) 上经 AMP 优化后，
batch_size=128 时显存占用约 5-6GB，留有余量。
"""

import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config
from utils import set_seed, EarlyStopping, save_checkpoint
from data.dataset import HDF5Dataset
from data.transforms import get_train_transforms, get_val_transforms
from model.vit import VisionTransformer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
) -> float:
    """执行一个训练 epoch。

    Args:
        model: ViT 模型（训练模式）
        dataloader: 训练数据加载器
        optimizer: AdamW 优化器
        criterion: 损失函数（CrossEntropyLoss with label smoothing）
        scaler: AMP 梯度缩放器
        device: 计算设备
        grad_accum_steps: 梯度累积步数

    Returns:
        本 epoch 的平均训练损失
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # AMP 自动混合精度前向传播
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps  # 梯度累积的损失缩放

        # AMP 反向传播
        scaler.scale(loss).backward()

        # 每 grad_accum_steps 步更新一次参数
        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """在验证集上评估模型。

    Args:
        model: ViT 模型（评估模式）
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备

    Returns:
        (平均损失, top-1 准确率, top-5 准确率)
    """
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        total += labels.size(0)

        # Top-1: 最高 logit 对应的类别
        _, pred_top1 = logits.topk(1, dim=1)
        correct_top1 += (pred_top1.squeeze() == labels).sum().item()

        # Top-5: 前 5 个最高 logit 中是否包含正确类别
        _, pred_top5 = logits.topk(5, dim=1)
        correct_top5 += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

    return (
        total_loss / len(dataloader),
        correct_top1 / total,
        correct_top5 / total,
    )


def main():
    parser = argparse.ArgumentParser(description="Train ViT on CASIA-HWDB1.1")
    parser.add_argument("--config", type=str, default="configs/vit_small_hwdb.yaml",
                        help="YAML 配置文件路径")
    args = parser.parse_args()

    # ---- 加载配置 ----
    cfg = Config.from_yaml(args.config)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 构建数据加载器 ----
    train_ds = HDF5Dataset(
        cfg.data.hdf5_path, split="train",
        transform=get_train_transforms(cfg.model.image_size),
    )
    val_ds = HDF5Dataset(
        cfg.data.hdf5_path, split="val",
        transform=get_val_transforms(cfg.model.image_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    print(f"训练样本: {len(train_ds):,}, 验证样本: {len(val_ds):,}")

    # ---- 构建模型 ----
    model = VisionTransformer(
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数总量: {total_params:,} (可训练: {trainable_params:,})")

    # ---- 优化器 & 损失函数 & 调度器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # Label Smoothing: 将 hard label [0,0,1,0] 平滑为 [ε, ε, 1-ε, ε]
    # 防止模型对预测过于自信，提升泛化能力
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

    # Cosine Warmup 学习率调度
    # 前 warmup_epochs 线性增加到峰值，然后 cosine 衰减到 0
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.train.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs - cfg.train.warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.train.warmup_epochs],
    )

    # ---- 训练循环 ----
    scaler = GradScaler()  # AMP 梯度缩放器
    stopper = EarlyStopping(patience=10, mode="max")
    best_acc = 0.0
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n开始训练 — {cfg.train.epochs} epochs, batch_size={cfg.train.batch_size}")
    print("=" * 60)

    for epoch in range(1, cfg.train.epochs + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, cfg.train.grad_accum_steps,
        )

        # 验证
        val_loss, top1, top5 = validate(model, val_loader, criterion, device)

        # 输出日志
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{cfg.train.epochs} | "
            f"LR: {lr_now:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Top-1: {top1:.4f} | "
            f"Top-5: {top5:.4f}"
        )

        # 保存最佳模型
        if top1 > best_acc:
            best_acc = top1
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                checkpoint_dir / "best.pt",
            )
            print(f"  → 保存最佳模型 (Top-1: {best_acc:.4f})")

        # 保存最新 checkpoint（用于断点续训）
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            checkpoint_dir / "last.pt",
        )

        # 早停检查
        if stopper(top1):
            print(f"早停触发于 epoch {epoch}，最佳 Top-1: {best_acc:.4f}")
            break

        scheduler.step()

    print(f"\n训练完成！最佳 Top-1: {best_acc:.4f}")
    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法正确**

```bash
conda run -n paddle python -c "import py_compile; py_compile.compile('src/train.py', doraise=True); print('Syntax OK')"
```

- [ ] **Step 3: 提交**

```bash
git add src/train.py && git commit -m "feat: implement training script with AMP and early stopping"
```

---

### Task 14: 评估脚本

**Files:**
- Create: `src/evaluate.py`

**Interfaces:**
- Consumes: `Config` (Task 2), `VisionTransformer` (Task 9), `HDF5Dataset` (Task 11)
- Produces: `python -m src.evaluate --config configs/vit_small_hwdb.yaml --checkpoint outputs/checkpoints/best.pt`

- [ ] **Step 1: 实现评估脚本**

```python
"""评估脚本 — 加载训练好的 checkpoint 在测试集上计算最终指标。

用法:
    python -m src.evaluate \
        --config configs/vit_small_hwdb.yaml \
        --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import HDF5Dataset
from data.transforms import get_val_transforms
from model.vit import VisionTransformer


@torch.no_grad()
def evaluate(model, dataloader, device):
    """在测试集上评估模型，输出详细的 Top-1/Top-5 准确率。

    Args:
        model: 已加载权重的 ViT 模型
        dataloader: 测试数据加载器
        device: 计算设备

    Returns:
        (top1_acc, top5_acc, total_samples)
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    pbar = tqdm(dataloader, desc="Evaluating")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        _, pred_top1 = logits.topk(1, dim=1)
        correct_top1 += (pred_top1.squeeze() == labels).sum().item()

        _, pred_top5 = logits.topk(5, dim=1)
        correct_top5 += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)
        pbar.set_postfix({
            "top1": f"{correct_top1/total:.4f}",
            "top5": f"{correct_top5/total:.4f}",
        })

    return correct_top1 / total, correct_top5 / total, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT on test set")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载测试数据
    test_ds = HDF5Dataset(
        cfg.data.hdf5_path,
        split="test",
        transform=get_val_transforms(cfg.model.image_size),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    print(f"测试样本: {len(test_ds):,}")

    # 加载模型
    model = VisionTransformer(
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        num_classes=cfg.model.num_classes,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"已加载 checkpoint: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")

    # 评估
    top1, top5, total = evaluate(model, test_loader, device)
    print(f"\n{'='*40}")
    print(f"测试集结果: {total:,} 样本")
    print(f"Top-1 准确率: {top1:.4f} ({top1*100:.2f}%)")
    print(f"Top-5 准确率: {top5:.4f} ({top5*100:.2f}%)")
    print(f"{'='*40}")

    test_ds.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法**

```bash
conda run -n paddle python -c "import py_compile; py_compile.compile('src/evaluate.py', doraise=True); print('Syntax OK')"
```

- [ ] **Step 3: 提交**

```bash
git add src/evaluate.py && git commit -m "feat: add evaluation script for test set metrics"
```

---

### Task 15: CLI 推理接口

**Files:**
- Create: `src/inference.py`

**Interfaces:**
- Consumes: `VisionTransformer` (Task 9), `Config` (Task 2)
- Produces: `python -m src.inference --image path/to/image.png --checkpoint outputs/checkpoints/best.pt`

- [ ] **Step 1: 实现推理脚本**

```python
"""命令行推理 — 对单张手写汉字图片进行识别。

用法:
    python -m src.inference \
        --image my_handwriting.png \
        --checkpoint outputs/checkpoints/best.pt \
        --config configs/vit_small_hwdb.yaml

输出: Top-5 预测汉字及对应置信度。
"""

import argparse
import torch
import numpy as np
from PIL import Image

from config import Config
from data.transforms import get_val_transforms
from model.vit import VisionTransformer


def load_label_list(hdf5_path: str) -> list[str]:
    """从 HDF5 属性中加载汉字标签列表。

    预处理时已将 GB2312 编码→汉字的映射保存到 HDF5 根组属性中，
    这里直接读取即可获取按类别索引排列的汉字字符串。

    Args:
        hdf5_path: HDF5 文件路径

    Returns:
        汉字字符串列表，索引对应类别 ID
    """
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        if 'labels' in f.attrs:
            return list(f.attrs['labels'])
    raise FileNotFoundError(
        f"HDF5 文件中未找到 labels 属性，请重新运行预处理脚本: {hdf5_path}"
    )


def predict(
    model: VisionTransformer,
    image_path: str,
    label_list: list[str],
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """对单张图片进行推理。

    Args:
        model: 已加载权重的 ViT 模型
        image_path: 输入图片路径
        label_list: 类别标签列表
        device: 计算设备
        top_k: 返回前 k 个预测结果

    Returns:
        [(汉字, 置信度), ...] 列表，按置信度降序排列
    """
    # 加载图片并预处理
    img = Image.open(image_path).convert("L")  # 灰度图
    img = img.resize((112, 112), Image.BILINEAR)
    img_array = np.array(img, dtype=np.uint8)

    transform = get_val_transforms(112)
    tensor = transform(img_array).unsqueeze(0).to(device)  # (1, 1, 112, 112)

    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    # 取 Top-K
    topk_probs, topk_indices = probs.topk(top_k, dim=1)
    results = [
        (label_list[idx], prob.item())
        for idx, prob in zip(topk_indices[0], topk_probs[0])
    ]
    return results


def main():
    parser = argparse.ArgumentParser(description="单张手写汉字识别")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--config", type=str, default="configs/vit_small_hwdb.yaml")
    parser.add_argument("--top_k", type=int, default=5, help="显示 Top-K 结果")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = VisionTransformer(
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        in_channels=cfg.model.in_channels,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        num_classes=cfg.model.num_classes,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"已加载模型 (epoch {checkpoint['epoch']})")

    # 加载标签列表
    label_list = load_label_list(cfg.data.hdf5_path)

    # 预测
    results = predict(model, args.image, label_list, device, args.top_k)

    print(f"\n{'='*40}")
    print(f"图片: {args.image}")
    print(f"识别结果 (Top-{args.top_k}):")
    print("-" * 40)
    for rank, (char, conf) in enumerate(results, 1):
        bar = "█" * int(conf * 20)
        print(f"  {rank}. {char}  [{bar}] {conf:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法**

```bash
conda run -n paddle python -c "import py_compile; py_compile.compile('src/inference.py', doraise=True); print('Syntax OK')"
```

- [ ] **Step 3: 提交**

```bash
git add src/inference.py && git commit -m "feat: add CLI inference for single image recognition"
```

---

### Task 16: Gradio 推理 Demo

**Files:**
- Create: `demo/app.py`

**Interfaces:**
- Consumes: `VisionTransformer` (Task 9), `Config` (Task 2)
- Produces: Gradio Web 界面

- [ ] **Step 1: 实现 Gradio Demo**

```python
"""Gradio Web Demo — 手写汉字识别交互界面。

用法:
    python demo/app.py --config configs/vit_small_hwdb.yaml \
                       --checkpoint outputs/checkpoints/best.pt

提供两种输入方式:
    1. 鼠标手写画板 — 在画布上书写汉字
    2. 图片上传 — 上传已有的手写汉字图片
"""

import argparse
import torch
import numpy as np
from PIL import Image
import gradio as gr

import sys
sys.path.insert(0, "src")
from config import Config
from data.transforms import get_val_transforms
from model.vit import VisionTransformer


class HandwritingRecognizer:
    """封装模型加载和推理逻辑，供 Gradio 界面调用。"""

    def __init__(self, config_path: str, checkpoint_path: str):
        self.cfg = Config.from_yaml(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VisionTransformer(
            image_size=self.cfg.model.image_size,
            patch_size=self.cfg.model.patch_size,
            in_channels=self.cfg.model.in_channels,
            hidden_dim=self.cfg.model.hidden_dim,
            num_layers=self.cfg.model.num_layers,
            num_heads=self.cfg.model.num_heads,
            num_classes=self.cfg.model.num_classes,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = get_val_transforms(self.cfg.model.image_size)
        print(f"模型已加载 (epoch {checkpoint['epoch']})")

    @torch.no_grad()
    def recognize_sketch(self, sketch_img) -> dict:
        """识别画板输入的图片。

        Gradio 画板返回 RGB PIL Image，需要转为灰度 numpy 后再处理。
        """
        if sketch_img is None:
            return {}

        # Gradio Sketchpad 返回 RGB PIL Image
        img = sketch_img.convert("L")  # 转灰度
        img = img.resize((112, 112), Image.BILINEAR)
        img_array = np.array(img, dtype=np.uint8)

        tensor = self.transform(img_array).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

        topk_probs, topk_indices = probs.topk(5)
        return {
            f"第{i+1}名": f"类_{idx.item()} ({prob.item():.1%})"
            for i, (idx, prob) in enumerate(zip(topk_indices, topk_probs))
        }

    @torch.no_grad()
    def recognize_upload(self, upload_img):
        """识别上传的图片文件。"""
        if upload_img is None:
            return {}

        img = upload_img.convert("L").resize((112, 112), Image.BILINEAR)
        img_array = np.array(img, dtype=np.uint8)

        tensor = self.transform(img_array).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

        topk_probs, topk_indices = probs.topk(5)
        return {
            f"第{i+1}名": f"类_{idx.item()} ({prob.item():.1%})"
            for i, (idx, prob) in enumerate(zip(topk_indices, topk_probs))
        }


def build_interface(recognizer: HandwritingRecognizer):
    """构建 Gradio 界面。"""
    with gr.Blocks(title="手写汉字识别") as demo:
        gr.Markdown(
            """
            # ✍️ 手写汉字识别
            基于 Vision Transformer (ViT-Small)，支持 3,755 个常用汉字识别。

            **两种使用方式**: 左侧画板鼠标书写，或右侧上传手写图片。
            """
        )

        with gr.Row():
            with gr.Column():
                sketch = gr.Sketchpad(
                    label="手写画板（写一个汉字）",
                    brush_radius=3,
                    shape=(280, 280),
                )
                sketch_btn = gr.Button("识别画板文字", variant="primary")

            with gr.Column():
                upload = gr.Image(
                    label="上传手写图片",
                    type="pil",
                    image_mode="L",
                )
                upload_btn = gr.Button("识别上传图片", variant="primary")

        with gr.Row():
            sketch_result = gr.Label(label="画板识别结果", num_top_classes=5)
            upload_result = gr.Label(label="上传识别结果", num_top_classes=5)

        sketch_btn.click(
            fn=recognizer.recognize_sketch,
            inputs=[sketch],
            outputs=[sketch_result],
        )
        upload_btn.click(
            fn=recognizer.recognize_upload,
            inputs=[upload],
            outputs=[upload_result],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio 手写汉字识别 Demo")
    parser.add_argument("--config", type=str, default="configs/vit_small_hwdb.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    args = parser.parse_args()

    recognizer = HandwritingRecognizer(args.config, args.checkpoint)
    demo = build_interface(recognizer)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add demo/app.py && git commit -m "feat: add Gradio web demo for handwriting recognition"
```

---

### Task 17: README.md 文档

**Files:**
- Create: `README.md`

- [ ] **Step 1: 编写项目文档**

```markdown
# ✍️ Transformer 手写汉字识别

基于 **Vision Transformer (ViT-Small)** 的离线手写汉字识别系统。

- **数据集**: CASIA-HWDB1.1（3,755 个常用汉字，~240 万样本）
- **模型**: ViT-Small（12层 Transformer，~22M 参数）
- **框架**: PyTorch 2.10 + CUDA 12.6
- **硬件需求**: NVIDIA GPU ≥8GB VRAM

## 项目结构

```
├── configs/                  # YAML 配置文件
├── src/
│   ├── config.py             # 配置解析
│   ├── utils.py              # 工具函数（种子、早停、Checkpoint）
│   ├── data/
│   │   ├── preprocess.py     # GNT → HDF5 预处理
│   │   ├── dataset.py        # HDF5 PyTorch Dataset
│   │   └── transforms.py     # 数据增强管道
│   ├── model/
│   │   ├── patch_embed.py    # 图像分块 + 线性投影
│   │   ├── attention.py      # 多头自注意力
│   │   ├── mlp.py            # 前馈网络
│   │   ├── encoder.py        # Transformer Encoder 层
│   │   ├── transformer.py    # 堆叠 Encoder
│   │   └── vit.py            # 完整 ViT 模型
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   └── inference.py          # CLI 推理
├── demo/
│   └── app.py                # Gradio Web Demo
├── tests/                    # 单元测试
└── outputs/                  # 训练输出（checkpoints, logs）
```

## 快速开始

### 1. 环境准备

```bash
conda activate paddle
pip install -r requirements.txt
```

### 2. 数据预处理

下载 CASIA-HWDB1.1 数据集后：

```bash
python -m src.data.preprocess \
    --raw_dir data/raw \
    --output data/processed/hwdb.h5
```

### 3. 训练

```bash
python -m src.train --config configs/vit_small_hwdb.yaml
```

### 4. 评估

```bash
python -m src.evaluate \
    --config configs/vit_small_hwdb.yaml \
    --checkpoint outputs/checkpoints/best.pt
```

### 5. 推理

```bash
# 命令行
python -m src.inference \
    --image my_char.png \
    --checkpoint outputs/checkpoints/best.pt

# Web Demo
python demo/app.py \
    --checkpoint outputs/checkpoints/best.pt
```

## 模型架构

| 组件 | 配置 |
|------|------|
| 输入 | 112×112 灰度图 |
| Patch | 4×4（28×28 = 784 patches） |
| Hidden Dim | 384 |
| 层数 | 12 |
| 头数 | 6 |
| MLP 扩展 | 4×（384→1536） |
| 参数量 | ~22M |

## 运行测试

```bash
pytest tests/ -v
```
```

- [ ] **Step 2: 提交并推送**

```bash
git add README.md && git commit -m "docs: add README with project overview and usage"
git push -u origin main
```

---

### Task 18: 全量测试验证

**Files:**
- 无新建文件，运行全部已有测试

- [ ] **Step 1: 运行全部单元测试**

```bash
conda run -n paddle python -m pytest tests/ -v --tb=short
```
期望: 所有测试通过（约 20+ tests）

- [ ] **Step 2: 验证模型前向传播（GPU 设备检查）**

```bash
conda run -n paddle python -c "
import torch
from src.model.vit import VisionTransformer

model = VisionTransformer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = torch.randn(2, 1, 112, 112, device=device)
out = model(x)
print(f'设备: {device}')
print(f'输入: {x.shape}')
print(f'输出: {out.shape}')
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
print(f'显存: {torch.cuda.memory_allocated() // 1024**2 if device.type == \"cuda\" else 0} MB')
print('✓ 模型前向传播正常')
"
```

- [ ] **Step 3: 最终提交**

```bash
git add -A && git commit -m "chore: finalize tests and verify all modules" && git push
```
```

---

## 自我审查

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="Read">
<｜｜DSML｜｜parameter name="file_path" string="true">C:/Users/XL/Desktop/手写汉字识别/docs/superpowers/plans/2026-06-23-transformer-handwritten-chinese-recognition-plan.md