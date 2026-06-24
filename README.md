# ✍️ 手写汉字识别

基于 **CNN (ResNet)** 的离线手写汉字识别系统，支持 3,490 个汉字分类，**Top-1 准确率 98.5%**。

## 数据流

```
┌────────── 原始数据 ──────────┐
│  data/raw/train/*.gnt  ×240  │  ← CASIA GNT 二进制
│  data/raw/test/*.gnt   ×60   │
│  [4B size][2B tag][2B w][2B h][bitmap]  │
└────────────┬─────────────────┘
             │  src/data/preprocess.py    ← 解析 GNT → 提取图像+标签
             ▼
┌────────── 预处理后 ──────────┐
│  data/processed/hwdb.h5       │  ← gzip 压缩 (弃用: 随机读太慢)
│  data/processed/hwdb_fast.h5  │  ← 无压缩, 随机读 0.04s/128张
│  112×112 uint8 灰度, 1,041,835 张     │
└────────────┬─────────────────┘
             │  src/data/dataset.py      ← HDF5Dataset
             │  src/data/transforms.py   ← 旋转/平移/缩放/归一化
             ▼
┌────────── 训练 ──────────────┐
│  DataLoader(batch=128, shuffle)        │
│  src/model/resnet.py          │  ← ResNet CNN (~12.8M)
│  src/train.py                 │  ← AMP + Warmup + 早停
│  src/utils.py                 │  ← 种子/早停/Checkpoint
│  configs/vit_small_hwdb.yaml  │  ← 切换模型/调超参
└────────────┬─────────────────┘
             │  src/train.py  每 epoch 验证 → save_checkpoint
             ▼
┌────────── 输出 ──────────────┐
│  outputs/checkpoints/best.pt     ← 最佳模型
│  outputs/checkpoints/curve.png   ← 训练曲线
│  outputs/checkpoints/history.json← 训练数据
└────────────┬─────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
  CLI 推理           Web Demo
  src/inference.py   demo/app.py
  单张命令行识别      上传图片识别
```

## 模型架构

```
输入 (B, 1, 112, 112) 灰度手写汉字
        │
        ▼  src/model/resnet.py
┌─── Conv1 ─────────────────────────────┐
│  Conv2d(1→64, k=7, s=2) + BN + ReLU   │  → (B, 64, 56, 56)
│  MaxPool2d(3, s=2)                     │  → (B, 64, 28, 28)
└────────────────────────────────────────┘
        │
        ▼  Stage1 ×2  BasicBlock(64→128)
┌─── Stage1 ────────────────────────────┐
│  ┌─ Conv3 BN ReLU ─┐                  │
│  ├─ Conv3 BN ──────┤→ +残差 → ReLU    │  → (B, 128, 28, 28)
│  └─ shortcut(1×1) ─┘                  │
└────────────────────────────────────────┘
        │
        ▼  Stage2 ×2  BasicBlock(128→256, s=2)
┌─── Stage2 ────────────────────────────┐
│  同上, stride=2 降采样                 │  → (B, 256, 14, 14)
└────────────────────────────────────────┘
        │
        ▼  Stage3 ×2  BasicBlock(256→512, s=2)
┌─── Stage3 ────────────────────────────┐
│  同上, stride=2 降采样                 │  → (B, 512, 7, 7)
└────────────────────────────────────────┘
        │
        ▼
┌─── Head ──────────────────────────────┐
│  GlobalAvgPool        → (B, 512, 1, 1) │
│  Flatten              → (B, 512)       │
│  Linear(512 → 3490)   → (B, 3490)     │  ← 分类 logits
└────────────────────────────────────────┘

参数: ~12.8M  |  训练: batch=128, lr=3e-4, AdamW, CosineWarmup, AMP
准确率: Top-1 98.5%  |  Top-5 99.8%
```

## 可选模型: Vision Transformer

```yaml
# configs/vit_small_hwdb.yaml  改一行即可切换
model:
  type: "vit"       # ← 切换为 Transformer
```

```
输入 (B, 1, 112, 112) 灰度手写汉字
        │
        ▼  src/model/patch_embed.py
┌─── Patch Embedding ────────────────────┐
│  Conv2d(1→384, k=8, s=8)               │  切块 + 线性投影
│  Flatten → Transpose                    │  → (B, 196, 384)
└────────────────────────────────────────┘
        │
        ▼
┌─── + [CLS] Token + Position Embed ─────┐
│  [CLS] learnable (1, 1, 384)           │  汇总全图信息的「哨兵」
│  Pos Embed learnable (1, 197, 384)     │  标记空间位置
│  拼接 → (B, 197, 384)                  │
└────────────────────────────────────────┘
        │
        ▼  src/model/encoder.py ×6  src/model/transformer.py
┌─── Transformer Encoder ×6 ─────────────┐
│  ┌─ LayerNorm ─────────────────────┐    │
│  │  src/model/attention.py         │    │
│  │  Multi-Head Self-Attention (6头) │   │
│  │  QKV(384→1152) → 拆6头每头64维  │   │
│  │  Attn = Softmax(Q·Kᵀ/√64)·V     │   │
│  │  → 合并头 → Proj(384→384)       │   │
│  └────────────────── +残差 ────────┘    │
│  ┌─ LayerNorm ─────────────────────┐    │
│  │  src/model/mlp.py               │    │
│  │  Linear(384→1536) → GELU → Drop  │   │
│  │  Linear(1536→384) → Drop        │    │
│  └────────────────── +残差 ────────┘    │
│                  ×6 层                   │  src/model/transformer.py 堆叠
│             最终 LayerNorm              │
└────────────────────────────────────────┘
        │
        ▼  src/model/vit.py (Head)
┌─── Head ──────────────────────────────┐
│  取 [CLS] token       → (B, 384)       │  CLS 聚合了全图信息
│  Linear(384 → 3490)   → (B, 3490)     │  分类 logits
└────────────────────────────────────────┘

参数: ~7.6M  |  token=197, 6层6头, dim=384
速度: 比 ResNet 慢 5-10× (自注意力 O(N²))
```

## Model 文件依赖关系

```
ViT 组件链:
  patch_embed.py ──┐
  attention.py  ───┤
  mlp.py        ───┼──→ encoder.py ──→ transformer.py ──→ vit.py
                   │      (单层)          (堆叠×6)        (完整模型)

CNN (独立):
  resnet.py ── 零依赖, 纯 PyTorch 原生组件
```

## 项目结构

| 文件 | 功能 |
|------|------|
| `configs/vit_small_hwdb.yaml` | 超参数配置 (`type: resnet` 切换模型) |
| **数据** | |
| `src/data/preprocess.py` | GNT 二进制 → HDF5, 按作者划分 80/10/10 |
| `src/data/dataset.py` | HDF5Dataset, h5py 随机读取, pickle 安全 |
| `src/data/transforms.py` | 训练增强 (旋转/平移/缩放) / 验证归一化 |
| `src/data/rebuild_hdf5.py` | 批量重建成无压缩 HDF5 |
| **模型** | |
| `src/model/resnet.py` | **CNN**, 3 stage, ~12.8M 参数 |
| `src/model/vit.py` | Vision Transformer (备选) |
| **训练/评估/推理** | |
| `src/train.py` | 完整训练 (AMP + Warmup + 早停 + 曲线) |
| `src/train_quick.py` | 快速验证, 默认 10% 数据 |
| `src/evaluate.py` | 测试集评估 Top-1/Top-5 |
| `src/inference.py` | CLI 单张图片识别 |
| `demo/app.py` | **Gradio Web Demo**, 上传图片识别 |
| **工具** | |
| `src/config.py` | YAML 配置解析 |
| `src/utils.py` | set_seed / EarlyStopping / save_checkpoint |
| `tests/` | 37 个单元测试, `pytest tests/ -v` |

## 快速开始

```bash
# 1. 环境
conda activate paddle
pip install -r requirements.txt

# 2. 预处理 (已完成, 跳过)
# python src/data/preprocess.py --raw_dir data/raw --output data/processed/hwdb.h5
# python src/data/rebuild_hdf5.py

# 3. 快速验证 (~10分钟)
python src/train_quick.py

# 4. 完整训练 (25% 数据 ~8h, 100% ~35h)
python src/train.py --fraction 0.25

# 5. 评估
python src/evaluate.py --config configs/vit_small_hwdb.yaml --checkpoint outputs/checkpoints/best.pt

# 6. Web Demo (PyCharm 右键运行 demo/app.py)
python demo/app.py

# 7. CLI 推理
python src/inference.py --image test.png --checkpoint outputs/checkpoints/best.pt
```

## 训练结果

| 指标 | 值 |
|------|-----|
| 模型 | ResNet CNN, 12.8M 参数 |
| 数据 | CASIA-HWDB1.2, 3490 类, 83 万训练集 |
| **Top-1** | **98.5%** |
| **Top-5** | **99.8%** |
| 训练轮数 | 50 epoch (25% 数据) |
| 硬件 | RTX 4060 Laptop 8GB |

```
Epoch  1: Loss 6.30 → 4.37, Top-1 41.5%
Epoch 10: Loss 1.29 → 1.31, Top-1 97.6%
Epoch 30: Loss 1.19 → 1.25, Top-1 98.3%
Epoch 50: Loss 1.16 → 1.23, Top-1 98.5%
```

## 模型切换

```yaml
# configs/vit_small_hwdb.yaml
model:
  type: "resnet"    # CNN, ~12.8M, 快
  type: "vit"       # Transformer, ~7.6M, 慢
```

所有脚本 (`train.py` / `evaluate.py` / `inference.py` / `demo/app.py`) 自动适配。
