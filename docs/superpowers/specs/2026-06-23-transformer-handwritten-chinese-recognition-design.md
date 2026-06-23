# Transformer 手写汉字识别系统 — 设计文档

> 日期: 2026-06-23 | 状态: 已完成设计

## 1. 项目目标

构建基于 Vision Transformer (ViT) 的离线手写汉字识别系统，支持 CASIA-HWDB1.1 数据集上 3755 类常用汉字（GB2312 一级字库）的分类识别。

项目定位为**个人深度学习项目**，目标是通过实战深入理解 Transformer 在视觉任务中的应用。代码追求：**模块划分清晰、代码简洁、注释详细明确**。

## 2. 环境与硬件

| 项目 | 详情 |
|------|------|
| 框架 | PyTorch 2.10.0+cu126 |
| GPU | NVIDIA RTX 4060 Laptop 8GB VRAM |
| CUDA | 12.6 |
| Python | 3.10.18 |
| 虚拟环境 | conda `paddle` |
| 依赖管理 | pip + requirements.txt |

## 3. 数据

### 数据集
- **CASIA-HWDB1.1**: 离线手写汉字数据集（由中科院自动化研究所发布）
- **字符范围**: GB2312 一级字库，3,755 个常用汉字
- **样本量**: 约 240 万张手写汉字图像（每字符约 240 个作者 × 约 2-3 样本 ≈ 每类约 600 样本）
- **图像格式**: 原始为 GNT 二进制格式，需预处理转为 PNG/NumPy/HDF5
- **图像尺寸**: 原始尺寸不一，统一 resize 到 112×112 作为模型输入

### 数据预处理管道
```
原始 GNT 文件 → 解析二进制 → 提取图像 + 标签 → Resize(112×112) → 归一化 → HDF5存储（供训练快速读取）
```

### 数据增强（训练时在线）
- 随机旋转 ±5°
- 随机平移 ±5%
- 轻微缩放 0.95~1.05
- （可选）弹性变形模拟手写形变

### 数据集划分
- 训练集: ~80%，约 192 万样本
- 验证集: ~10%，约 24 万样本
- 测试集: ~10%，约 24 万样本
- 按作者划分，避免数据泄漏（同一作者的样本不会同时出现在训练和测试集中）

## 4. 模型架构 — ViT-Small

### 整体流程
```
输入图像 (1, 112, 112)
  ↓ Patch Embedding: 4×4 卷积核, stride=4 → (384, 28, 28)
  ↓ Flatten → (784, 384)  # 784 = 28×28 个 patch
  ↓ 前置 [CLS] token → (785, 384)
  ↓ 加可学习位置编码
  ↓ 12 × Transformer Encoder Layer
      ├── Multi-Head Self-Attention (6 heads)
      └── MLP (384 → 1536 → 384)
  ↓ 取 [CLS] token → (384,)
  ↓ Classification Head (384 → 3755)
  ↓ Softmax
```

### 超参数

| 参数 | 值 |
|------|------|
| 输入尺寸 | 112×112 灰度图 |
| Patch 大小 | 4×4 |
| Patch 数量 | 28×28 = 784 |
| 隐藏维度 | 384 |
| Transformer 层数 | 12 |
| 注意力头数 | 6 |
| MLP 扩展比 | 4 (384→1536) |
| [CLS] Token | 可学习，维度 384 |
| 位置编码 | 可学习，维度 (785, 384) |
| Dropout | 0.1 |
| 分类类别 | 3755 |

### 参数量
- 总参数: 约 22M（ViT-Small 级别）
- Patch Embedding: ~6K
- Transformer: ~21M（12层 × ~1.7M/层）
- Classification Head: ~1.4M

## 5. 训练策略

| 配置 | 值 | 说明 |
|------|------|------|
| 优化器 | AdamW | weight_decay=0.05 |
| 学习率 | 3e-4 | 峰值学习率 |
| 学习率调度 | Cosine Warmup | 10 epochs warmup → cosine decay |
| 损失函数 | CrossEntropyLoss | 标准多分类 |
| Batch Size | 128 | 每步处理 128 张图 |
| Epochs | 100 | |
| AMP | ✅ 混合精度 | 节省显存 ~40%，支持更大 batch |
| 梯度累积 | 可选 (×2) | 等效 batch 256 |
| 早停 | patience=10 | 验证 top-1 精度不再提升时停止 |
| 模型保存 | best + last | 保存最佳验证精度的 checkpoint |

### 评估指标
- Top-1 准确率（主要）
- Top-5 准确率
- 每 epoch 的验证损失曲线

### 预期效果
- 训练时间: RTX 4060 8GB 下约 15-20 小时（100 epochs）
- Top-1 准确率: 预期 85%+（3755 类汉字分类，纯 ViT 从头训练）
- Top-5 准确率: 预期 95%+

## 6. 项目结构

```
手写汉字识别/
├── configs/
│   └── vit_small_hwdb.yaml      # 训练/模型超参数配置
├── data/
│   ├── raw/                      # 原始 CASIA-HWDB1.1 数据
│   └── processed/                # 预处理后的 HDF5 文件
├── src/
│   ├── __init__.py
│   ├── config.py                 # 配置解析类
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py         # 原始 GNT → HDF5 预处理脚本
│   │   ├── dataset.py            # HDF5Dataset 类，封装 __getitem__
│   │   └── transforms.py         # 数据增强（torchvision.transforms）
│   ├── model/
│   │   ├── __init__.py
│   │   ├── patch_embed.py        # Patch Embedding 模块
│   │   ├── attention.py          # Multi-Head Self-Attention
│   │   ├── mlp.py                # MLP 前馈网络
│   │   ├── encoder.py            # 单层 Transformer Encoder
│   │   ├── transformer.py        # 堆叠 N 层 Encoder 的完整 Transformer
│   │   └── vit.py                # ViT 整体模型（组装 Embedding + Transformer + Head）
│   ├── train.py                  # 训练脚本（含验证循环）
│   ├── evaluate.py               # 测试集评估 + 指标报告
│   ├── inference.py              # 单张图片推理接口
│   └── utils.py                  # 工具函数（日志、Checkpoint、早停等）
├── notebooks/
│   └── visualization.ipynb       # Attention Map + 特征嵌入可视化
├── outputs/                      # 训练输出（checkpoints, logs, TensorBoard）
│   ├── checkpoints/
│   └── logs/
├── demo/
│   └── app.py                    # Gradio 推理 Demo（上传/鼠标写汉字 → 识别）
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_preprocess.py
├── requirements.txt
├── .gitignore
└── README.md
```

### 模块职责划分

| 模块 | 职责 | 依赖 |
|------|------|------|
| `config.py` | 解析 YAML 配置，提供全局 `Config` 对象 | 无 |
| `dataset.py` | 从 HDF5 读取数据，返回 `(image, label)` | transforms |
| `transforms.py` | 训练/验证数据增强管道 | 无 |
| `preprocess.py` | 解析 CASIA GNT 文件，生成 HDF5 | 无 |
| `patch_embed.py` | 将图像分割为固定大小 patch 并进行线性投影 | 无 |
| `attention.py` | 多头自注意力（Q、K、V 投影 + Scaled Dot-Product Attention） | 无 |
| `mlp.py` | Transformer 中的两层 MLP（GELU 激活） | 无 |
| `encoder.py` | 单个 Transformer Encoder 层（Attention + MLP + LayerNorm + Residual） | attention, mlp |
| `transformer.py` | 堆叠 N 个 Encoder 层 | encoder |
| `vit.py` | 完整 ViT 模型：PatchEmbed → Transformer → ClassificationHead | patch_embed, transformer |
| `train.py` | 训练循环、验证循环、日志记录 | 所有 model 模块, dataset |
| `evaluate.py` | 加载 checkpoint 在测试集上评估 | 所有 model 模块 |
| `inference.py` | 单张图像推理（命令行接口） | vit, config |
| `utils.py` | 工具：早停、Checkpoint 管理、日志、设置随机种子 | 无 |

## 7. 推理 Demo

使用 **Gradio** 搭建交互式 Web 界面，提供两种输入方式：
- **图片上传**: 用户上传手写汉字图片
- **手写画板**: 鼠标在画布上写字（输出 112×112 灰度图）

输出: Top-5 预测结果及对应的置信度。

## 8. 代码规范

- 每个函数有 docstring 说明参数、返回值和功能
- 类有 docstring 说明职责和使用方式
- 关键步骤添加行内注释解释"为什么"
- 变量命名遵循 PyTorch 社区惯例（`x`, `logits`, `probs` 等）
- 每个模块保持单一职责，文件长度控制在 200 行以内
- 使用 `black` 风格（不做强制要求，但代码需整洁一致）

## 9. 非目标（YAGNI）

本项目不包括：
- 多 GPU 分布式训练（DDP）
- 模型量化 / 剪枝 / 蒸馏
- ONNX / TensorRT 部署
- 移动端适配
- 手写汉字定位 / 检测（仅做分类）
- 其他数据集支持

## 10. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 数据集过大（240万张），预处理耗时长 | 分批次预处理为 HDF5；支持断点续处理 |
| 从头训练 ViT 可能欠拟合 | 使用 Label Smoothing、数据增强；可切换为 ViT-Base 微调 |
| 8GB 显存不足 | AMP 混合精度 + 梯度累积 + 适当减小 batch size |
| 3755 类分类难度高 | 关注 Top-5 准确率而非仅 Top-1 |
