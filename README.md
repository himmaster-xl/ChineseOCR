# ✍️ Transformer 手写汉字识别

基于 **Vision Transformer (ViT-Small)** 的离线手写汉字识别系统。

- **数据集**: CASIA-HWDB1.1（3,755 个常用汉字，~240 万样本）
- **模型**: ViT-Small（12层 Transformer，~22M 参数）
- **框架**: PyTorch 2.10 + CUDA 12.6
- **硬件需求**: NVIDIA GPU >= 8GB VRAM

## 项目结构

```
├── configs/                  # YAML 配置文件
├── src/
│   ├── config.py             # 配置解析
│   ├── utils.py              # 工具函数（种子、早停、Checkpoint）
│   ├── data/
│   │   ├── preprocess.py     # GNT -> HDF5 预处理
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
| 输入 | 112x112 灰度图 |
| Patch | 4x4（28x28 = 784 patches） |
| Hidden Dim | 384 |
| 层数 | 12 |
| 头数 | 6 |
| MLP 扩展 | 4x（384 -> 1536） |
| 参数量 | ~22M |

## 运行测试

```bash
pytest tests/ -v
```
