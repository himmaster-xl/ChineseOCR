"""Gradio Web Demo — 手写汉字识别交互界面。

用法:
    python demo/app.py --config configs/vit_small_hwdb.yaml \
                       --checkpoint outputs/checkpoints/best.pt

提供两种输入方式:
    1. 鼠标手写画板 — 在画布上书写汉字
    2. 图片上传 — 上传已有的手写汉字图片
"""

import argparse
import sys
from pathlib import Path

# 支持 PyCharm 一键运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image
import gradio as gr

from src.config import Config
from src.data.transforms import get_val_transforms
from src.model.vit import VisionTransformer


class HandwritingRecognizer:
    """封装模型加载和推理逻辑，供 Gradio 界面调用。"""

    def __init__(self, config_path: str, checkpoint_path: str):
        self.cfg = Config.from_yaml(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_type = getattr(self.cfg.model, 'type', 'vit')
        if model_type == 'resnet':
            from src.model.resnet import ResNet
            self.model = ResNet(
                in_channels=self.cfg.model.in_channels,
                num_classes=self.cfg.model.num_classes,
            ).to(self.device)
        else:
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

        # 加载标签
        self._load_labels()

    def _load_labels(self):
        """从 HDF5 加载汉字标签列表。"""
        import h5py
        from pathlib import Path
        h5_path = Path(self.cfg.data.hdf5_path)
        if not h5_path.is_absolute():
            h5_path = Path(__file__).resolve().parent.parent / h5_path
        with h5py.File(str(h5_path), 'r') as f:
            if 'labels' in f.attrs:
                self.labels = list(f.attrs['labels'])
            else:
                self.labels = [f"class_{i}" for i in range(self.cfg.model.num_classes)]

    def _to_pil(self, output) -> Image.Image:
        """将 Gradio 各种可能的输出统一转为 PIL Image。"""
        if output is None:
            return None
        # dict → 提取图像数据
        if isinstance(output, dict):
            for key in ["composite", "image", "background"]:
                val = output.get(key)
                if val is not None:
                    output = val
                    break
        # numpy array → PIL
        if isinstance(output, np.ndarray):
            output = Image.fromarray(output)
        # 确保是 PIL Image
        if not isinstance(output, Image.Image):
            return None
        return output

    @torch.no_grad()
    def _predict(self, output) -> dict:
        """通用推理方法。"""
        pil_img = self._to_pil(output)
        if pil_img is None:
            return {}

        img = pil_img.convert("L").resize((112, 112), Image.BILINEAR)
        img_array = np.array(img, dtype=np.uint8)
        tensor = self.transform(img_array).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        topk_probs, topk_indices = probs.topk(5)

        # Gradio 6 Label: {label: confidence_float}
        return {
            self.labels[idx.item()]: round(prob.item(), 4)
            for idx, prob in zip(topk_indices, topk_probs)
        }

    def recognize_sketch(self, sketch_output) -> dict:
        """识别画板输入。"""
        return self._predict(sketch_output)

    def recognize_upload(self, upload_output):
        """识别上传的图片。"""
        return self._predict(upload_output)


def build_interface(recognizer: HandwritingRecognizer):
    """构建 Gradio 界面。"""
    with gr.Blocks(title="手写汉字识别") as demo:
        gr.Markdown(
            """
            # ✍️ 手写汉字识别
            基于 CNN (ResNet) / ViT，支持 3,490 个常用汉字识别。
            **两种使用方式**: 左侧画板鼠标书写，或右侧上传手写图片。
            """
        )

        with gr.Row():
            with gr.Column():
                sketch = gr.Paint(
                    label="手写画板（写一个汉字）",
                    width=280, height=280,
                    image_mode="L",
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
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型 checkpoint (默认自动查找 quick.pt 或 best.pt)")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    args = parser.parse_args()

    # 自动查找 checkpoint
    project_root = Path(__file__).resolve().parent.parent
    if args.checkpoint is None:
        for candidate in [
            "outputs/checkpoints/quick.pt",
            "outputs/checkpoints/best.pt",
        ]:
            p = project_root / candidate
            if p.exists():
                args.checkpoint = str(p)
                print(f"自动选择 checkpoint: {p}")
                break
        if args.checkpoint is None:
            print("未找到 checkpoint，请先训练或指定 --checkpoint")
            return

    # 解析相对路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    recognizer = HandwritingRecognizer(str(config_path), str(checkpoint_path))
    demo = build_interface(recognizer)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
