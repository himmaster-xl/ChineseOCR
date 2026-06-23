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

        # 加载标签
        self._load_labels()

    def _load_labels(self):
        """从 HDF5 加载汉字标签列表。"""
        import h5py
        with h5py.File(self.cfg.data.hdf5_path, 'r') as f:
            if 'labels' in f.attrs:
                self.labels = list(f.attrs['labels'])
            else:
                self.labels = [f"class_{i}" for i in range(self.cfg.model.num_classes)]

    @torch.no_grad()
    def _predict(self, pil_img) -> dict:
        """通用推理方法。"""
        if pil_img is None:
            return {}

        img = pil_img.convert("L").resize((112, 112), Image.BILINEAR)
        img_array = np.array(img, dtype=np.uint8)
        tensor = self.transform(img_array).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        topk_probs, topk_indices = probs.topk(5)

        return {
            f"Top-{i+1}": f"{self.labels[idx.item()]} ({prob.item():.1%})"
            for i, (idx, prob) in enumerate(zip(topk_indices, topk_probs))
        }

    def recognize_sketch(self, sketch_img) -> dict:
        """识别画板输入的图片（Gradio Sketchpad 返回 RGB PIL Image）。"""
        return self._predict(sketch_img)

    def recognize_upload(self, upload_img):
        """识别上传的图片文件。"""
        return self._predict(upload_img)


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
