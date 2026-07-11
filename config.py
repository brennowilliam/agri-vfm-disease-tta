"""Central configuration. Dataset roots come from env vars with Kaggle defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 2

    # Dataset roots (override via env vars; defaults target a Kaggle notebook).
    pv_root: str = field(default_factory=lambda: _env(
        "PV_ROOT", "/kaggle/input/plantvillage-dataset/color"))
    plantdoc_root: str = field(default_factory=lambda: _env(
        "PLANTDOC_ROOT", "/kaggle/input/plantdoc-dataset"))
    plantwild_root: str = field(default_factory=lambda: _env(
        "PLANTWILD_ROOT", "/kaggle/input/plantwild"))
    cassava_root: str = field(default_factory=lambda: _env(
        "CASSAVA_ROOT", "/kaggle/input/cassava-leaf-disease-classification"))

    # Where checkpoints and cached features are written.
    out_root: str = field(default_factory=lambda: _env("OUT_ROOT", "/kaggle/working"))

    # ImageNet stats (for ResNet/ViT source training; CLIP/DINOv2 use their own preprocessing).
    imagenet_mean: tuple = (0.485, 0.456, 0.406)
    imagenet_std: tuple = (0.229, 0.224, 0.225)

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.out_root, "checkpoints")

    @property
    def feat_dir(self) -> str:
        return os.path.join(self.out_root, "features")


CFG = Config()

# Frozen backbones we sweep (timm / open_clip identifiers).
BACKBONES = {
    "dinov2_vits14": ("timm", "vit_small_patch14_dinov2.lvd142m"),
    "dinov2_vitb14": ("timm", "vit_base_patch14_dinov2.lvd142m"),
    "clip_vitb32": ("open_clip", ("ViT-B-32-quickgelu", "openai")),
    # Source-training backbones:
    "resnet50": ("timm", "resnet50.a1_in1k"),
    "vit_b16": ("timm", "vit_base_patch16_224.augreg2_in21k_ft_in1k"),
}
