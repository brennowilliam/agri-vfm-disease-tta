"""Aligned datasets for the lab->field benchmark.

PlantVillage / PlantDoc are mapped to the unified shared label space via
`class_alignment.resolve_dirs`. PlantWild and Cassava are provided as extra
target domains (PlantWild for scale; Cassava as an unseen-crop OOD probe).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from class_alignment import resolve_dirs
from config import CFG

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


def seed_everything(seed: int = CFG.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(train: bool, img_size: int = CFG.img_size, preprocess=None):
    """Return a transform. If `preprocess` (a model's own transform) is given, use it."""
    if preprocess is not None:
        return preprocess
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(CFG.imagenet_mean, CFG.imagenet_std),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(CFG.imagenet_mean, CFG.imagenet_std),
    ])


class AlignedFolderDataset(Dataset):
    """Images from a set of class directories, each mapped to a unified label index."""

    def __init__(self, dir_to_label: dict[str, int], transform=None, return_path: bool = False):
        self.transform = transform
        self.return_path = return_path
        self.samples: list[tuple[str, int]] = []
        for d, label in dir_to_label.items():
            for fn in sorted(os.listdir(d)):
                if fn.endswith(_IMG_EXT):
                    self.samples.append((os.path.join(d, fn), label))
        if not self.samples:
            raise RuntimeError(f"No images found for {len(dir_to_label)} class dirs.")
        self.labels = np.array([lbl for _, lbl in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, label, path
        return img, label


def plantvillage(transform=None, return_path: bool = False) -> AlignedFolderDataset:
    return AlignedFolderDataset(resolve_dirs(CFG.pv_root, "pv"), transform, return_path)


def plantdoc(transform=None, return_path: bool = False, split: str = "all") -> AlignedFolderDataset:
    """PlantDoc field target. split='all' (default) uses train+test = full field set."""
    return AlignedFolderDataset(
        resolve_dirs(CFG.plantdoc_root, "plantdoc", split), transform, return_path)


def stratified_split(ds: AlignedFolderDataset, val_frac: float = 0.2, seed: int = CFG.seed):
    """Class-stratified train/val index split (for PlantVillage source training)."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(ds.labels):
        idx = np.where(ds.labels == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * val_frac))
        val_idx.extend(idx[:cut].tolist())
        train_idx.extend(idx[cut:].tolist())
    return sorted(train_idx), sorted(val_idx)
