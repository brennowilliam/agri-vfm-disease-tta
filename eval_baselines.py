"""Non-adapting foundation-model baselines from cached features.

These contextualize "why a foundation model helps" before any test-time adaptation:
  - DINOv2 (or any backbone) linear probe trained on PlantVillage features, eval on PlantDoc
  - DINOv2 kNN (source prototypes)
  - CLIP zero-shot (text prompts) — run with --backbone clip_vitb32

    python eval_baselines.py --backbone dinov2_vitb14
    python eval_baselines.py --backbone clip_vitb32 --clip_zeroshot
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from class_alignment import NUM_CLASSES, IDX_TO_CLASS, clip_prompts
from config import CFG, BACKBONES
import metrics as M


def _load(backbone: str, domain: str):
    path = os.path.join(CFG.feat_dir, backbone, f"{domain}.npz")
    d = np.load(path, allow_pickle=True)
    return d["feats"], d["labels"]


def linear_probe(backbone: str):
    Xtr, ytr = _load(backbone, "plantvillage")
    Xte, yte = _load(backbone, "plantdoc")
    Xtr, Xte = normalize(Xtr), normalize(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    M.print_summary(f"{backbone} linear-probe (PV->PlantDoc, field)",
                    M.summarize(yte, clf.predict(Xte), NUM_CLASSES, IDX_TO_CLASS))


def knn(backbone: str, k: int = 20):
    Xtr, ytr = _load(backbone, "plantvillage")
    Xte, yte = _load(backbone, "plantdoc")
    Xtr, Xte = normalize(Xtr), normalize(Xte)
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    clf.fit(Xtr, ytr)
    M.print_summary(f"{backbone} kNN(k={k}) (PV->PlantDoc, field)",
                    M.summarize(yte, clf.predict(Xte), NUM_CLASSES, IDX_TO_CLASS))


def clip_zeroshot():
    """CLIP zero-shot with per-class text prompts (no training)."""
    import open_clip
    import torch
    arch, pretrained = BACKBONES["clip_vitb32"][1]
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    with torch.no_grad():
        text = tokenizer([f"a photo of {p}" for p in clip_prompts()]).to(device)
        tfeat = model.encode_text(text)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    Xte, yte = _load("clip_vitb32", "plantdoc")
    ifeat = torch.tensor(Xte, dtype=torch.float32, device=device)
    ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
    preds = (ifeat @ tfeat.T).argmax(1).cpu().numpy()
    M.print_summary("CLIP zero-shot (PlantDoc, field)",
                    M.summarize(yte, preds, NUM_CLASSES, IDX_TO_CLASS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--clip_zeroshot", action="store_true",
                    help="run CLIP zero-shot (requires --backbone clip_vitb32 features cached)")
    args = ap.parse_args()

    if args.clip_zeroshot:
        clip_zeroshot()
    else:
        linear_probe(args.backbone)
        knn(args.backbone)


if __name__ == "__main__":
    main()
