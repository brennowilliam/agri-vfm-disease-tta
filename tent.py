"""TENT baseline (Wang et al., ICLR 2021) — the canonical backprop test-time adaptation.

Reviewer M2 wants the standard TTA comparators. TENT differs in kind from our method: it
UN-FREEZES the backbone's LayerNorm affine params and updates them by entropy minimisation,
so it needs the model, images, GPU, and back-propagation (not cached features). We keep the
classifier fixed as the source prototypes, matching our setup, and adapt online over the
target stream.

    python tent.py --backbone dinov2_vitb14 --target plantdoc --lr 1e-4
"""
from __future__ import annotations

import argparse
import os

# Reduce CUDA fragmentation OOMs on the 15 GB T4 (must be set before torch loads CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from class_alignment import NUM_CLASSES, IDX_TO_CLASS
from config import CFG
import datasets as D
import metrics as M
from tta import _load, l2norm, source_prototypes
from extract_features import load_backbone, DOMAIN_BUILDERS


def configure_tent(model):
    """Freeze everything except LayerNorm affine params; put those in train mode."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            m.train()                       # LN uses no running stats, but keep affine trainable
            params += [p for p in m.parameters() if p.requires_grad]
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitb14",
                    choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    ap.add_argument("--target", default="plantdoc", choices=["plantdoc", "plantwild"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = auto (16 for ViT-L to fit a 15GB T4 under backprop, else 64)")
    args = ap.parse_args()
    if args.batch == 0:
        args.batch = 8 if args.backbone == "dinov2_vitl14" else 64

    D.seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C = NUM_CLASSES

    # Fixed classifier = source prototypes (from cached source features).
    Xs, ys = _load(args.backbone, "plantvillage")
    P = torch.tensor(source_prototypes(Xs, ys, C), dtype=torch.float32, device=device)  # (C,d)

    model, preprocess, _ = load_backbone(args.backbone, device)
    params = configure_tent(model)
    opt = torch.optim.Adam(params, lr=args.lr)
    print(f"TENT: {args.backbone} → {args.target} | {sum(p.numel() for p in params)} LN params | lr {args.lr}")

    ds = DOMAIN_BUILDERS[args.target](transform=preprocess, return_path=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=CFG.num_workers)

    if device == "cuda":
        torch.cuda.empty_cache()
    use_amp = device == "cuda"
    ys_all, yp_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            feat = model(imgs)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
            logits = (feat.float() @ P.T) / args.temp
            probs = logits.softmax(1)
            ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean()
        # predict (before this batch's update), then adapt on entropy
        yp_all.append(logits.argmax(1).detach().cpu().numpy())
        ys_all.append(labels.numpy())
        opt.zero_grad(); ent.backward(); opt.step()

    s = M.summarize(np.concatenate(ys_all), np.concatenate(yp_all), C, IDX_TO_CLASS)
    M.print_summary(f"TENT {args.backbone} → {args.target} (online, LN-affine)", s)
    print("Note: single online run; stochastic in batch order. Compare to source-only and our method.")


if __name__ == "__main__":
    main()
