"""Phase 2 — extract and cache frozen-backbone features for every dataset.

Features are the substrate for all TTA methods (backprop-free prototype/cache updates
run on top of these), so extract once and reuse. Cheap: forward pass only, <1 GB total.

    python extract_features.py --backbone dinov2_vitb14 --domains plantvillage plantdoc
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, BACKBONES
import datasets as D


def load_backbone(name: str, device: str):
    """Return (model, preprocess_transform, embed_dim). Frozen, eval mode."""
    kind, ident = BACKBONES[name]
    if kind == "timm":
        import timm
        from timm.data import resolve_data_config, create_transform
        model = timm.create_model(ident, pretrained=True, num_classes=0,
                                  dynamic_img_size=True).to(device).eval()
        cfg = resolve_data_config({}, model=model)
        cfg["input_size"] = (3, CFG.img_size, CFG.img_size)
        preprocess = create_transform(**cfg, is_training=False)
        embed_dim = model.num_features
    elif kind == "open_clip":
        import open_clip
        arch, pretrained = ident
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model = model.visual.to(device).eval()      # image tower only
        embed_dim = model.output_dim if hasattr(model, "output_dim") else None
    else:
        raise ValueError(kind)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, embed_dim


DOMAIN_BUILDERS = {
    "plantvillage": D.plantvillage,
    "plantdoc": D.plantdoc,
    "plantwild": D.plantwild,
}


@torch.no_grad()
def extract(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feats, labels, paths = [], [], []
    use_amp = device == "cuda"
    for imgs, lbls, pths in tqdm(loader, desc="extract", leave=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(imgs.to(device))
        if isinstance(out, (tuple, list)):
            out = out[0]
        feats.append(out.float().cpu().numpy())
        labels.append(np.asarray(lbls))
        paths.extend(pths)
    return np.concatenate(feats), np.concatenate(labels), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitb14",
                    choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "clip_vitb32"])
    ap.add_argument("--domains", nargs="+", default=["plantvillage", "plantdoc"],
                    choices=list(DOMAIN_BUILDERS))
    args = ap.parse_args()

    D.seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, dim = load_backbone(args.backbone, device)
    print(f"Backbone {args.backbone} | embed_dim={dim} | device={device}")

    out_dir = os.path.join(CFG.feat_dir, args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    for dom in args.domains:
        ds = DOMAIN_BUILDERS[dom](transform=preprocess, return_path=True)
        loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers)
        feats, labels, paths = extract(model, loader, device)
        np.savez(os.path.join(out_dir, f"{dom}.npz"),
                 feats=feats, labels=labels, paths=np.array(paths))
        print(f"  {dom}: {feats.shape} -> {out_dir}/{dom}.npz")


if __name__ == "__main__":
    main()
