"""Generate the two paper figures. Run on Kaggle after features are cached.

  Fig 1  scaling curve   — worst-group & macro-F1 vs backbone, both targets (from grid numbers)
  Fig 2  tail recovery   — per-class accuracy, source-prototypes vs +Sinkhorn (recomputed)

    python figures.py --outdir /kaggle/working/figures

Colours are colourblind-safe (Wong palette). Saves PNG + PDF.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from class_alignment import NUM_CLASSES, IDX_TO_CLASS
import metrics as M
from tta import _load, source_prototypes, sinkhorn, l2norm

# Wong colourblind-safe palette
BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"

# --- Fig 1 data: worst-group (n>=10) and macro-F1, source→Sinkhorn, from the results grid ---
GRID = {  # backbone: {target: (src_wg, sink_wg, src_mf1, sink_mf1)}
    "ViT-S": {"PlantDoc": (0.0, 7.69, 42.49, 45.40), "PlantWild": (1.27, 11.79, 35.85, 39.66)},
    "ViT-B": {"PlantDoc": (0.0, 15.38, 46.26, 50.24), "PlantWild": (1.27, 17.50, 41.20, 44.88)},
    "ViT-L": {"PlantDoc": (0.0, 17.27, 43.27, 51.74), "PlantWild": (0.0, 10.04, 38.19, 45.14)},
}
ORDER = ["ViT-S", "ViT-B", "ViT-L"]


def fig1_scaling(outdir: str) -> None:
    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, tgt in zip(axes, ["PlantDoc", "PlantWild"]):
        wg_src = [GRID[b][tgt][0] for b in ORDER]
        wg_snk = [GRID[b][tgt][1] for b in ORDER]
        mf_src = [GRID[b][tgt][2] for b in ORDER]
        mf_snk = [GRID[b][tgt][3] for b in ORDER]
        ax.plot(x, wg_src, "o--", color=VERM, alpha=0.5, label="worst-group (source)")
        ax.plot(x, wg_snk, "o-", color=VERM, label="worst-group (+Sinkhorn)")
        ax.plot(x, mf_src, "s--", color=BLUE, alpha=0.5, label="macro-F1 (source)")
        ax.plot(x, mf_snk, "s-", color=BLUE, label="macro-F1 (+Sinkhorn)")
        ax.set_title(tgt)
        ax.set_xticks(x); ax.set_xticklabels([f"DINOv2\n{b}" for b in ORDER])
        ax.set_ylim(0, 60); ax.grid(alpha=0.3)
        if tgt == "PlantDoc":
            ax.set_ylabel("score (%)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Balanced assignment lifts macro-F1 and the worst-group tail across backbones",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig1_scaling.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_scaling")


def fig2_tail_recovery(outdir: str, backbone: str = "dinov2_vitl14", target: str = "plantdoc") -> None:
    Xs, ys = _load(backbone, "plantvillage")
    Xt, yt = _load(backbone, target)
    Xt = l2norm(Xt)
    P = source_prototypes(Xs, ys, NUM_CLASSES)
    logits = Xt @ P.T
    pred_src = logits.argmax(1)
    pred_snk = sinkhorn(logits).argmax(1)

    acc_src = M.per_class_accuracy(yt, pred_src, NUM_CLASSES)
    acc_snk = M.per_class_accuracy(yt, pred_snk, NUM_CLASSES)
    sup = M.class_support(yt, NUM_CLASSES)
    keep = [c for c in range(NUM_CLASSES) if sup[c] >= 10]
    keep.sort(key=lambda c: acc_snk[c] - acc_src[c], reverse=True)  # biggest gains first

    names = [IDX_TO_CLASS[c].replace("_", " ") for c in keep]
    y = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(keep))))
    ax.barh(y - 0.2, [acc_src[c] * 100 for c in keep], height=0.4, color=VERM,
            label="source prototypes")
    ax.barh(y + 0.2, [acc_snk[c] * 100 for c in keep], height=0.4, color=GREEN,
            label="+ Sinkhorn (balanced)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("per-class accuracy (%)")
    ax.set_title(f"Per-class tail recovery ({backbone} → {target})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig2_tail_{target}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig2_tail_{target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/kaggle/working/figures")
    ap.add_argument("--backbone", default="dinov2_vitl14")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    fig1_scaling(args.outdir)
    fig2_tail_recovery(args.outdir, args.backbone, "plantdoc")
    fig2_tail_recovery(args.outdir, args.backbone, "plantwild")
    print("figures ->", args.outdir)


if __name__ == "__main__":
    main()
