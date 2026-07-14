"""Exploratory data analysis — characterise the datasets and quantify the shift.

Produces the evidence behind the paper's claims:
 1. Class support & imbalance per dataset (why worst-group matters).
 2. Source->target LABEL SHIFT: source prior vs target prior vs uniform, with TV distances
    (the same quantity the prior-distance control uses — this is *why* estimators fail).
 3. A t-SNE of frozen DINOv2 features showing the lab->field DOMAIN GAP and class structure.

Runs on the cached DINOv2 ViT-L features. Saves figures + prints a summary table.

    python eda.py --backbone dinov2_vitl14 --outdir /kaggle/working/eda
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from class_alignment import NUM_CLASSES, IDX_TO_CLASS
from tta import _load, l2norm

BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8f4fbf"


def prior(labels, C):
    p = np.bincount(labels, minlength=C).astype(float)
    return p / p.sum()


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitl14")
    ap.add_argument("--outdir", default="/kaggle/working/eda")
    ap.add_argument("--tsne_n", type=int, default=2500, help="per-domain subsample for t-SNE")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    C = NUM_CLASSES
    rng = np.random.default_rng(0)

    doms = {}
    for d in ["plantvillage", "plantdoc", "plantwild"]:
        try:
            X, y = _load(args.backbone, d)
            doms[d] = (X, y.astype(int))
        except FileNotFoundError:
            print(f"[skip] no cached features for {d}")

    # ---------- 1. class support & imbalance ----------
    print("\n" + "=" * 70)
    print("1. CLASS SUPPORT & IMBALANCE")
    print("=" * 70)
    names = [IDX_TO_CLASS[i] for i in range(C)]
    print(f"{'class':30s}" + "".join(f"{d[:9]:>11s}" for d in doms))
    supports = {d: np.bincount(y, minlength=C) for d, (X, y) in doms.items()}
    for c in range(C):
        print(f"{names[c]:30s}" + "".join(f"{supports[d][c]:>11d}" for d in doms))
    print("-" * (30 + 11 * len(doms)))
    for d in doms:
        s = supports[d][supports[d] > 0]
        print(f"{d}: total {s.sum()}, classes {len(s)}, min {s.min()}, "
              f"max {s.max()}, median {int(np.median(s))}, imbalance ratio {s.max()/s.min():.1f}x")

    # bar chart (targets)
    fig, axes = plt.subplots(1, len(doms), figsize=(5 * len(doms), 4.2), sharey=False)
    if len(doms) == 1:
        axes = [axes]
    for ax, (d, (X, y)) in zip(axes, doms.items()):
        sup = supports[d]
        order = np.argsort(sup)[::-1]
        present = [c for c in order if sup[c] > 0]
        ax.bar(range(len(present)), [sup[c] for c in present], color=GREEN, alpha=.85)
        ax.set_title(f"{d}  (n={sup.sum()}, {len(present)} classes)")
        ax.set_xlabel("class (sorted by frequency)"); ax.set_ylabel("images")
        ax.grid(axis="y", alpha=.3)
    fig.suptitle("Class imbalance — long-tailed in every dataset (why worst-group matters)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"eda_class_support.{ext}"), dpi=180, bbox_inches="tight")
    plt.close(fig); print("wrote eda_class_support")

    # ---------- 2. label shift ----------
    print("\n" + "=" * 70)
    print("2. LABEL SHIFT (source -> target)")
    print("=" * 70)
    if "plantvillage" in doms:
        src_p = prior(doms["plantvillage"][1], C)
        uni = np.ones(C) / C
        for d in ["plantdoc", "plantwild"]:
            if d in doms:
                tgt_p = prior(doms[d][1], C)
                print(f"{d:12s}: TV(source, target) = {tv(src_p, tgt_p):.3f} | "
                      f"TV(uniform, target) = {tv(uni, tgt_p):.3f}")
        print("(Note: TV(uniform,target) here is the 'best case' the method's uniform prior aims at; "
              "the estimators' priors land FARTHER from target than uniform — see the prior-distance control.)")
        # visualise the three distributions for PlantDoc
        if "plantdoc" in doms:
            tgt_p = prior(doms["plantdoc"][1], C)
            present = [c for c in range(C) if supports["plantdoc"][c] >= 10]
            present.sort(key=lambda c: tgt_p[c], reverse=True)
            x = np.arange(len(present)); w = 0.4
            fig, ax = plt.subplots(figsize=(10, 3.8))
            ax.bar(x - w/2, [tgt_p[c]*100 for c in present], w, label="PlantDoc (target)", color=VERM)
            ax.bar(x + w/2, [src_p[c]*100 for c in present], w, label="PlantVillage (source)", color=BLUE, alpha=.8)
            ax.axhline(100/len(present), ls="--", color=GREEN, label="uniform")
            ax.set_xticks(x); ax.set_xticklabels([IDX_TO_CLASS[c].replace("_", " ") for c in present],
                                                 rotation=60, ha="right", fontsize=7)
            ax.set_ylabel("class share (%)"); ax.legend(fontsize=8)
            ax.set_title("Label shift: source vs field-target class distribution")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(args.outdir, f"eda_label_shift.{ext}"), dpi=180, bbox_inches="tight")
            plt.close(fig); print("wrote eda_label_shift")

    # ---------- 3. domain gap t-SNE ----------
    print("\n" + "=" * 70)
    print("3. DOMAIN GAP (t-SNE of frozen DINOv2 features)")
    print("=" * 70)
    if "plantvillage" in doms and "plantdoc" in doms:
        def sub(d, n):
            X, y = doms[d]; idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
            return l2norm(X[idx]), y[idx]
        Xs, ys = sub("plantvillage", args.tsne_n)
        Xt, yt = sub("plantdoc", args.tsne_n)
        Xall = np.vstack([Xs, Xt])
        dom = np.array([0] * len(Xs) + [1] * len(Xt))
        yall = np.concatenate([ys, yt])
        print(f"t-SNE on {len(Xall)} points (PCA->50->t-SNE)...")
        Z = PCA(n_components=50, random_state=0).fit_transform(Xall)
        Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(Z)

        # (a) coloured by domain
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
        a1.scatter(Z[dom == 0, 0], Z[dom == 0, 1], s=6, c=BLUE, alpha=.45, label="PlantVillage (lab)")
        a1.scatter(Z[dom == 1, 0], Z[dom == 1, 1], s=6, c=VERM, alpha=.55, label="PlantDoc (field)")
        a1.set_title("By domain — the lab→field gap"); a1.legend(fontsize=8); a1.set_xticks([]); a1.set_yticks([])
        # (b) coloured by a few frequent shared classes
        top = [c for c in np.argsort(np.bincount(yt, minlength=C))[::-1] if (yt == c).sum() > 0][:6]
        palette = [BLUE, ORANGE, GREEN, VERM, PURPLE, "#444"]
        for col, c in zip(palette, top):
            m = yall == c
            a2.scatter(Z[m, 0], Z[m, 1], s=7, c=col, alpha=.6, label=IDX_TO_CLASS[c].replace("_", " "))
        a2.set_title("By class (6 most frequent)"); a2.legend(fontsize=7, markerscale=1.5)
        a2.set_xticks([]); a2.set_yticks([])
        fig.suptitle("Frozen DINOv2 features: domains partly overlap, classes form structure", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.outdir, f"eda_tsne.{ext}"), dpi=180, bbox_inches="tight")
        plt.close(fig); print("wrote eda_tsne")

    print(f"\nAll EDA outputs -> {args.outdir}")


if __name__ == "__main__":
    main()
