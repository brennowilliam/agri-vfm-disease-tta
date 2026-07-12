"""Hyperparameter sensitivity for uniform-Sinkhorn (reviewer A4 / MAJOR-3).

The method has three knobs — temperature tau, Sinkhorn eps, iterations T. In a source-free setting
they cannot be tuned on target labels, so we FIX them a priori (tau=0.1, eps=0.05, T=50) and show
here that the result is not knife-edge: accuracy / macro-F1 / worst-group are stable over a range.

    python sensitivity.py --backbone dinov2_vitl14 --target plantdoc
"""
from __future__ import annotations

import argparse

import numpy as np

from class_alignment import NUM_CLASSES
from tta import _load, l2norm, source_prototypes, sinkhorn
import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitl14")
    ap.add_argument("--target", default="plantdoc")
    args = ap.parse_args()
    C = NUM_CLASSES
    Xs, ys = _load(args.backbone, "plantvillage")
    Xt, yt = _load(args.backbone, args.target); yt = yt.astype(int)
    Xt = l2norm(Xt)
    P = source_prototypes(Xs, ys, C)

    def run(temp, eps, it):
        pred = sinkhorn((Xt @ P.T) / temp, n_iter=it, eps=eps).argmax(1)
        s = M.summarize(yt, pred, C, min_support=10)
        return s["accuracy"] * 100, s["macro_f1"] * 100, s["worst_group_supported"] * 100

    print(f"\n=== Sensitivity: {args.backbone} -> {args.target} (default tau=0.1, eps=0.05, T=50) ===")
    print("\n temp   acc   mF1    WG      (eps=0.05, T=50)")
    for temp in [0.05, 0.07, 0.1, 0.15, 0.2]:
        a, f, w = run(temp, 0.05, 50)
        print(f" {temp:5.2f} {a:5.1f} {f:5.1f} {w:5.1f}" + ("   <- default" if temp == 0.1 else ""))
    print("\n eps    acc   mF1    WG      (temp=0.1, T=50)")
    for eps in [0.02, 0.03, 0.05, 0.08, 0.12]:
        a, f, w = run(0.1, eps, 50)
        print(f" {eps:5.2f} {a:5.1f} {f:5.1f} {w:5.1f}" + ("   <- default" if eps == 0.05 else ""))
    print("\n iters  acc   mF1    WG      (temp=0.1, eps=0.05)")
    for it in [10, 25, 50, 100]:
        a, f, w = run(0.1, 0.05, it)
        print(f" {it:5d} {a:5.1f} {f:5.1f} {w:5.1f}" + ("   <- default" if it == 50 else ""))


if __name__ == "__main__":
    main()
