"""Per-disease recovery table for the agronomic discussion (CEA reviewer M2).

Dumps, for every shared crop-disease class, the per-class recall of source prototypes vs
+uniform-Sinkhorn (DINOv2 ViT-L), sorted by gain. Feeds the paper's per-disease table naming
which diseases the balancing rescues.

    python per_disease.py --backbone dinov2_vitl14 --target plantdoc
"""
from __future__ import annotations

import argparse

import numpy as np

from class_alignment import NUM_CLASSES, IDX_TO_CLASS
from tta import _load, l2norm, source_prototypes, sinkhorn
import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitl14")
    ap.add_argument("--target", default="plantdoc")
    ap.add_argument("--temp", type=float, default=0.1)
    args = ap.parse_args()
    C = NUM_CLASSES
    Xs, ys = _load(args.backbone, "plantvillage")
    Xt, yt = _load(args.backbone, args.target); yt = yt.astype(int)
    P = source_prototypes(Xs, ys, C)
    L = (l2norm(Xt) @ P.T) / args.temp
    src = M.per_class_accuracy(yt, L.argmax(1), C)
    snk = M.per_class_accuracy(yt, sinkhorn(L).argmax(1), C)
    sup = M.class_support(yt, C)

    rows = []
    for c in range(C):
        if sup[c] >= 10 and not np.isnan(src[c]):
            rows.append((IDX_TO_CLASS[c], int(sup[c]), src[c] * 100, snk[c] * 100,
                         (snk[c] - src[c]) * 100))
    rows.sort(key=lambda r: r[4], reverse=True)

    print(f"\n=== Per-disease recovery: {args.backbone} -> {args.target} (n>=10 classes) ===")
    print(f"{'class':30s} {'n':>4s} {'src%':>6s} {'+Sink%':>7s} {'delta':>7s}")
    for name, n, s, k, d in rows:
        print(f"{name:30s} {n:4d} {s:6.1f} {k:7.1f} {d:+7.1f}")
    n_rescued = sum(1 for r in rows if r[2] < 5 and r[3] >= 5)
    print(f"\nclasses rescued from 'dead' (recall <5% -> >=5%): {n_rescued}")


if __name__ == "__main__":
    main()
