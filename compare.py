"""Full method comparison with bootstrap CIs + timing (reviewer M1, M2, M4, M5).

Methods (all source-free, backprop-free, on cached features):
  - source prototypes / k-NN            (non-adapting baselines)
  - Ours: uniform-Sinkhorn (proto/kNN)  (balanced assignment, uniform prior)
  - OTTER: Sinkhorn + BBSE-estimated prior   (Shin et al., NeurIPS 2024, our reimpl.)
  - Sinkhorn + MLLS-estimated prior     (label-shift estimator variant)
  - T3A: confident-support template adjustment  (Iwasawa & Matsuo, 2021)

TENT/SHOT update backbone norm params via backprop and are a different class of method
(they un-freeze the backbone); TENT is provided separately in tent.py. Everything here is
cached-feature / CPU / deterministic point-estimates with bootstrap CIs.

    python compare.py --backbone dinov2_vitl14 --target plantdoc --bootstrap 200
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from class_alignment import NUM_CLASSES
from tta import _load, l2norm, source_prototypes, sinkhorn, knn_logits, softmax
import label_shift as LS
import bootstrap as BS


def _knn_scores(Xt, Xs, ys, C, k, temp):
    return knn_logits(Xt, Xs, ys, C, k=k, temp=temp)


def build_methods(Xs, ys, C, temp=0.1, k=20):
    """Return {name: predict_fn(X)->preds}. Xs/ys are the fixed source; X is the (possibly
    bootstrap-resampled) target feature matrix."""
    Xs_n = l2norm(Xs)
    P = source_prototypes(Xs, ys, C)
    # source confusion for BBSE/MLLS (a shipped C×C statistic; computed once from source)
    src_logits = (Xs_n @ P.T) / temp
    conf, src_prior = LS.source_confusion(src_logits, ys, C)

    def proto_logits(X):
        return (l2norm(X) @ P.T) / temp

    def m_source_proto(X):
        return proto_logits(X).argmax(1)

    def m_knn(X):
        return _knn_scores(l2norm(X), Xs_n, ys, C, k, temp).argmax(1)

    def m_uniform_proto(X):
        return sinkhorn(proto_logits(X)).argmax(1)

    def m_uniform_knn(X):
        return sinkhorn(_knn_scores(l2norm(X), Xs_n, ys, C, k, temp)).argmax(1)

    def m_otter(X):                      # OTTER = OT assignment to a BBSE-estimated prior
        L = proto_logits(X)
        r = LS.bbse_prior(conf, L, src_prior)
        return sinkhorn(L, r=r).argmax(1)

    def m_mlls(X):
        L = proto_logits(X)
        r = LS.mlls_prior(L, src_prior)
        return sinkhorn(L, r=r).argmax(1)

    def m_t3a(X, k_sup=20):
        Xn = l2norm(X)
        logits = (Xn @ P.T) / temp
        probs = np.apply_along_axis(softmax, 1, logits)
        ent = -(probs * np.log(probs + 1e-12)).sum(1)
        pred0 = logits.argmax(1)
        cent = P.copy()
        for c in range(C):
            m = np.where(pred0 == c)[0]
            if len(m):
                order = m[np.argsort(ent[m])][:k_sup]
                sup = np.vstack([P[c][None], Xn[order]])
                cent[c] = l2norm(sup.mean(0, keepdims=True))[0]
        return (Xn @ cent.T).argmax(1)

    return {
        "source prototypes": m_source_proto,
        "k-NN (k=20)": m_knn,
        "T3A": m_t3a,
        "OTTER (BBSE prior)": m_otter,
        "Sinkhorn + MLLS prior": m_mlls,
        "Ours: uniform-Sinkhorn (proto)": m_uniform_proto,
        "Ours: uniform-Sinkhorn (kNN)": m_uniform_knn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitl14")
    ap.add_argument("--target", default="plantdoc")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.1)
    args = ap.parse_args()

    Xs, ys = _load(args.backbone, "plantvillage")
    Xt, yt = _load(args.backbone, args.target)
    C = NUM_CLASSES
    methods = build_methods(Xs, ys, C, temp=args.temp)

    print(f"\n=== {args.backbone} → {args.target}  (n={len(yt)}, bootstrap={args.bootstrap}) ===")
    print("method                             accuracy [95% CI]     macro-F1 [95% CI]      worst-group [95% CI]")
    for name, fn in methods.items():
        t0 = time.perf_counter()
        _ = fn(Xt)                                   # timing on the full target set
        dt = (time.perf_counter() - t0) * 1000
        r = BS.bootstrap_ci(fn, Xt, yt, B=args.bootstrap)
        print(BS.fmt(name, r) + f"   [{dt:6.0f} ms]")


if __name__ == "__main__":
    main()
