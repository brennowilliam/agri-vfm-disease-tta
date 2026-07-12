"""Full method comparison with bootstrap CIs + timing + round-2 controls.

Round-2 rigor fixes baked in:
 - Held-out source split for the BBSE/MLLS confusion matrix (was in-sample → unfair to OTTER).
 - BCTS-calibrated MLLS (uncalibrated MLLS is a known strawman).
 - Diagnostics that decide the headline: ORACLE true-prior Sinkhorn, RANDOM-logit control,
   uniform<->estimated SHRINKAGE sweep, and a prior-distance (TV to true prior) table.
 - Within-backbone sklearn k-NN classifier baseline.
 - Frozen worst-group estimand + balanced-accuracy + dead-class count via the updated metrics/bootstrap.

    python compare.py --backbone dinov2_vitl14 --target plantdoc --bootstrap 2000
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

from class_alignment import NUM_CLASSES, IDX_TO_CLASS
from tta import _load, l2norm, source_prototypes, sinkhorn, knn_logits, softmax
import label_shift as LS
import bootstrap as BS
import metrics as M


def _stratified_split(ys, C, frac_fit=0.7, seed=0):
    rng = np.random.default_rng(seed)
    fit, cal = [], []
    for c in range(C):
        idx = np.where(ys == c)[0]
        rng.shuffle(idx)
        k = max(1, int(len(idx) * frac_fit))
        fit += idx[:k].tolist(); cal += idx[k:].tolist() if len(idx) > 1 else idx[:1].tolist()
    return np.array(sorted(fit)), np.array(sorted(cal))


def build(Xs, ys, C, temp=0.1, k=20):
    """Prototypes from a source FIT split; confusion/calibration from a held-out source CAL split."""
    fit, cal = _stratified_split(ys, C)
    Xs_fit_n = l2norm(Xs[fit])
    P = source_prototypes(Xs[fit], ys[fit], C)
    cal_logits = (l2norm(Xs[cal]) @ P.T) / temp                 # HELD-OUT source logits
    conf, src_prior = LS.source_confusion(cal_logits, ys[cal], C)
    T, b = LS.bcts_calibrate(cal_logits, ys[cal], C)            # BCTS on held-out source
    Xs_all_n = l2norm(Xs)
    ctx = dict(P=P, conf=conf, src_prior=src_prior, T=T, b=b, Xs_all_n=Xs_all_n, ys=ys,
               temp=temp, k=k, C=C)

    def proto_logits(X):
        return (l2norm(X) @ P.T) / temp

    ctx["proto_logits"] = proto_logits

    def m_source_proto(X): return proto_logits(X).argmax(1)
    def m_knn_affinity(X): return knn_logits(l2norm(X), Xs_all_n, ys, C, k, temp).argmax(1)
    def m_uniform_proto(X): return sinkhorn(proto_logits(X)).argmax(1)
    def m_uniform_knn(X): return sinkhorn(knn_logits(l2norm(X), Xs_all_n, ys, C, k, temp)).argmax(1)

    def m_otter(X):                       # OTTER = OT to a BBSE-estimated prior (held-out confusion)
        L = proto_logits(X); r = LS.bbse_prior(conf, L, src_prior)
        return sinkhorn(L, r=r).argmax(1)

    def m_mlls_cal(X):                     # calibrated MLLS prior -> Sinkhorn
        L = proto_logits(X)
        r = LS.mlls_prior(LS.apply_calibration(L, T, b), src_prior)
        return sinkhorn(L, r=r).argmax(1)

    # sklearn k-NN classifier (the "standard" k-NN, within THIS backbone)
    knn_clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn_clf.fit(normalize(Xs), ys)
    def m_knn_clf(X): return knn_clf.predict(normalize(X))

    methods = {
        "source prototypes": m_source_proto,
        "k-NN classifier (sklearn)": m_knn_clf,
        "k-NN affinity (scorer)": m_knn_affinity,
        "OTTER (BBSE, held-out)": m_otter,
        "Sinkhorn + calib-MLLS": m_mlls_cal,
        "Ours: uniform-Sinkhorn (proto)": m_uniform_proto,
        "Ours: uniform-Sinkhorn (kNN)": m_uniform_knn,
    }
    return methods, ctx


def diagnostics(ctx, Xt, yt):
    """Point-estimate controls that decide the headline (use yt; not deployable methods)."""
    C = ctx["C"]
    L = ctx["proto_logits"](Xt)
    true_prior = np.bincount(yt, minlength=C).astype(float); true_prior /= true_prior.sum()
    uniform = np.ones(C) / C
    bbse = LS.bbse_prior(ctx["conf"], L, ctx["src_prior"])
    mlls_c = LS.mlls_prior(LS.apply_calibration(L, ctx["T"], ctx["b"]), ctx["src_prior"])

    print("\n--- Prior-distance diagnostic (TV to TRUE target prior; lower = better estimate) ---")
    for name, r in [("uniform", uniform), ("BBSE", bbse), ("calib-MLLS", mlls_c)]:
        print(f"  TV({name:10s}, true) = {LS.tv_distance(r, true_prior):.3f}")

    print("\n--- Oracle / control (point estimates, min_support=10) ---")
    def rep(tag, pred):
        s = M.summarize(yt, pred, C, min_support=10)
        print(f"  {tag:34s} acc {s['accuracy']*100:5.1f}  mF1 {s['macro_f1']*100:5.1f}  "
              f"bal {s['balanced_acc']*100:5.1f}  WG {s['worst_group_supported']*100:5.1f}  "
              f"dead {s['dead_classes']}/{s['n_eligible']}")
    rep("ORACLE Sinkhorn (true prior)", sinkhorn(L, r=true_prior).argmax(1))
    # random-logit control: shuffle features so class signal is destroyed, then uniform-Sinkhorn
    rng = np.random.default_rng(0)
    Xrand = l2norm(rng.standard_normal(size=Xt.shape))
    rep("RANDOM-logit + uniform-Sinkhorn", sinkhorn((Xrand @ ctx["P"].T) / ctx["temp"]).argmax(1))

    print("\n--- Shrinkage sweep  r = (1-lam)*uniform + lam*BBSE  ---")
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = (1 - lam) * uniform + lam * bbse
        rep(f"lambda={lam:.2f}", sinkhorn(L, r=r).argmax(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitl14")
    ap.add_argument("--target", default="plantdoc")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--temp", type=float, default=0.1)
    args = ap.parse_args()

    Xs, ys = _load(args.backbone, "plantvillage")
    Xt, yt = _load(args.backbone, args.target)
    yt = yt.astype(int)
    C = NUM_CLASSES
    methods, ctx = build(Xs, ys, C, temp=args.temp)

    print(f"\n=== {args.backbone} -> {args.target}  (n={len(yt)}, bootstrap={args.bootstrap}) ===")
    print("Hyperparameters fixed a priori (not tuned on target): temp=0.1, sinkhorn eps=0.05, iters=50, k=20, min_support=10")
    for name, fn in methods.items():
        t0 = time.perf_counter(); _ = fn(Xt); dt = (time.perf_counter() - t0) * 1000
        r = BS.bootstrap_ci(fn, Xt, yt, B=args.bootstrap)
        print(BS.fmt(name, r) + f"   [{dt:6.0f} ms]")

    diagnostics(ctx, Xt, yt)


if __name__ == "__main__":
    main()
