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


def _onehot_logits(preds, C):
    """Turn hard predictions into a trivial logit matrix so a non-transductive method plugs into
    the logit-based bootstrap (argmax recovers the prediction)."""
    L = np.full((len(preds), C), -1.0); L[np.arange(len(preds)), preds] = 1.0
    return L


def build(Xs, ys, C, Xt, temp=0.1, k=20):
    """Precompute per-sample logits ONCE for the full target; return (assign_fn, L) per method plus
    a context dict. Prototypes from a source FIT split; confusion/calibration from a held-out CAL split."""
    fit, cal = _stratified_split(ys, C)
    P = source_prototypes(Xs[fit], ys[fit], C)
    cal_logits = (l2norm(Xs[cal]) @ P.T) / temp                 # HELD-OUT source logits
    conf, src_prior = LS.source_confusion(cal_logits, ys[cal], C)
    T, b = LS.bcts_calibrate(cal_logits, ys[cal], C)
    Xs_all_n = l2norm(Xs)

    # --- precompute per-sample logits for the FULL target (expensive parts done once) ---
    L_proto = (l2norm(Xt) @ P.T) / temp
    L_knn = knn_logits(l2norm(Xt), Xs_all_n, ys, C, k, temp)
    knn_clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn_clf.fit(normalize(Xs), ys)
    L_knnclf = _onehot_logits(knn_clf.predict(normalize(Xt)), C)

    def a_argmax(L): return L.argmax(1)
    def a_uniform(L): return sinkhorn(L).argmax(1)
    def a_otter(L): return sinkhorn(L, r=LS.bbse_prior(conf, L, src_prior)).argmax(1)
    def a_mlls(L): return sinkhorn(L, r=LS.mlls_prior(LS.apply_calibration(L, T, b), src_prior)).argmax(1)

    # (name, assign_fn, precomputed logit matrix)
    methods = [
        ("source prototypes", a_argmax, L_proto),
        ("k-NN classifier (sklearn)", a_argmax, L_knnclf),
        ("k-NN affinity (scorer)", a_argmax, L_knn),
        ("OTTER (BBSE, held-out)", a_otter, L_proto),
        ("Sinkhorn + calib-MLLS", a_mlls, L_proto),
        ("Ours: uniform-Sinkhorn (proto)", a_uniform, L_proto),
        ("Ours: uniform-Sinkhorn (kNN)", a_uniform, L_knn),
    ]
    ctx = dict(P=P, conf=conf, src_prior=src_prior, T=T, b=b, temp=temp, C=C, L_proto=L_proto)
    return methods, ctx


def diagnostics(ctx, Xt, yt):
    """Point-estimate controls that decide the headline (use yt; not deployable methods)."""
    C = ctx["C"]
    L = ctx["L_proto"]
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
    print(f"precomputing logits (source neighbour search, calibration)...", flush=True)
    methods, ctx = build(Xs, ys, C, Xt, temp=args.temp)

    print(f"\n=== {args.backbone} -> {args.target}  (n={len(yt)}, bootstrap={args.bootstrap}) ===")
    print("Hyperparameters fixed a priori (not tuned on target): temp=0.1, sinkhorn eps=0.05, iters=50, k=20, min_support=10", flush=True)
    for name, assign_fn, L in methods:
        t0 = time.perf_counter(); _ = assign_fn(L); dt = (time.perf_counter() - t0) * 1000
        r = BS.bootstrap_ci_logits(assign_fn, L, yt, B=args.bootstrap)
        print(BS.fmt(name, r) + f"   [{dt:6.0f} ms/assign]", flush=True)

    diagnostics(ctx, Xt, yt)


if __name__ == "__main__":
    main()
