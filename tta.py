"""Source-free, backprop-free online prototype test-time adaptation on cached features.

The method contribution. Prototypes are initialized from PlantVillage class-mean
features (source-free init), then evolved online from confident PlantDoc (field)
features. Mechanisms (each a flag) target the worst-group failure:

  --adapt        : enable online prototype updates (EMA from confident samples)
  --anchor A     : pull each update back toward the frozen SOURCE prototype (anti-drift)
  --balanced     : per-class learning rate scaled by inverse frequency (imbalance-aware)
  --reset K      : RDumb-style periodic reset to source prototypes (collapse insurance)
  --prior T      : online logit adjustment by running predicted-class frequency, so
                   starved tail classes get predicted (the worst-group fix)
  --sinkhorn     : extra final-pass line using balanced (Sinkhorn) assignment — a
                   diagnostic for whether class-balancing can rescue the 0% tail

Runs on cached .npz features only — CPU, seconds. Compare configs by worst-group.

    python tta.py --backbone dinov2_vitb14 --sinkhorn                        # balancing on source protos
    python tta.py --backbone dinov2_vitb14 --adapt --anchor 0.3 --prior 1.0  # anchored + prior-corrected
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from class_alignment import NUM_CLASSES, IDX_TO_CLASS
from config import CFG
import metrics as M


def _load(backbone: str, domain: str):
    d = np.load(os.path.join(CFG.feat_dir, backbone, f"{domain}.npz"), allow_pickle=True)
    return d["feats"].astype(np.float64), d["labels"].astype(int)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def source_prototypes(Xs: np.ndarray, ys: np.ndarray, C: int) -> np.ndarray:
    dim = Xs.shape[1]
    protos = np.zeros((C, dim))
    Xs = l2norm(Xs)
    for c in range(C):
        m = ys == c
        if m.any():
            protos[c] = Xs[m].mean(0)
    return l2norm(protos)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def knn_logits(Xt: np.ndarray, Xs: np.ndarray, ys: np.ndarray, C: int,
               k: int = 20, temp: float = 0.1) -> np.ndarray:
    """Per-class soft kNN score: for each target sample and class c, the mean cosine
    similarity to its k nearest source neighbours *of class c*. (N,C) logits.

    Captures each class's multi-modal structure (why kNN out-accuracies nearest-mean
    prototypes) while still producing a dense logit matrix Sinkhorn can balance —
    the 'best-of-both' candidate: kNN accuracy + balanced-assignment tail rescue."""
    Xt = l2norm(Xt)
    Xs = l2norm(Xs)
    N = Xt.shape[0]
    L = np.full((N, C), -1.0)
    for c in range(C):
        Xc = Xs[ys == c]
        if Xc.shape[0] == 0:
            continue
        sims = Xt @ Xc.T                              # (N, M_c)
        kk = min(k, Xc.shape[0])
        topk = np.partition(sims, -kk, axis=1)[:, -kk:]  # k best per sample
        L[:, c] = topk.mean(1)
    return L / temp


def sinkhorn(logits: np.ndarray, r: np.ndarray | None = None,
             n_iter: int = 50, eps: float = 0.05) -> np.ndarray:
    """Balanced assignment via Sinkhorn-Knopp. logits (N,C) -> probs (N,C).

    `r` is the target CLASS marginal (length C): uniform (1/C each) forces perfectly
    balanced classes; an estimated distribution relaxes that toward the real target."""
    N, K = logits.shape
    if r is None:
        r = np.ones(K) / K
    r = r / (r.sum() + 1e-12)
    Q = np.exp((logits - logits.max()) / eps).T          # (C, N)
    Q /= (Q.sum() + 1e-12)
    for _ in range(n_iter):
        Q *= (r[:, None] / (Q.sum(1, keepdims=True) + 1e-12))   # each class row -> r_c
        Q *= ((1.0 / N) / (Q.sum(0, keepdims=True) + 1e-12))    # each sample col -> 1/N
    Q /= (Q.sum(0, keepdims=True) + 1e-12)               # per-sample distribution
    return Q.T


def estimate_class_marginal(logits: np.ndarray, C: int, smooth: float = 1.0) -> np.ndarray:
    """Estimate target class distribution from raw argmax predictions (add-one smoothed)."""
    counts = np.bincount(logits.argmax(1), minlength=C).astype(np.float64)
    return (counts + smooth) / (counts.sum() + smooth * C)


def run(args) -> None:
    C = NUM_CLASSES
    Xs, ys = _load(args.backbone, "plantvillage")
    Xt, yt = _load(args.backbone, args.target)
    Xt = l2norm(Xt)

    # --- kNN scorer path: best-of-both candidate (kNN accuracy + optional balancing) ---
    if args.scorer == "knn":
        L = knn_logits(Xt, Xs, ys, C, k=args.k, temp=args.temp)
        tag = f"{args.backbone}->{args.target} kNN-scorer(k={args.k})"
        M.print_summary(f"{tag} — raw", M.summarize(yt, L.argmax(1), C, IDX_TO_CLASS))
        if args.sinkhorn:
            r = estimate_class_marginal(L, C) if args.prior_mode == "estimated" else None
            mode = "estimated-prior" if args.prior_mode == "estimated" else "uniform"
            M.print_summary(f"{tag} + Sinkhorn ({mode})",
                            M.summarize(yt, sinkhorn(L, r=r).argmax(1), C, IDX_TO_CLASS))
        return

    P_src = source_prototypes(Xs, ys, C)   # frozen anchor
    P = P_src.copy()                       # working prototypes
    counts = np.zeros(C)                   # per-class update counts (imbalance gating)
    q = np.ones(C) / C                     # running predicted-class distribution (prior correction)

    order = np.arange(len(Xt))
    if args.shuffle:
        np.random.default_rng(args.seed).shuffle(order)

    online_true, online_pred = [], []
    for step, i in enumerate(order):
        x = Xt[i]
        logits = (P @ x) / args.temp
        if args.prior > 0:
            logits = logits - args.prior * np.log(q + 1e-6)   # boost under-predicted classes
        prob = softmax(logits)
        pred = int(prob.argmax())
        conf = float(prob.max())
        online_true.append(int(yt[i]))
        online_pred.append(pred)

        if args.prior > 0:
            onehot = np.zeros(C); onehot[pred] = 1.0
            q = 0.99 * q + 0.01 * onehot

        if args.adapt and conf >= args.conf:
            a = args.alpha
            if args.balanced:
                a *= min(1.0, np.sqrt((counts.mean() + 1.0) / (counts[pred] + 1.0)))
            new = (1 - a) * P[pred] + a * x
            if args.anchor > 0:
                new = (1 - args.anchor) * new + args.anchor * P_src[pred]
            P[pred] = l2norm(new[None])[0]
            counts[pred] += 1

        if args.reset > 0 and (step + 1) % args.reset == 0:
            P = P_src.copy()
            counts[:] = 0

    tag = (f"{args.backbone}->{args.target} TTA["
           f"{'adapt' if args.adapt else 'noadapt'}"
           f"{',anchor%.2f' % args.anchor if args.anchor > 0 else ''}"
           f"{',balanced' if args.balanced else ''}"
           f"{',prior%.1f' % args.prior if args.prior > 0 else ''}"
           f"{',reset%d' % args.reset if args.reset > 0 else ''}]")

    M.print_summary(f"{tag} — ONLINE (predict-then-adapt)",
                    M.summarize(online_true, online_pred, C, IDX_TO_CLASS))

    final_logits = (Xt @ P.T) / args.temp
    if args.prior > 0:
        final_logits = final_logits - args.prior * np.log(q + 1e-6)
    M.print_summary(f"{tag} — FINAL PASS (adapted prototypes)",
                    M.summarize(yt, final_logits.argmax(1), C, IDX_TO_CLASS))

    if args.sinkhorn:
        sk_logits = Xt @ P.T / args.temp
        r = None
        mode = "uniform"
        if args.prior_mode == "estimated":
            r = estimate_class_marginal(sk_logits, C)
            mode = "estimated-prior"
        bal_pred = sinkhorn(sk_logits, r=r).argmax(1)
        M.print_summary(f"{tag} — FINAL PASS + Sinkhorn ({mode})",
                        M.summarize(yt, bal_pred, C, IDX_TO_CLASS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--target", default="plantdoc", help="field target: plantdoc | plantwild")
    ap.add_argument("--scorer", default="proto", choices=["proto", "knn"],
                    help="proto = class-mean prototypes; knn = per-class kNN affinity (best-of-both)")
    ap.add_argument("--k", type=int, default=20, help="neighbours for the kNN scorer")
    ap.add_argument("--adapt", action="store_true", help="enable online prototype updates")
    ap.add_argument("--anchor", type=float, default=0.0, help="pull-back toward source proto (0..1)")
    ap.add_argument("--balanced", action="store_true", help="imbalance-aware per-class LR")
    ap.add_argument("--prior", type=float, default=0.0, help="online logit-adjustment strength (0=off)")
    ap.add_argument("--sinkhorn", action="store_true", help="add a balanced-assignment final-pass line")
    ap.add_argument("--prior_mode", default="uniform", choices=["uniform", "estimated"],
                    help="Sinkhorn class marginal: uniform (balanced) or estimated from predictions")
    ap.add_argument("--reset", type=int, default=0, help="steps between RDumb resets (0=off)")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold to update")
    ap.add_argument("--alpha", type=float, default=0.1, help="base EMA step")
    ap.add_argument("--temp", type=float, default=0.1, help="softmax temperature (lower = peakier)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle the online stream")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
