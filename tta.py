"""Source-free, backprop-free online prototype test-time adaptation on cached features.

The method contribution (Proposal 2). Prototypes are initialized from PlantVillage
class-mean features (source-free init), then evolved online from confident PlantDoc
(field) features, with two mechanisms aimed at the worst-group failure:
  --anchor    : each update is pulled back toward the frozen SOURCE prototype (anti-drift)
  --balanced  : per-class learning rate scaled by inverse frequency so common classes
                (e.g. healthy leaves) don't overwrite rare diseases (imbalance-aware gating)
  --reset K   : RDumb-style periodic reset to source prototypes (collapse insurance)

Runs on cached .npz features only — CPU, seconds. Compare configs by worst-group.

Examples:
    python tta.py --backbone dinov2_vitb14                          # source prototypes, no adapt
    python tta.py --backbone dinov2_vitb14 --adapt                  # naive online update
    python tta.py --backbone dinov2_vitb14 --adapt --anchor 0.3 --balanced   # full recipe
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
    """Normalized class-mean prototypes from source features. Empty classes -> zeros."""
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


def run(args) -> None:
    C = NUM_CLASSES
    Xs, ys = _load(args.backbone, "plantvillage")
    Xt, yt = _load(args.backbone, "plantdoc")
    Xt = l2norm(Xt)

    P_src = source_prototypes(Xs, ys, C)   # frozen anchor
    P = P_src.copy()                       # working prototypes
    counts = np.zeros(C)                   # per-class update counts (for imbalance gating)

    order = np.arange(len(Xt))
    if args.shuffle:
        np.random.default_rng(args.seed).shuffle(order)

    online_true, online_pred = [], []
    for step, i in enumerate(order):
        x = Xt[i]
        sim = P @ x
        pred = int(sim.argmax())
        conf = float(softmax(sim / args.temp).max())
        online_true.append(int(yt[i]))
        online_pred.append(pred)

        if args.adapt and conf >= args.conf:
            a = args.alpha
            if args.balanced:
                # rare classes keep full step; frequent classes are damped.
                a *= min(1.0, np.sqrt((counts.mean() + 1.0) / (counts[pred] + 1.0)))
            new = (1 - a) * P[pred] + a * x
            if args.anchor > 0:
                new = (1 - args.anchor) * new + args.anchor * P_src[pred]
            P[pred] = l2norm(new[None])[0]
            counts[pred] += 1

        if args.reset > 0 and (step + 1) % args.reset == 0:
            P = P_src.copy()
            counts[:] = 0

    tag = (f"{args.backbone} TTA["
           f"{'adapt' if args.adapt else 'noadapt'}"
           f"{',anchor%.2f' % args.anchor if args.anchor > 0 else ''}"
           f"{',balanced' if args.balanced else ''}"
           f"{',reset%d' % args.reset if args.reset > 0 else ''}]")

    M.print_summary(f"{tag} — ONLINE (predict-then-adapt)",
                    M.summarize(online_true, online_pred, C, IDX_TO_CLASS))
    # Final-pass: re-classify the whole target with the adapted prototypes.
    final_pred = (Xt @ P.T).argmax(1)
    M.print_summary(f"{tag} — FINAL PASS (adapted prototypes)",
                    M.summarize(yt, final_pred, C, IDX_TO_CLASS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--adapt", action="store_true", help="enable online prototype updates")
    ap.add_argument("--anchor", type=float, default=0.0, help="pull-back toward source proto (0..1)")
    ap.add_argument("--balanced", action="store_true", help="imbalance-aware per-class LR")
    ap.add_argument("--reset", type=int, default=0, help="steps between RDumb resets (0=off)")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold to update")
    ap.add_argument("--alpha", type=float, default=0.1, help="base EMA step")
    ap.add_argument("--temp", type=float, default=0.1, help="softmax temperature (lower = peakier)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle the online stream")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
