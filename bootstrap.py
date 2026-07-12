"""Bootstrap confidence intervals for the target metrics (round-2 rigor fixes).

Fixes from the stats review:
 - MAJOR-1: the worst-group eligible-class set is FROZEN from the full sample and reused in every
   resample, so the estimand does not move (previously the n>=min_support filter was recomputed per
   resample, mixing different functionals and inflating/skewing the CI).
 - MAJOR-2: default B raised to 2000; we also report the bootstrap MEDIAN (the point estimate of a
   min-type statistic sits high in its own skewed CI, so the median is the honest central value).
 - Percentile CIs for the extremal worst-group are approximate near the 0 boundary — reported as such.
 - We resample the target set and RE-RUN the transductive predictor per resample (correct for a
   method that balances over the whole set).
"""
from __future__ import annotations

import numpy as np

from class_alignment import NUM_CLASSES, IDX_TO_CLASS
import metrics as M


def bootstrap_ci(predict_fn, Xt: np.ndarray, yt: np.ndarray, B: int = 2000,
                 seed: int = 42, min_support: int = 10) -> dict:
    """predict_fn(X_subset) -> y_pred for that subset. Returns point, median, 95% CI for
    accuracy, macro-F1, balanced accuracy, worst-group (fixed estimand), and dead-class count."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(yt)
    N = len(yt)
    eligible = M.eligible_classes(yt, NUM_CLASSES, min_support)   # FROZEN estimand
    point = M.summarize(yt, predict_fn(Xt), NUM_CLASSES, IDX_TO_CLASS, min_support, eligible=eligible)

    acc, mf1, bal, wg, dead = [], [], [], [], []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        yb, pb = yt[idx], predict_fn(Xt[idx])
        acc.append(M.accuracy(yb, pb))
        mf1.append(M.macro_f1(yb, pb, NUM_CLASSES))
        bal.append(M.balanced_accuracy(yb, pb, NUM_CLASSES))
        wg.append(M.worst_group_accuracy(yb, pb, NUM_CLASSES, eligible=eligible))  # fixed set
        dead.append(M.dead_classes(yb, pb, NUM_CLASSES, min_support=min_support))

    def stat(a):
        a = np.asarray(a) * 100
        return {"med": float(np.median(a)),
                "ci": (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))}

    return {
        "n_eligible": point["n_eligible"],
        "acc": point["accuracy"] * 100, **{f"acc_{k}": v for k, v in stat(acc).items()},
        "mf1": point["macro_f1"] * 100, **{f"mf1_{k}": v for k, v in stat(mf1).items()},
        "bal": point["balanced_acc"] * 100, **{f"bal_{k}": v for k, v in stat(bal).items()},
        "wg": point["worst_group_supported"] * 100, **{f"wg_{k}": v for k, v in stat(wg).items()},
        "dead": point["dead_classes"],
    }


def fmt(name: str, r: dict) -> str:
    return (f"{name:32s} "
            f"acc {r['acc']:5.1f} [{r['acc_ci'][0]:4.1f},{r['acc_ci'][1]:4.1f}]  "
            f"mF1 {r['mf1']:5.1f} [{r['mf1_ci'][0]:4.1f},{r['mf1_ci'][1]:4.1f}]  "
            f"bal {r['bal']:5.1f}  "
            f"WG {r['wg']:5.1f} (med {r['wg_med']:4.1f}) [{r['wg_ci'][0]:4.1f},{r['wg_ci'][1]:4.1f}]  "
            f"dead {r['dead']}/{r['n_eligible']}")
