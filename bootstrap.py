"""Bootstrap confidence intervals for the target metrics (reviewer point M4).

Worst-group is computed over classes with only n≈10–57 images, so the headline numbers
carry real sampling variance. We resample the target set with replacement B times and,
because our method is TRANSDUCTIVE (Sinkhorn balances over the whole target set), we
RE-RUN the full predictor on each resample rather than resampling fixed predictions.
"""
from __future__ import annotations

import numpy as np

from class_alignment import NUM_CLASSES, IDX_TO_CLASS
import metrics as M


def bootstrap_ci(predict_fn, Xt: np.ndarray, yt: np.ndarray, B: int = 1000,
                 seed: int = 42, min_support: int = 10) -> dict:
    """predict_fn(X_subset) -> y_pred for that subset (re-runs the full method).

    Returns point estimate + 95% percentile CI for accuracy, macro-F1, worst-group."""
    rng = np.random.default_rng(seed)
    N = len(yt)
    # point estimate on the full set
    point = M.summarize(yt, predict_fn(Xt), NUM_CLASSES, IDX_TO_CLASS, min_support)
    acc, mf1, wg = [], [], []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)               # resample with replacement
        yb = yt[idx]
        pb = predict_fn(Xt[idx])
        acc.append(M.accuracy(yb, pb))
        mf1.append(M.macro_f1(yb, pb, NUM_CLASSES))
        wg.append(M.worst_group_accuracy(yb, pb, NUM_CLASSES, min_support))

    def ci(a):
        a = np.asarray(a)
        return float(np.percentile(a, 2.5) * 100), float(np.percentile(a, 97.5) * 100)

    return {
        "acc": point["accuracy"] * 100, "acc_ci": ci(acc),
        "mf1": point["macro_f1"] * 100, "mf1_ci": ci(mf1),
        "wg": point["worst_group_supported"] * 100, "wg_ci": ci(wg),
    }


def fmt(name: str, r: dict) -> str:
    return (f"{name:34s} "
            f"acc {r['acc']:5.1f} [{r['acc_ci'][0]:4.1f},{r['acc_ci'][1]:4.1f}]  "
            f"mF1 {r['mf1']:5.1f} [{r['mf1_ci'][0]:4.1f},{r['mf1_ci'][1]:4.1f}]  "
            f"WG {r['wg']:5.1f} [{r['wg_ci'][0]:4.1f},{r['wg_ci'][1]:4.1f}]")
