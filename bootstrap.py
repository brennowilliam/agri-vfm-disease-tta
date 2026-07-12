"""Bootstrap confidence intervals for the target metrics (round-2 rigor + efficiency).

Design: per-sample SCORING (source-neighbour search, prototype logits) is independent of the
resample and is precomputed ONCE per method as a logit matrix L (N x C). The bootstrap then
resamples rows of L and re-applies only the (cheap, transductive) ASSIGN step — so B=2000 is
seconds even for k-NN-scored methods, instead of hours.

Rigor fixes:
 - worst-group eligible-class set FROZEN from the full sample (was recomputed per resample);
 - B default 2000; report bootstrap MEDIAN alongside the point estimate;
 - also report balanced accuracy and dead-class count.
Percentile CI for the extremal worst-group is approximate near the 0 boundary (noted in the paper).
"""
from __future__ import annotations

import numpy as np

from class_alignment import NUM_CLASSES, IDX_TO_CLASS
import metrics as M


def bootstrap_ci_logits(assign_fn, L: np.ndarray, yt: np.ndarray, B: int = 2000,
                        seed: int = 42, min_support: int = 10) -> dict:
    """assign_fn(L_subset) -> predictions. L is the precomputed per-sample logit matrix (N x C)."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(yt)
    N = len(yt)
    eligible = M.eligible_classes(yt, NUM_CLASSES, min_support)          # FROZEN estimand
    point = M.summarize(yt, assign_fn(L), NUM_CLASSES, IDX_TO_CLASS, min_support, eligible=eligible)

    acc, mf1, bal, wg, dead = [], [], [], [], []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        yb, pb = yt[idx], assign_fn(L[idx])
        acc.append(M.accuracy(yb, pb))
        mf1.append(M.macro_f1(yb, pb, NUM_CLASSES))
        bal.append(M.balanced_accuracy(yb, pb, NUM_CLASSES))
        wg.append(M.worst_group_accuracy(yb, pb, NUM_CLASSES, eligible=eligible))
        dead.append(M.dead_classes(yb, pb, NUM_CLASSES, min_support=min_support))

    def stat(a):
        a = np.asarray(a) * 100.0
        return float(np.median(a)), (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    m_acc, ci_acc = stat(acc); m_mf1, ci_mf1 = stat(mf1)
    m_bal, ci_bal = stat(bal); m_wg, ci_wg = stat(wg)
    return {
        "n_eligible": point["n_eligible"],
        "acc": point["accuracy"] * 100, "acc_med": m_acc, "acc_ci": ci_acc,
        "mf1": point["macro_f1"] * 100, "mf1_med": m_mf1, "mf1_ci": ci_mf1,
        "bal": point["balanced_acc"] * 100, "bal_med": m_bal, "bal_ci": ci_bal,
        "wg": point["worst_group_supported"] * 100, "wg_med": m_wg, "wg_ci": ci_wg,
        "dead": point["dead_classes"],
    }


def fmt(name: str, r: dict) -> str:
    return (f"{name:32s} "
            f"acc {r['acc']:5.1f} [{r['acc_ci'][0]:4.1f},{r['acc_ci'][1]:4.1f}]  "
            f"mF1 {r['mf1']:5.1f} [{r['mf1_ci'][0]:4.1f},{r['mf1_ci'][1]:4.1f}]  "
            f"bal {r['bal']:5.1f}  "
            f"WG {r['wg']:5.1f}(med{r['wg_med']:4.1f})[{r['wg_ci'][0]:4.1f},{r['wg_ci'][1]:4.1f}]  "
            f"dead {r['dead']}/{r['n_eligible']}")
