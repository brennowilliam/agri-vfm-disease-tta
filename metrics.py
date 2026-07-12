"""Evaluation metrics.

Worst-group (lowest per-class recall) is the headline but fragile — computed over classes with
small support. Two rigor fixes from the round-2 stats review:
  (1) the eligible-class set can be FROZEN (pass `eligible=`) so a bootstrap does not silently
      change the estimand per resample;
  (2) we also report stable tail metrics — balanced accuracy (mean per-class recall) and the
      count of "dead" classes (recall < 5%) — which are less sensitive to a single small class.
`present_only`: restrict the label space to classes actually present in y_true (e.g. PlantWild
has 27 of 28 classes) so an absent class doesn't structurally deflate macro-F1.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true, y_pred) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def class_support(y_true, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true)
    return np.array([(y_true == c).sum() for c in range(num_classes)])


def per_class_accuracy(y_true, y_pred, num_classes: int) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs = np.full(num_classes, np.nan)
    for c in range(num_classes):
        m = y_true == c
        if m.any():
            accs[c] = (y_pred[m] == c).mean()
    return accs


def present_labels(y_true, num_classes: int) -> list[int]:
    return [c for c in range(num_classes) if (np.asarray(y_true) == c).any()]


def macro_f1(y_true, y_pred, num_classes: int, present_only: bool = True) -> float:
    labels = present_labels(y_true, num_classes) if present_only else list(range(num_classes))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def eligible_classes(y_true, num_classes: int, min_support: int = 10) -> np.ndarray:
    """Classes with support >= min_support on THIS y_true. Freeze once on the full sample and
    reuse across bootstrap resamples so the worst-group estimand does not move."""
    sup = class_support(y_true, num_classes)
    return np.where(sup >= min_support)[0]


def worst_group_accuracy(y_true, y_pred, num_classes: int, min_support: int = 1,
                         eligible: np.ndarray | None = None) -> float:
    """Lowest per-class recall. If `eligible` is given, min is taken over exactly those classes
    (the fixed estimand); otherwise classes are filtered by min_support on the current y_true."""
    accs = per_class_accuracy(y_true, y_pred, num_classes)
    if eligible is None:
        eligible = eligible_classes(y_true, num_classes, min_support)
    vals = [accs[c] for c in eligible if not np.isnan(accs[c])]
    return float(min(vals)) if vals else float("nan")


def balanced_accuracy(y_true, y_pred, num_classes: int) -> float:
    """Mean per-class recall over present classes — the stable tail metric."""
    pca = per_class_accuracy(y_true, y_pred, num_classes)
    return float(np.nanmean(pca))


def dead_classes(y_true, y_pred, num_classes: int, thresh: float = 0.05,
                 min_support: int = 10) -> int:
    """Number of supported classes with recall < thresh (near-never recognised)."""
    accs = per_class_accuracy(y_true, y_pred, num_classes)
    elig = eligible_classes(y_true, num_classes, min_support)
    return int(sum(1 for c in elig if not np.isnan(accs[c]) and accs[c] < thresh))


def summarize(y_true, y_pred, num_classes: int, idx_to_class: dict | None = None,
              min_support: int = 10, eligible: np.ndarray | None = None,
              present_only: bool = True) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if eligible is None:
        eligible = eligible_classes(y_true, num_classes, min_support)
    pca = per_class_accuracy(y_true, y_pred, num_classes)
    sup = class_support(y_true, num_classes)
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes, present_only),
        "balanced_acc": balanced_accuracy(y_true, y_pred, num_classes),
        "worst_group_acc": worst_group_accuracy(y_true, y_pred, num_classes, 1),
        "worst_group_supported": worst_group_accuracy(y_true, y_pred, num_classes, eligible=eligible),
        "dead_classes": dead_classes(y_true, y_pred, num_classes, min_support=min_support),
        "n_eligible": int(len(eligible)),
        "min_support": min_support,
        "n": int(len(y_true)),
    }
    if idx_to_class is not None:
        present = [(idx_to_class[c], float(pca[c]), int(sup[c])) for c in eligible
                   if not np.isnan(pca[c])]
        out["worst5"] = sorted(present, key=lambda t: t[1])[:5]
    return out


def print_summary(title: str, s: dict) -> None:
    print(f"\n=== {title} (n={s['n']}, eligible={s['n_eligible']}) ===")
    print(f"  accuracy              : {s['accuracy']*100:6.2f}%")
    print(f"  macro-F1              : {s['macro_f1']*100:6.2f}%")
    print(f"  balanced acc (mean/cl): {s['balanced_acc']*100:6.2f}%")
    print(f"  worst-group (n>={s['min_support']:<3d}) : {s['worst_group_supported']*100:6.2f}%")
    print(f"  dead classes (<5%)    : {s['dead_classes']} of {s['n_eligible']}")
    if "worst5" in s:
        worst = ", ".join(f"{k}={v*100:.0f}%(n={n})" for k, v, n in s["worst5"])
        print(f"  worst 5 classes       : {worst}")
