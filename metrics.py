"""Evaluation metrics. Emphasis on the tail — averages hide field failures, but a
tiny-support class (e.g. n=2) can also falsely pin worst-group at 0, so we report
worst-group over adequately-supported classes and always show per-class support."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def class_support(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    return np.array([(y_true == c).sum() for c in range(num_classes)])


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Recall per class; NaN for classes absent from y_true."""
    accs = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = y_true == c
        if mask.any():
            accs[c] = (y_pred[mask] == c).mean()
    return accs


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)),
                          average="macro", zero_division=0))


def worst_group_accuracy(y_true, y_pred, num_classes: int, min_support: int = 1) -> float:
    """Lowest per-class recall over classes with support >= min_support."""
    accs = per_class_accuracy(y_true, y_pred, num_classes)
    sup = class_support(y_true, num_classes)
    keep = (sup >= min_support) & ~np.isnan(accs)
    return float(accs[keep].min()) if keep.any() else float("nan")


def summarize(y_true, y_pred, num_classes: int, idx_to_class: dict | None = None,
              min_support: int = 10) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pca = per_class_accuracy(y_true, y_pred, num_classes)
    sup = class_support(y_true, num_classes)
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "worst_group_acc": worst_group_accuracy(y_true, y_pred, num_classes, 1),
        "worst_group_supported": worst_group_accuracy(y_true, y_pred, num_classes, min_support),
        "mean_per_class_acc": float(np.nanmean(pca)),
        "min_support": min_support,
        "n": int(len(y_true)),
    }
    if idx_to_class is not None:
        present = [(idx_to_class[c], float(pca[c]), int(sup[c]))
                   for c in range(num_classes) if not np.isnan(pca[c])]
        out["worst5"] = sorted(present, key=lambda t: t[1])[:5]
    return out


def print_summary(title: str, s: dict) -> None:
    print(f"\n=== {title} (n={s['n']}) ===")
    print(f"  accuracy              : {s['accuracy']*100:6.2f}%")
    print(f"  macro-F1              : {s['macro_f1']*100:6.2f}%")
    print(f"  mean per-class        : {s['mean_per_class_acc']*100:6.2f}%")
    print(f"  worst-group (all)     : {s['worst_group_acc']*100:6.2f}%")
    print(f"  worst-group (n>={s['min_support']:<3d}) : {s['worst_group_supported']*100:6.2f}%")
    if "worst5" in s:
        worst = ", ".join(f"{k}={v*100:.0f}%(n={n})" for k, v, n in s["worst5"])
        print(f"  worst 5 classes       : {worst}")
