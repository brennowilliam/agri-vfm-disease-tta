"""Evaluation metrics. Emphasis on worst-group accuracy — averages hide field failures."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


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


def worst_group_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Lowest per-class recall over classes present in y_true."""
    accs = per_class_accuracy(y_true, y_pred, num_classes)
    present = accs[~np.isnan(accs)]
    return float(present.min()) if present.size else float("nan")


def summarize(y_true, y_pred, num_classes: int, idx_to_class: dict | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pca = per_class_accuracy(y_true, y_pred, num_classes)
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "worst_group_acc": worst_group_accuracy(y_true, y_pred, num_classes),
        "mean_per_class_acc": float(np.nanmean(pca)),
        "n": int(len(y_true)),
    }
    if idx_to_class is not None:
        present = [(idx_to_class[c], float(pca[c])) for c in range(num_classes) if not np.isnan(pca[c])]
        out["worst5"] = sorted(present, key=lambda kv: kv[1])[:5]
    return out


def print_summary(title: str, s: dict) -> None:
    print(f"\n=== {title} (n={s['n']}) ===")
    print(f"  accuracy         : {s['accuracy']*100:6.2f}%")
    print(f"  macro-F1         : {s['macro_f1']*100:6.2f}%")
    print(f"  mean per-class   : {s['mean_per_class_acc']*100:6.2f}%")
    print(f"  worst-group acc  : {s['worst_group_acc']*100:6.2f}%")
    if "worst5" in s:
        worst = ", ".join(f"{k}={v*100:.0f}%" for k, v in s["worst5"])
        print(f"  worst 5 classes  : {worst}")
