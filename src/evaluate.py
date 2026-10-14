"""evaluate.py — NumPy metrics and compact comparison helpers."""

from __future__ import annotations

import numpy as np


def _as_float_arrays(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return a, b


# Regression -----------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# Classification --------------------------------------------------------------
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def precision_recall_f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def roc_auc(y_true, scores) -> float:
    """ROC-AUC using average ranks (Mann-Whitney U), including score ties."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same shape")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # Ranks are 1-based; ties receive their average rank.
        avg_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = ranks[y == 1].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def regression_metrics(y_true, y_pred):
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "R2": r2(y_true, y_pred)}


def classification_metrics(y_true, y_pred, scores=None):
    p, r, f = precision_recall_f1(y_true, y_pred)
    out = {"Accuracy": accuracy(y_true, y_pred), "Precision": p, "Recall": r, "F1": f}
    if scores is not None:
        out["ROC_AUC"] = roc_auc(y_true, scores)
    return out


def compare(name_to_metrics: dict) -> None:
    """Print an aligned table for models sharing the same metric names."""
    if not name_to_metrics:
        print("(no models)")
        return
    metric_names = []
    for metrics in name_to_metrics.values():
        for key in metrics:
            if key not in metric_names:
                metric_names.append(key)

    name_width = max(12, max(len(str(k)) for k in name_to_metrics))
    widths = {m: max(10, len(m) + 2) for m in metric_names}
    header = f"{'Model':<{name_width}}  " + "  ".join(f"{m:>{widths[m]}}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for name, metrics in name_to_metrics.items():
        row = f"{name:<{name_width}}  "
        for m in metric_names:
            val = metrics.get(m, np.nan)
            if isinstance(val, (int, np.integer)):
                cell = str(int(val))
            elif np.isfinite(val):
                cell = f"{float(val):.4f}"
            else:
                cell = "-"
            row += f"{cell:>{widths[m]}}  "
        print(row)
