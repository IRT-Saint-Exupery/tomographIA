"""
Binary classification metrics (numpy / scikit-learn only).

F1-optimal threshold, point metrics, ROC-AUC / PR-AUC, expected calibration
error, Brier score and (cluster) bootstrap confidence intervals.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

POINT_METRICS = ("accuracy", "precision", "recall", "f1", "specificity")


def f1_optimal_threshold(y_true: Sequence[int], prob: Sequence[float],
                         eps: float = 1e-7) -> float:
    """
    Threshold maximising F1 on (validation) scores (predict positive when
    ``prob >= threshold``).

    Returns 0.5 only when the curve is degenerate (no positive, or no finite
    F1).  An optimum at a saturated score (e.g. exactly 1.0) is returned as is;
    compute probabilities in float64 to limit saturation.
    """
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(prob).ravel().astype(float)
    if p.min() < -1e-6 or p.max() > 1 + 1e-6:
        raise ValueError("prob must lie in [0, 1]; pass probabilities, not logits")
    prec, rec, thr = precision_recall_curve(y, p)
    if thr.size == 0:
        return 0.5
    # sklearn: precision[i] / recall[i] belong to thresholds[i]; the last
    # element (precision=1, recall=0) has no threshold and is dropped.
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / (prec + rec + eps)
    if not np.isfinite(f1).any() or y.sum() == 0:
        return 0.5
    return float(thr[int(np.nanargmax(f1))])


def point_metrics(y_true: Sequence[int], prob: Sequence[float],
                  threshold: float = 0.5) -> Dict[str, float]:
    """Accuracy, precision, recall, F1, specificity and confusion counts."""
    y = np.asarray(y_true).ravel().astype(int)
    pred = (np.asarray(prob).ravel() >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    def _div(a, b):
        return float(a / b) if b > 0 else 0.0

    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    return {
        "accuracy": _div(tp + tn, len(y)),
        "precision": precision,
        "recall": recall,
        "f1": _div(2 * precision * recall, precision + recall),
        "specificity": _div(tn, tn + fp),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def ranking_metrics(y_true: Sequence[int], prob: Sequence[float]) -> Dict[str, float]:
    """ROC-AUC and PR-AUC (NaN when only one class is present)."""
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(prob).ravel().astype(float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p))}


def ece_score(y_true: Sequence[int], prob: Sequence[float], n_bins: int = 15) -> float:
    """Expected calibration error of P(y=1) with equal-width bins."""
    y = np.asarray(y_true).ravel().astype(float)
    p = np.asarray(prob).ravel().astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            m = (p >= edges[i]) & (p < edges[i + 1])
        else:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def brier_score(y_true: Sequence[int], prob: Sequence[float]) -> float:
    """Mean squared error between P(y=1) and the label."""
    y = np.asarray(y_true).ravel().astype(float)
    p = np.asarray(prob).ravel().astype(float)
    return float(np.mean((p - y) ** 2))


def bootstrap_ci(
    y_true: Sequence[int],
    prob: Sequence[float],
    threshold: float,
    groups: Optional[Sequence] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, tuple]:
    """
    Percentile bootstrap confidence intervals.

    Parameters
    ----------
    groups : optional group id per sample (e.g. specimen); when given, whole
             groups are resampled (cluster bootstrap), which accounts for the
             correlation between patches of the same specimen.

    Returns
    -------
    ``{metric: (low, high)}`` for accuracy, precision, recall, F1,
    specificity, ROC-AUC and PR-AUC.
    """
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(prob).ravel().astype(float)
    rng = np.random.default_rng(seed)
    keys = list(POINT_METRICS) + ["roc_auc", "pr_auc"]

    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in uniq]

    stats = []
    for _ in range(n_bootstrap):
        if groups is None:
            idx = rng.integers(0, len(y), size=len(y))
        else:
            pick = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([members[k] for k in pick])
        pm = point_metrics(y[idx], p[idx], threshold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rm = ranking_metrics(y[idx], p[idx])
        stats.append([pm[k] for k in POINT_METRICS] + [rm["roc_auc"], rm["pr_auc"]])

    arr = np.asarray(stats, dtype=float)
    lo = np.nanpercentile(arr, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(arr, 100 * (1 - alpha / 2), axis=0)
    return {k: (float(a), float(b)) for k, a, b in zip(keys, lo, hi)}
