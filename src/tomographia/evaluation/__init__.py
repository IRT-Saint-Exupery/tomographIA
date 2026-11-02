"""Evaluation of binary patch classifiers: metrics, calibration, bootstrap CIs.

The metric functions are numpy/scikit-learn only; temperature fitting needs torch.

Quick-start
-----------
::

    from tomographia.evaluation import evaluate_calibrated

    report = evaluate_calibrated(val_logits, val_y, test_logits, test_y,
                                 test_groups=test_specimen_ids)
    print(report["temperature"], report["threshold"], report["test"]["f1"])
"""
from .metrics import (
    bootstrap_ci,
    brier_score,
    ece_score,
    f1_optimal_threshold,
    point_metrics,
    ranking_metrics,
)
from .calibration import apply_temperature, evaluate_calibrated, fit_temperature

__all__ = [
    "f1_optimal_threshold",
    "point_metrics",
    "ranking_metrics",
    "ece_score",
    "brier_score",
    "bootstrap_ci",
    "fit_temperature",
    "apply_temperature",
    "evaluate_calibrated",
]
