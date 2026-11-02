"""Tests for tomographia.evaluation (metrics, calibration, bootstrap)."""
import numpy as np
import pytest

from tomographia.evaluation import (
    bootstrap_ci,
    brier_score,
    ece_score,
    f1_optimal_threshold,
    point_metrics,
)


def test_f1_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.35, 0.6, 0.8, 0.9])
    thr = f1_optimal_threshold(y, p)
    assert point_metrics(y, p, thr)["f1"] == 1.0


def test_f1_threshold_rejects_logits():
    with pytest.raises(ValueError):
        f1_optimal_threshold([0, 1], [-2.0, 3.0])


def test_point_metrics_counts():
    m = point_metrics([1, 1, 0, 0], [0.9, 0.2, 0.7, 0.1], 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["specificity"] == 0.5


def test_ece_and_brier_perfect():
    y = np.array([0, 1, 0, 1])
    assert ece_score(y, y.astype(float)) == 0.0
    assert brier_score(y, y.astype(float)) == 0.0


def test_bootstrap_grouped_and_plain():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    ci = bootstrap_ci(y, p, 0.5, n_bootstrap=50)
    assert set(ci) >= {"accuracy", "f1", "roc_auc", "pr_auc"}
    lo, hi = ci["accuracy"]
    assert 0 <= lo <= point_metrics(y, p, 0.5)["accuracy"] <= hi <= 1
    groups = np.repeat(np.arange(20), 10)
    ci_g = bootstrap_ci(y, p, 0.5, groups=groups, n_bootstrap=50)
    assert ci_g["f1"][0] <= ci_g["f1"][1]


def test_temperature_recovers_scale():
    pytest.importorskip("torch")
    from tomographia.evaluation import apply_temperature, fit_temperature

    rng = np.random.default_rng(1)
    true_logit = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    t = fit_temperature(true_logit * 3.0, y)  # over-confident by 3x
    assert 2.5 < t < 3.5
    p = apply_temperature(true_logit * 3.0, t)
    assert ece_score(y, p) < 0.05


def test_evaluate_calibrated_protocol():
    pytest.importorskip("torch")
    from tomographia.evaluation import evaluate_calibrated

    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    z = (y * 2 - 1) * 2.0 + rng.normal(0, 1, 300)
    r = evaluate_calibrated(z[:150], y[:150], z[150:], y[150:], n_bootstrap=20)
    assert r["temperature"] > 0
    assert 0 < r["threshold"] < 1
    assert r["test"]["roc_auc"] > 0.9
    assert {"ece", "brier", "f1"} <= set(r["test"])
