"""
Temperature scaling and the calibrated evaluation protocol.

- The temperature is optimised as ``log T`` (it stays positive).
- Binary classifiers provide a single logit; prototype networks provide
  ``class_logits[:, 1] - class_logits[:, 0]`` (see
  :func:`tomographia.prototypes.binary_logit`), for which binary temperature
  scaling is exactly two-class softmax temperature scaling.
- :func:`evaluate_calibrated` packages the protocol: fit T on validation,
  choose the F1-optimal threshold on *calibrated* validation probabilities,
  report everything on test.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .metrics import (
    bootstrap_ci,
    brier_score,
    ece_score,
    f1_optimal_score_threshold,
    point_metrics,
    ranking_metrics,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(logits: Sequence[float], y_true: Sequence[int],
                    max_iter: int = 200,
                    bounds: Tuple[float, float] = (0.05, 20.0)) -> float:
    """
    Fit a scalar temperature ``T`` minimising the NLL of ``sigmoid(logit / T)``.

    The NLL is unimodal in ``log T``, so the optimum is clipped to *bounds*.
    On a perfectly separated validation set the NLL keeps decreasing as
    ``T -> 0`` (no finite optimum): calibration is then skipped (``T = 1``)
    with a warning.

    Parameters
    ----------
    logits : binary logits (validation set).
    y_true : binary labels.
    bounds : admissible ``(T_min, T_max)``.

    Returns
    -------
    ``T > 0`` (``T < 1`` sharpens, ``T > 1`` softens the probabilities).
    """
    import torch

    z = torch.as_tensor(np.asarray(logits, dtype=np.float64).ravel())
    y = torch.as_tensor(np.asarray(y_true, dtype=np.float64).ravel())
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=max_iter, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(torch.exp(log_t).item())
    if not np.isfinite(t):
        t = bounds[0]
    yv, zv = y.numpy(), z.numpy()
    pos, neg = zv[yv > 0.5], zv[yv <= 0.5]
    if pos.size and neg.size and pos.min() > neg.max():
        warnings.warn("Validation logits separate the classes perfectly: the temperature "
                      "has no finite optimum; calibration is skipped (T = 1). Calibration "
                      "needs a validation set with errors", stacklevel=2)
        return 1.0
    t_clip = float(np.clip(t, *bounds))
    if t_clip != t:
        warnings.warn(f"Fitted temperature {t:.3g} clipped to {t_clip:.3g}", stacklevel=2)
    return t_clip


def apply_temperature(logits: Sequence[float], temperature: float) -> np.ndarray:
    """Calibrated probabilities ``sigmoid(logit / T)``."""
    return _sigmoid(np.asarray(logits, dtype=float) / float(temperature))


def evaluate_calibrated(
    val_logits: Sequence[float],
    val_y: Sequence[int],
    test_logits: Sequence[float],
    test_y: Sequence[int],
    calibrate: bool = True,
    threshold: Optional[float] = None,
    test_groups: Optional[Sequence] = None,
    n_bootstrap: int = 1000,
    n_bins: int = 15,
    seed: int = 0,
) -> Dict:
    """
    Calibrated evaluation protocol on binary logits.

    1. Fit the temperature on validation (if ``calibrate``).
    2. Pick the F1-optimal threshold on calibrated validation probabilities
       (unless ``threshold`` is given; a given threshold is applied to the
       *calibrated* probabilities when ``calibrate=True``).
    3. Report point metrics, ROC/PR-AUC, ECE, Brier and bootstrap 95 % CIs on
       test.

    Returns
    -------
    Dict with ``temperature``, ``threshold`` (calibrated probability) and
    ``threshold_logit`` (the same on ``logit / T``), ``test`` (metrics),
    ``ci`` (bootstrap intervals of the point and ranking metrics, conditional
    on the fitted temperature and threshold) and the calibrated ``test_prob``
    array.  The threshold is selected, and test decisions and ROC / PR-AUC are
    computed, on ``logit / T``: equivalent to the probabilities but free of
    floating-point saturation at 0 / 1.
    """
    t = fit_temperature(val_logits, val_y) if calibrate else 1.0
    s_val = np.asarray(val_logits, dtype=float).ravel() / t
    s_test = np.asarray(test_logits, dtype=float).ravel() / t
    p_test = apply_temperature(test_logits, t)
    if threshold is None:
        thr_logit = (f1_optimal_score_threshold(val_y, s_val)
                     if np.asarray(val_y).any() else 0.0)
    else:
        q = float(np.clip(threshold, 1e-12, 1 - 1e-12))
        thr_logit = float(np.log(q) - np.log1p(-q))
    thr = float(_sigmoid(np.array([thr_logit]))[0])

    test = point_metrics(test_y, s_test, thr_logit)
    test.update(ranking_metrics(test_y, s_test))
    test["ece"] = ece_score(test_y, p_test, n_bins=n_bins)
    test["brier"] = brier_score(test_y, p_test)

    ci = bootstrap_ci(test_y, s_test, thr_logit, groups=test_groups,
                      n_bootstrap=n_bootstrap, seed=seed) if n_bootstrap else {}
    return {"temperature": t, "threshold": thr, "threshold_logit": thr_logit,
            "test": test, "ci": ci,
            "test_prob": p_test}
