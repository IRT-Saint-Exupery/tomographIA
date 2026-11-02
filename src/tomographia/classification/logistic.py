"""
Logistic-regression classification of defects (pore vs. crack) from the
descriptors of :func:`tomographia.components.compute_descriptors`.

Bundle format
-------------
Produced by :func:`fit_classifier`::

    {
        "model"          : sklearn Pipeline (median imputer → StandardScaler →
                           LogisticRegression) with ``predict_proba``,
        "features"       : list[str],  # feature columns, in order
        "threshold"      : float,      # P(crack) threshold, F1-optimal on
                                       # out-of-fold training predictions
        "holdout"        : "random" | "volume",
        "holdout_results": dict,       # metrics on the held-out split
        "volumes"        : list[str],  # descriptor tables used for training
        "n_train", "n_test": int,
        "tomographia_version": str,
    }
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .. import __version__
from ..components.pipeline2 import CANONICAL_FEATURES

logger = logging.getLogger(__name__)

#: Label strings accepted in the training tables, and their class index.
LABELS = {"pore": 0, "crack": 1}


def _load_bundle(bundle_or_path) -> dict:
    """Accept a dict, a path to a .joblib file, or a directory containing exactly one."""
    if isinstance(bundle_or_path, dict):
        return bundle_or_path
    import joblib
    p = Path(bundle_or_path)
    if p.is_dir():
        hits = sorted(p.glob("*.joblib"))
        if len(hits) != 1:
            raise FileNotFoundError(f"Expected exactly one .joblib file in {p}, found {len(hits)}")
        p = hits[0]
    return joblib.load(str(p))


def _feature_matrix(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Feature matrix with ±inf turned into NaN (NaN is imputed by the model)."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Descriptor table lacks the model features {missing}; compute them with "
                       "compute_descriptors (int_std_rel needs grayscale_volume=)")
    X = np.array(df[list(features)].to_numpy(dtype=float), copy=True)
    X[~np.isfinite(X)] = np.nan
    return X


def classify_defects_logistic(
    df: pd.DataFrame,
    bundle: Union[dict, str, Path],
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Classify defects with a bundle from :func:`fit_classifier`.

    Parameters
    ----------
    df        : descriptor DataFrame (output of ``compute_descriptors``).
    bundle    : bundle dict, path to a ``.joblib`` file, or a directory
                containing exactly one ``.joblib`` file.
    threshold : override the bundle's threshold.

    Returns
    -------
    Copy of *df* with ``cls_proba`` (P(crack)), ``cls_pred`` (0 = pore,
    1 = crack) and ``cls_label`` (``"pore"`` / ``"crack"``).

    Raises
    ------
    KeyError : a feature used by the model is missing from *df*.

    Missing (NaN / inf) feature values are replaced by the training median;
    a warning gives their number.
    """
    b = _load_bundle(bundle)
    if "scaler" in b:
        raise ValueError("This bundle was produced by an incompatible version; retrain it "
                         "with fit_classifier")
    features = b["features"]
    thr = float(b["threshold"] if threshold is None else threshold)

    X = _feature_matrix(df, features)
    n_nan = int(np.isnan(X).any(axis=1).sum())
    if n_nan:
        logger.warning("%d / %d components have missing feature values (imputed with the "
                       "training median)", n_nan, len(X))
    out = df.copy()
    if len(X) == 0:
        out["cls_proba"] = pd.Series(dtype=float)
        out["cls_pred"] = pd.Series(dtype=int)
        out["cls_label"] = pd.Series(dtype=object)
        return out
    proba = b["model"].predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)
    out["cls_proba"] = proba
    out["cls_pred"] = pred
    out["cls_label"] = np.where(pred == 1, "crack", "pore")
    return out


def build_classified_volume(
    cc_vol: np.ndarray,
    df: pd.DataFrame,
    label_col: str = "label_id",
) -> np.ndarray:
    """
    ``uint8`` volume with 0 = background, 1 = pore, 2 = crack.

    Parameters
    ----------
    cc_vol    : integer label array (e.g. from ``build_cc_mask``).
    df        : DataFrame with ``cls_pred`` (0/1) and *label_col* columns.
                Components absent from *df* are written as background.
    label_col : column of *df* holding the component label.
    """
    max_label = int(cc_vol.max()) if cc_vol.size else 0
    lut = np.zeros(max_label + 1, dtype=np.uint8)
    if len(df):
        ids = df[label_col].to_numpy(dtype=np.int64)
        codes = np.where(df["cls_pred"].to_numpy(dtype=int) == 1, 2, 1).astype(np.uint8)
        ok = (ids > 0) & (ids <= max_label)
        lut[ids[ok]] = codes[ok]
    return np.take(lut, cc_vol, mode="clip")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _optimal_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """F1-optimal threshold (predict crack when ``score >= threshold``)."""
    from ..evaluation.metrics import f1_optimal_threshold
    return f1_optimal_threshold(y_true, y_score)


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1_pore":         float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_crack":        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_crack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_crack":    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc":         float(roc_auc_score(y_true, y_proba)) if both else float("nan"),
        "n_pore":          int((y_true == 0).sum()),
        "n_crack":         int((y_true == 1).sum()),
    }


def _parse_labels(series: pd.Series) -> np.ndarray:
    """Map ``"pore"`` / ``"crack"`` (case-insensitive) to 0 / 1; NaN stays NaN."""
    s = series.astype("string").str.strip().str.lower()
    unknown = sorted(set(s.dropna()) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels {unknown}; expected {sorted(LABELS)} (or empty for "
                         "unlabelled rows)")
    return s.map(LABELS).to_numpy(dtype=float)


def _make_model(max_iter: int, random_state: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=max_iter,
                           random_state=random_state),
    )


def fit_classifier(
    descriptors: Sequence[Union[str, Path, pd.DataFrame]],
    label_col: str = "cls_label",
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    holdout: str = "random",
    max_iter: int = 1000,
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Train a logistic-regression pore / crack classifier and return a bundle.

    Parameters
    ----------
    descriptors : one descriptor table per volume (CSV path or DataFrame from
                  ``compute_descriptors``), with a *label_col* column holding
                  ``"pore"`` / ``"crack"``.  Rows with an empty label are
                  ignored.  Descriptors are in physical units, so volumes
                  scanned at different voxel sizes can be pooled.
    label_col   : label column.
    features    : feature columns; default
                  :data:`tomographia.components.CANONICAL_FEATURES`.  Every
                  feature must be present; a feature with no finite value
                  raises.
    test_size   : fraction held out for evaluation.
    holdout     : ``"random"`` — stratified split of the components;
                  ``"volume"`` — whole volumes are held out (needs ≥ 2
                  volumes), which measures generalisation to new specimens.
    max_iter    : ``LogisticRegression(max_iter=)``.
    random_state: RNG seed.
    save_path   : if given, the bundle is written with ``joblib.dump``.

    Returns
    -------
    Bundle dict (see the module docstring).

    Notes
    -----
    The model is a median imputer, a standard scaler and a class-balanced
    logistic regression, all fitted on the training split only.  The
    decision threshold maximises F1 on 5-fold **out-of-fold** predictions of
    the training split; the held-out split is used only for the reported
    metrics.
    """
    from sklearn.model_selection import (
        GroupShuffleSplit, StratifiedKFold, cross_val_predict, train_test_split,
    )

    features = list(CANONICAL_FEATURES if features is None else features)
    if holdout not in ("random", "volume"):
        raise ValueError("holdout must be 'random' or 'volume'")

    tables, names = [], []
    for i, d in enumerate(descriptors):
        if isinstance(d, pd.DataFrame):
            tables.append(d.assign(_volume=i))
            names.append(f"<DataFrame {i}>")
        else:
            tables.append(pd.read_csv(d).assign(_volume=i))
            names.append(str(d))
    if not tables:
        raise ValueError("descriptors is empty")
    all_df = pd.concat(tables, ignore_index=True)
    if label_col not in all_df.columns:
        raise KeyError(f"Label column {label_col!r} not found; columns: {list(all_df.columns)}")

    y = _parse_labels(all_df[label_col])
    keep = ~np.isnan(y)
    if (~keep).any():
        logger.info("Ignoring %d unlabelled rows", int((~keep).sum()))
    all_df, y = all_df.loc[keep].reset_index(drop=True), y[keep].astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain both 'pore' and 'crack' labels")

    X = _feature_matrix(all_df, features)
    empty = [f for f, ok in zip(features, np.isfinite(X).any(axis=0)) if not ok]
    if empty:
        raise ValueError(f"Features with no finite value: {empty}")

    if holdout == "volume":
        groups = all_df["_volume"].to_numpy()
        if len(np.unique(groups)) < 2:
            raise ValueError("holdout='volume' needs at least two descriptor tables")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr, te = next(splitter.split(X, y, groups))
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y,
                                  random_state=random_state)
    if len(np.unique(y[tr])) < 2:
        raise ValueError("The training split holds a single class")

    model = _make_model(max_iter, random_state)
    n_folds = int(min(5, np.bincount(y[tr]).min()))
    if n_folds >= 2:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        oof = cross_val_predict(model, X[tr], y[tr], cv=cv, method="predict_proba")[:, 1]
    else:
        logger.warning("Too few training samples of one class for cross-validation; the "
                       "threshold is selected on in-sample predictions")
        oof = model.fit(X[tr], y[tr]).predict_proba(X[tr])[:, 1]
    threshold = _optimal_f1_threshold(y[tr], oof)

    model.fit(X[tr], y[tr])
    test_proba = model.predict_proba(X[te])[:, 1]
    holdout_results = _eval_metrics(y[te], (test_proba >= threshold).astype(int), test_proba)

    bundle = {
        "model":           model,
        "features":        features,
        "threshold":       float(threshold),
        "holdout":         holdout,
        "holdout_results": holdout_results,
        "volumes":         names,
        "n_train":         int(len(tr)),
        "n_test":          int(len(te)),
        "tomographia_version": __version__,
    }
    if save_path is not None:
        import joblib
        joblib.dump(bundle, str(save_path))
    return bundle
