"""
Logistic-regression (sklearn-bundle) classification of defects.

Inference path extracted from ``_cls_worker()`` in ``tomo_app/backend/main.py``.
Training path (``fit_classifier``) follows the notebook
``notebooks/magma/classification/06_supervised_benchmark.ipynb``.

Bundle format
-------------
Produced by :func:`fit_classifier` or the training notebook::

    {
        "model"            : sklearn estimator with ``predict_proba()``,
        "scaler"           : ``StandardScaler``,
        "features"         : list[str],   # feats_avail — actually used at training time
        "threshold"        : float,       # optimal P(crack) threshold (TRAIN set PR curve)
        "holdout_results"  : dict,        # metrics on the held-out test split
        "volumes"          : list[str],   # CSV paths used for training
        "voxel_sizes"      : list[float], # one per volume
    }

The 12 canonical features are::

    sphericity, anisotropy, flatness, elongation, fill_frac,
    r1, r2, ar_zx, ar_zy, ar_yx, int_std_rel, log_voxels
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

#: Default ordered feature list.  ``fit_classifier`` stores the actual usable
#: subset (``feats_avail``) in the bundle, not necessarily this full list.
CANONICAL_FEATURES: List[str] = [
    "sphericity", "anisotropy", "flatness", "elongation", "fill_frac",
    "r1", "r2", "ar_zx", "ar_zy", "ar_yx", "int_std_rel", "log_voxels",
]


def _load_bundle(bundle_or_path) -> dict:
    """Accept a dict, a path to a .joblib file, or a directory containing one."""
    if isinstance(bundle_or_path, dict):
        return bundle_or_path
    import joblib
    p = Path(bundle_or_path)
    if p.is_dir():
        hits = sorted(p.glob("*.joblib"))
        if not hits:
            raise FileNotFoundError(f"No .joblib file found in {p}")
        return joblib.load(str(hits[0]))
    return joblib.load(str(p))


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive backward-compat columns absent from older descriptor CSVs."""
    df = df.copy()
    # r1/r2 from raw eigenvalues (old CSVs pre-dating the field)
    if "r1" not in df.columns and "eig0" in df.columns:
        e0 = df["eig0"].to_numpy(float)
        df["r1"] = np.where(e0 > 0, df["eig1"] / e0, np.nan)
        df["r2"] = np.where(e0 > 0, df["eig2"] / e0, np.nan)
    # log_voxels (physical log-volume)
    if "log_voxels" not in df.columns:
        if "volume" in df.columns:
            df["log_voxels"] = np.log(df["volume"].clip(lower=1e-12))
        elif "voxels" in df.columns:
            df["log_voxels"] = np.log(df["voxels"].clip(lower=1))
    # int_std_rel (relative intensity std)
    if (
        "int_std_rel" not in df.columns
        and "int_std" in df.columns
        and "int_mean" in df.columns
    ):
        df["int_std_rel"] = df["int_std"] / (df["int_mean"] + 1e-7)
    return df


def classify_defects_logistic(
    df: pd.DataFrame,
    bundle: Union[dict, str, Path],
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Classify defects using a pre-trained sklearn bundle.

    Parameters
    ----------
    df        : descriptor DataFrame (output of ``compute_descriptors``).
    bundle    : joblib bundle dict, path to a ``.joblib`` file, or a
                directory containing exactly one ``.joblib`` file.
    threshold : override the bundle's stored threshold.  Falls back to
                ``bundle["threshold"]`` then to ``0.5`` if absent.

    Returns
    -------
    Copy of *df* with three new columns added:

    ``cls_proba``  float  — P(crack) from the classifier.
    ``cls_pred``   int    — 0 (pore) or 1 (crack).
    ``cls_label``  str    — ``"pore"`` or ``"crack"``.
    """
    b = _load_bundle(bundle)
    model    = b["model"]
    scaler   = b["scaler"]
    features = b["features"]
    thr = threshold if threshold is not None else b.get("threshold", 0.5)

    df = _ensure_derived_columns(df)

    # Any feature the model expects but missing from the CSV → NaN → 0.0
    for f in features:
        if f not in df.columns:
            df[f] = np.nan

    X = df[features].to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    pred  = (proba >= thr).astype(int)

    df = df.copy()
    df["cls_proba"] = proba
    df["cls_pred"]  = pred
    df["cls_label"] = np.where(pred == 1, "crack", "pore")
    return df


def build_classified_volume(
    cc_vol: np.ndarray,
    df: pd.DataFrame,
    label_col: str = "label_id",
) -> np.ndarray:
    """
    Build a uint8 classified volume from a CC label array and classified DataFrame.

    Encoding: ``0`` = background, ``1`` = pore, ``2`` = crack.

    Parameters
    ----------
    cc_vol    : integer CC label array (e.g. from ``build_cc_mask``).
    df        : DataFrame with ``cls_pred`` (0/1) and *label_col* columns.
    label_col : name of the column in *df* that holds the CC integer label.

    Notes
    -----
    This function is deliberately slice-friendly: it builds a look-up table
    rather than iterating over the volume, so memory usage scales with the
    number of components, not voxels.
    """
    max_label = int(cc_vol.max())
    lut = np.zeros(max_label + 1, dtype=np.uint8)
    for _, row in df.iterrows():
        lid = int(row[label_col])
        if 0 < lid <= max_label:
            lut[lid] = 2 if int(row["cls_pred"]) == 1 else 1

    flat = np.clip(cc_vol.ravel().astype(np.int64), 0, max_label)
    return lut[flat].reshape(cc_vol.shape)


# ---------------------------------------------------------------------------
# Training helpers (used by fit_classifier)
# ---------------------------------------------------------------------------

def _preprocess_for_training(df: pd.DataFrame, voxel_size: float) -> pd.DataFrame:
    """Derive r1/r2, log_voxels, int_std_rel with physical voxel calibration.

    Unlike :func:`_ensure_derived_columns` (inference-time helper that accepts
    a pre-computed ``log_voxels``), this function always recomputes
    ``log_voxels`` from the raw voxel count:

        log_voxels = log(voxels) + log(voxel_size ** 3)

    This matches the notebook formula exactly (``np.log(voxels) +
    np.log(VOXEL_SIZES[i]**3)``), ensuring that volumes with different
    physical sizes are properly calibrated before fitting.
    """
    df = df.copy()
    voxel_vol = voxel_size ** 3

    if "r1" not in df.columns and "eig0" in df.columns:
        e0 = df["eig0"].to_numpy(float)
        df["r1"] = np.where(e0 > 0, df["eig1"] / e0, np.nan)
        df["r2"] = np.where(e0 > 0, df["eig2"] / e0, np.nan)

    if "voxels" in df.columns:
        df["log_voxels"] = np.log(df["voxels"].clip(lower=1)) + np.log(voxel_vol)

    if (
        "int_std_rel" not in df.columns
        and "int_std" in df.columns
        and "int_mean" in df.columns
    ):
        df["int_std_rel"] = df["int_std"] / (df["int_mean"] + 1e-7)

    return df


def _optimal_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Find the decision threshold that maximises F1-crack on the PR curve.

    **Must be called on training predictions only.**  Calling with test /
    held-out predictions constitutes threshold-selection leakage: the threshold
    is then tuned to the test distribution rather than the training
    distribution, artificially inflating held-out F1-crack.

    Deviation from original notebook
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The notebook's ``optimal_threshold()`` does NOT trim the arrays before
    computing F1, i.e. it uses the full-length ``precision``/``recall`` and
    then indexes ``thresholds[np.argmax(f1)]``.  This is a latent
    off-by-one: ``precision_recall_curve`` returns arrays of length
    ``n_thresholds + 1`` but ``thresholds`` has length ``n_thresholds``,
    so on degenerate data (all F1 = 0) the indexing would raise
    ``IndexError``.

    **In practice, this trimming produces the same threshold as the
    notebook** because the sentinel's F1 is always 0.0 (P=1, R=0), so
    ``argmax`` never selects it on a non-trivial model.  The fix is a
    safety improvement only; it does not change threshold selection on
    real training data.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precision[:-1], recall[:-1]      # drop (P=1, R=0) sentinel
    f1 = 2 * p * r / (p + r + 1e-8)
    return float(thresholds[int(np.argmax(f1))])


def _eval_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1_pore":         float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_crack":        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_crack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_crack":    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc":         float(roc_auc_score(y_true, y_proba)),
        "n_pore":          int((y_true == 0).sum()),
        "n_crack":         int((y_true == 1).sum()),
    }


# ---------------------------------------------------------------------------
# fit_classifier
# ---------------------------------------------------------------------------

def fit_classifier(
    descriptor_csvs: List[Union[str, Path]],
    voxel_sizes: List[float],
    label_col: str = "cls_label",
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    max_iter: int = 1000,
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Train a logistic-regression defect classifier and return a joblib bundle.

    Parameters
    ----------
    descriptor_csvs : list of paths to per-volume descriptor CSVs.
                      Each CSV must contain a label column (default ``cls_label``)
                      with values ``"pore"`` and ``"crack"``.
    voxel_sizes     : isotropic physical voxel size in mm, **one per CSV**.
                      ``voxel_sizes[i]`` must correspond to ``descriptor_csvs[i]``.
                      Example: ``[0.014, 0.014]`` for two 14-µm volumes.
                      Used to compute ``log_voxels = log(voxels) + log(vs**3)``,
                      matching the notebook formula exactly.
    label_col       : column name holding ground-truth labels.
    features        : feature columns to attempt using.  Defaults to
                      :data:`CANONICAL_FEATURES`.  **The bundle stores
                      ``feats_avail`` (the subset that is actually present and
                      non-null in the loaded data) not this parameter verbatim.**
                      A bundle claiming features that weren't fitted would break
                      inference silently.
    test_size       : fraction held out for held-out evaluation.
    max_iter        : ``LogisticRegression(max_iter=)`` — increase if convergence
                      warnings appear.
    random_state    : RNG seed for reproducibility.
    save_path       : if given, serialise the bundle with ``joblib.dump``.

    Returns
    -------
    bundle dict compatible with :func:`classify_defects_logistic`::

        "model"           : fitted LogisticRegression (bare — not a Pipeline)
        "scaler"          : fitted StandardScaler (bare)
        "features"        : feats_avail — what the model was actually trained on
        "threshold"       : optimal P(crack) threshold (computed on TRAIN only)
        "holdout_results" : evaluation metrics on the held-out test split
        "volumes"         : list of CSV paths (str) for provenance
        "voxel_sizes"     : list of voxel sizes used

    Notes
    -----
    **Threshold computation — no leakage**

    The threshold is computed on *training-set* predictions only::

        # 1. Split first — X_test is never seen before step 7
        X_train, X_test, y_train, y_test = train_test_split(...)

        # 2. Fit scaler + model on TRAIN
        scaler.fit_transform(X_train)
        clf.fit(X_train_sc, y_train)

        # 3. Threshold on TRAIN predictions → no leakage
        threshold = _optimal_f1_threshold(y_train, clf.predict_proba(X_train_sc)[:,1])

        # 4. Held-out evaluation with that threshold
        test_pred = (clf.predict_proba(X_test_sc)[:,1] >= threshold).astype(int)

    Applying ``_optimal_f1_threshold`` to test predictions would constitute
    threshold-selection leakage and inflate held-out F1-crack.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if features is None:
        features = CANONICAL_FEATURES

    # ── 0. Defensive input validation ─────────────────────────────────────
    if len(descriptor_csvs) != len(voxel_sizes):
        raise ValueError(
            f"descriptor_csvs and voxel_sizes must have the same length; "
            f"got {len(descriptor_csvs)} CSV(s) and {len(voxel_sizes)} voxel size(s). "
            "Each voxel_sizes[i] must correspond to descriptor_csvs[i]."
        )

    # ── 1. Load and preprocess all CSVs ──────────────────────────────────
    dfs: list = []
    for csv_path, vs in zip(descriptor_csvs, voxel_sizes):
        df_i = pd.read_csv(csv_path)
        df_i = _preprocess_for_training(df_i, voxel_size=vs)
        dfs.append(df_i)
    all_df = pd.concat(dfs, ignore_index=True)

    if label_col not in all_df.columns:
        raise KeyError(
            f"Label column {label_col!r} not found in the loaded data. "
            f"Available columns: {list(all_df.columns)}"
        )

    # ── 2. Determine feats_avail ──────────────────────────────────────────
    # Include only features that (a) exist in the DataFrame and (b) have at
    # least one finite non-null value.  The bundle stores feats_avail exactly
    # as used for fitting — not the caller's features parameter.
    def _usable(col: str) -> bool:
        if col not in all_df.columns:
            return False
        return all_df[col].replace([np.inf, -np.inf], np.nan).notna().any()

    feats_avail = [f for f in features if _usable(f)]
    if not feats_avail:
        raise ValueError(
            "No usable features found after loading and preprocessing. "
            f"Requested features: {features}"
        )

    # ── 3. Build feature matrix and binary labels ─────────────────────────
    X = all_df[feats_avail].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = (all_df[label_col].str.lower().str.strip() == "crack").astype(int).to_numpy()

    # ── 4. Stratified train / test split ─────────────────────────────────
    # X_test is quarantined from this point until step 7.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # ── 5. Fit scaler + model on TRAIN only ───────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit + transform TRAIN
    X_test_sc  = scaler.transform(X_test)         # transform only (no fit)

    clf = LogisticRegression(
        class_weight="balanced",    # mitigates pore/crack imbalance
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X_train_sc, y_train)

    # ── 6. Optimal threshold — TRAIN predictions only (no leakage) ────────
    train_proba = clf.predict_proba(X_train_sc)[:, 1]
    threshold   = _optimal_f1_threshold(y_train, train_proba)

    # ── 7. Held-out evaluation ────────────────────────────────────────────
    test_proba = clf.predict_proba(X_test_sc)[:, 1]
    test_pred  = (test_proba >= threshold).astype(int)
    holdout_results = _eval_metrics(y_test, test_pred, test_proba)

    # ── 8. Assemble bundle ────────────────────────────────────────────────
    bundle = {
        "model":           clf,
        "scaler":          scaler,
        "features":        feats_avail,      # actual features used (not `features`)
        "threshold":       float(threshold),
        "holdout_results": holdout_results,
        "volumes":         [str(p) for p in descriptor_csvs],
        "voxel_sizes":     list(voxel_sizes),
    }

    if save_path is not None:
        import joblib
        joblib.dump(bundle, str(save_path))

    return bundle
