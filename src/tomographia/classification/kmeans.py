"""KMeans-based unsupervised defect clustering."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_defects(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple:
    """
    KMeans clustering of standardised descriptor *features*.

    Rows with NaN / ±Inf in any feature column are excluded from clustering;
    their ``cluster`` value is NaN in the returned DataFrame.

    Parameters
    ----------
    df          : descriptor DataFrame (output of ``compute_descriptors``).
    features    : columns to cluster on; default: the shape features of
                  :data:`tomographia.components.CANONICAL_FEATURES`
                  (without ``int_std_rel``).
    n_clusters  : number of KMeans clusters.
    random_state: RNG seed for reproducibility.

    Returns
    -------
    (df_with_cluster_col, fitted_KMeans, fitted_StandardScaler)
    """
    if features is None:
        from ..components.pipeline2 import CANONICAL_FEATURES
        features = [f for f in CANONICAL_FEATURES if f != "int_std_rel"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns {missing}")
    X = df[features].copy().replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    X_clean = X.loc[valid]
    if len(X_clean) < n_clusters:
        raise ValueError(f"{len(X_clean)} complete rows for {n_clusters} clusters")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    col = np.full(len(df), np.nan)
    col[valid.to_numpy()] = cluster_labels.astype(float)
    df = df.copy()
    df["cluster"] = col
    return df, km, scaler


def make_cluster_volume(
    cc: np.ndarray,
    df: pd.DataFrame,
    out_dtype=np.int16,
) -> np.ndarray:
    """
    Build a 3-D volume where each voxel holds its cluster id + 1.

    Encoding: 0 = background, 1..K = cluster ids.

    Parameters
    ----------
    cc       : integer CC label array (e.g. from ``build_cc_mask``).
    df       : DataFrame with ``label_id`` and ``cluster`` columns.
    out_dtype: NumPy dtype for the output volume.
    """
    lv = cc.astype(np.int64, copy=False)
    if not lv.flags["C_CONTIGUOUS"]:
        lv = np.ascontiguousarray(lv)

    valid = df["cluster"].notna()
    keys = df.loc[valid, "label_id"].to_numpy(dtype=np.int64)
    vals = df.loc[valid, "cluster"].to_numpy(dtype=np.int64)

    if keys.size == 0:
        return np.zeros_like(lv, dtype=out_dtype)

    lut = np.zeros(int(lv.max()) + 1, dtype=out_dtype)
    lut[keys] = (vals + 1).astype(out_dtype)
    return np.take(lut, lv, mode="clip")
