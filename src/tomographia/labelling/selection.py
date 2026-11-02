"""Selecting the patches that still need a label."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

COORDS = ("x", "y", "z")


def df_recap(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    """Counts of unlabelled patches and of labels 0 / 1 / 2."""
    col = df[label_col]
    return {"unlabelled": int(col.isna().sum()), "label_0": int((col == 0).sum()),
            "label_1": int((col == 1).sum()), "label_2": int((col == 2).sum())}


def apply_threshold_labels(df: pd.DataFrame, label_col: str, intensity_col: str,
                           threshold: float) -> pd.DataFrame:
    """
    Label 0 every *unlabelled* patch with ``intensity < threshold`` (in place,
    returns *df*).  Existing labels are never overwritten; nothing happens if
    the intensity column is absent.
    """
    if label_col not in df.columns:
        df[label_col] = np.nan
    if intensity_col not in df.columns:
        return df
    fill = df[label_col].isna() & (df[intensity_col] < threshold)
    df.loc[fill, label_col] = 0
    return df


def get_unlabelled_patches(df: pd.DataFrame, label_col: str, shuffle: bool = True,
                           seed: int = 42) -> pd.DataFrame:
    """Unlabelled rows (optionally shuffled) with integer coordinates."""
    out = df[df[label_col].isna()].copy()
    if shuffle and len(out):
        out = out.sample(frac=1, random_state=seed)
    for c in COORDS:
        if c in out.columns:
            out[c] = out[c].astype(int)
    return out
