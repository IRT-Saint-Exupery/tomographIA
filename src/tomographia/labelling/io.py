"""CSV and slice-file helpers for the labelling tool."""
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd


def find_label_csv(data_path: str, identifier: str) -> str:
    """Path of the first ``.csv`` in *data_path* whose name contains *identifier*."""
    for f in sorted(os.listdir(data_path)):
        if identifier in f and f.lower().endswith(".csv"):
            return os.path.join(data_path, f)
    raise FileNotFoundError(f"No CSV containing '{identifier}' in {data_path}")


def load_label_df(data_path: str, identifier: str) -> pd.DataFrame:
    """Load the patch table to be labelled."""
    return pd.read_csv(find_label_csv(data_path, identifier))


def save_label_df(df: pd.DataFrame, data_path: str, identifier: str) -> None:
    """Write the patch table back to its CSV."""
    df.to_csv(find_label_csv(data_path, identifier), index=False)


def list_slice_images(data_path: str, max_count: Optional[int] = None) -> List[str]:
    """Sorted ``.tif`` slice file names (slice ``z`` is the ``z``-th file)."""
    imgs = sorted(f for f in os.listdir(data_path) if f.lower().endswith((".tif", ".tiff")))
    return imgs[:max_count] if max_count is not None else imgs
