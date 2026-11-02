"""Patch-by-patch labelling session (shared by the CLI and the Streamlit app)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from .config import LabelingConfig
from .io import list_slice_images, save_label_df
from .preprocess import treat_image
from .selection import get_unlabelled_patches
from .visualize import extract_patch, patch_overview_figure

VALID_LABELS = (0, 1, 2)  # 0: no defect, 1: defect, 2: unsure


@dataclass
class PatchRecord:
    """One patch to review (centre ``x, y`` on slice ``z``)."""
    id_global: str
    x: int
    y: int
    z: int
    intensity: float
    patch_size: int = 64
    prediction: Optional[float] = None


class LabelingSession:
    """
    Iterate over unlabelled patches, display them and record labels.

    The patch table needs the columns ``id_global``, ``x``, ``y``, ``z`` and,
    optionally, ``patch_size`` (default 64), ``intensity`` and ``prediction``
    (model probability shown as a hint).
    """

    def __init__(self, df: pd.DataFrame, cfg: LabelingConfig) -> None:
        self.df = df
        self.cfg = cfg
        self.image_list = list_slice_images(cfg.data_path)
        if not self.image_list:
            raise FileNotFoundError(f"No .tif slices found in {cfg.data_path}")
        self.label_col = cfg.label_col
        if self.label_col not in self.df.columns:
            self.df[self.label_col] = np.nan
        self.patch_size = int(self.df["patch_size"].iloc[0]) if "patch_size" in df.columns else 64
        self._stop = False
        self._since_save = 0
        self._queue = get_unlabelled_patches(self.df, self.label_col).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._queue)

    def iter_patches(self) -> Iterator[PatchRecord]:
        """Yield unlabelled patches whose slice exists, until :meth:`stop`."""
        has_int = "intensity" in self._queue.columns
        has_pred = "prediction" in self._queue.columns
        for _, row in self._queue.iterrows():
            if self._stop:
                break
            if not 0 <= int(row["z"]) < len(self.image_list):
                continue
            pred = row["prediction"] if has_pred else None
            yield PatchRecord(
                id_global=str(row["id_global"]), x=int(row["x"]), y=int(row["y"]),
                z=int(row["z"]),
                intensity=float(row["intensity"]) if has_int else float("nan"),
                patch_size=self.patch_size,
                prediction=float(pred) if pred is not None and pd.notna(pred) else None,
            )

    def load_slice(self, patch: PatchRecord) -> np.ndarray:
        """Slice ``z`` of *patch* with the display contrast of the config."""
        path = os.path.join(self.cfg.data_path, self.image_list[patch.z])
        return treat_image(path, self.cfg.scale, self.cfg.offset)

    def figure_for_patch(self, image: np.ndarray, patch: PatchRecord) -> Tuple[object, object]:
        """Patch + context figure (matplotlib)."""
        patch_img, bounds = extract_patch(image, patch.x, patch.y, patch.patch_size)
        return patch_overview_figure(image, patch_img, bounds, self.cfg.focus_unzoom)

    def assign_label(self, patch: PatchRecord, label: int) -> None:
        """Write *label* (0, 1 or 2) for the patch; autosave every N labels."""
        if label not in VALID_LABELS:
            raise ValueError(f"Label must be one of {VALID_LABELS}")
        mask = self.df["id_global"].astype(str) == patch.id_global
        if not mask.any():
            raise KeyError(f"Unknown patch id {patch.id_global}")
        self.df.loc[mask, self.label_col] = int(label)
        self._since_save += 1
        if self._since_save >= self.cfg.save_after_x_labels:
            self.save()

    def stop(self) -> None:
        """Stop :meth:`iter_patches` after the current patch."""
        self._stop = True

    def save(self) -> None:
        """Write the table back to its CSV."""
        save_label_df(self.df, self.cfg.data_path, self.cfg.identifier)
        self._since_save = 0
