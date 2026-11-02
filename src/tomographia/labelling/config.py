"""Configuration of a patch-labelling session."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabelingConfig:
    """
    Settings of a labelling session.

    Attributes
    ----------
    data_path           : folder containing the patch CSV and the ``.tif`` slices.
    identifier          : substring identifying the CSV file in ``data_path``.
    label_col           : CSV column where labels are written.
    intensity_col       : CSV column used for automatic labelling.
    threshold           : patches with ``intensity < threshold`` are auto-labelled 0.
    focus_unzoom        : margin (pixels) around the patch in the context view.
    save_after_x_labels : autosave cadence.
    scale, offset       : display contrast, ``img * scale + offset`` clipped to [0, 255].
    """
    data_path: str
    identifier: str
    label_col: str = "defect"
    intensity_col: str = "intensity"
    threshold: float = 50.0
    focus_unzoom: int = 256
    save_after_x_labels: int = 20
    scale: float = 1.0
    offset: float = 0.0
