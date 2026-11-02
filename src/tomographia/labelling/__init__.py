"""Interactive labelling of 2-D CT patches (terminal CLI and Streamlit app).

The patch table is a CSV with ``id_global, x, y, z`` (patch centre and slice
index), optionally ``patch_size``, an ``intensity`` column used to auto-label
obvious background, and a ``prediction`` column (model hint).

- Terminal: ``tomographia-label --data-path ... --identifier ...``
- Browser:  ``tomographia-label-app`` (requires the ``labelling`` extra)

Ported from ``tomographIA.labelling.label_2d``; see the CHANGELOG for fixes.
"""
from .config import LabelingConfig
from .io import find_label_csv, list_slice_images, load_label_df, save_label_df
from .preprocess import treat_image
from .selection import apply_threshold_labels, df_recap, get_unlabelled_patches
from .session import LabelingSession, PatchRecord
from .visualize import extract_patch, patch_overview_figure

__all__ = [
    "LabelingConfig",
    "find_label_csv", "load_label_df", "save_label_df", "list_slice_images",
    "treat_image",
    "df_recap", "apply_threshold_labels", "get_unlabelled_patches",
    "extract_patch", "patch_overview_figure",
    "LabelingSession", "PatchRecord",
    "launch_app",
]


def launch_app() -> None:
    """Start the Streamlit labelling app (console script ``tomographia-label-app``)."""
    import os
    import subprocess
    import sys

    app = os.path.join(os.path.dirname(__file__), "app.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", app]))
