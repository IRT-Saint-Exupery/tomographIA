"""Connected-component analysis for segmentation outputs.

Pipeline 2 (default — skimage):
    :func:`build_cc_mask`, :func:`compute_descriptors`,
    :func:`load_grayscale_volume`, :data:`CANONICAL_FEATURES`

Pipeline 1 (size / bbox / centroid only — cc3d + scipy):
    :func:`close_image`, :func:`cc_report`, :func:`dict_to_df`
"""
from .pipeline2 import CANONICAL_FEATURES, build_cc_mask, compute_descriptors, load_grayscale_volume
from .pipeline1 import close_image, dict_to_df
from .pipeline1 import report as cc_report

__all__ = [
    # Pipeline 2 (default)
    "build_cc_mask",
    "compute_descriptors",
    "load_grayscale_volume",
    "CANONICAL_FEATURES",
    # Pipeline 1
    "close_image",
    "cc_report",
    "dict_to_df",
]
