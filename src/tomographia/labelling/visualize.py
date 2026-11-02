"""Patch / context figure for the labelling tool."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def extract_patch(image: np.ndarray, x: int, y: int, patch_size: int
                  ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Square patch centred on ``(x, y)`` clamped to the image.

    Returns ``(patch, (x_min, y_min, x_max, y_max))`` with unclamped bounds.
    """
    h, w = image.shape
    half = patch_size // 2
    x0, x1, y0, y1 = x - half, x + half, y - half, y + half
    patch = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    return patch, (x0, y0, x1, y1)


def patch_overview_figure(image: np.ndarray, patch: np.ndarray,
                          rect_bounds: Tuple[int, int, int, int], focus_unzoom: int = 256):
    """Two panels: the patch, and the slice zoomed around it with a red box."""
    import matplotlib.pyplot as plt

    x0, y0, x1, y1 = rect_bounds
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(patch, cmap="gray")
    ax[0].set_title("Patch")
    ax[1].imshow(image, cmap="gray")
    ax[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="r", facecolor="none"))
    ax[1].set_xlim([x0 - focus_unzoom, x1 + focus_unzoom])
    ax[1].set_ylim([y1 + focus_unzoom, y0 - focus_unzoom])
    ax[1].set_title("Context")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    return fig, ax
