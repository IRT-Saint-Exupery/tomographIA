"""
Post-processing and rendering helpers for stacked patch predictions.

- ``remove_sparse_voxels`` removes isolated voxels (6-neighbour count).
- ``isosurface_mesh`` returns vertices in (z, y, x) order scaled by the given
  spacing.
- ``plot_slice_overlay`` draws patch predictions over a slice (matplotlib).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, convolve
from skimage.measure import block_reduce, marching_cubes

from .predict import SlicePrediction


def downsample_volume(volume: np.ndarray, factor: Tuple[int, int, int] = (2, 2, 2)) -> np.ndarray:
    """Block-average downsampling of a ``(Z, Y, X)`` volume."""
    return block_reduce(volume, block_size=tuple(factor), func=np.mean)


def remove_sparse_voxels(volume: np.ndarray, min_neighbors: int = 1) -> np.ndarray:
    """Drop positive voxels with fewer than *min_neighbors* 6-connected positive neighbours."""
    v = (np.asarray(volume) > 0).astype(np.int32)
    kernel = np.zeros((3, 3, 3), dtype=np.int32)
    kernel[0, 1, 1] = kernel[2, 1, 1] = kernel[1, 0, 1] = 1
    kernel[1, 2, 1] = kernel[1, 1, 0] = kernel[1, 1, 2] = 1
    n = convolve(v, kernel, mode="constant", cval=0)
    return ((v == 1) & (n >= min_neighbors)).astype(np.uint8)


def surface_voxels(binary_volume: np.ndarray, size: int = 3) -> np.ndarray:
    """Keep only the outer shell of a binary volume (erosion with a ``size³`` cube)."""
    b = np.asarray(binary_volume).astype(bool)
    eroded = binary_erosion(b, structure=np.ones((size, size, size)))
    return (b & ~eroded).astype(np.uint8)


def isosurface_mesh(volume: np.ndarray, level: float = 0.5,
                    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """
    Marching-cubes mesh of a ``(Z, Y, X)`` volume.

    Returns ``(verts [N, 3] in (z, y, x) physical units, faces [M, 3])``;
    both are empty when the volume does not cross ``level`` (e.g. no defect).
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.size == 0 or not (vol.min() < level < vol.max()):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    verts, faces, _, _ = marching_cubes(vol, level=level, spacing=tuple(spacing))
    return verts, faces


def plot_slice_overlay(image: np.ndarray, prediction: SlicePrediction,
                       threshold: Optional[float] = None, cmap: str = "RdYlGn_r",
                       alpha: float = 0.35, zoom: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
                       title: Optional[str] = None, ax=None):
    """
    Slice with patch probabilities overlaid as coloured squares.

    Parameters
    ----------
    threshold : if given, only patches with ``prob >= threshold`` are drawn.
    zoom      : ``((x_min, x_max), (y_min, y_max))`` region to display.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib: pip install 'tomographia[report]'") from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    cm = plt.get_cmap(cmap)
    p = prediction.patch_size
    for i, r in enumerate(prediction.rows):
        for j, c in enumerate(prediction.cols):
            v = float(prediction.prob[i, j])
            if threshold is not None and v < threshold:
                continue
            ax.add_patch(Rectangle((c, r), p, p, facecolor=cm(v), alpha=alpha,
                                   edgecolor=cm(v), linewidth=0.5))
    if zoom is not None:
        ax.set_xlim(*zoom[0])
        ax.set_ylim(zoom[1][1], zoom[1][0])
    if title:
        ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return ax.figure
