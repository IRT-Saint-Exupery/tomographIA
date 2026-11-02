"""
Pipeline 1 — connected-component summary with ``cc3d`` + SciPy.

A lightweight alternative to :mod:`.pipeline2` that reports only size,
bounding box and centroid of each component (no shape descriptors).  As in
Pipeline 2, closing only groups fragments: sizes are those of the predicted
voxels.

Requires: ``connected-components-3d`` (imports as ``cc3d``).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import ndimage


def close_image(
    volume: np.ndarray,
    iterations: int = 1,
    structuring_dim: int = 5,
    connectivity: int = 26,
) -> Tuple[np.ndarray, int]:
    """
    Group defect fragments by binary closing, then label them with cc3d.

    The closing (cubic element of side *structuring_dim*, zero-padded so that
    defects touching the border are not eroded) decides which voxels belong
    to the same defect; the returned labels cover exactly the non-zero voxels
    of *volume*.

    Parameters
    ----------
    volume          : binary (or non-zero = defect) 3-D array.
    iterations      : number of closing iterations.
    structuring_dim : side length of the cubic structuring element.
    connectivity    : cc3d connectivity (6, 18, or 26).

    Returns
    -------
    ``(labels, n_labels)`` with labels ``1..n_labels``.
    """
    import cc3d  # optional dependency: pip install connected-components-3d

    mask = np.asarray(volume) > 0
    pad = (structuring_dim // 2 + 1) * iterations
    se = np.ones((structuring_dim,) * 3, dtype=bool)
    closed = ndimage.binary_closing(np.pad(mask, pad), structure=se, iterations=iterations)
    closed = closed[(slice(pad, -pad),) * 3] | mask
    labels = cc3d.connected_components(closed, connectivity=connectivity)
    labels[~mask] = 0
    from skimage.segmentation import relabel_sequential
    labels = relabel_sequential(labels)[0]
    return labels, int(labels.max())


def report(labels_out: np.ndarray, min_pixels: int = 10) -> dict:
    """
    Summarise labelled components: size, bounding box, centroid.

    Parameters
    ----------
    labels_out : label array (e.g. from :func:`close_image`); its labels are
                 used as they are.
    min_pixels : components with fewer voxels are excluded.

    Returns
    -------
    dict keyed by label.  Bounding boxes are voxel indices with **exclusive**
    max, as in :func:`compute_descriptors`.
    """
    lab = np.asarray(labels_out)
    counts = np.bincount(lab.ravel())
    slices = ndimage.find_objects(lab)
    ids = [i for i in range(1, len(counts)) if counts[i] >= max(min_pixels, 1)]
    centroids = ndimage.center_of_mass(lab > 0, lab, ids) if ids else []

    info: dict = {}
    for lbl, c in zip(ids, centroids):
        s = slices[lbl - 1]
        info[lbl] = {
            "size": int(counts[lbl]),
            "bbox": {
                "xmin": s[2].start, "xmax": s[2].stop,
                "ymin": s[1].start, "ymax": s[1].stop,
                "zmin": s[0].start, "zmax": s[0].stop,
            },
            "centroid": {"x": c[2], "y": c[1], "z": c[0]},
        }
    return info


def dict_to_df(report_dict: dict) -> pd.DataFrame:
    """Convert the dict from :func:`report` to a flat DataFrame."""
    rows = []
    for key, val in report_dict.items():
        row: dict = {"label": key, "size": val["size"]}
        for coord in ("x", "y", "z"):
            row[f"centroid_{coord}"] = val["centroid"][coord]
            row[f"bbox_{coord}_min"] = val["bbox"][f"{coord}min"]
            row[f"bbox_{coord}_max"] = val["bbox"][f"{coord}max"]
        rows.append(row)
    return pd.DataFrame(rows)
