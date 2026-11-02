"""
Pipeline 1 — connected-component summary with ``cc3d`` + SciPy.

A lightweight alternative to :mod:`.pipeline2` that reports only size,
bounding box and centroid of each component (no shape descriptors).

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
    connectivity: int = 6,
) -> Tuple[np.ndarray, int]:
    """
    Binary closing then 3-D connected-component labelling (cc3d).

    Parameters
    ----------
    volume          : binary (or non-zero = defect) 3-D array.
    iterations      : number of closing iterations.
    structuring_dim : side length of the cubic structuring element.
    connectivity    : cc3d connectivity (6, 18, or 26).

    Returns
    -------
    (labels_array, n_labels)
    """
    import cc3d  # optional dependency: pip install connected-components-3d

    se = np.ones((structuring_dim, structuring_dim, structuring_dim))
    closed = ndimage.binary_closing(volume, structure=se, iterations=iterations)
    return cc3d.connected_components(closed, connectivity=connectivity, return_N=True)


def report(labels_out: np.ndarray, min_pixels: int = 10) -> dict:
    """
    Summarise connected components: size, bounding box, centroid.

    Parameters
    ----------
    labels_out : label array (e.g. from :func:`close_image`).
    min_pixels : components with fewer voxels than this are excluded.

    Returns
    -------
    dict keyed by integer label.
    """
    labeled, n_labels = ndimage.label(labels_out)
    slices = ndimage.find_objects(labeled)
    centroids = ndimage.center_of_mass(labels_out, labeled, range(1, n_labels + 1))
    counts = np.bincount(labeled.ravel())

    info: dict = {}
    for lbl in range(1, n_labels + 1):
        if counts[lbl] < min_pixels:
            continue
        s = slices[lbl - 1]
        info[lbl] = {
            "size": int(counts[lbl]),
            "bbox": {
                "xmin": s[2].start, "xmax": s[2].stop - 1,
                "ymin": s[1].start, "ymax": s[1].stop - 1,
                "zmin": s[0].start, "zmax": s[0].stop - 1,
            },
            "centroid": {
                "x": centroids[lbl - 1][2],
                "y": centroids[lbl - 1][1],
                "z": centroids[lbl - 1][0],
            },
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
