"""
Intensity-window yield computation.

Yield definition
----------------
A voxel is *matrix* if:

  1. its grey value lies within the window
     ``[ref_value * (1 - lower),  ref_value * (1 + upper)]``, and
  2. it is **not** classified as a defect (``classified_vol == 0``).

Overall yield per slice:

  yield = defect_voxels / (defect_voxels + matrix_voxels)

This is a **defect fraction** (higher = more defects), not a conventional
"yield" (fraction of sound material); the keys keep the name ``*_yield``.
A slice with neither defect nor matrix voxels gives NaN.  Every predicted
defect voxel counts (false positives in air or at the part surface included),
and sound material outside the grey-value window (beam hardening, partial
volume around pores) is left out: both raise the fraction, so choose the
window on the matrix of the scan at hand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from .._slices import list_slices, read_slice


def compute_yield_report(
    classified_vol: np.ndarray,
    tif_dir: Union[str, Path],
    ref_value: float,
    lower: float,
    upper: float,
    slice_start: int = 0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Compute per-slice and overall defect yields.

    Parameters
    ----------
    classified_vol : uint8 volume — 0=background, 1=pore, 2=crack.
    tif_dir        : directory of raw grayscale TIF slices (natural order, as
                     read by the segmentation).
    ref_value      : reference grey value defining the matrix intensity window.
    lower          : fractional lower half-width below *ref_value* (e.g. 0.3).
    upper          : fractional upper half-width above *ref_value* (e.g. 0.7).
    slice_start    : absolute TIF index that corresponds to ``classified_vol[0]``.
                     Non-zero when inference was run on a subset of slices.
    progress_cb    : optional ``(done: int, total: int) → None`` callback.

    Returns
    -------
    dict with keys::

        "pore_yield"    : float (%)
        "crack_yield"   : float (%)
        "total_yield"   : float (%)
        "matrix_voxels" : int
        "pore_voxels"   : int
        "crack_voxels"  : int
        "per_slice"     : {
            "z"    : list[int],   # absolute TIF index
            "pore" : list[float], # %
            "crack": list[float], # %
            "total": list[float], # %
        }
    """
    if ref_value <= 0:
        raise ValueError("ref_value must be the (positive) reference grey value of the matrix")
    tifs = list_slices(str(tif_dir))
    nz = classified_vol.shape[0]
    min_val = ref_value * (1.0 - lower)
    max_val = ref_value * (1.0 + upper)
    n_slices = min(nz, max(0, len(tifs) - slice_start))
    if n_slices != nz:
        raise ValueError(f"classified_vol has {nz} slices but {tif_dir} holds only "
                         f"{max(0, len(tifs) - slice_start)} slices from slice_start="
                         f"{slice_start}; check slice_start and the slice folder")

    per_z: list = []
    per_pore: list = []
    per_crack: list = []
    per_total: list = []
    total_matrix = total_pore = total_crack = 0

    for z in range(n_slices):
        arr = read_slice(tifs[slice_start + z]).astype(np.float32)
        cls_slc = np.array(classified_vol[z])

        pore_count  = int((cls_slc == 1).sum())
        crack_count = int((cls_slc == 2).sum())
        defect_count = pore_count + crack_count

        in_range     = (arr >= min_val) & (arr <= max_val)
        matrix_count = int((in_range & (cls_slc == 0)).sum())

        total_matrix += matrix_count
        total_pore   += pore_count
        total_crack  += crack_count

        denom = (defect_count + matrix_count) or float("nan")
        per_z.append(slice_start + z)
        per_pore.append(round(pore_count   / denom * 100, 4))
        per_crack.append(round(crack_count  / denom * 100, 4))
        per_total.append(round(defect_count / denom * 100, 4))

        if progress_cb:
            progress_cb(z + 1, n_slices)

    total_defect = total_pore + total_crack
    denom_t = (total_defect + total_matrix) or float("nan")

    return {
        "pore_yield":    round(total_pore   / denom_t * 100, 4),
        "crack_yield":   round(total_crack  / denom_t * 100, 4),
        "total_yield":   round(total_defect / denom_t * 100, 4),
        "matrix_voxels": total_matrix,
        "pore_voxels":   total_pore,
        "crack_voxels":  total_crack,
        "per_slice": {
            "z":     per_z,
            "pore":  per_pore,
            "crack": per_crack,
            "total": per_total,
        },
    }
