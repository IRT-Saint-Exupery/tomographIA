"""
Intensity-window yield computation.

Extracted from ``launch_report()`` in ``tomo_app/backend/main.py``.

Yield definition
----------------
A voxel is *matrix* if:

  1. Its grey value lies within the window
     ``[ref_value * (1 - lower),  ref_value * (1 + upper)]``.
  2. It is **not** classified as a defect (``classified_vol == 0``).

Overall yield per slice:

  yield = defect_voxels / (defect_voxels + matrix_voxels)

Note: this is a defect-density metric (higher = more defects), not a
conventional "yield" (fraction of good material).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np


def _sorted_tifs(directory: Union[str, Path]) -> List[str]:
    """Return .tif/.tiff paths from *directory* sorted by slice index."""
    def _key(name: str) -> int:
        nums = re.findall(r"(\d+)", os.path.splitext(name)[0])
        return int(nums[-1]) if nums else 0

    p = Path(directory)
    return [
        str(f)
        for f in sorted(p.iterdir(), key=lambda f: _key(f.name))
        if f.suffix.lower() in (".tif", ".tiff")
    ]


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
    tif_dir        : directory of raw grayscale TIF slices.
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
    from PIL import Image

    tifs = _sorted_tifs(tif_dir)
    nz = classified_vol.shape[0]
    min_val = ref_value * (1.0 - lower)
    max_val = ref_value * (1.0 + upper)
    n_slices = min(nz, max(0, len(tifs) - slice_start))

    per_z: list = []
    per_pore: list = []
    per_crack: list = []
    per_total: list = []
    total_matrix = total_pore = total_crack = 0

    for z in range(n_slices):
        arr = np.array(Image.open(tifs[slice_start + z]), dtype=np.float32)
        cls_slc = np.array(classified_vol[z])

        pore_count  = int((cls_slc == 1).sum())
        crack_count = int((cls_slc == 2).sum())
        defect_count = pore_count + crack_count

        in_range     = (arr >= min_val) & (arr <= max_val)
        matrix_count = int((in_range & (cls_slc == 0)).sum())

        total_matrix += matrix_count
        total_pore   += pore_count
        total_crack  += crack_count

        denom = defect_count + matrix_count or 1
        per_z.append(slice_start + z)
        per_pore.append(round(pore_count   / denom * 100, 4))
        per_crack.append(round(crack_count  / denom * 100, 4))
        per_total.append(round(defect_count / denom * 100, 4))

        if progress_cb:
            progress_cb(z + 1, n_slices)

    total_defect = total_pore + total_crack
    denom_t = total_defect + total_matrix or 1

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
