"""Slice-stack I/O and volume normalisation for 3-D segmentation."""
from __future__ import annotations

import os
import re
import warnings
from typing import List

import numpy as np
from PIL import Image

from .._slices import list_slices, read_slice, read_volume

#: Normalisations accepted by :func:`apply_normalization_by_name`.
NORMALIZATIONS = ("robust_minmax", "minmax", "standardization", "basic", "no_norm")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def get_tif_files(directory: str) -> List[str]:
    """``.tif`` / ``.tiff`` slices of *directory* in natural order."""
    files = list_slices(directory)
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff slices found in {directory}")
    return files


def get_tif_labels(directory: str, paths: List[str], prefix: str = "slice") -> List[str]:
    """Label file of each input slice, matched on the slice number.

    The slice number is the last integer of the file name, compared as a
    number: ``raw/scan_1.tif`` and ``raw/scan_0001_rec.tif`` both map to
    ``labels/<prefix>_0001.tif`` (or ``<prefix>_1.tif``; ``.tif`` / ``.tiff``).

    Raises
    ------
    ValueError        : a name has no number, or two slices map to one label.
    FileNotFoundError : a label file is missing.
    """
    index = {}
    for name in os.listdir(directory):
        stem, ext = os.path.splitext(name)
        if ext.lower() in (".tif", ".tiff") and stem.startswith(prefix + "_"):
            nums = re.findall(r"\d+", stem[len(prefix) + 1:])
            if nums:
                index.setdefault(int(nums[-1]), os.path.join(directory, name))
    out = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        nums = re.findall(r"\d+", stem)
        if not nums:
            raise ValueError(f"No slice number in file name {os.path.basename(path)!r}")
        hit = index.get(int(nums[-1]))
        if hit is None:
            raise FileNotFoundError(f"No label file {prefix}_<{int(nums[-1])}>.tif in {directory}")
        out.append(hit)
    if len(set(out)) != len(out):
        raise ValueError("Several input slices map to the same label file; slice numbers "
                         "must be unique")
    return out


def get_image_as_numpy_array(path: str) -> np.ndarray:
    return read_slice(path)


def get_volume_as_numpy_array(paths: List[str]) -> np.ndarray:
    """Stack slices into a ``(Z, Y, X)`` array."""
    return read_volume(paths)


# ---------------------------------------------------------------------------
# Normalisation (all statistics are computed on the whole volume)
# ---------------------------------------------------------------------------

def basic_normalization_3d(data: np.ndarray) -> np.ndarray:
    """Fixed 16-bit scaling ``x / 65536 · 2 − 1`` (assumes 16-bit input)."""
    if np.asarray(data).dtype != np.uint16:
        warnings.warn(f"'basic' normalisation assumes uint16 data, got {np.asarray(data).dtype}",
                      stacklevel=3)
    arr = np.asarray(data, dtype=np.float32)
    return arr / (2 ** 16) * 2 - 1


def minmax_normalization_3d(data: np.ndarray) -> np.ndarray:
    """Scale the volume to ``[-1, 1]`` with its global minimum and maximum."""
    arr = np.asarray(data, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (1e-7 + hi - lo) * 2 - 1


def standardization_3d(data: np.ndarray) -> np.ndarray:
    """Standardise the volume to zero mean and unit variance."""
    arr = np.asarray(data, dtype=np.float32)
    return (arr - arr.mean()) / (arr.std() + 1e-7)


def robust_minmax_normalization_3d(
    data: np.ndarray,
    lower_p: float = 0.5,
    upper_p: float = 99.5,
) -> np.ndarray:
    """Clip to the volume's ``[lower_p, upper_p]`` percentiles, then scale to ``[-1, 1]``."""
    arr = np.asarray(data, dtype=np.float32)
    lo = float(np.percentile(arr, lower_p))
    hi = float(np.percentile(arr, upper_p))
    return (np.clip(arr, lo, hi) - lo) / (1e-7 + hi - lo) * 2 - 1


def apply_normalization_by_name(name: str, input_volume: np.ndarray) -> np.ndarray:
    """Apply one of :data:`NORMALIZATIONS` to a volume.

    Raises
    ------
    ValueError : unknown *name*.
    """
    if name == "minmax":
        return minmax_normalization_3d(input_volume)
    if name == "standardization":
        return standardization_3d(input_volume)
    if name == "basic":
        return basic_normalization_3d(input_volume)
    if name == "robust_minmax":
        return robust_minmax_normalization_3d(input_volume)
    if name == "no_norm":
        return input_volume
    raise ValueError(f"Unknown normalization {name!r}; choose one of {NORMALIZATIONS}")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_input_as_png_image(image: np.ndarray, path: str) -> None:
    mn, mx = float(image.min()), float(image.max())
    img = ((image - mn) / (mx - mn + 1e-12) * 255).astype(np.uint8)
    Image.fromarray(img).save(path, format="PNG")


def save_mask_as_png_image(image: np.ndarray, path: str) -> None:
    Image.fromarray((np.uint8(image != 0) * 255)).save(path, format="PNG")


def save_all_inputs(directory: str, inputs: np.ndarray) -> None:
    """Write each slice of *inputs* as ``inputs_<z>.png`` (contrast-stretched per slice)."""
    os.makedirs(directory, exist_ok=True)
    for i, slc in enumerate(inputs):
        save_input_as_png_image(slc, os.path.join(directory, f"inputs_{i}.png"))


def save_all_labels(directory: str, labels: np.ndarray, prefix: str = "labels",
                    save_np: bool = True) -> None:
    """Write each slice of a binary volume as ``<prefix>_<z>.png`` and the volume
    as ``labeled_volume.npy``."""
    os.makedirs(directory, exist_ok=True)
    for i, slc in enumerate(labels):
        save_mask_as_png_image(slc, os.path.join(directory, f"{prefix}_{i}.png"))
    if save_np:
        np.save(os.path.join(directory, "labeled_volume.npy"), labels)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Boolean *mask* without the 26-connected components smaller than *min_size* voxels."""
    from scipy.ndimage import label

    mask = np.asarray(mask, dtype=bool)
    if min_size <= 1:
        return mask
    lab, _ = label(mask, structure=np.ones((3,) * mask.ndim))
    keep = np.bincount(lab.ravel()) >= min_size
    keep[0] = False
    return keep[lab]
