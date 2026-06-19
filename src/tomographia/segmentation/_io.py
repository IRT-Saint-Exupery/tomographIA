"""
Low-level TIF I/O and volume normalisation utilities for segmentation.

Fixed relative to the original ``tomographIA.data_processing.utils``:
- Removed self-import (was ``import tomographIA.data_processing.utils as tomo_utils``
  on line 11 of the original, causing a circular import).
- Removed torch dependency (not needed in I/O layer).
- Removed ad-hoc file-renaming helpers (``delete_brute_from_name``, etc.).
- ``apply_normalization_by_name`` now calls the local functions directly.
- Print statements removed (noisy for library use).
"""
from __future__ import annotations

import glob
import os
import re
from typing import List, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def sort_files_by_depth(file_list: List[str]) -> List[str]:
    """Sort filenames by the last integer in the base name (slice index)."""
    def _key(filename: str) -> int:
        base = os.path.splitext(os.path.basename(filename))[0]
        nums = re.findall(r"(\d+)", base)
        return int(nums[-1]) if nums else 0
    return sorted(file_list, key=_key)


def get_tif_files(directory: str) -> List[str]:
    """Return all .tif files in *directory*, sorted by depth (slice) index."""
    files = glob.glob(os.path.join(directory, "*.tif"))
    return sort_files_by_depth(files)


def get_tif_labels(
    directory: str,
    paths: List[str],
    prefix: str = "slice",
) -> List[str]:
    """Find the label counterpart for each input path using *prefix* + suffix."""
    out = []
    for path in paths:
        filename = prefix + "_" + os.path.basename(path).split("_")[-1]
        label_path = os.path.join(directory, filename)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        out.append(label_path)
    return out


# ---------------------------------------------------------------------------
# Array loading
# ---------------------------------------------------------------------------

def get_image_as_numpy_array(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def get_volume_as_numpy_array(paths: List[str]) -> np.ndarray:
    """Stack TIF slices into a (Z, Y, X) array."""
    return np.stack([get_image_as_numpy_array(p) for p in paths], axis=0)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def basic_normalization_3d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    return arr / (2 ** 16) * 2 - 1


def minmax_normalization_3d(data: np.ndarray) -> np.ndarray:
    out = []
    for matrix in data:
        lo, hi = float(matrix.min()), float(matrix.max())
        out.append((matrix - lo) / (1e-7 + hi - lo) * 2 - 1)
    return np.array(out, dtype=np.float32)


def robust_normalization_3d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    return (arr - arr.mean()) / (arr.std() + 1e-7)


def robust_minmax_normalization_3d(
    data: np.ndarray,
    lower_p: float = 0.5,
    upper_p: float = 99.5,
) -> np.ndarray:
    """
    Clip to percentile range then scale to [-1, 1].
    Computes percentiles on the whole 3-D volume for consistent contrast.
    """
    arr = np.asarray(data, dtype=np.float32)
    lo = float(np.percentile(arr, lower_p))
    hi = float(np.percentile(arr, upper_p))
    return (np.clip(arr, lo, hi) - lo) / (1e-7 + hi - lo) * 2 - 1


def apply_normalization_by_name(
    name: str, input_volume: np.ndarray
) -> np.ndarray:
    if name == "minmax":
        return minmax_normalization_3d(input_volume)
    if name == "standardization":
        return robust_normalization_3d(input_volume)
    if name == "basic":
        return basic_normalization_3d(input_volume)
    if name == "robust_minmax":
        return robust_minmax_normalization_3d(input_volume)
    return input_volume  # "no_norm" or unrecognised


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_input_as_png_image(image: np.ndarray, path: str) -> None:
    mn, mx = float(image.min()), float(image.max())
    img = ((image - mn) / (mx - mn + 1e-12) * 255).astype(np.uint8)
    Image.fromarray(img).save(path, format="PNG")


def save_mask_as_png_image(image: np.ndarray, path: str) -> None:
    Image.fromarray((np.uint8(image != 0) * 255)).save(path, format="PNG")


def save_all_inputs(
    directory: str,
    inputs: np.ndarray,
    image_format: str = "png",
) -> None:
    os.makedirs(directory, exist_ok=True)
    fmt = image_format.lower()
    for i, slc in enumerate(inputs):
        path = os.path.join(directory, f"inputs_{i}.{fmt}")
        if fmt == "png":
            save_input_as_png_image(slc, path)
        else:
            save_input_as_png_image(slc, os.path.join(directory, f"inputs_{i}.png"))


def save_all_labels(
    directory: str,
    labels: np.ndarray,
    image_format: str = "png",
    prefix: str = "labels",
    save_np: bool = True,
) -> None:
    os.makedirs(directory, exist_ok=True)
    fmt = image_format.lower()
    for i, slc in enumerate(labels):
        path = os.path.join(directory, f"{prefix}_{i}.{fmt}")
        if fmt == "png":
            save_mask_as_png_image(slc, path)
        else:
            save_mask_as_png_image(slc, os.path.join(directory, f"{prefix}_{i}.png"))
    if save_np:
        np.save(os.path.join(directory, "labeled_volume.npy"), labels)
