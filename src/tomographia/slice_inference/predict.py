"""
Whole-slice and whole-volume inference with a 2-D patch classifier or a
prototype network.

Slices are tiled with ``unfold``, normalised exactly as during training
(:func:`tomographia.patches.normalize_patch`) and predicted in batches.
Results are :class:`SlicePrediction` grids of probabilities;
:func:`predictions_to_volume` stacks them into a 3-D patch grid.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from .._slices import list_slices as _list_slices
from ..patches.dataset import load_patch, normalize_patch

SLICE_EXTENSIONS = (".tif", ".tiff", ".png")


@dataclass
class SlicePrediction:
    """
    Patch-grid prediction for one slice.

    Attributes
    ----------
    prob      : ``[n_rows, n_cols]`` P(defect) per patch.
    rows      : top-left row (y) of each grid row.
    cols      : top-left column (x) of each grid column.
    patch_size: patch side in pixels.
    """
    prob: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    patch_size: int

    def binary(self, threshold: float = 0.5) -> np.ndarray:
        return (self.prob >= threshold).astype(np.uint8)


def tile_slice(image: np.ndarray, patch_size: int = 64, stride: Optional[int] = None,
               max_pixel_value: float = 255.0):
    """
    Cut a slice into (possibly overlapping) square patches.

    Returns
    -------
    ``(patches [N, 1, p, p] normalised tensor, rows, cols)`` where the grid is
    ``len(rows) × len(cols)`` in row-major order.  Border pixels that do not
    fill a whole patch are ignored.
    """
    stride = stride or patch_size
    t = torch.as_tensor(np.asarray(image, dtype=np.float32))
    h, w = t.shape
    if h < patch_size or w < patch_size:
        return torch.empty(0, 1, patch_size, patch_size), np.array([], int), np.array([], int)
    tiles = t.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
    rows = np.arange(tiles.shape[0]) * stride
    cols = np.arange(tiles.shape[1]) * stride
    patches = tiles.reshape(-1, 1, patch_size, patch_size)
    return normalize_patch(patches, max_pixel_value), rows, cols


@torch.no_grad()
def predict_patches(model: nn.Module, patches: torch.Tensor, device: str = "cpu",
                    batch_size: int = 512, temperature: float = 1.0) -> np.ndarray:
    """
    P(defect) for a batch of normalised patches.

    Handles single-logit classifiers (``sigmoid(z / T)``), multi-logit
    classifiers (``softmax(z / T)``, class 1) and two-class prototype networks
    (``sigmoid(binary_logit / T)``, identical to ``class_probs[:, 1]`` for
    ``T = 1``).

    temperature : the ``report["temperature"]`` of
                  :func:`tomographia.evaluation.evaluate_calibrated`.  Use it
                  whenever the threshold comes from that report, which is
                  chosen on calibrated probabilities.
    """
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    model.eval()
    out_probs = []
    for i in range(0, len(patches), batch_size):
        out = model(patches[i:i + batch_size].to(device))
        if isinstance(out, dict):
            cl = out["class_logits"]
            if cl.shape[1] == 2:
                p = torch.sigmoid((cl[:, 1] - cl[:, 0]).double() / temperature)
            else:
                p = torch.softmax(cl / temperature, dim=1)[:, 1]
        elif out.ndim == 2 and out.shape[1] > 1:
            p = torch.softmax(out / temperature, dim=1)[:, 1]
        else:
            p = torch.sigmoid(out.reshape(-1).double() / temperature)
        out_probs.append(p.double().cpu().numpy())
    return np.concatenate(out_probs) if out_probs else np.zeros(0, np.float64)


def predict_slice(model: nn.Module, image: Union[str, np.ndarray], patch_size: int = 64,
                  stride: Optional[int] = None, device: str = "cpu", batch_size: int = 512,
                  max_pixel_value: float = 255.0, temperature: float = 1.0) -> SlicePrediction:
    """Predict every patch of one slice (path or ``[H, W]`` array); see
    :func:`predict_patches` for *temperature*."""
    model.to(device)
    if isinstance(image, (str, os.PathLike)):
        image = load_patch(os.fspath(image))
    patches, rows, cols = tile_slice(image, patch_size, stride, max_pixel_value)
    prob = predict_patches(model, patches, device, batch_size, temperature)
    return SlicePrediction(prob.reshape(len(rows), len(cols)), rows, cols, patch_size)


def list_slices(folder: str) -> List[str]:
    """Slice image paths of a volume folder in natural order (``2`` before ``10``)."""
    return _list_slices(folder, SLICE_EXTENSIONS)


def predict_volume(model: nn.Module, folder: str, patch_size: int = 64,
                   stride: Optional[int] = None, device: str = "cpu",
                   start: int = 0, end: Optional[int] = None, batch_size: int = 512,
                   max_pixel_value: float = 255.0, progress: bool = False,
                   temperature: float = 1.0) -> List[SlicePrediction]:
    """Predict a folder of slices (natural file-name order), ``[start, end)``."""
    model.to(device)
    paths = list_slices(folder)[start:end]
    if not paths:
        raise FileNotFoundError(f"No slice images in {folder} for range [{start}, {end})")
    if progress:
        from tqdm.auto import tqdm
        paths = tqdm(paths, desc="slices")
    return [predict_slice(model, p, patch_size, stride, device, batch_size, max_pixel_value,
                          temperature)
            for p in paths]


def predictions_to_volume(predictions: Sequence[SlicePrediction],
                          threshold: Optional[float] = None) -> np.ndarray:
    """
    Stack slice predictions into a ``[n_slices, n_rows, n_cols]`` patch-grid
    volume (probabilities, or binary if ``threshold`` is given).
    """
    if not predictions:
        raise ValueError("No predictions to stack")
    shapes = {p.prob.shape for p in predictions}
    if len(shapes) != 1:
        raise ValueError(f"Slices have different grid shapes: {shapes}")
    vol = np.stack([p.prob for p in predictions])
    return (vol >= threshold).astype(np.uint8) if threshold is not None else vol
