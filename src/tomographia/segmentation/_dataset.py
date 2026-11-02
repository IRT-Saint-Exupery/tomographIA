"""Lazy overlapping-patch datasets for large CT volumes (inference)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as torch_utils

from ._io import (
    apply_normalization_by_name,
    get_tif_files,
    get_tif_labels,
    get_volume_as_numpy_array,
)


class PatchedVolumeBig(torch_utils.Dataset):
    """
    Lazy overlapping 3-D patcher: no disk I/O.

    Patch coordinates are computed once; slices are extracted on the fly.
    Along an axis shorter than the patch, patches are edge-padded to the patch
    size and cropped back on rebuild.

    Parameters
    ----------
    volume      : 3-D numpy array (Z, Y, X).
    patch_shape : (D, H, W) spatial patch size.
    padding     : overlap used to derive the default stride
                  (``stride = patch_shape - 2 * padding``).
    stride      : explicit (Sz, Sy, Sx). If ``None``, derived from *padding*.
    """

    def __init__(
        self,
        volume: np.ndarray,
        patch_shape: Tuple[int, int, int],
        padding: int = 3,
        stride: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        if volume.ndim != 3:
            raise ValueError("volume must be 3-D (Z, Y, X)")
        self.volume = volume
        self.volume_shape = volume.shape
        self.patch_shape = tuple(int(p) for p in patch_shape)
        self.padding = padding
        self.stride = (
            tuple(max(ps - 2 * padding, 1) for ps in self.patch_shape)
            if stride is None
            else tuple(int(s) for s in stride)
        )
        self.coords: List[Tuple[int, int, int, int, int, int]] = (
            self._compute_patch_coords()
        )

    def _compute_patch_coords(self) -> List[Tuple[int, int, int, int, int, int]]:
        Z, Y, X = self.volume_shape
        Dz, Dy, Dx = self.patch_shape
        Sz, Sy, Sx = self.stride

        def _starts(size: int, patch: int, stride: int) -> List[int]:
            if size <= patch:
                return [0]
            starts = list(range(0, size - patch + 1, stride))
            last = size - patch
            if starts[-1] != last:
                starts.append(last)
            return starts

        coords = []
        for z0 in _starts(Z, Dz, Sz):
            for y0 in _starts(Y, Dy, Sy):
                for x0 in _starts(X, Dx, Sx):
                    coords.append((z0, min(z0 + Dz, Z), y0, min(y0 + Dy, Y),
                                   x0, min(x0 + Dx, X)))
        return coords

    def rebuild_volume_from_patches(self, patches) -> np.ndarray:
        """Average overlapping patches back into the volume shape (float32)."""
        if len(patches) != len(self.coords):
            raise ValueError(
                f"Expected {len(self.coords)} patches, got {len(patches)}"
            )
        out = np.zeros(self.volume_shape, dtype=np.float32)
        cnt = np.zeros(self.volume_shape, dtype=np.uint16)
        for idx, patch in enumerate(patches):
            z0, z1, y0, y1, x0, x1 = self.coords[idx]
            out[z0:z1, y0:y1, x0:x1] += np.asarray(patch)[: z1 - z0, : y1 - y0, : x1 - x0]
            cnt[z0:z1, y0:y1, x0:x1] += 1
        out /= np.maximum(cnt, 1)
        return out

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> torch.Tensor:
        z0, z1, y0, y1, x0, x1 = self.coords[index]
        patch = self.volume[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
        pad = [(0, p - n) for p, n in zip(self.patch_shape, patch.shape)]
        if any(after for _, after in pad):
            patch = np.pad(patch, pad, mode="edge")
        return torch.from_numpy(np.ascontiguousarray(patch))


class SingleVolumeBig(torch_utils.Dataset):
    """
    Load a TIF-stack volume from disk and expose it as overlapping 3-D patches.

    Parameters
    ----------
    directories       : ``{"volume": (input_dir, label_dir)}``.
                        Pass ``""`` for *label_dir* in inference mode.
    patch_shape       : (D, H, W) spatial patch size.
    normalization_name: ``"robust_minmax"`` | ``"minmax"`` | ``"standardization"``
                        | ``"basic"`` | ``"no_norm"``.
    prefix            : filename prefix used to locate label TIFs.
    padding           : overlap for default stride computation.
    stride            : explicit stride. ``None`` → derived from *padding*.
    min_file          : first slice index (0-based) to load.
    max_file          : one-past-last slice index to load (``None`` = all).
    """

    def __init__(
        self,
        directories: Dict[str, Tuple[str, str]],
        patch_shape: Tuple[int, int, int],
        normalization_name: str = "no_norm",
        prefix: str = "slice",
        padding: int = 3,
        stride: Optional[Tuple[int, int, int]] = None,
        min_file: Optional[int] = None,
        max_file: Optional[int] = None,
    ) -> None:
        self.min_file = min_file
        self.max_file = max_file
        self.directories = directories
        self.patch_shape = patch_shape
        self.normalization_name = normalization_name
        self.prefix = prefix
        self.volume, self.labels = self._load()
        self.patched_volume = PatchedVolumeBig(
            volume=self.volume,
            patch_shape=patch_shape,
            padding=padding,
            stride=stride,
        )

    def _load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        inputs_dir, labels_dir = self.directories["volume"]
        paths = get_tif_files(inputs_dir)
        paths = paths[self.min_file or 0 : self.max_file]
        vol = get_volume_as_numpy_array(paths)
        labels: Optional[np.ndarray] = None
        if labels_dir:
            lpaths = get_tif_labels(labels_dir, paths, self.prefix)
            raw = get_volume_as_numpy_array(lpaths)
            labels = (raw != 0).astype(np.int64)
        if labels is not None and vol.shape != labels.shape:
            raise ValueError(
                f"Volume shape {vol.shape} does not match label shape {labels.shape}"
            )
        vol = apply_normalization_by_name(self.normalization_name, vol)
        return vol, labels

    def rebuild_volume_from_patches(self, patches) -> np.ndarray:
        return self.patched_volume.rebuild_volume_from_patches(patches)

    def __len__(self) -> int:
        return len(self.patched_volume)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.patched_volume[index]
