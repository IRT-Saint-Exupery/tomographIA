"""HDF5-backed patch extraction and dataset for segmentation training."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch.utils.data as torch_utils

from ._io import (
    apply_normalization_by_name,
    get_tif_files,
    get_tif_labels,
    get_volume_as_numpy_array,
)

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def _patch_starts(size: int, patch: int, stride: int) -> List[int]:
    """Patch start positions covering ``[0, size)``; the last patch is shifted
    back to end at the border (no padding, no mirrored content)."""
    if size < patch:
        raise ValueError(f"Volume dimension {size} is smaller than the patch dimension {patch}")
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _save_patch(patch: np.ndarray, path: str, name: str = "data") -> None:
    import h5py  # only needed when writing patches

    with h5py.File(path, "w") as f:
        f.create_dataset(name, data=patch)


class PatchGenerator:
    """Extract HDF5 patches from TIF-stack volumes and split them into
    train / val / test.

    Parameters
    ----------
    directories : dict
        ``{volume_name: (input_dir, label_dir)}``.
    patch_shape : (D, H, W)
        3-D patch size in voxels, e.g. ``(64, 64, 64)``.
    patch_directory : str
        Directory where the ``.h5`` patches and the ``splits.pkl`` cache are
        written.
    seed : int
        RNG seed of the per-patch split.
    normalization_name : str
        Volume-level normalisation applied before extraction (one of
        :data:`tomographia.segmentation.NORMALIZATIONS`).
    prefix : str
        Label file prefix (see :func:`get_tif_labels`).
    overwrite : bool
        Re-extract even if a cache with the same settings exists.
    stride : tuple of int, optional
        Patch stride; ``None`` = ``patch_shape`` (non-overlapping).  Along
        each axis the last patch is aligned with the volume border.
    split_volumes : dict, optional
        ``{volume_name: "train" | "val" | "test"}``.  Assigns whole volumes to
        splits, so test metrics measure generalisation to unseen volumes.
        If ``None``, patches are assigned at random with probabilities
        ``ratios``: neighbouring patches of the same volume then fall into
        different splits and validation / test metrics are optimistic.
    ratios : (float, float, float)
        Train / val / test probabilities of the per-patch split.

    Notes
    -----
    The patches and ``splits.pkl`` are reused on later runs with the same
    settings; changing any setting (or ``overwrite=True``) re-extracts them.
    """

    def __init__(
        self,
        directories: Dict[str, Tuple[str, str]],
        patch_shape: Tuple[int, int, int],
        patch_directory: str,
        seed: int = 42,
        normalization_name: str = "no_norm",
        prefix: str = "slice",
        overwrite: bool = False,
        stride=None,
        split_volumes: Optional[Dict[str, str]] = None,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> None:
        self.directories = directories
        self.patch_shape = tuple(int(p) for p in patch_shape)
        self.stride = self.patch_shape if stride is None else tuple(int(s) for s in stride)
        self.patch_directory = patch_directory
        self.seed = seed
        self.normalization_name = normalization_name
        self.prefix = prefix
        self.split_volumes = dict(split_volumes) if split_volumes is not None else None
        self.ratios = tuple(float(r) for r in ratios)

        if self.split_volumes is not None:
            missing = set(directories) - set(self.split_volumes)
            bad = {v for v in self.split_volumes.values() if v not in SPLITS}
            if missing or bad:
                raise ValueError(f"split_volumes must map every volume to one of {SPLITS}; "
                                 f"missing {sorted(missing)}, invalid {sorted(bad)}")
        if len(self.ratios) != 3 or abs(sum(self.ratios) - 1.0) > 1e-6:
            raise ValueError("ratios must be three numbers summing to 1")

        config = {
            "directories": {k: tuple(map(os.path.abspath, v)) for k, v in directories.items()},
            "patch_shape": self.patch_shape, "stride": self.stride, "seed": seed,
            "normalization": normalization_name, "prefix": prefix,
            "split_volumes": self.split_volumes, "ratios": self.ratios,
        }
        splits_path = os.path.join(patch_directory, "splits.pkl")
        cached = None
        if os.path.exists(splits_path) and not overwrite:
            cached = pd.read_pickle(splits_path)
            if not (isinstance(cached, dict) and cached.get("config") == config):
                logger.info("Patch cache settings changed: re-extracting patches")
                cached = None
        if cached is not None:
            self.splits: Dict[str, pd.DataFrame] = cached["splits"]
        else:
            self.splits = self._generate_samples()
            os.makedirs(patch_directory, exist_ok=True)
            pd.to_pickle({"config": config, "splits": self.splits}, splits_path)

    def _extract(self, volume: np.ndarray, name: str, kind: str) -> List[str]:
        os.makedirs(self.patch_directory, exist_ok=True)
        pz, py, px = self.patch_shape
        paths: List[str] = []
        for z in _patch_starts(volume.shape[0], pz, self.stride[0]):
            for y in _patch_starts(volume.shape[1], py, self.stride[1]):
                for x in _patch_starts(volume.shape[2], px, self.stride[2]):
                    path = os.path.join(self.patch_directory, f"{name}_patch_{kind}_{z}_{y}_{x}.h5")
                    _save_patch(volume[z:z + pz, y:y + py, x:x + px], path)
                    paths.append(path)
        return paths

    def _create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if self.split_volumes is not None:
            df["split"] = df["volume"].map(self.split_volumes)
        else:
            rng = np.random.RandomState(self.seed)
            df["split"] = rng.choice(list(SPLITS), size=len(df), p=list(self.ratios))
        return {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}

    def _generate_samples(self) -> Dict[str, pd.DataFrame]:
        inputs: List[str] = []
        labels: List[str] = []
        volumes: List[str] = []

        for key, (input_dir, label_dir) in self.directories.items():
            input_paths = get_tif_files(input_dir)
            input_vol = get_volume_as_numpy_array(input_paths)
            label_paths = get_tif_labels(label_dir, input_paths, self.prefix)
            label_vol = get_volume_as_numpy_array(label_paths)

            if input_vol.shape != label_vol.shape:
                raise ValueError(
                    f"Volume '{key}': input shape {input_vol.shape} != "
                    f"label shape {label_vol.shape}"
                )
            input_vol = apply_normalization_by_name(self.normalization_name, input_vol)
            label_vol = (label_vol != 0).astype(np.uint8)

            input_patches = self._extract(input_vol, key, "raw")
            label_patches = self._extract(label_vol, key, "labels")
            inputs.extend(input_patches)
            labels.extend(label_patches)
            volumes.extend([key] * len(input_patches))

        df = pd.DataFrame({"inputs": inputs, "labels": labels, "volume": volumes})
        return self._create_splits(df)

    def get_splits(self) -> Dict[str, pd.DataFrame]:
        """Return the ``{"train": df, "val": df, "test": df}`` split dict."""
        return self.splits


class SegmentationDataset(torch_utils.Dataset):
    """PyTorch dataset backed by the HDF5 patches written by :class:`PatchGenerator`.

    Parameters
    ----------
    dataframe : pd.DataFrame
        One row per patch with columns ``"inputs"`` and ``"labels"`` (paths to
        ``.h5`` files) and ``"volume"`` (volume name), e.g.
        ``PatchGenerator.get_splits()["train"]``.

    Items are ``(x, y, vol_id)``: float32 arrays of shape ``patch_shape`` and
    the source volume name.  ``h5py`` is only needed when items are read.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        super().__init__()
        self.dataframe = dataframe

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def _load_patch(path: str, name: str = "data") -> np.ndarray:
        import h5py

        with h5py.File(path, "r") as f:
            if name in f:
                return np.array(f[name])
            raise KeyError(f"Dataset '{name}' not found in {path}")

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
        row = self.dataframe.iloc[index]
        x = self._load_patch(row["inputs"]).astype(np.float32)
        y = self._load_patch(row["labels"]).astype(np.float32)
        return x, y, str(row["volume"])
