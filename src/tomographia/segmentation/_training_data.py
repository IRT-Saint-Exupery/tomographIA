"""
HDF5-backed patch extractor and dataset for segmentation training.

Ported from:
  ``tomographIA.data_processing.patchers.Patcher``       (L11–153)
  ``tomographIA.data_processing.dataset.PatchGenerator`` (L17–131)
  ``tomographIA.data_processing.dataset.Dataset``        (L135–161)

Porting changes vs. source
--------------------------
* ``h5py`` imported lazily inside the methods that use it (was module-level
  in both source files — would crash on import if h5py is not installed).
* ``cv2`` dependency removed entirely — only used by the classification-specific
  ``PathGenerator_class.treat_image`` which is out of scope here.
* ``tomo_utils`` / ``tomo_patchers`` replaced with the local ``_io`` helpers.
* ``artifacts_directory`` / ``tomo_analysis.create_pixel_histogram`` removed
  (optional diagnostic artifact, not part of the public segmentation API).
* ``np.random.seed`` (global state) replaced with ``np.random.RandomState``
  (instance-local RNG) for reproducible, side-effect-free splits.
* Class renamed ``SegmentationDataset`` (was ``Dataset``) to be unambiguous
  alongside the inference-side ``SingleVolumeBig`` / ``PatchedVolumeBig``.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_utils

from ._io import (
    apply_normalization_by_name,
    get_tif_files,
    get_tif_labels,
    get_volume_as_numpy_array,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal patch writer
# (ported from tomographIA.data_processing.patchers.Patcher L11–110)
# ─────────────────────────────────────────────────────────────────────────────

class _Patcher:
    """Extract 3-D patches from a volume and write them as HDF5 files.

    Source: ``tomographIA/data_processing/patchers.py`` ``Patcher`` L11–110.

    Parameters
    ----------
    patch_shape : (D, H, W)
        Patch dimensions.
    padding : int
        Symmetric reflect-padding applied to the volume before extraction.
        Use ``0`` for non-overlapping patches.
    patch_directory : str
        Directory where ``.h5`` patch files are written.
    overwrite : bool
        If ``True``, re-extract even if the file already exists.
    stride : array-like of int, optional
        Extraction stride.  ``None`` → ``patch_shape - 2*padding`` per axis.
    """

    def __init__(
        self,
        patch_shape: Tuple[int, ...],
        padding: int,
        patch_directory: str,
        overwrite: bool = False,
        stride=None,
    ) -> None:
        self.patch_shape = tuple(patch_shape)
        self.padding = padding
        self.patch_directory = patch_directory
        self.overwrite = overwrite
        if stride is None:
            # Source: Patcher.compute_stride L34-36
            self.stride = np.array(self.patch_shape) - np.array((2 * self.padding,) * 3)
        else:
            self.stride = np.array(stride)
        self._padded_shape: Optional[Tuple[int, ...]] = None

    # -- padding helpers (source: Patcher.apply_hard_padding L38-53) ----------

    def _apply_hard_padding(self, volume: np.ndarray) -> np.ndarray:
        if isinstance(self.padding, list):
            final_padding = self.padding
        elif isinstance(self.padding, int):
            final_padding = [(self.padding, self.padding)] * volume.ndim
        else:
            raise ValueError(
                f"padding must be int or list of tuples, got {type(self.padding)}"
            )
        return np.pad(volume, final_padding, mode="reflect")

    def _apply_padding_with_stride(self, volume: np.ndarray) -> np.ndarray:
        # Source: Patcher.apply_padding_with_stride L55-63
        padded_shape = []
        for dim, patch_dim, stride_dim in zip(
            volume.shape, self.patch_shape, self.stride
        ):
            remainder = int(dim) % int(stride_dim)
            padding_needed = int(patch_dim) - remainder
            padded_shape.append((0, padding_needed))
        return np.pad(volume, padded_shape, mode="reflect")

    # -- HDF5 write (source: Patcher.save_patch L66-69) -----------------------

    @staticmethod
    def _save_patch(patch: np.ndarray, path: str, name: str = "data") -> None:
        import h5py  # lazy: only needed when actually writing patches

        with h5py.File(path, "w") as f:
            f.create_dataset(name, data=patch)

    # -- inner extraction loop (source: Patcher._generate_overlapping_patches L71-97)

    def _extract_patches(
        self, volume: np.ndarray, name: str, prefix: str
    ) -> List[str]:
        # Loop bounds come from the HARD-padded shape (self._padded_shape), not
        # from the stride-padded volume passed in.  The source stores this as
        # self.constant_padded_volume_shape (patchers.py L105) and uses it as
        # the range upper bound (L78-84).  Using volume.shape instead would
        # produce one extra iteration per axis that is exactly divisible by the
        # stride — those patches have the correct shape and pass the guard,
        # generating a spurious extra patch from the stride-padding region.
        if self._padded_shape is None:
            raise RuntimeError(
                "_extract_patches called before generate_overlapping_patches "
                "set self._padded_shape"
            )
        os.makedirs(self.patch_directory, exist_ok=True)
        paths: List[str] = []
        sz, sy, sx = (int(s) for s in self.stride)
        pz, py, px = self.patch_shape
        depth, height, width = self._padded_shape

        for z in range(0, depth, sz):
            for y in range(0, height, sy):
                for x in range(0, width, sx):
                    path = os.path.join(
                        self.patch_directory,
                        f"{name}_patch_{prefix}_{z}_{y}_{x}.h5",
                    )
                    if os.path.exists(path) and not self.overwrite:
                        paths.append(path)
                    else:
                        patch = volume[z : z + pz, y : y + py, x : x + px]
                        if patch.shape == self.patch_shape:
                            self._save_patch(patch, path)
                            paths.append(path)
        return paths

    # -- public entry point (source: Patcher.generate_overlapping_patches L99-110)

    def generate_overlapping_patches(
        self, volume: np.ndarray, name: str, prefix: str
    ) -> List[str]:
        """Apply padding, then extract and save all patches.

        Returns
        -------
        List[str]
            Absolute paths to the saved ``.h5`` files, in Z-Y-X scan order.
        """
        padded = self._apply_hard_padding(volume)
        self._padded_shape = padded.shape
        padded = self._apply_padding_with_stride(padded)
        return self._extract_patches(padded, name, prefix)


# ─────────────────────────────────────────────────────────────────────────────
# Patch generator
# (ported from tomographIA.data_processing.dataset.PatchGenerator L17–131)
# ─────────────────────────────────────────────────────────────────────────────

class PatchGenerator:
    """Extract HDF5 patches from TIF-stack volumes and create 80/10/10 splits.

    Source: ``tomographIA/data_processing/dataset.py`` ``PatchGenerator`` L17–131.

    Parameters
    ----------
    directories : dict
        ``{volume_name: (input_dir, label_dir)}``.  Multiple volumes are
        processed in order; their patches are pooled before splitting.
    patch_shape : (D, H, W)
        3-D patch size in voxels.  Paper sweep anchor: ``(64, 64, 64)``.
    patch_directory : str
        Directory where ``.h5`` patch files and the ``splits.pkl`` cache are
        written.
    seed : int
        RNG seed for the per-patch train/val/test split.  Anchor: ``999``.
    normalization_name : str
        Volume-level normalisation applied before extraction.
        One of ``"robust_minmax"`` *(anchor)*, ``"minmax"``,
        ``"standardization"``, ``"basic"``, ``"no_norm"``.
    prefix : str
        Filename prefix used by :func:`get_tif_labels` to locate label TIFs.
        Anchor: ``"slice"``.
    overwrite : bool
        Re-extract patches even if ``splits.pkl`` already exists.
    stride : tuple of int, optional
        Patch stride.  ``None`` → non-overlapping (stride = patch_shape).
        Anchor: ``None`` (script passes ``stride=opt.patch_shape``, which with
        ``padding=0`` gives the same result).

    Notes
    -----
    On the first run, all patches are extracted to disk and a ``splits.pkl``
    is written.  Subsequent runs with the same *patch_directory* load the
    cached splits and skip re-extraction unless ``overwrite=True``.
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
    ) -> None:
        self.directories = directories
        self.patch_shape = tuple(patch_shape)
        self.patch_directory = patch_directory
        self.seed = seed
        self.normalization_name = normalization_name
        self._split_names = ["train", "val", "test"]

        self.patcher = _Patcher(
            patch_shape=patch_shape,
            padding=0,
            patch_directory=patch_directory,
            overwrite=overwrite,
            stride=stride,
        )

        splits_path = os.path.join(patch_directory, "splits.pkl")
        if os.path.exists(splits_path) and not overwrite:
            self.splits: Dict[str, pd.DataFrame] = pd.read_pickle(splits_path)
        else:
            self.splits = self._generate_samples(prefix=prefix)
            os.makedirs(patch_directory, exist_ok=True)
            pd.to_pickle(self.splits, splits_path)

    # -- internals (source: PatchGenerator.create_dataframe L57-73) -----------

    def _create_splits(
        self,
        inputs: List[str],
        labels: List[str],
        volumes: List[str],
    ) -> Dict[str, pd.DataFrame]:
        df = pd.DataFrame(
            {"inputs": inputs, "labels": labels, "volume": volumes}
        )
        rng = np.random.RandomState(self.seed)
        df["split"] = rng.choice(
            self._split_names, size=len(df), p=[0.8, 0.1, 0.1]
        )
        return {
            s: df[df["split"] == s].reset_index(drop=True)
            for s in self._split_names
        }

    def _generate_samples(
        self, prefix: str = "slice"
    ) -> Dict[str, pd.DataFrame]:
        # Source: PatchGenerator.generate_samples L75-110
        inputs: List[str] = []
        labels: List[str] = []
        volumes: List[str] = []

        for key, (input_dir, label_dir) in self.directories.items():
            input_paths = get_tif_files(input_dir)
            input_vol = get_volume_as_numpy_array(input_paths)
            label_paths = get_tif_labels(label_dir, input_paths, prefix)
            label_vol = get_volume_as_numpy_array(label_paths)

            input_vol = apply_normalization_by_name(self.normalization_name, input_vol)
            label_vol = (label_vol != 0).astype(np.int64)

            if input_vol.shape != label_vol.shape:
                raise ValueError(
                    f"Volume '{key}': input shape {input_vol.shape} != "
                    f"label shape {label_vol.shape}"
                )

            input_patches = self.patcher.generate_overlapping_patches(
                input_vol, key, "raw"
            )
            label_patches = self.patcher.generate_overlapping_patches(
                label_vol, key, "labels"
            )

            inputs.extend(input_patches)
            labels.extend(label_patches)
            volumes.extend([key] * len(input_patches))

        return self._create_splits(inputs, labels, volumes)

    def get_splits(self) -> Dict[str, pd.DataFrame]:
        """Return the ``{"train": df, "val": df, "test": df}`` split dict."""
        return self.splits


# ─────────────────────────────────────────────────────────────────────────────
# HDF5-backed training dataset
# (ported from tomographIA.data_processing.dataset.Dataset L135–161)
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDataset(torch_utils.Dataset):
    """PyTorch dataset backed by HDF5 patch files written by :class:`PatchGenerator`.

    Source: ``tomographIA/data_processing/dataset.py`` ``Dataset`` L135–161.

    Parameters
    ----------
    dataframe : pd.DataFrame
        One row per patch.  Required columns: ``"inputs"`` (path to input
        ``.h5``), ``"labels"`` (path to label ``.h5``), ``"volume"``
        (volume name string).  Obtained from ``PatchGenerator.get_splits()``.

    Returns (per item)
    ------------------
    ``(x, y, vol_id)`` where *x* and *y* are float32 numpy arrays of shape
    ``patch_shape``, and *vol_id* is a string identifying the source volume.

    Notes
    -----
    ``h5py`` is imported lazily inside :meth:`_load_patch`; installing h5py
    is only required when the dataset is actually iterated, not on import.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        super().__init__()
        self.dataframe = dataframe

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def _load_patch(path: str, name: str = "data") -> np.ndarray:
        # Source: Dataset.load_patch L143-149. h5py imported lazily.
        import h5py  # lazy: only needed when iterating the dataset

        with h5py.File(path, "r") as f:
            if name in f:
                return np.array(f[name])
            raise KeyError(f"Dataset '{name}' not found in {path}")

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
        # Source: Dataset.__getitem__ L151-161
        row = self.dataframe.iloc[index]
        x = self._load_patch(row["inputs"]).astype(np.float32)
        y = self._load_patch(row["labels"]).astype(np.float32)
        return x, y, str(row["volume"])
