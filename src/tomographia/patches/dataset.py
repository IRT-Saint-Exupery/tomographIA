"""
2-D patch datasets: indexing, splitting, loading and augmentation.

Patch folders follow the layout::

    <folder>/0/*.tif   # patches without defect
    <folder>/1/*.tif   # patches containing a defect

Several folders can be combined (e.g. one per slicing direction xy / xz / yz).

- Files are listed in natural order, so splits are reproducible.
- ``split_patches`` splits per folder at random, or with ``group_col`` keeps
  all patches of a group (specimen, volume, ...) in the same split, which
  avoids leakage between neighbouring slices / orthogonal views.
- :class:`PatchDataset` returns ``[1, H, W]`` tensors (optionally with the
  global sample index used by the expert anchors).
- :class:`PatchTransform`: H/V flips (p=0.5), rotation ±15° (p=0.5) with
  reflected borders, normalisation ``(x / max_pixel_value - 0.5) / 0.5``.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._slices import natural_key, read_slice

try:  # torch is an optional dependency of the package
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None
    _TorchDataset = object

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png")
SPLITS = ("train", "val", "test")


# ─────────────────────────────────────────────────────────────────────────────
# Indexing and splitting
# ─────────────────────────────────────────────────────────────────────────────

def build_patch_index(
    folders: Union[str, Sequence[str]],
    group_fn: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """
    Index labelled patches stored as ``<folder>/<label>/<image>``.

    Parameters
    ----------
    folders  : one folder or a list of folders, each containing ``0/`` and ``1/``.
    group_fn : optional ``path -> group id`` callable (e.g. parse the specimen
               from the file name).  Defaults to the folder name.

    Returns
    -------
    DataFrame with columns ``path``, ``label`` (int), ``source`` (folder name)
    and ``group``; sorted, with a fresh ``RangeIndex``.
    """
    if isinstance(folders, (str, os.PathLike)):
        folders = [folders]

    rows = []
    for folder in folders:
        folder = os.fspath(folder)
        source = os.path.basename(os.path.normpath(folder))
        for label in (0, 1):
            label_dir = os.path.join(folder, str(label))
            if not os.path.isdir(label_dir):
                raise FileNotFoundError(f"Missing class folder: {label_dir}")
            for name in sorted(os.listdir(label_dir), key=natural_key):
                if not name.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                path = os.path.join(label_dir, name)
                rows.append({
                    "path": path,
                    "label": label,
                    "source": source,
                    "group": group_fn(path) if group_fn else source,
                })

    if not rows:
        raise FileNotFoundError(f"No patch images found in {list(folders)}")
    return pd.DataFrame(rows)


def split_patches(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    group_col: Optional[str] = None,
    per_source: bool = True,
) -> pd.DataFrame:
    """
    Add a ``split`` column (``"train"`` / ``"val"`` / ``"test"``).

    Parameters
    ----------
    df         : patch index from :func:`build_patch_index`.
    ratios     : train / val / test fractions.
    seed       : shuffling seed.
    group_col  : if given, whole groups are assigned to a single split
                 (recommended to avoid leakage between neighbouring patches).
                 Groups are shuffled, ordered by decreasing size, and each
                 goes to the split that is relatively furthest below its
                 share of *patches*; with at least three groups no split is
                 left empty.
    per_source : split each ``source`` folder independently (every slicing
                 direction contributes 80/10/10).  Ignored when
                 ``group_col`` is given.

    Returns
    -------
    Copy of *df* with a ``split`` column; the index is preserved, so it can be
    used as the global sample index for anchors.
    """
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    if not df.index.is_unique:
        raise ValueError("df.index must be unique (it is used as the global sample id); "
                         "call df.reset_index(drop=True) first")
    if (group_col is not None and "source" in df.columns and df["source"].nunique() > 1
            and (df[group_col].astype(str) == df["source"].astype(str)).all()):
        warnings.warn(f"'{group_col}' equals the source folder (build_patch_index was called "
                      "without group_fn): each folder, e.g. each slicing direction, becomes "
                      "one split, so the same specimens appear in several splits. Pass a "
                      "group_fn that returns the specimen id", stacklevel=2)
    out = df.copy()
    out["split"] = "train"
    out = _assign_splits(out, ratios, seed, group_col, per_source)
    if "label" in out.columns:
        for s, r in zip(SPLITS, ratios):
            labels = set(out.loc[out["split"] == s, "label"])
            if r > 0 and labels and len(labels) < 2:
                warnings.warn(f"Split '{s}' contains a single class {labels}: metrics such "
                              "as ROC-AUC are undefined on it", stacklevel=2)
    return out


def _assign_splits(out, ratios, seed, group_col, per_source):

    if group_col is not None:
        rng = np.random.default_rng(seed)
        sizes = out.groupby(group_col).size()
        groups = sizes.index.to_numpy().copy()  # may be a read-only view (pandas CoW)
        rng.shuffle(groups)
        # largest groups first (shuffle order breaks ties) -> tighter ratios
        groups = groups[np.argsort(-sizes[groups].to_numpy(), kind="stable")]
        targets = np.asarray(ratios, dtype=float) * len(out)
        filled = np.zeros(3)
        assigned = {}
        for g in groups:
            # give the group to the split that is relatively the most under-filled
            deficit = np.where(targets > 0, (targets - filled) / np.maximum(targets, 1e-12), -np.inf)
            k = int(np.argmax(deficit))
            assigned[g] = SPLITS[k]
            filled[k] += sizes[g]
        out["split"] = out[group_col].map(assigned)
        empty = [s for s, r in zip(SPLITS, ratios) if r > 0 and not (out["split"] == s).any()]
        if empty:
            warnings.warn(f"Grouped split left {empty} empty: {len(groups)} groups are too "
                          "few for these ratios", stacklevel=3)
        return out

    blocks = [out] if not per_source else [g for _, g in out.groupby("source", sort=True)]
    for block in blocks:
        order = block.sample(frac=1, random_state=seed).index
        n = len(order)
        n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
        n_test = int(n * ratios[2])
        labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        labels += ["train"] * (n - len(labels))  # rounding remainder goes to train
        out.loc[order, "split"] = labels
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

_RANGE_WARNED = False


def normalize_patch(x, max_pixel_value: float = 255.0):
    """
    Map raw intensities to ``(x / max_pixel_value - 0.5) / 0.5`` (≈ [-1, 1]).

    Warns (once) if values exceed ``max_pixel_value`` — e.g. 16-bit data
    normalised with the 8-bit default; pass ``max_pixel_value=65535`` then,
    and use the same value at training and inference.
    """
    global _RANGE_WARNED
    if not _RANGE_WARNED and float(x.max()) > max_pixel_value * 1.001:
        _RANGE_WARNED = True
        warnings.warn(f"Pixel values up to {float(x.max()):.0f} exceed max_pixel_value="
                      f"{max_pixel_value:g}; inputs will fall outside [-1, 1]. Set "
                      "max_pixel_value to the bit depth of your data.", stacklevel=2)
    return (x / max_pixel_value - 0.5) / 0.5


class PatchTransform:
    """
    Normalisation and optional augmentation of a ``[1, H, W]`` float tensor.

    Parameters
    ----------
    augment         : apply random H/V flips (p=0.5 each) and a rotation
                      (p=0.5, angle ~ U[-max_rotation, max_rotation]).
    max_rotation    : maximum rotation angle in degrees.
    max_pixel_value : intensity mapped to 1.0 before centring (255 for 8-bit).
    """

    def __init__(self, augment: bool = False, max_rotation: float = 15.0,
                 max_pixel_value: float = 255.0) -> None:
        self.augment = augment
        self.max_rotation = max_rotation
        self.max_pixel_value = max_pixel_value

    def _rotate(self, x: "torch.Tensor", angle: float) -> "torch.Tensor":
        from torchvision.transforms.functional import InterpolationMode, rotate
        _, h, w = x.shape
        # reflect-pad so the rotated corners are filled with image content
        pad = int(math.ceil(max(h, w) * (math.sqrt(2) - 1) / 2)) + 1
        pad = min(pad, h - 1, w - 1)
        xp = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
        xr = rotate(xp, angle, interpolation=InterpolationMode.BILINEAR)
        return xr[:, pad:pad + h, pad:pad + w]

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        if self.augment:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-1])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-2])
            if torch.rand(()) < 0.5:
                angle = float(torch.empty(()).uniform_(-self.max_rotation, self.max_rotation))
                x = self._rotate(x, angle)
        return normalize_patch(x, self.max_pixel_value)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_patch(path: str) -> np.ndarray:
    """Read a single-channel patch image as ``float32`` ``[H, W]``."""
    arr = read_slice(os.fspath(path))
    if arr.ndim == 3:  # RGB(A) saved patches: keep the first channel
        arr = arr[..., 0]
    return arr.astype(np.float32)


class PatchDataset(_TorchDataset):
    """
    Torch dataset over a patch DataFrame (columns ``path`` and ``label``).

    Parameters
    ----------
    df           : patch table, typically ``splits[splits.split == "train"]``.
    transform    : callable on a ``[1, H, W]`` tensor; defaults to
                   ``PatchTransform(augment=False)`` (normalisation only).
    return_index : also return the DataFrame index of each sample (needed to
                   look up expert anchors during prototype training).

    Items are ``(x, y)`` or ``(x, y, index)`` with ``x`` of shape ``[1, H, W]``
    (float32) and ``y`` a ``long`` scalar.
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None,
                 return_index: bool = False) -> None:
        if torch is None:
            raise ImportError("PatchDataset requires torch: pip install 'tomographia[vision]'")
        self.df = df
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).to_numpy()
        self.indices = df.index.to_numpy()
        self.transform = transform if transform is not None else PatchTransform()
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x = torch.from_numpy(load_patch(self.paths[i])).unsqueeze(0)
        x = self.transform(x)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        if self.return_index:
            return x, y, int(self.indices[i])
        return x, y


def class_counts(labels: Iterable[int]) -> Dict[int, int]:
    """Return ``{0: n0, 1: n1}`` for a binary label sequence."""
    labels = np.asarray(list(labels), dtype=int)
    return {0: int((labels == 0).sum()), 1: int((labels == 1).sum())}
