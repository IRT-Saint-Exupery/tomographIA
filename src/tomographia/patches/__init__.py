"""2-D patch data: indexing labelled patch folders, splitting, loading.

Quick-start
-----------
::

    from tomographia.patches import build_patch_index, split_patches, PatchDataset, PatchTransform

    df = build_patch_index(["patches/xy", "patches/xz", "patches/yz"])
    df = split_patches(df, ratios=(0.8, 0.1, 0.1), seed=42)
    train_ds = PatchDataset(df[df.split == "train"], PatchTransform(augment=True))
"""
from .dataset import (
    PatchDataset,
    PatchTransform,
    build_patch_index,
    class_counts,
    load_patch,
    normalize_patch,
    split_patches,
)

__all__ = [
    "build_patch_index",
    "split_patches",
    "PatchDataset",
    "PatchTransform",
    "load_patch",
    "normalize_patch",
    "class_counts",
]
