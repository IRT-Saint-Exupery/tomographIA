"""Tests for tomographia.patches: indexing, splitting, dataset, transforms."""
import os

import numpy as np
import pytest

from tomographia.patches import build_patch_index, split_patches


def test_build_patch_index_sorted_and_labelled(patch_folders):
    df = build_patch_index(patch_folders)
    assert len(df) == 2 * 36
    assert set(df.label) == {0, 1}
    assert set(df.source) == {"xy", "xz"}
    # deterministic order within each class folder
    sub = df[(df.source == "xy") & (df.label == 0)].path.tolist()
    assert sub == sorted(sub)


def test_build_patch_index_missing_class(tmp_path):
    os.makedirs(tmp_path / "0")
    with pytest.raises(FileNotFoundError):
        build_patch_index(str(tmp_path))


def test_split_per_source_ratios(patch_folders):
    df = split_patches(build_patch_index(patch_folders), seed=0)
    for _, g in df.groupby("source"):
        counts = g.split.value_counts()
        assert counts["val"] == int(36 * 0.1) and counts["test"] == int(36 * 0.1)
        assert counts.sum() == 36
    # reproducible
    df2 = split_patches(build_patch_index(patch_folders), seed=0)
    assert (df.split == df2.split).all()


def test_split_by_group_has_no_leakage(patch_folders):
    df = build_patch_index(patch_folders,
                           group_fn=lambda p: os.path.basename(p).split("_")[0])
    df = split_patches(df, ratios=(0.6, 0.2, 0.2), group_col="group", seed=3)
    assert df.groupby("group").split.nunique().max() == 1
    assert set(df.split) == {"train", "val", "test"}  # 3 groups -> one per split


def test_split_by_group_approximates_ratios(patch_folders):
    df = build_patch_index(patch_folders,
                           group_fn=lambda p: os.path.basename(p).split("_")[1])  # 36 groups
    df = split_patches(df, ratios=(0.8, 0.1, 0.1), group_col="group", seed=0)
    frac = df.split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.8) < 0.06 and frac["val"] > 0.05 and frac["test"] > 0.05


def test_split_ratios_must_sum_to_one(patch_folders):
    with pytest.raises(ValueError):
        split_patches(build_patch_index(patch_folders), ratios=(0.5, 0.2, 0.2))


def _torch():
    pytest.importorskip("torchvision")
    return pytest.importorskip("torch")


def test_dataset_items_and_index(patch_folders):
    torch = _torch()
    from tomographia.patches import PatchDataset, PatchTransform

    df = split_patches(build_patch_index(patch_folders))
    ds = PatchDataset(df, PatchTransform(augment=True), return_index=True)
    x, y, idx = ds[5]
    assert x.shape == (1, 32, 32) and x.dtype == torch.float32
    assert y.dtype == torch.long
    assert idx == df.index[5]
    assert -1.01 <= float(x.min()) and float(x.max()) <= 1.01


def test_transform_matches_albumentations_normalisation():
    torch = _torch()
    from tomographia.patches import PatchTransform

    x = torch.full((1, 4, 4), 255.0)
    assert torch.allclose(PatchTransform()(x), torch.ones(1, 4, 4))
    x = torch.zeros(1, 4, 4)
    assert torch.allclose(PatchTransform()(x), -torch.ones(1, 4, 4))


def test_rotation_keeps_shape_and_fills_borders():
    torch = _torch()
    from tomographia.patches import PatchTransform

    t = PatchTransform(augment=True)
    x = torch.rand(1, 64, 64) * 255
    r = t._rotate(x, 15.0)
    assert r.shape == x.shape
    assert float(r[0, 0, 0]) != 0.0  # reflect padding, no black corners
