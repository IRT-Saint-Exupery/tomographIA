"""Shared fixtures: synthetic 2-D patch folders."""
import os

import numpy as np
import pytest
from PIL import Image


def _make_patch(rng, defect: bool, size: int = 32) -> np.ndarray:
    img = rng.normal(120, 10, (size, size))
    if defect:  # dark pore-like disc
        yy, xx = np.mgrid[:size, :size]
        cy, cx = rng.integers(8, size - 8, 2)
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < 16] = 20
    return np.clip(img, 0, 255).astype(np.uint8)


def write_patch_folder(root, n_neg=24, n_pos=12, size=32, seed=0):
    rng = np.random.default_rng(seed)
    for label, n in ((0, n_neg), (1, n_pos)):
        d = os.path.join(root, str(label))
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            Image.fromarray(_make_patch(rng, bool(label), size)).save(
                os.path.join(d, f"spec{i % 3}_{i:03d}.tif"))
    return str(root)


@pytest.fixture
def patch_folders(tmp_path):
    """Two folders (e.g. two slicing directions) of synthetic labelled patches."""
    return [write_patch_folder(tmp_path / "xy", seed=0),
            write_patch_folder(tmp_path / "xz", seed=1)]
