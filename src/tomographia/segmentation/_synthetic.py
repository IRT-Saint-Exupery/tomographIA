"""
Synthetic 3-D volume generator for smoke-test / end-to-end pipeline validation.

WARNING: volumes produced by :func:`make_synthetic_volume` are Gaussian noise
with embedded geometric blobs.  They are NOT physically realistic; a model
trained on them will not segment real CT data.  Use real data for any
scientific purpose — see the "Training on your own data" section in README.md.
"""
from __future__ import annotations

import numpy as np


def _solid_ellipsoid(rz: int, ry: int, rx: int, padding: int = 3) -> np.ndarray:
    d = (2*(rz+padding)+1, 2*(ry+padding)+1, 2*(rx+padding)+1)
    arr = np.zeros(d, dtype=np.uint8)
    cz, cy, cx = rz+padding, ry+padding, rx+padding
    z, y, x = np.ogrid[:d[0], :d[1], :d[2]]
    arr[((z-cz)/rz)**2 + ((y-cy)/ry)**2 + ((x-cx)/rx)**2 <= 1.0] = 1
    return arr


def make_synthetic_volume(
    shape: tuple[int, int, int] = (64, 64, 64),
    n_defects: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a (volume, label) pair for end-to-end pipeline smoke tests.

    The volume is Gaussian noise with synthetic ellipsoidal blobs embedded as
    "defects".  It is sized so that the training pipeline (patch extraction →
    SegmentationTrainer) completes quickly on a CPU.

    WARNING: this is a test fixture, not a data-generation tool.  A model
    trained on this output will not generalise to real CT volumes.

    Parameters
    ----------
    shape     : (D, H, W) voxel dimensions.  Default (64, 64, 64) gives
                exactly 8 non-overlapping 32³ patches.  Use (128, 128, 128)
                for a stable 80/10/10 split (64 patches with seed=999 →
                49 train / 7 val / 8 test).
    n_defects : number of ellipsoidal blobs to embed.
    seed      : RNG seed.

    Returns
    -------
    volume : float32 ndarray, shape *shape*.
    label  : uint8 ndarray, same shape; 1 inside blobs, 0 elsewhere.
    """
    rng = np.random.default_rng(seed)
    volume = rng.standard_normal(shape).astype(np.float32)
    label  = np.zeros(shape, dtype=np.uint8)

    for _ in range(n_defects):
        rz, ry, rx = rng.integers(3, 9, size=3)
        blob = _solid_ellipsoid(int(rz), int(ry), int(rx))
        dz, dy, dx = blob.shape

        if dz >= shape[0] or dy >= shape[1] or dx >= shape[2]:
            continue

        z0 = int(rng.integers(0, shape[0] - dz))
        y0 = int(rng.integers(0, shape[1] - dy))
        x0 = int(rng.integers(0, shape[2] - dx))

        volume[z0:z0+dz, y0:y0+dy, x0:x0+dx] += blob.astype(np.float32) * 3.0
        label [z0:z0+dz, y0:y0+dy, x0:x0+dx] |= blob

    return volume, label
