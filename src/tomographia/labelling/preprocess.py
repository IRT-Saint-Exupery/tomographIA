"""Display preprocessing for the labelling tool."""
from __future__ import annotations

import numpy as np

from .._slices import read_slice


def treat_image(img_path: str, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Read a grey-level slice and return ``clip(img * scale + offset, 0, 255)`` as uint8."""
    arr = read_slice(img_path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.clip(arr.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)
