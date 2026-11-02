"""Display preprocessing for the labelling tool."""
from __future__ import annotations

import numpy as np
from PIL import Image


def treat_image(img_path: str, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Read a grey-level slice and return ``clip(img * scale + offset, 0, 255)`` as uint8."""
    with Image.open(img_path) as img:
        arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.clip(arr.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)
