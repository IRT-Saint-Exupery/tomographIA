"""Slice-file discovery shared by every module that reads a stack of slices.

All readers order slices the same way (natural order: ``slice_2`` before
``slice_10``) so that index ``z`` refers to the same image everywhere.
"""
from __future__ import annotations

import os
import re
from typing import List, Sequence

TIF_EXTENSIONS = (".tif", ".tiff")


def natural_key(name: str):
    """Sort key that orders ``slice_2`` before ``slice_10`` (case-insensitive)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.fspath(name))]


def list_slices(directory: str, extensions: Sequence[str] = TIF_EXTENSIONS) -> List[str]:
    """Full paths of the slice images in *directory*, in natural order.

    Hidden files are skipped; the extension match is case-insensitive.
    """
    exts = tuple(e.lower() for e in extensions)
    names = [f for f in os.listdir(directory)
             if not f.startswith(".") and f.lower().endswith(exts)]
    return [os.path.join(directory, f) for f in sorted(names, key=natural_key)]


def read_slice(path: str):
    """Read one slice image as a numpy array (any TIFF byte order or bit depth).

    TIFFs are read with ``tifffile``; other formats (and TIFFs needing a codec
    that is not installed) fall back to Pillow.
    """
    import numpy as np

    if path.lower().endswith(TIF_EXTENSIONS):
        import tifffile
        try:
            return np.asarray(tifffile.imread(path))
        except ValueError as exc:
            if "imagecodecs" not in str(exc):
                raise
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im)


def read_volume(paths: Sequence[str]):
    """Stack slice images into a ``(Z, Y, X)`` array."""
    import numpy as np

    if not paths:
        raise FileNotFoundError("No slice images to read")
    return np.stack([read_slice(p) for p in paths], axis=0)
