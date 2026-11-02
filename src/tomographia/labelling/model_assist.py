"""Optional model suggestions while labelling (requires torch)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .session import PatchRecord


def load_patch_classifier(checkpoint_path: str, arch: str = "resnet50",
                          device: Optional[str] = None):
    """Load a :class:`~tomographia.classification.ResNetPatchClassifier` checkpoint.

    Accepts a raw state dict or a dict with a ``"model"`` entry.
    """
    import torch
    from ..classification.resnet2d import ResNetPatchClassifier

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetPatchClassifier(arch, num_classes=1)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.eval().to(device), device


def suggest_for_patch(image: np.ndarray, patch: PatchRecord, model, device: str = "cpu",
                      threshold: float = 0.5, max_pixel_value: float = 255.0) -> Tuple[int, float]:
    """
    Model suggestion ``(label, probability)`` for a patch.

    *image* must hold the raw intensities the model was trained on (not the
    contrast-adjusted display image) and ``max_pixel_value`` must match
    training.  Border patches are edge-padded on the side that was clipped.
    """
    import torch
    from ..patches.dataset import normalize_patch
    from ..slice_inference.predict import predict_patches

    ps = patch.patch_size
    h, w = image.shape
    y0, x0 = patch.y - ps // 2, patch.x - ps // 2
    y1, x1 = y0 + ps, x0 + ps
    p = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    pad = ((max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w)))
    if any(v for pair in pad for v in pair):
        p = np.pad(p, pad, mode="edge")
    x = normalize_patch(torch.as_tensor(p, dtype=torch.float32),
                        max_pixel_value).view(1, 1, ps, ps)
    prob = float(predict_patches(model, x, device)[0])
    return int(prob >= threshold), prob
