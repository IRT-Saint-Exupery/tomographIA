"""Optional model suggestions while labelling (requires torch)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .session import PatchRecord
from .visualize import extract_patch


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
                      threshold: float = 0.5) -> Tuple[int, float]:
    """
    Model suggestion ``(label, probability)`` for a patch.

    *image* must hold the raw intensities the model was trained on (not the
    contrast-adjusted display image).  Border patches are edge-padded.
    """
    import torch
    from ..patches.dataset import normalize_patch
    from ..slice_inference.predict import predict_patches

    p, _ = extract_patch(image, patch.x, patch.y, patch.patch_size)
    ps = patch.patch_size
    if p.shape != (ps, ps):
        p = np.pad(p, ((0, ps - p.shape[0]), (0, ps - p.shape[1])), mode="edge")
    x = normalize_patch(torch.as_tensor(p, dtype=torch.float32)).view(1, 1, ps, ps)
    prob = float(predict_patches(model, x, device)[0])
    return int(prob >= threshold), prob
