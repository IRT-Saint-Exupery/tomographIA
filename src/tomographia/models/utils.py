"""Model factory — build segmentation and 2-D classification architectures by name."""
from __future__ import annotations

import torch


def select_model(name: str, num_channels: int = 1, num_classes: int = 1) -> torch.nn.Module:
    """
    Instantiate a model by *name* (case-insensitive).

    ``"unet3d"``
        3-D U-Net for volumetric segmentation (:class:`tomographia.models.Unet3D`).
    ``"resnet18"`` / ``"resnet34"`` / ``"resnet50"`` / ``"resnet101"`` / ``"resnet152"``
        (``"resnet"`` = resnet18) — 2-D patch classifiers
        (:class:`tomographia.classification.ResNetPatchClassifier`); require
        ``torchvision`` (``vision`` extra).

    Parameters
    ----------
    name         : model identifier.
    num_channels : number of input channels (1 for greyscale CT).
    num_classes  : number of output channels / classes (1 for binary tasks).

    Raises
    ------
    ValueError  : unknown name.
    ImportError : optional dependency missing.
    """
    n = name.lower()

    if n == "unet3d":
        from .unet import Unet3D
        return Unet3D(in_channels=num_channels, num_classes=num_classes)

    if n in ("resnet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        try:
            from ..classification.resnet2d import ResNetPatchClassifier
            import torchvision  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "2-D ResNet classifiers require torchvision: "
                "pip install 'tomographia[vision]'"
            ) from exc
        return ResNetPatchClassifier("resnet18" if n == "resnet" else n,
                                     num_classes=num_classes, in_channels=num_channels)

    raise ValueError(
        f"Unknown model name {name!r}. "
        "Supported: unet3d, resnet18, resnet34, resnet50, resnet101, resnet152."
    )
