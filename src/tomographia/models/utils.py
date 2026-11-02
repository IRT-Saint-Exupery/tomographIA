"""Model factory — load segmentation and 2-D classification architectures by name."""
from __future__ import annotations

from typing import Tuple

import torch


def select_model(
    name: str,
    num_channels: int,
    num_classes: int,
    patch_shape: Tuple,
    feature_size: int = 48,
) -> torch.nn.Module:
    """
    Instantiate a segmentation model by *name* (case-insensitive).

    Built-in (no extra deps):
      ``"unet3d"``, ``"unet3d_classifier"``

    Optional (require additional packages):
      ``"swin_unetr"``, ``"unetr"`` — require ``monai``
      ``"resnet"`` (= resnet18) / ``"resnet18"`` / ``"resnet34"`` /
      ``"resnet50"`` / ``"resnet101"`` / ``"resnet152"`` — 2-D patch
      classifiers (:class:`tomographia.classification.ResNetPatchClassifier`),
      require ``torchvision`` (``vision`` extra)

    Parameters
    ----------
    name         : model identifier string.
    num_channels : number of input channels (1 for greyscale CT).
    num_classes  : number of output classes (1 for binary segmentation).
    patch_shape  : spatial patch shape passed to architectures that need it.
                   Note: the original ``predict_volume_big.py`` script passes
                   ``patch_shape[::-1]`` (reversed from (D,H,W) → (W,H,D))
                   — callers should handle this reversal themselves.
    feature_size : feature map size for transformer-based models.

    Raises
    ------
    ValueError  : unknown name.
    ImportError : optional dependency missing.
    """
    n = name.lower()

    if n == "unet3d":
        from .unet import Unet3D
        return Unet3D(in_channels=num_channels, num_classes=num_classes)

    if n == "unet3d_classifier":
        from .unet import Unet3d_classifier
        return Unet3d_classifier(in_channels=num_channels, num_classes=num_classes)

    if n == "swin_unetr":
        try:
            from .swin_unetr import SwappedDimsSwinUNETR  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "swin_unetr requires the 'monai' package: pip install monai"
            ) from exc
        return SwappedDimsSwinUNETR(
            img_size=patch_shape,
            in_channels=num_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=True,
        )

    if n == "unetr":
        try:
            from .unetr import UNETR  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "unetr requires the 'monai' package: pip install monai"
            ) from exc
        return UNETR(
            in_channels=num_channels,
            out_channels=num_classes,
            img_size=patch_shape,
            feature_size=feature_size,
        )

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
        "Supported built-ins: unet3d, unet3d_classifier. "
        "Optional: swin_unetr, unetr, resnet18/34/50/101/152."
    )
