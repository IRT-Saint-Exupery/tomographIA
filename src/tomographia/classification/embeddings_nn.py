"""
Neural-network architectures and utilities for defect classification.

:class:`SimpleNN` and :class:`MulticlassNN` classify tabular descriptor
features (same 12-feature list as the logistic path).

:class:`HeatmapEmbedding` embeds a raw 3-D heatmap patch (Conv3D → pool → FC).

:func:`find_heatmap_box` was previously duplicated across three modules
(``data_processing/defects.py``, ``inference/classify_defects.py``,
``inference/classify_defects_multiclass.py``).  It is consolidated here.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def find_heatmap_box(heatmap: np.ndarray):
    """
    Return a closure that extracts the bounding-box region from *heatmap*.

    Parameters
    ----------
    heatmap : 3-D array (Z, Y, X).

    Returns
    -------
    callable ``coord_box(bbox: dict) → np.ndarray``

    Example
    -------
    >>> box_fn = find_heatmap_box(heatmap_vol)
    >>> patch = box_fn({"zmin": 10, "zmax": 20, "ymin": 5, "ymax": 15,
    ...                  "xmin": 3, "xmax": 8})
    """
    def coord_box(bbox: dict) -> np.ndarray:
        return heatmap[
            bbox["zmin"] : bbox["zmax"] + 1,
            bbox["ymin"] : bbox["ymax"] + 1,
            bbox["xmin"] : bbox["xmax"] + 1,
        ]
    return coord_box


class SimpleNN(nn.Module):
    """Binary MLP for tabular descriptor features.

    Output is a single sigmoid probability (P(crack)).
    Use BCELoss or BCEWithLogitsLoss during training.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))


class MulticlassNN(nn.Module):
    """3-class MLP for tabular descriptor features (logit output).

    Classes: 0=background, 1=pore, 2=crack.
    Use CrossEntropyLoss during training (applies softmax internally).
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # raw logits


class HeatmapEmbedding(nn.Module):
    """
    Embed a 3-D defect-probability heatmap into a fixed-size vector.

    Architecture: Conv3D(1→32, k=3) → AdaptiveAvgPool3d(8³) → Flatten → Linear(→dim).
    The FC layer is initialised lazily on the first forward pass so this module
    works with any heatmap spatial size.
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3)
        self.pool1 = nn.AdaptiveAvgPool3d((8, 8, 8))
        self.flatten = nn.Flatten()
        self.embedding_dim = embedding_dim
        self.fc1: nn.Linear | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure spatial dims are > kernel size 3
        for dim_idx, dim in enumerate(x.shape[2:]):
            if dim < 8:
                pad = max(0, 8 - dim)
                padding = [0] * 6
                padding[(2 - dim_idx) * 2] = pad
                padding[(2 - dim_idx) * 2 + 1] = pad + 2
                x = nn.functional.pad(x, padding, mode="constant")
        out = torch.relu(self.conv1(x))
        out = self.pool1(out)
        out = self.flatten(out)
        if self.fc1 is None:
            self.fc1 = nn.Linear(out.shape[1], self.embedding_dim).to(out.device)
        return self.fc1(out)


def neural_network_dict() -> dict:
    """Return a ``{name: class}`` mapping for available NN classifiers.

    Fixed from original: was ``"multiclassNN"`` (lower m); now ``"MulticlassNN"``.
    """
    return {"SimpleNN": SimpleNN, "MulticlassNN": MulticlassNN}
