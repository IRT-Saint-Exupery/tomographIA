"""
2-D ResNet patch classifiers for single-channel CT patches.

Ported from ``tomographIA.models.resnet`` (``ResNet{18,34,50,101,152}_classifier``).

Porting changes vs. source
--------------------------
- The five near-identical classes are replaced by one parametrised
  :class:`ResNetPatchClassifier`.  The module layout (``self.model`` holding a
  torchvision ResNet) is unchanged, so state dicts saved by the original
  classes load directly.
- ``features()`` is a differentiable embedding method (``extract_features``
  in the source was wrapped in ``@torch.no_grad`` and could not be used for
  fine-tuning); ``extract_features()`` is kept as its no-grad counterpart.
- ``forward`` is expressed as ``fc(features(x))`` so that embedding and
  logit paths can never diverge.
- The source ignored the ``pretrained`` argument for ResNet-18/101/152 and
  always downloaded ImageNet weights; it is now honoured for every depth and
  uses the torchvision ``weights=`` API.  The paper models were trained from
  ImageNet weights (``pretrained=True``); the default here is ``False`` so
  that the library works offline.

Note: ``conv1`` is replaced by a 3×3 / stride-1 single-channel convolution
(randomly initialised), as in the paper, which suits 64×64 patches.
"""
from __future__ import annotations

import torch
from torch import nn

ARCHITECTURES = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")


class ResNetPatchClassifier(nn.Module):
    """
    Torchvision ResNet adapted to 1-channel 2-D patches.

    Parameters
    ----------
    arch        : one of ``resnet18``, ``resnet34``, ``resnet50``,
                  ``resnet101``, ``resnet152``.
    num_classes : output logits (1 for binary defect / no-defect).
    in_channels : input channels (1 for grey-level CT).
    pretrained  : initialise from ImageNet weights (except ``conv1``/``fc``).
    """

    def __init__(self, arch: str = "resnet50", num_classes: int = 1,
                 in_channels: int = 1, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision import models

        arch = arch.lower()
        if arch not in ARCHITECTURES:
            raise ValueError(f"Unknown arch {arch!r}; choose from {ARCHITECTURES}")
        self.arch = arch
        self.model = getattr(models, arch)(weights="DEFAULT" if pretrained else None)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                                     padding=1, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    @property
    def embed_dim(self) -> int:
        """Dimension of the pooled embedding (512 or 2048)."""
        return self.model.fc.in_features

    @property
    def last_block(self) -> nn.Module:
        """The last residual stage (``layer4``)."""
        return self.model.layer4

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled penultimate embedding ``[B, embed_dim]`` (differentiable)."""
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        return torch.flatten(m.avgpool(x), 1)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """No-grad version of :meth:`features` (embedding extraction)."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits ``[B, num_classes]``."""
        return self.model.fc(self.features(x))
