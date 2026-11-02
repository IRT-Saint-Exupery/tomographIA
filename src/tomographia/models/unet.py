"""3-D U-Net for volumetric segmentation."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class Conv3DModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.batch_norm(self.conv3d(x)))


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.a = Conv3DModule(in_channels, out_channels)
        self.b = Conv3DModule(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class Encoder3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = Conv3DBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return skip, self.pool(skip)


class Decoder3DBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = Conv3DBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


#: Every spatial input dimension must be a multiple of this value.
DIVISOR = 16


class Unet3D(nn.Module):
    """
    3-D U-Net for binary or multi-class volumetric segmentation.

    Four 2× poolings: every spatial dimension of the input must be divisible
    by 16 (:data:`DIVISOR`).

    Parameters
    ----------
    in_channels : number of input channels (typically 1 for greyscale CT).
    num_classes : number of output classes (1 for binary segmentation).
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.enc0 = Encoder3DBlock(in_channels, 64)
        self.enc1 = Encoder3DBlock(64, 128)
        self.enc2 = Encoder3DBlock(128, 256)
        self.enc3 = Encoder3DBlock(256, 512)
        self.bridge = Conv3DBlock(512, 1024)
        self.dec0 = Decoder3DBlock(1024, 512, 512)
        self.dec1 = Decoder3DBlock(512, 256, 256)
        self.dec2 = Decoder3DBlock(256, 128, 128)
        self.dec3 = Decoder3DBlock(128, 64, 64)
        self.segmenter = nn.Conv3d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(int(d) % DIVISOR for d in x.shape[2:]):
            raise ValueError(f"Unet3D input spatial shape {tuple(x.shape[2:])} must be "
                             f"divisible by {DIVISOR}")
        s0, p0 = self.enc0(x)
        s1, p1 = self.enc1(p0)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        lat = self.bridge(p3)
        out = self.dec0(lat, s3)
        out = self.dec1(out, s2)
        out = self.dec2(out, s1)
        out = self.dec3(out, s0)
        return self.segmenter(out)

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Run a forward pass and threshold the sigmoid output."""
        logits = self.forward(x)
        predictions = (self.sigmoid(logits) > threshold).float()
        if return_logits:
            return logits, predictions
        return predictions

