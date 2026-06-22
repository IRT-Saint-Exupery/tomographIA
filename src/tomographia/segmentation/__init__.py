"""3-D segmentation: inference, training, and patch data utilities."""
from .inference import run_inference
from .training import (
    AugmentedDataset,
    FocalLoss,
    SegmentationTrainer,
    SoftDiceLoss,
    build_pos_sampler,
    estimate_pos_weight,
)
from ._training_data import PatchGenerator, SegmentationDataset
from ._synthetic import make_synthetic_volume

__all__ = [
    # Inference
    "run_inference",
    # Training — losses
    "SoftDiceLoss",
    "FocalLoss",
    # Training — trainer
    "SegmentationTrainer",
    # Training — sampling / augmentation
    "build_pos_sampler",
    "estimate_pos_weight",
    "AugmentedDataset",
    # Training — data pipeline
    "PatchGenerator",
    "SegmentationDataset",
    # Smoke-test fixture
    "make_synthetic_volume",
]
