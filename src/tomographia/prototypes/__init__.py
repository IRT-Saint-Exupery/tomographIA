"""Interpretable prototype networks (p-ResNet) with expert anchors.

Workflow (see the README for a complete example)
------------------------------------------------
1. Train a black-box :class:`~tomographia.classification.ResNetPatchClassifier`.
2. Embed the training set (:func:`extract_embeddings`) and fit a scaler.
3. Build medoid prototypes from expert anchors (:func:`prototypes_from_anchors`).
4. :func:`build_prototype_network` then :func:`train_prototype_network`.
5. Evaluate (:func:`predict_prototype_network`,
   :func:`tomographia.evaluation.evaluate_calibrated`) and inspect
   (:func:`nearest_examples`, :func:`coalescence_stats`, plotting helpers).
"""
from .model import PrototypeHead, PrototypeNetwork, binary_logit, build_prototype_network
from .anchors import AnchorSet, extract_embeddings, fit_embedding_scaler, prototypes_from_anchors
from .losses import ProtoLossWeights, medoid_regulariser, prototype_loss
from .training import (
    ProtoTrainConfig,
    ProtoTrainResult,
    inverse_frequency_weights,
    predict_prototype_network,
    train_prototype_network,
)
from .diagnostics import (
    anchor_assignment_accuracy,
    assign_prototypes,
    coalescence_stats,
    learned_anchor_purity,
    nearest_examples,
    per_prototype_confusion,
    prototype_classes,
    prototype_drift,
)

__all__ = [
    # model
    "PrototypeHead", "PrototypeNetwork", "build_prototype_network", "binary_logit",
    # initialisation
    "extract_embeddings", "fit_embedding_scaler", "prototypes_from_anchors", "AnchorSet",
    # loss / training
    "ProtoLossWeights", "prototype_loss", "medoid_regulariser",
    "ProtoTrainConfig", "ProtoTrainResult", "train_prototype_network",
    "predict_prototype_network", "inverse_frequency_weights",
    # diagnostics
    "coalescence_stats", "prototype_drift", "assign_prototypes",
    "anchor_assignment_accuracy", "nearest_examples", "learned_anchor_purity",
    "per_prototype_confusion", "prototype_classes",
]
