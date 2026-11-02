"""Defect classification: logistic regression, KMeans, neural network.

Quick-start
-----------
Logistic (sklearn bundle)::

    from tomographia.classification import classify_defects
    df = classify_defects(df, method="logistic", bundle="path/to/bundle.joblib")

KMeans (unsupervised)::

    df = classify_defects(df, method="kmeans", features=["sphericity", ...])

2-D patch classification (ResNet, requires the ``vision`` extra)::

    from tomographia.classification import ResNetPatchClassifier, PatchClassifierTrainer
    model = ResNetPatchClassifier("resnet50", pretrained=True)
    result = PatchClassifierTrainer(model).fit(train_loader, val_loader)
"""
from __future__ import annotations

import pandas as pd

from .logistic import classify_defects_logistic, build_classified_volume, fit_classifier
from .kmeans import cluster_defects, make_cluster_volume
from .embeddings_nn import (
    SimpleNN,
    MulticlassNN,
    HeatmapEmbedding,
    find_heatmap_box,
    neural_network_dict,
)
from .resnet2d import ARCHITECTURES, ResNetPatchClassifier
from .patch_training import (
    PatchClassifierTrainer,
    PatchTrainConfig,
    PatchTrainResult,
    predict_logits,
    predict_proba,
)

__all__ = [
    "classify_defects",
    "classify_defects_logistic",
    "build_classified_volume",
    "fit_classifier",
    "cluster_defects",
    "make_cluster_volume",
    "SimpleNN",
    "MulticlassNN",
    "HeatmapEmbedding",
    "find_heatmap_box",
    "neural_network_dict",
    # 2-D patch classification
    "ResNetPatchClassifier",
    "ARCHITECTURES",
    "PatchClassifierTrainer",
    "PatchTrainConfig",
    "PatchTrainResult",
    "predict_logits",
    "predict_proba",
]


def classify_defects(
    df: pd.DataFrame,
    method: str = "logistic",
    bundle=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Classify defects using the specified *method*.

    Parameters
    ----------
    df     : descriptor DataFrame from
             ``tomographia.components.compute_descriptors``.
    method : ``"logistic"`` (sklearn bundle) or ``"kmeans"`` (unsupervised).
    bundle : joblib bundle dict / path — required for ``method="logistic"``.
    **kwargs: forwarded verbatim to the underlying function.

    Returns
    -------
    Copy of *df* with classification columns added.

    See Also
    --------
    :func:`classify_defects_logistic` — full logistic-path signature.
    :func:`cluster_defects`           — full KMeans-path signature.
    """
    if method == "logistic":
        if bundle is None:
            raise ValueError("bundle= is required for method='logistic'")
        return classify_defects_logistic(df, bundle, **kwargs)
    if method == "kmeans":
        result, _, _ = cluster_defects(df, **kwargs)
        return result
    raise ValueError(
        f"Unknown method {method!r}. Choose 'logistic' or 'kmeans'."
    )
