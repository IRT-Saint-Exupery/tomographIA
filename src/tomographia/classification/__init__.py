"""Defect classification: logistic regression and KMeans (3-D descriptors), ResNet (2-D patches).

Quick-start
-----------
Logistic (sklearn bundle)::

    from tomographia.classification import classify_defects
    df = classify_defects(df, method="logistic", bundle="path/to/bundle.joblib")

KMeans (unsupervised)::

    df = classify_defects(df, method="kmeans", n_clusters=4)

2-D patch classification (ResNet, requires the ``vision`` extra)::

    from tomographia.classification import ResNetPatchClassifier, PatchClassifierTrainer
    model = ResNetPatchClassifier("resnet50", pretrained=True)
    result = PatchClassifierTrainer(model).fit(train_loader, val_loader)
"""
from __future__ import annotations

import importlib

import pandas as pd

from .logistic import classify_defects_logistic, build_classified_volume, fit_classifier
from .kmeans import cluster_defects, make_cluster_volume

# torch-based members are imported lazily so that the logistic / KMeans path
# works with the core dependencies only (torch is an optional extra).
_LAZY = {
    "ARCHITECTURES": "resnet2d",
    "ResNetPatchClassifier": "resnet2d",
    "PatchClassifierTrainer": "patch_training",
    "PatchTrainConfig": "patch_training",
    "PatchTrainResult": "patch_training",
    "predict_logits": "patch_training",
    "predict_proba": "patch_training",
}


def __getattr__(name: str):
    if name in _LAZY:
        try:
            module = importlib.import_module(f".{_LAZY[name]}", __name__)
        except ImportError as exc:
            raise ImportError(
                f"tomographia.classification.{name} requires torch "
                "and torchvision: pip install 'tomographia[vision]'"
            ) from exc
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "classify_defects",
    "classify_defects_logistic",
    "build_classified_volume",
    "fit_classifier",
    "cluster_defects",
    "make_cluster_volume",
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
