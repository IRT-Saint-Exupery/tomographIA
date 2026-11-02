"""
Prototype network (p-ResNet): a distance-based prototype head on a CNN embedding.

Model
-----
For an embedding ``z`` standardised with the training-set scaler,
``d_k = ||z - p_k||² / D`` is the normalised squared distance to prototype
``k``; prototype logits are ``-d_k / τ``; the logit of class ``c`` is the
``logsumexp`` of the logits of its prototypes.

Any backbone exposing ``features(x) -> [B, D]`` (and ``last_block`` for
partial fine-tuning) can be used.  :class:`PrototypeNetwork` checks the
embedding dimension, so a backbone returning logits instead of the
embedding raises instead of broadcasting.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PrototypeHead(nn.Module):
    """
    Distance-to-prototype classification head.

    Parameters
    ----------
    prototypes      : ``[K, D]`` initial prototypes in *standardised* space,
                      ordered class by class.
    scaler_mean     : ``[D]`` mean of the training embeddings.
    scaler_scale    : ``[D]`` standard deviation of the training embeddings.
    k_per_class     : number of prototypes per class, e.g. ``[3, 3]``.
    learnable_prototypes : optimise the prototype positions.
    init_tau        : softmax temperature τ.
    learnable_tau   : optimise τ (default: fixed).
    tau_bounds      : clamp range for τ in the forward pass.
    """

    def __init__(self, prototypes: np.ndarray, scaler_mean: np.ndarray,
                 scaler_scale: np.ndarray, k_per_class: Sequence[int],
                 learnable_prototypes: bool = True, init_tau: float = 1.0,
                 learnable_tau: bool = False,
                 tau_bounds: Tuple[float, float] = (0.5, 5.0)) -> None:
        super().__init__()
        prototypes = np.asarray(prototypes, dtype=np.float32)
        k_total, dim = prototypes.shape
        if sum(k_per_class) != k_total:
            raise ValueError(f"sum(k_per_class)={sum(k_per_class)} != {k_total} prototypes")
        if np.shape(scaler_mean) != (dim,) or np.shape(scaler_scale) != (dim,):
            raise ValueError("scaler_mean/scaler_scale must have shape [D]")

        self.embed_dim = dim
        self.k_per_class: List[int] = [int(k) for k in k_per_class]
        self.class_slices: List[slice] = []
        start = 0
        for k in self.k_per_class:
            self.class_slices.append(slice(start, start + k))
            start += k
        if not tau_bounds[0] <= init_tau <= tau_bounds[1]:
            raise ValueError(f"init_tau={init_tau} lies outside tau_bounds={tau_bounds}; "
                             "it would be clamped silently in the forward pass")
        self.tau_bounds = tau_bounds

        self.register_buffer("scaler_mean", torch.as_tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.as_tensor(scaler_scale, dtype=torch.float32))
        p = torch.from_numpy(prototypes)
        if learnable_prototypes:
            self.prototypes = nn.Parameter(p)
        else:
            self.register_buffer("prototypes", p)
        tau = torch.tensor(float(init_tau))
        if learnable_tau:
            self.tau = nn.Parameter(tau)
        else:
            self.register_buffer("tau", tau)

    @property
    def num_classes(self) -> int:
        return len(self.k_per_class)

    def standardize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.scaler_mean) / self.scaler_scale

    def forward(self, z_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        z_raw : ``[B, D]`` raw (unstandardised) backbone embeddings.

        Returns
        -------
        dict with ``dist2`` ``[B, K]`` (normalised squared distances),
        ``proto_logits``, ``proto_probs`` ``[B, K]``, ``class_logits``,
        ``class_probs`` ``[B, C]``.
        """
        if z_raw.ndim != 2 or z_raw.shape[1] != self.embed_dim:
            raise ValueError(
                f"PrototypeHead expects embeddings of shape [B, {self.embed_dim}], "
                f"got {tuple(z_raw.shape)}")
        z = self.standardize(z_raw)
        p = self.prototypes
        dist2 = (z.pow(2).sum(1, keepdim=True) + p.pow(2).sum(1).unsqueeze(0)
                 - 2.0 * z @ p.T) / self.embed_dim
        tau = torch.clamp(self.tau, *self.tau_bounds)
        proto_logits = -dist2 / tau
        class_logits = torch.stack(
            [torch.logsumexp(proto_logits[:, s], dim=1) for s in self.class_slices], dim=1)
        return {
            "dist2": dist2,
            "proto_logits": proto_logits,
            "proto_probs": F.softmax(proto_logits, dim=1),
            "class_logits": class_logits,
            "class_probs": F.softmax(class_logits, dim=1),
        }


class PrototypeNetwork(nn.Module):
    """
    Backbone embedding + :class:`PrototypeHead`.

    Parameters
    ----------
    backbone : module exposing ``features(x) -> [B, D]`` (the pooled embedding,
               **not** the classifier logits) and, for partial fine-tuning,
               ``last_block``; e.g.
               :class:`tomographia.classification.ResNetPatchClassifier`.
    head     : prototype head with ``embed_dim == D``.
    """

    def __init__(self, backbone: nn.Module, head: PrototypeHead) -> None:
        super().__init__()
        if not hasattr(backbone, "features"):
            raise TypeError("backbone must expose features(x) returning the embedding")
        self.backbone = backbone
        self.head = head

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Raw backbone embedding ``[B, D]``."""
        return self.backbone.features(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.embed(x)
        out = self.head(z)
        out["embedding"] = z
        return out

    def set_trainable(self, mode: str = "last_block") -> None:
        """
        Choose which backbone parameters are trained (the head always is).

        ``"head"``: backbone frozen; ``"last_block"``: only ``last_block``
        (``layer4`` for ResNets, default); ``"all"``: full fine-tuning.
        """
        if mode not in ("head", "last_block", "all"):
            raise ValueError(f"Unknown mode {mode!r}")
        for p in self.backbone.parameters():
            p.requires_grad = mode == "all"
        if mode == "last_block":
            for p in self.backbone.last_block.parameters():
                p.requires_grad = True
        self.train(self.training)  # re-apply BatchNorm freezing

    def train(self, mode: bool = True) -> "PrototypeNetwork":
        """
        Like :meth:`nn.Module.train`, but BatchNorm layers whose parameters are
        all frozen stay in eval mode, so the running statistics of frozen
        backbone blocks (and hence the embedding space the scaler and
        prototypes were fitted on) do not drift during training.
        """
        super().train(mode)
        if mode and getattr(self, "freeze_frozen_bn", True):
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    params = list(m.parameters(recurse=False))
                    if params and not any(p.requires_grad for p in params):
                        m.eval()
        return self


def binary_logit(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """``class_logits[:, 1] - class_logits[:, 0]`` (so that ``sigmoid`` gives P(y=1))."""
    cl = out["class_logits"]
    return cl[:, 1] - cl[:, 0]


def build_prototype_network(
    classifier: nn.Module,
    centroids_by_class: Dict[int, np.ndarray],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    learnable_prototypes: bool = True,
    init_tau: float = 1.0,
    learnable_tau: bool = False,
    tau_bounds: Tuple[float, float] = (0.5, 5.0),
    copy_backbone: bool = True,
) -> PrototypeNetwork:
    """
    Wrap a trained black-box classifier into a prototype network.

    Parameters
    ----------
    classifier          : trained classifier exposing ``features`` (its
                          ``fc`` layer is not used by the prototype network).
    centroids_by_class  : ``{class_id: [K_c, D]}`` initial prototypes in
                          standardised space (see
                          :func:`tomographia.prototypes.prototypes_from_anchors`).
    scaler_mean, scaler_scale : embedding standardisation statistics.
    init_tau, learnable_tau, tau_bounds : temperature τ of the head; τ is
                          clamped to *tau_bounds* in the forward pass and
                          ``init_tau`` must lie inside them.
    copy_backbone       : deep-copy the classifier so the original is untouched.
    """
    classes = sorted(centroids_by_class)
    mats = [np.asarray(centroids_by_class[c], dtype=np.float32) for c in classes]
    head = PrototypeHead(np.vstack(mats), scaler_mean, scaler_scale,
                         [m.shape[0] for m in mats],
                         learnable_prototypes=learnable_prototypes,
                         init_tau=init_tau, learnable_tau=learnable_tau,
                         tau_bounds=tau_bounds)
    backbone = copy.deepcopy(classifier) if copy_backbone else classifier
    return PrototypeNetwork(backbone, head)
