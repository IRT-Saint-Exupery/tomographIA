"""
Training loop for prototype networks.

Protocol
--------
- Prototypes initialised at expert-anchor medoids; ``L_medoid`` pulls them
  towards these **fixed** medoids.
- Only the head and the last backbone block (``layer4``) are trained.
- AdamW, separate learning rates for head and backbone, grad clipping 1.0.
- Validation objective = prototype loss (no anchor term: validation has no
  anchors) + ``w_medoid * L_medoid``.  ``ReduceLROnPlateau`` on it; the best
  model is the one with the lowest validation objective; early stopping.
- The F1-optimal threshold is chosen on validation for the best model.

Note: BatchNorm layers of frozen backbone blocks are kept in eval mode
(:meth:`PrototypeNetwork.train`), so their running statistics do not drift
away from the embedding space the scaler and prototypes were fitted on
(set ``pnet.freeze_frozen_bn = False`` to let them update).

Note on ``tau_push``: distances are divided by the embedding dimension, so
for standardised embeddings ``dist2`` is O(1) (≈ 2 between unrelated
samples) and ``exp(-dist2 / 0.1)`` is tiny; with the default ``tau_push=0.1``
the push term contributes little.  Increase ``tau_push`` (e.g. 1–2) for a
stronger inter-class repulsion.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..evaluation.metrics import f1_optimal_threshold, ranking_metrics
from .losses import ProtoLossWeights, prototype_loss
from .model import PrototypeNetwork, binary_logit

logger = logging.getLogger(__name__)


@dataclass
class ProtoTrainConfig:
    """Optimisation settings."""
    trainable: str = "last_block"
    lr_head: float = 5e-4
    lr_backbone: float = 5e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    tau_push: float = 0.1
    delta: float = 0.7
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-7
    early_stop_patience: int = 50
    max_epochs: int = 1000
    use_class_weights: bool = True
    device: str = "cpu"
    loss_weights: ProtoLossWeights = field(default_factory=ProtoLossWeights)


@dataclass
class ProtoTrainResult:
    best_epoch: int
    best_val_loss: float
    threshold: float
    best_val_auc: float = float("nan")
    history: List[Dict] = field(default_factory=list)


def inverse_frequency_weights(labels: Sequence[int], num_classes: int = 2) -> torch.Tensor:
    """``w_c = N / (C · n_c)`` class weights for the cross-entropy term."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    if (counts == 0).any():
        raise ValueError(f"Every class needs at least one sample, got counts {counts}")
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


@torch.no_grad()
def predict_prototype_network(pnet: PrototypeNetwork, loader, device: str = "cpu") -> Dict[str, np.ndarray]:
    """
    Run a prototype network over a loader.

    Returns
    -------
    dict with ``logit`` (binary logit), ``prob`` (P(y=1)), ``proto_probs``
    ``[N, K]``, ``dist2`` ``[N, K]``, ``embedding`` ``[N, D]``, ``label`` and,
    if available, ``index``.
    """
    pnet.eval().to(device)
    keys = ("logit", "prob", "proto_probs", "dist2", "embedding", "label", "index")
    acc: Dict[str, list] = {k: [] for k in keys}
    for batch in loader:
        out = pnet(batch[0].to(device))
        acc["logit"].append(binary_logit(out).cpu().numpy())
        acc["prob"].append(out["class_probs"][:, 1].cpu().numpy())
        for k in ("proto_probs", "dist2", "embedding"):
            acc[k].append(out[k].cpu().numpy())
        acc["label"].append(batch[1].numpy().ravel())
        if len(batch) > 2:
            acc["index"].append(np.asarray(batch[2]).ravel())
    res = {k: np.concatenate(v) for k, v in acc.items() if v}
    # float64 sigmoid of the logit: identical to class_probs[:, 1] but without
    # float32 saturation at 0 / 1 (matters for threshold selection)
    res["prob"] = 1.0 / (1.0 + np.exp(-res["logit"].astype(np.float64)))
    return res


def _validation_objective(pnet, loader, cfg, class_weights, medoids) -> float:
    """Sample-weighted mean of the per-batch objective (batch-level terms such
    as usage and diversity still depend on the validation batch size)."""
    pnet.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(cfg.device), batch[1].long().to(cfg.device)
            terms = prototype_loss(pnet(x), y, pnet.head, cfg.loss_weights,
                                   class_weights=class_weights, tau_push=cfg.tau_push,
                                   delta=cfg.delta, medoids=medoids)
            total += float(terms["total"]) * len(y)
            n += len(y)
    return total / max(n, 1)


def train_prototype_network(
    pnet: PrototypeNetwork,
    train_loader,
    val_loader,
    medoids: Optional[np.ndarray] = None,
    anchor_class: Optional[torch.Tensor] = None,
    anchor_proto: Optional[torch.Tensor] = None,
    config: Optional[ProtoTrainConfig] = None,
    on_epoch_end: Optional[Callable[[Dict], None]] = None,
) -> ProtoTrainResult:
    """
    Train *pnet* in place and load its best weights.

    Parameters
    ----------
    train_loader : yields ``(x, y, index)`` — build the training
                   :class:`~tomographia.patches.PatchDataset` with
                   ``return_index=True`` so anchors can be looked up.
    val_loader   : yields ``(x, y)`` or ``(x, y, index)``.
    medoids      : ``[K, D]`` fixed medoids (``AnchorSet.medoids``); defaults to
                   the initial prototypes.
    anchor_class, anchor_proto : anchor targets (``AnchorSet``); omit to train
                   without the anchor term.
    """
    cfg = config or ProtoTrainConfig()
    dev = cfg.device
    pnet.to(dev)
    pnet.set_trainable(cfg.trainable)

    if medoids is None:
        medoids_t = pnet.head.prototypes.detach().clone()
    else:
        medoids_t = torch.as_tensor(np.asarray(medoids), dtype=torch.float32, device=dev)
    if anchor_class is not None:
        anchor_class, anchor_proto = anchor_class.to(dev), anchor_proto.to(dev)

    class_weights = None
    if cfg.use_class_weights:
        ds = train_loader.dataset
        labels = ds.labels if hasattr(ds, "labels") else np.concatenate(
            [b[1].numpy().ravel() for b in train_loader])
        class_weights = inverse_frequency_weights(labels, pnet.head.num_classes).to(dev)

    head_params = [p for p in pnet.head.parameters() if p.requires_grad]
    bb_params = [p for p in pnet.backbone.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": cfg.lr_head}]
    if bb_params:
        groups.append({"params": bb_params, "lr": cfg.lr_backbone})
    optim = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    clip_params = head_params + bb_params

    best_loss, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history: List[Dict] = []

    for epoch in range(cfg.max_epochs):
        pnet.train()
        train_terms: List[Dict[str, float]] = []
        for batch in train_loader:
            if len(batch) < 3 and anchor_class is not None:
                raise ValueError("train_loader must yield (x, y, index) when anchors are used")
            x, y = batch[0].to(dev), batch[1].long().to(dev)
            idx = torch.as_tensor(batch[2], device=dev) if len(batch) > 2 else None
            terms = prototype_loss(pnet(x), y, pnet.head, cfg.loss_weights,
                                   class_weights=class_weights, tau_push=cfg.tau_push,
                                   delta=cfg.delta, sample_indices=idx,
                                   anchor_class=anchor_class, anchor_proto=anchor_proto,
                                   medoids=medoids_t)
            optim.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
            optim.step()
            train_terms.append({k: float(v.detach()) for k, v in terms.items()})

        val_loss = _validation_objective(pnet, val_loader, cfg, class_weights, medoids_t)
        sched.step(val_loss)
        record = {"epoch": epoch, "val_loss": val_loss,
                  "lr_head": optim.param_groups[0]["lr"]}
        record.update({f"train_{k}": float(np.mean([t[k] for t in train_terms]))
                       for k in train_terms[0]})

        if val_loss < best_loss:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(pnet.state_dict())
        else:
            bad += 1
        history.append(record)
        logger.info("epoch %d  val_loss=%.5f", epoch, val_loss)
        if on_epoch_end:
            on_epoch_end(record)
        if bad >= cfg.early_stop_patience:
            break

    if best_state is not None:
        pnet.load_state_dict(best_state)
    val = predict_prototype_network(pnet, val_loader, dev)
    threshold = f1_optimal_threshold(val["label"], val["prob"])
    return ProtoTrainResult(best_epoch=best_epoch, best_val_loss=best_loss,
                            threshold=threshold,
                            best_val_auc=ranking_metrics(val["label"], val["prob"])["roc_auc"],
                            history=history)
