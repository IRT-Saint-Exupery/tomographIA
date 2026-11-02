"""
Training loop for binary 2-D patch classifiers (black-box ResNet baseline).

- ``BCEWithLogitsLoss`` with ``pos_weight = n_neg / n_pos`` (optional cap).
- AdamW (``weight_decay=1e-4``), gradient clipping at 1.0, optional AMP.
- After every epoch: validation ROC-AUC and the F1-optimal threshold.
  The best state is the one with the highest validation AUC.
- "Revert and decay": if validation AUC drops more than ``max_drop`` below the
  best, or ``patience_bad`` epochs pass without improvement, the best weights
  and optimiser state are restored and every learning rate is multiplied by
  ``lr_reduction``.
- Early stopping after ``early_stop_patience`` epochs without improvement.
- An optional ``on_epoch_end`` callback receives the epoch metrics.
- Predictions are returned with the dataset indices (:func:`predict_proba`).
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ..evaluation.metrics import f1_optimal_threshold, ranking_metrics

logger = logging.getLogger(__name__)


@dataclass
class PatchTrainConfig:
    """Hyper-parameters of :class:`PatchClassifierTrainer`."""
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    pos_weight_cap: Optional[float] = None
    grad_clip: float = 1.0
    use_amp: bool = False
    max_drop: float = 0.02
    lr_reduction: float = 0.8
    patience_bad: int = 10
    early_stop_patience: int = 30
    manual_threshold: Optional[float] = None
    device: str = "cpu"


@dataclass
class PatchTrainResult:
    best_epoch: int
    best_val_auc: float
    threshold: float
    history: List[Dict] = field(default_factory=list)


def _dataset_labels(loader) -> np.ndarray:
    ds = loader.dataset
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels, dtype=int)
    return np.concatenate([b[1].numpy().ravel() for b in loader]).astype(int)


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: str = "cpu"
                   ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Run a binary classifier over a loader.

    Returns
    -------
    ``(logits, labels, indices)``; ``indices`` is ``None`` unless the dataset
    was built with ``return_index=True``.
    """
    model.eval()
    logits, labels, idx = [], [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        out = model(x).float()
        if out.ndim == 2 and out.shape[1] != 1:
            raise ValueError("predict_logits expects a single-logit binary classifier "
                             "(ResNetPatchClassifier(num_classes=1))")
        logits.append(out.view(-1).cpu().numpy())
        labels.append(y.numpy().ravel())
        if len(batch) > 2:
            idx.append(np.asarray(batch[2]).ravel())
    return (np.concatenate(logits), np.concatenate(labels).astype(int),
            np.concatenate(idx) if idx else None)


def predict_proba(model: nn.Module, loader, device: str = "cpu"):
    """Like :func:`predict_logits` but returns sigmoid probabilities."""
    z, y, idx = predict_logits(model, loader, device)
    z = z.astype(np.float64)  # float32 sigmoid saturates at |logit| > ~17
    return 1.0 / (1.0 + np.exp(-z)), y, idx


class PatchClassifierTrainer:
    """
    Train a binary patch classifier (e.g. :class:`ResNetPatchClassifier`).

    Example
    -------
    ::

        model = ResNetPatchClassifier("resnet50", pretrained=True)
        trainer = PatchClassifierTrainer(model, PatchTrainConfig(device="cuda:0"))
        result = trainer.fit(train_loader, val_loader)
        probs, y, idx = predict_proba(model, test_loader, "cuda:0")
    """

    def __init__(self, model: nn.Module, config: Optional[PatchTrainConfig] = None,
                 on_epoch_end: Optional[Callable[[Dict], None]] = None) -> None:
        self.cfg = config or PatchTrainConfig()
        self.model = model.to(self.cfg.device)
        self.on_epoch_end = on_epoch_end
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr,
                                           weight_decay=self.cfg.weight_decay)
        self.use_amp = self.cfg.use_amp and "cuda" in str(self.cfg.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.criterion: Optional[nn.Module] = None
        self.threshold = 0.5

    def _build_criterion(self, train_loader) -> None:
        labels = _dataset_labels(train_loader)
        n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
        if n1 == 0:
            raise ValueError("Training set contains no positive patch")
        ratio = n0 / n1
        if self.cfg.pos_weight_cap is not None:
            ratio = min(ratio, float(self.cfg.pos_weight_cap))
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([ratio], dtype=torch.float32, device=self.cfg.device))
        logger.info("Train class counts: 0=%d 1=%d, pos_weight=%.3f", n0, n1, ratio)

    def _train_epoch(self, loader) -> float:
        self.model.train()
        losses = []
        for batch in loader:
            x = batch[0].to(self.cfg.device, non_blocking=True)
            y = batch[1].float().view(-1, 1).to(self.cfg.device, non_blocking=True)
            with torch.autocast("cuda", enabled=self.use_amp):
                loss = self.criterion(self.model(x), y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        return float(np.mean(losses)) if losses else float("nan")

    def fit(self, train_loader, val_loader) -> PatchTrainResult:
        """Train, then load the best (highest validation AUC) weights into the model."""
        cfg = self.cfg
        self._build_criterion(train_loader)
        best_auc, best_state, best_opt, best_epoch = -np.inf, None, None, -1
        bad_epochs, since_best = 0, 0
        history: List[Dict] = []

        for epoch in range(cfg.max_epochs):
            train_loss = self._train_epoch(train_loader)
            prob, y, _ = predict_proba(self.model, val_loader, cfg.device)
            val_auc = ranking_metrics(y, prob)["roc_auc"]
            if epoch == 0 and not np.isfinite(val_auc):
                raise ValueError("Validation ROC-AUC is undefined (the validation set "
                                 "contains a single class); check the data split")
            record = {"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc,
                      "lr": self.optimizer.param_groups[0]["lr"]}

            if np.isfinite(val_auc) and val_auc > best_auc:
                best_auc, best_epoch = val_auc, epoch
                best_state = copy.deepcopy(self.model.state_dict())
                best_opt = copy.deepcopy(self.optimizer.state_dict())
                self.threshold = (cfg.manual_threshold if cfg.manual_threshold is not None
                                  else f1_optimal_threshold(y, prob))
                bad_epochs, since_best = 0, 0
            else:
                bad_epochs += 1
                since_best += 1
                dropped = np.isfinite(val_auc) and val_auc < best_auc - cfg.max_drop
                if best_state is not None and (dropped or bad_epochs >= cfg.patience_bad):
                    lrs = [g["lr"] * cfg.lr_reduction for g in self.optimizer.param_groups]
                    self.model.load_state_dict(best_state)
                    self.optimizer.load_state_dict(best_opt)  # moments of the best epoch
                    for g, lr in zip(self.optimizer.param_groups, lrs):
                        g["lr"] = lr
                    bad_epochs = 0
                    record["reverted"] = True

            record["threshold"] = self.threshold
            history.append(record)
            logger.info("epoch %d  loss=%.4f  val_auc=%.4f", epoch, train_loss, val_auc)
            if self.on_epoch_end:
                self.on_epoch_end(record)
            if since_best >= cfg.early_stop_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return PatchTrainResult(best_epoch=best_epoch, best_val_auc=float(best_auc),
                                threshold=float(self.threshold), history=history)
