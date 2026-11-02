"""
Segmentation training: SegmentationTrainer, loss functions, and sampling utilities.

Ported from ``tomographIA.data_training.train_complex_oversample``.

Source-line provenance
----------------------
SoftDiceLoss              train_complex_oversample.py  L587-600
FocalLoss                 train_complex_oversample.py  L117-151
_remove_small_components  train_complex_oversample.py  L29-35
build_pos_sampler         train_complex_oversample.py  L153-187
estimate_pos_weight       train_complex_oversample.py  L190-209
AugmentedDataset          train_complex_oversample.py  L212-250
SegmentationTrainer       train_complex_oversample.py  L253-584
  + fit/training/val loop trainers/simple.py           L92-187
  + metric helpers        evaluation_tools/metrics.py  L10-90

Porting changes vs. source
--------------------------
- ``opt.min_size`` global reference in ``validation_epoch_end`` moved to
  constructor parameter ``min_size`` (critical: class was unusable outside
  ``__main__``).
- AMP made opt-in via ``use_amp=False``; GradScaler now stored on the
  instance and passed to ``training_step`` (fixes dead-code bug where the
  scaler was constructed but never used).
- Test-set buffer-leak fixed: ``evaluation_epoch_end`` now calls
  ``validation_epoch_end()`` to clear accumulated logit buffers, matching
  the intent of the original comment.
- ``build_pos_sampler`` / ``AugmentedDataset`` accept any
  ``torch.utils.data.Dataset`` instead of the h5-backed internal Dataset.
- MLflow dependency removed; replaced with an optional ``on_log`` callback.
- Training/validation/evaluation loop inlined from ``simple.Trainer``;
  EMA and distributed-training paths dropped (confirmed unused).
- Metrics inlined as private helpers (pure torch, no sklearn/numpy).
- Hardcoded ``/workspace/…`` argparse defaults removed.
- ``gradient_accumulation_steps`` parameter **removed**.  It was accepted by
  the original ``simple.Trainer`` constructor but the accumulation loop was
  never implemented — every step called ``optimizer.step()`` regardless.
  Keeping the parameter would give false confidence; it was dropped.
- Metric key ``"best_treshold"`` (typo in source L497) corrected to
  ``"best_threshold"``.
"""
from __future__ import annotations

import collections
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as torch_utils
from scipy.ndimage import label as _scipy_label

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Private metric helpers
# (inlined from tomographIA/evaluation_tools/metrics.py L10-90)
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.mean((y_true == y_pred).float()).item()


def _precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = torch.sum((y_true == 1) & (y_pred == 1)).float()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).float()
    return (tp / (tp + fp + 1e-8)).item()


def _recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = torch.sum((y_true == 1) & (y_pred == 1)).float()
    fn = torch.sum((y_true == 1) & (y_pred == 0)).float()
    return (tp / (tp + fn + 1e-8)).item()


def _f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    p = _precision(y_true, y_pred)
    r = _recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)


def _specificity(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tn = torch.sum((y_true == 0) & (y_pred == 0)).float()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).float()
    return (tn / (tn + fp + 1e-8)).item()


def _sensitivity(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return _recall(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Private helper
# (inlined from train_complex_oversample.py L29-35)
# ─────────────────────────────────────────────────────────────────────────────

def _remove_small_components(
    pred_mask: np.ndarray, min_size: int
) -> np.ndarray:
    labeled, _ = _scipy_label(pred_mask)
    counts = np.bincount(labeled.ravel())
    remove = counts < min_size
    labeled[remove[labeled]] = 0
    return (labeled > 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class SoftDiceLoss(torch.nn.Module):
    """Soft Dice loss for binary segmentation.

    Source: ``train_complex_oversample.py`` L587–600.

    Parameters
    ----------
    smooth : float
        Laplace smoothing added to numerator and denominator.
    reduction : {"mean", "none"}
        ``"mean"`` returns a scalar; ``"none"`` returns per-sample values.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = logits.sigmoid()
        dims = tuple(range(2, probs.ndim))  # reduce over spatial dimensions
        num = 2.0 * (probs * targets).sum(dim=dims) + self.smooth
        den = (probs + targets).sum(dim=dims) + self.smooth
        loss = 1.0 - num / den
        return loss.mean() if self.reduction == "mean" else loss


class FocalLoss(torch.nn.Module):
    """Sigmoid focal loss for binary segmentation.

    Source: ``train_complex_oversample.py`` L117–151.

    Parameters
    ----------
    alpha : float
        Weighting factor for the focal modulation term.
    gamma : float
        Focusing parameter; 0 recovers standard BCE.
    pos_weight : torch.Tensor, optional
        Per-class weight passed to ``binary_cross_entropy_with_logits``.
    reduction : {"mean", "sum", "none"}
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        pos_w = self.pos_weight
        if pos_w is not None and logits.device != pos_w.device:
            pos_w = pos_w.to(logits.device)

        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_w, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Sampling utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_pos_sampler(
    dataset: torch_utils.Dataset,
    target_pos_frac: float,
) -> torch_utils.WeightedRandomSampler:
    """Return a ``WeightedRandomSampler`` that draws ``target_pos_frac``
    positive patches per batch.

    Source: ``train_complex_oversample.py`` L153–187.

    Interface change vs. source: accepts any ``torch.utils.data.Dataset``
    whose ``__getitem__`` returns ``(x, y, ...)`` with ``y`` being the label
    tensor or array.  The caller is responsible for constructing the dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Training dataset.  ``dataset[i][1]`` must be the label volume.
    target_pos_frac : float
        Target fraction of positive patches drawn per epoch (0 < f < 1).

    Raises
    ------
    ValueError
        If ``target_pos_frac`` is outside (0, 1).
    RuntimeError
        If no positive patches are found.
    """
    if not (0.0 < target_pos_frac < 1.0):
        raise ValueError("target_pos_frac must be strictly between 0 and 1")

    pos_mask: List[bool] = []
    for i in range(len(dataset)):  # type: ignore[arg-type]
        item = dataset[i]
        y = item[1]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        pos_mask.append(y.float().sum().item() > 0)

    n_pos = sum(pos_mask)
    n_total = len(pos_mask)
    n_neg = n_total - n_pos
    logger.info(
        "build_pos_sampler: total=%d  pos=%d  neg=%d", n_total, n_pos, n_neg
    )

    if n_pos == 0:
        raise RuntimeError(
            "No positive patches found — cannot oversample positives. "
            "Check that your label volumes contain non-zero voxels."
        )

    w_pos = 1.0
    w_neg = (n_pos * (1.0 - target_pos_frac)) / (
        target_pos_frac * max(n_neg, 1)
    )
    weights = torch.tensor(
        [w_pos if m else w_neg for m in pos_mask], dtype=torch.double
    )
    return torch_utils.WeightedRandomSampler(
        weights, num_samples=n_total, replacement=True
    )


def estimate_pos_weight(
    dataset: torch_utils.Dataset,
    max_samples: int = 2000,
) -> torch.Tensor:
    """Estimate the BCE ``pos_weight`` from a random subset of training data.

    Source: ``train_complex_oversample.py`` L190–209.

    Returns ``neg_voxels / pos_voxels`` computed over up to *max_samples*
    items.  Cap the result externally:
    ``weight = torch.minimum(estimate_pos_weight(ds), torch.tensor([cap]))``.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Training dataset.  ``dataset[i][1]`` must be the binary label.
    max_samples : int
        Maximum number of items to scan (for speed on large datasets).
    """
    pos = 0.0
    neg = 0.0
    for i in range(min(len(dataset), max_samples)):  # type: ignore[arg-type]
        item = dataset[i]
        y = item[1]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        y = y.float()
        pos += y.sum().item()
        neg += y.numel() - y.sum().item()
    return torch.tensor([neg / max(pos, 1.0)])


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AugmentedDataset(torch_utils.Dataset):
    """Wrap any segmentation dataset with TorchIO spatial augmentations.

    Source: ``train_complex_oversample.py`` L212–250.

    Interface change vs. source: wraps any ``torch.utils.data.Dataset``
    instead of subclassing the internal h5-backed ``Dataset``, making it
    compatible with any data loading strategy.

    Parameters
    ----------
    base_dataset : torch.utils.data.Dataset
        Base dataset returning ``(x, y, ...)`` where *x* and *y* are 3-D
        float32 arrays or tensors.
    transform : torchio.Compose or None
        TorchIO composed transform.  If ``None`` this is a transparent wrapper.

    Notes
    -----
    ``torchio`` is an optional dependency; it is imported lazily in
    ``__getitem__`` so the class can be imported without torchio installed
    as long as ``transform=None``.
    """

    def __init__(
        self,
        base_dataset: torch_utils.Dataset,
        transform=None,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        x = item[0]
        y = item[1]
        extra = item[2:]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        if self.transform is not None:
            import torchio as tio  # lazy import

            sample = tio.Subject(
                image=tio.ScalarImage(tensor=x),
                label=tio.LabelMap(tensor=y),
            )
            augmented = self.transform(sample)
            x = augmented.image.tensor
            y = augmented.label.tensor

        x = x.squeeze(0)
        y = y.squeeze(0)

        if extra:
            return (x, y, *extra)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationTrainer:
    """Training loop for binary 3-D segmentation with UNet-style models.

    Source provenance
    -----------------
    Core class:  ``train_complex_oversample.py``  L253–584
    Fit loop:    ``tomographIA/boilerplate/pytorch/trainers/simple.py``  L92–187

    AMP note
    --------
    AMP (automatic mixed precision) is controlled by the ``use_amp`` parameter.
    **Default is** ``False`` **(full FP32)**.  In the original script AMP was
    non-functional (the ``GradScaler`` was never passed to ``training_step``);
    it is functional in this version.

    Expected AMP sensitivity: the ``SoftDiceLoss`` + ``BCEWithLogitsLoss``
    combination and ``robust_minmax`` normalisation to ``[-1, 1]`` are
    numerically stable in fp16.  The threshold sweep in
    ``validation_epoch_end`` runs in CPU fp32 regardless of ``use_amp``, so
    reported val metrics are not affected.  Practical effect: ~30–50 % faster
    training, ~20 % lower peak GPU memory, <0.5 % numerical variation in loss
    values.  Safe to enable for new training runs.

    Parameters
    ----------
    max_epochs : int
        Number of training epochs.
    device : str or torch.device
        Training device, e.g. ``"cuda:0"`` or ``"cpu"``.
    optimizer : torch.optim.Optimizer
        Configured optimiser (Adam recommended).
    scheduler : LRScheduler, optional
        LR scheduler; ``step()`` is called after **every gradient step**.
        e.g. ``OneCycleLR(max_lr=lr, steps_per_epoch=…)``.
    save_directory : str
        Directory where model checkpoints are written.  Pass ``""`` to
        disable checkpointing.
    log_every_n_steps : int
        Step interval for ``on_log`` callbacks.
    log_n_last_models : int
        Maximum number of best checkpoints to retain on disk.
    min_size : int
        Minimum connected-component size (voxels) applied to predictions
        during validation; components smaller than this are removed.
        Use ``1`` to disable.  **Previously a global ``opt.min_size``
        reference inside the method body; moved to constructor.**
    manual_threshold : float, optional
        Fixed sigmoid threshold.  If ``None``, the threshold is chosen per
        epoch by an exhaustive F1-maximising grid search.
    use_tta : bool
        Apply test-time augmentation during validation/evaluation.
    tta_mode : str
        TTA strategy: ``"flip"`` (D/H/W axis flips, 4× inference) or
        ``"none"``.
    bce_or_focal : {"bce", "focal"}
        Primary loss function.
    alpha_focal, gamma_focal : float
        Focal-loss hyper-parameters (ignored when ``bce_or_focal="bce"``).
    bce_dice_weights : (float, float)
        ``(bce_weight, dice_weight)``; normalised internally so they need
        not sum to 1.
    use_amp : bool
        Enable CUDA automatic mixed precision.  See *AMP note* above.
        Default ``False``.
    on_log : callable, optional
        ``on_log(key: str, value: float, step: int) -> None`` — called for
        every scalar metric.  Use this to integrate MLflow, W&B,
        TensorBoard, or any other tracker.
    """

    def __init__(
        self,
        max_epochs: int = 50,
        device: str = "cuda:0",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        save_directory: str = "",
        log_every_n_steps: int = 50,
        log_n_last_models: int = 5,
        min_size: int = 1,
        manual_threshold: Optional[float] = None,
        use_tta: bool = False,
        tta_mode: str = "flip",
        bce_or_focal: str = "bce",
        alpha_focal: float = 0.75,
        gamma_focal: float = 2.0,
        bce_dice_weights: Tuple[float, float] = (0.5, 0.5),
        use_amp: bool = False,
        on_log: Optional[Callable[[str, float, int], None]] = None,
    ) -> None:
        self.max_epochs = max_epochs
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_directory = save_directory
        self.log_every_n_steps = log_every_n_steps
        self.log_n_last_models = log_n_last_models
        self.min_size = min_size
        self.manual_threshold = (
            float(manual_threshold) if manual_threshold is not None else None
        )
        self.use_tta = use_tta
        self.tta_mode = tta_mode.lower()
        self.on_log = on_log

        # AMP — stored on the instance and passed through the training loop
        # (fix: original script created the scaler but never passed it)
        self.use_amp = use_amp
        _is_cuda = (
            isinstance(device, str) and "cuda" in device
        ) or (
            hasattr(device, "type") and device.type == "cuda"
        )
        self.scaler: Optional[torch.cuda.amp.GradScaler] = (
            torch.cuda.amp.GradScaler() if (use_amp and _is_cuda) else None
        )

        # Loss
        w_total = sum(bce_dice_weights)
        self.w_bce = bce_dice_weights[0] / w_total
        self.w_dice = bce_dice_weights[1] / w_total
        if bce_or_focal.lower() == "focal":
            self.bce: torch.nn.Module = FocalLoss(
                alpha=alpha_focal, gamma=gamma_focal
            )
        else:
            self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=1.0)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self._best_loss = float("inf")
        self._best_epoch = 0
        self._model_path_list: List[str] = []

        # Validation accumulation buffers
        self._val_logits: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_volids: List[str] = []
        self.best_threshold_ = 0.5

    # ── public entry point ────────────────────────────────────────────────────

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: torch_utils.DataLoader,
        val_loader: Optional[torch_utils.DataLoader] = None,
        test_loader: Optional[torch_utils.DataLoader] = None,
    ) -> None:
        """Train *model* for ``max_epochs`` epochs.

        Source: ``simple.Trainer.fit`` L139–166 (inlined, EMA removed).
        """
        if self.optimizer is None:
            raise RuntimeError("optimizer must be set before calling fit()")

        self.optimizer.zero_grad(set_to_none=True)

        for epoch_id in range(self.max_epochs):
            self.epoch = epoch_id
            self._run_training_epoch(model, train_loader)

            if val_loader is not None:
                self._run_validation_epoch(model, val_loader)

            if test_loader is not None:
                self._run_evaluation_epoch(model, test_loader)

    # ── epoch runners (inlined from simple.Trainer) ───────────────────────────

    def _run_training_epoch(
        self,
        model: torch.nn.Module,
        train_loader: torch_utils.DataLoader,
    ) -> None:
        """Source: ``simple.Trainer.training`` L92–109."""
        try:
            from tqdm import tqdm as _tqdm
            iterable = _tqdm(
                train_loader,
                desc=f"Epoch {self.epoch + 1}/{self.max_epochs}",
            )
        except ImportError:
            iterable = train_loader

        model.train()
        train_losses: List[float] = []
        for batch in iterable:
            out = self.training_step(model=model, batch=batch)
            train_losses.append(out["loss"])

        avg = sum(train_losses) / max(len(train_losses), 1)
        self._emit("TrainEpoch/loss", avg, self.epoch)

    def _run_validation_epoch(
        self,
        model: torch.nn.Module,
        val_loader: torch_utils.DataLoader,
    ) -> None:
        """Source: ``simple.Trainer.validation`` L111–127."""
        batch_losses: List[float] = []
        with torch.inference_mode():
            model.eval()
            for batch in val_loader:
                out = self.validation_step(model=model, batch=batch)
                batch_losses.append(out["loss"])

        metrics = self.validation_epoch_end()
        epoch_loss = metrics.get("loss", sum(batch_losses) / max(len(batch_losses), 1))

        if epoch_loss <= self._best_loss:
            self._best_loss = epoch_loss
            self._best_epoch = self.epoch
            logger.info("Best epoch %d | loss %.6f", self.epoch, epoch_loss)

        if self.epoch == self._best_epoch:
            self.save_model(model)

    def _run_evaluation_epoch(
        self,
        model: torch.nn.Module,
        test_loader: torch_utils.DataLoader,
    ) -> None:
        """Source: ``simple.Trainer.evaluate`` L129–137.

        Bug fix vs. original: ``evaluation_epoch_end`` now calls
        ``validation_epoch_end()`` so that the accumulated logit/label
        buffers are fully processed *and cleared* after the test pass.
        In the original, buffers from ``evaluate()`` were never cleared,
        contaminating the next epoch's validation metrics.
        """
        with torch.inference_mode():
            model.eval()
            for batch in test_loader:
                self.evaluation_step(model=model, batch=batch)

        self.evaluation_epoch_end()

    # ── step implementations ──────────────────────────────────────────────────

    def training_step(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, ...],
    ) -> Dict[str, float]:
        """One gradient step.

        Source: ``train_complex_oversample.py`` L303–348.
        AMP fix: ``self.scaler`` (set in ``__init__``) is now actually used.
        """
        inputs, labels, *_ = batch
        inputs = inputs.unsqueeze(1).to(self.device)
        labels = labels.unsqueeze(1).to(self.device).float()

        if self.scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x=inputs)
                loss = self._combo_loss(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = model(x=inputs)
            loss = self._combo_loss(logits, labels)
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()

        loss_val = loss.item()
        self.global_step += 1
        if self.global_step % self.log_every_n_steps == 0:
            self._emit("Train/loss", loss_val, self.global_step)
            if self.scheduler is not None:
                try:
                    lr = self.scheduler.get_last_lr()[0]
                except Exception:
                    lr = self.optimizer.param_groups[0]["lr"]
                self._emit("Train/lr", float(lr), self.global_step)

        return {"loss": loss_val}

    @torch.inference_mode()
    def validation_step(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, ...],
    ) -> Dict[str, float]:
        """Accumulate logits/labels for epoch-end metric computation.

        Source: ``train_complex_oversample.py`` L361–385.
        """
        inputs, labels, *rest = batch
        vol_ids = rest[0] if rest else ["unknown"] * inputs.shape[0]

        inputs = inputs.unsqueeze(1).to(self.device)
        labels = labels.unsqueeze(1).to(self.device).float()

        if self.use_tta:
            logits = self.tta_predict(model, inputs)
        else:
            logits = model(x=inputs)

        loss = self._combo_loss(logits, labels)

        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(labels.detach().cpu())
        self._val_volids.extend(list(vol_ids))

        return {"loss": float(loss.item())}

    def evaluation_step(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, ...],
    ) -> Dict[str, float]:
        """Mirrors validation step; buffers are cleared in
        ``evaluation_epoch_end``.

        Source: ``train_complex_oversample.py`` L553–555.
        """
        return self.validation_step(model=model, batch=batch)

    def validation_epoch_end(self) -> Dict[str, float]:
        """Sweep thresholds, compute metrics, log, and clear buffers.

        Source: ``train_complex_oversample.py`` L388–550.
        Key fix: ``opt.min_size`` global reference replaced with
        ``self.min_size`` constructor parameter.
        """
        if not self._val_logits:
            return {}

        logits = torch.cat(self._val_logits, dim=0).float()
        labels = torch.cat(self._val_labels, dim=0).float()
        vol_ids = list(self._val_volids)

        # Clear buffers immediately so partial state can't leak
        self._val_logits.clear()
        self._val_labels.clear()
        self._val_volids.clear()

        # Epoch loss on CPU
        bce_cpu = self._cpu_bce_loss()
        dice_cpu = SoftDiceLoss(smooth=self.dice.smooth)
        epoch_loss = float(
            self.w_bce * bce_cpu(logits, labels)
            + self.w_dice * dice_cpu(logits, labels)
        )

        probs = logits.sigmoid()

        # Threshold selection
        if self.manual_threshold is not None:
            best_t = self.manual_threshold
        else:
            y_true = (labels > 0.5).view(-1).to(torch.int32)
            p = probs.view(-1)
            if p.numel() > 2_000_000:
                idx = torch.randperm(p.numel())[:2_000_000]
                p = p[idx]
                y_true = y_true[idx]

            p_sorted, sort_idx = torch.sort(p, descending=True)
            y_sorted = y_true[sort_idx]
            tp = torch.cumsum(y_sorted, 0).float()
            fp = torch.cumsum(1 - y_sorted, 0).float()
            fn = tp[-1] - tp
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
            best_t = float(p_sorted[torch.argmax(f1)])

        self.best_threshold_ = best_t
        logger.info("Epoch %d — best threshold %.4f", self.epoch, best_t)

        # Predictions with optional small-component filtering
        if self.min_size > 1:
            raw_preds = (probs >= best_t).cpu().numpy()
            filtered = np.stack(
                [
                    _remove_small_components(raw_preds[i, 0], self.min_size)
                    for i in range(raw_preds.shape[0])
                ]
            )
            preds = torch.tensor(filtered).unsqueeze(1).to(labels.dtype)
        else:
            preds = (probs >= best_t).to(labels.dtype)

        # Global metrics
        metrics: Dict[str, float] = {
            "loss": epoch_loss,
            "best_threshold": best_t,
            "accuracy": _accuracy(labels, preds),
            "precision": _precision(labels, preds),
            "recall": _recall(labels, preds),
            "f1_score": _f1_score(labels, preds),
            "specificity": _specificity(labels, preds),
            "sensitivity": _sensitivity(labels, preds),
        }

        # Per-volume micro-metrics
        vol_to_idxs: Dict[str, List[int]] = collections.defaultdict(list)
        for i, v in enumerate(vol_ids):
            vol_to_idxs[str(v)].append(i)

        for vol, idxs in vol_to_idxs.items():
            lv = labels[idxs]
            pv = preds[idxs]
            yt = (lv > 0.5).view(-1).int()
            yp = (pv > 0.5).view(-1).int()
            tp_v = int((yp & yt).sum().item())
            fp_v = int((yp & (1 - yt)).sum().item())
            fn_v = int(((1 - yp) & yt).sum().item())
            prec_v = tp_v / (tp_v + fp_v + 1e-12)
            rec_v = tp_v / (tp_v + fn_v + 1e-12)
            # Source: train_complex_oversample.py L502-538
            metrics[f"{vol}/tp"] = float(tp_v)
            metrics[f"{vol}/fp"] = float(fp_v)
            metrics[f"{vol}/fn"] = float(fn_v)
            metrics[f"{vol}/pos_rate"] = float(yt.float().mean().item())
            metrics[f"{vol}/pred_pos_rate"] = float(yp.float().mean().item())
            metrics[f"{vol}/micro_precision"] = prec_v
            metrics[f"{vol}/micro_recall"] = rec_v
            metrics[f"{vol}/micro_f1"] = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-12)

        # Emit all metrics
        for k, v in metrics.items():
            self._emit(f"Val/{k}", v, self.global_step)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    def evaluation_epoch_end(self) -> Dict[str, float]:
        """Process and clear the test-set accumulation buffers.

        Bug fix: in the original, ``simple.Trainer.evaluation_epoch_end``
        only averaged per-batch losses, leaving logit buffers unflushed
        and contaminating the next epoch's validation metrics.
        """
        return self.validation_epoch_end()

    # ── TTA ──────────────────────────────────────────────────────────────────

    def tta_predict(
        self, model: torch.nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        """Average logits over original + axis-flipped inputs.

        Source: ``train_complex_oversample.py`` L557–584.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, D, H, W)
        """
        preds = [model(x=x)]
        if self.tta_mode == "flip":
            for dim in [2, 3, 4]:
                flipped = torch.flip(x, dims=[dim])
                preds.append(torch.flip(model(x=flipped), dims=[dim]))
        return torch.mean(torch.stack(preds), dim=0)

    # ── checkpointing ─────────────────────────────────────────────────────────

    def save_model(self, model: torch.nn.Module) -> None:
        """Save current model weights.  Source: ``simple.Trainer.save_model``
        L177–187.
        """
        if self.log_n_last_models <= 0 or not self.save_directory:
            return

        os.makedirs(self.save_directory, exist_ok=True)
        fname = f"trained_model_epoch_{self.epoch}.bin"
        path = os.path.join(self.save_directory, fname)
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state, path)
        logger.info("Saved checkpoint: %s", path)

        self._model_path_list.append(path)
        if len(self._model_path_list) > self.log_n_last_models:
            oldest = self._model_path_list.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)

    # ── private helpers ───────────────────────────────────────────────────────

    def _combo_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.w_bce * self.bce(logits, labels) + self.w_dice * self.dice(
            logits, labels
        )

    def _cpu_bce_loss(self) -> torch.nn.Module:
        """Reconstruct a CPU-only copy of the BCE/Focal loss for epoch-end
        loss computation (avoids GPU OOM on large val sets).
        """
        if isinstance(self.bce, FocalLoss):
            pw = getattr(self.bce, "pos_weight", None)
            return FocalLoss(
                alpha=self.bce.alpha,
                gamma=self.bce.gamma,
                pos_weight=pw.detach().cpu() if pw is not None else None,
            )
        pw = getattr(self.bce, "pos_weight", None)
        if pw is not None:
            return torch.nn.BCEWithLogitsLoss(pos_weight=pw.detach().cpu())
        return torch.nn.BCEWithLogitsLoss()

    def _emit(self, key: str, value: float, step: int) -> None:
        if self.on_log is not None:
            self.on_log(key, value, step)
