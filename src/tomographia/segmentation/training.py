"""Segmentation training: SegmentationTrainer, loss functions and sampling utilities."""
from __future__ import annotations

import collections
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as torch_utils

from ._io import remove_small_components

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Private metric helpers
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


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class SoftDiceLoss(torch.nn.Module):
    """Soft Dice loss for binary segmentation (per sample, over spatial dims).

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
    """Sigmoid focal loss (Lin et al., 2017) for binary segmentation.

    ``FL = -α_t (1 - p_t)^γ log(p_t)`` with ``α_t = α`` for positive voxels and
    ``1 - α`` for negative voxels.

    Parameters
    ----------
    alpha : float
        Weight of the positive class in ``[0, 1]``; ``0.5`` weights both
        classes equally.
    gamma : float
        Focusing parameter; ``0`` recovers (α-weighted) BCE.
    pos_weight : torch.Tensor, optional
        Extra positive-class weight passed to ``binary_cross_entropy_with_logits``.
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
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))
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
        alpha_t = torch.where(targets == 1, torch.full_like(probs, self.alpha),
                              torch.full_like(probs, 1.0 - self.alpha))
        loss = alpha_t * (1 - pt) ** self.gamma * bce

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
    """Return a ``WeightedRandomSampler`` that draws a fraction
    ``target_pos_frac`` of patches containing at least one positive voxel.

    Works with any ``torch.utils.data.Dataset`` whose ``__getitem__`` returns
    ``(x, y, ...)`` with ``y`` the label tensor or array.

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

    base = getattr(dataset, "base_dataset", dataset)  # skip augmentation
    pos_mask: List[bool] = []
    for i in range(len(base)):  # type: ignore[arg-type]
        item = base[i]
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
    """Estimate the BCE ``pos_weight`` from the training data.

    Returns ``neg_voxels / pos_voxels`` computed over up to *max_samples*
    items drawn at random (seed 0).  Cap the result externally:
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
    n = len(dataset)  # type: ignore[arg-type]
    idx = np.random.default_rng(0).permutation(n)[:max_samples]
    for i in idx:
        item = dataset[int(i)]
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


def flip_tta_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Logit of the probability averaged over *x* and its D, H and W flips."""
    probs = torch.sigmoid(model(x=x).float())
    for dim in (2, 3, 4):
        probs = probs + torch.flip(torch.sigmoid(model(x=torch.flip(x, dims=[dim])).float()),
                                   dims=[dim])
    p = (probs / 4).clamp(1e-7, 1 - 1e-7)
    return torch.log(p) - torch.log1p(-p)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationTrainer:
    """Training loop for binary 3-D segmentation with UNet-style models.

    Protocol
    --------
    * Every epoch the model is evaluated on the validation loader; the
      probability threshold maximising voxel F1 on validation is selected.
    * The epoch with the lowest validation loss is kept (weights and its
      validation threshold); its checkpoint is written to ``save_directory``.
    * If a ``test_loader`` is given to :meth:`fit`, the test split is
      evaluated **once**, after training, with the best weights and the
      validation threshold (no threshold tuning on test).
    * Metrics are voxel metrics over the patches; voxels shared by
      overlapping patches are counted once when the dataset provides an
      ownership mask (:class:`SegmentationDataset` does).  ``min_size`` is
      applied per patch.

    Parameters
    ----------
    max_epochs : int
        Number of training epochs.
    device : str or torch.device
        Training device, e.g. ``"cuda:0"`` or ``"cpu"``.
    optimizer : torch.optim.Optimizer
        Configured optimiser.
    scheduler : LRScheduler, optional
        LR scheduler; ``step()`` is called after **every gradient step**
        (e.g. ``OneCycleLR``).
    save_directory : str
        Directory where checkpoints are written (``""`` disables it).
    log_every_n_steps : int
        Step interval for ``on_log`` callbacks.
    log_n_last_models : int
        Maximum number of best checkpoints kept on disk.
    min_size : int
        Predicted components smaller than this (voxels, 26-connectivity,
        per patch) are removed before computing metrics; ``1`` disables it.
        Pass the same value to :func:`run_inference`.
    manual_threshold : float, optional
        Fixed probability threshold.  If ``None``, it is selected on
        validation each epoch.
    use_tta : bool
        Average probabilities over flipped inputs during validation / test.
        Stored in the checkpoint; :func:`run_inference` then uses it too, so
        the threshold is applied to the predictor it was selected for.
    tta_mode : str
        ``"flip"`` (D/H/W flips, 4 forward passes) or ``"none"``.
    bce_or_focal : {"bce", "focal"}
        Voxel-wise loss combined with Dice.
    alpha_focal, gamma_focal : float
        Focal-loss parameters (class-balanced ``alpha``; see :class:`FocalLoss`).
    pos_weight : float, optional
        Positive-class weight of the BCE / focal term (e.g. from
        :func:`estimate_pos_weight`).
    bce_dice_weights : (float, float)
        ``(bce_weight, dice_weight)``, normalised to sum to 1.
    use_amp : bool
        CUDA automatic mixed precision for the training forward/backward pass.
        Validation metrics are computed in float32.
    on_log : callable, optional
        ``on_log(key: str, value: float, step: int) -> None`` for every scalar
        metric (MLflow, W&B, TensorBoard...).
    metadata : dict, optional
        Stored in every checkpoint and used by :func:`run_inference` as
        defaults, e.g. ``PatchGenerator.metadata`` (normalisation, patch
        shape).  ``min_size``, ``use_tta`` and ``model_name`` are added
        automatically.
    model_name : str
        Architecture key for :func:`~tomographia.models.select_model`,
        stored in the checkpoint.

    Attributes (after :meth:`fit`)
    ------------------------------
    best_epoch_, best_threshold_, best_val_metrics_, test_metrics_
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
        pos_weight: Optional[float] = None,
        bce_dice_weights: Tuple[float, float] = (0.5, 0.5),
        use_amp: bool = False,
        on_log: Optional[Callable[[str, float, int], None]] = None,
        metadata: Optional[Dict] = None,
        model_name: str = "unet3d",
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
        if self.tta_mode not in ("flip", "none"):
            raise ValueError("tta_mode must be 'flip' or 'none'")
        self.on_log = on_log
        self.metadata = {**(metadata or {}), "model_name": model_name,
                         "min_size": int(min_size),
                         "use_tta": bool(use_tta and self.tta_mode == "flip")}
        missing = [k for k in ("normalization", "patch_shape") if k not in self.metadata]
        if missing:
            logger.warning("metadata lacks %s: checkpoints will not record them and "
                           "run_inference will require them explicitly (pass "
                           "metadata=PatchGenerator.metadata)", missing)

        self.use_amp = use_amp
        _is_cuda = (
            isinstance(device, str) and "cuda" in device
        ) or (
            hasattr(device, "type") and device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda") if (use_amp and _is_cuda) else None

        # Loss
        w_total = sum(bce_dice_weights)
        self.w_bce = bce_dice_weights[0] / w_total
        self.w_dice = bce_dice_weights[1] / w_total
        pw = None if pos_weight is None else torch.as_tensor([float(pos_weight)])
        if bce_or_focal.lower() == "focal":
            self.bce: torch.nn.Module = FocalLoss(
                alpha=alpha_focal, gamma=gamma_focal, pos_weight=pw
            )
        elif bce_or_focal.lower() == "bce":
            self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            raise ValueError("bce_or_focal must be 'bce' or 'focal'")
        self.dice = SoftDiceLoss(smooth=1.0)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self._best_loss = float("inf")
        self.best_epoch_ = -1
        self.best_threshold_: Optional[float] = self.manual_threshold
        self.best_val_metrics_: Dict[str, float] = {}
        self.test_metrics_: Dict[str, float] = {}
        self._best_state: Optional[Dict[str, torch.Tensor]] = None
        self._model_path_list: List[str] = []

        # Accumulation buffers (validation / test)
        self._val_logits: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_volids: List[str] = []
        self._val_masks: List[torch.Tensor] = []

    # ── public entry point ────────────────────────────────────────────────────

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: torch_utils.DataLoader,
        val_loader: Optional[torch_utils.DataLoader] = None,
        test_loader: Optional[torch_utils.DataLoader] = None,
    ) -> Dict[str, float]:
        """Train *model* for ``max_epochs`` epochs.

        Without a validation loader every epoch's weights are checkpointed and
        no threshold is selected.  On return *model* holds the best weights
        (lowest validation loss).

        Returns
        -------
        Test metrics (empty if no ``test_loader``), also in ``test_metrics_``.
        """
        if self.optimizer is None:
            raise RuntimeError("optimizer must be set before calling fit()")
        if test_loader is not None and val_loader is None:
            raise ValueError("a val_loader is required to select the weights and "
                             "threshold evaluated on test_loader")
        for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
            if loader is not None and len(loader) == 0:
                raise ValueError(f"The {name} loader is empty (check the data split)")

        self.optimizer.zero_grad(set_to_none=True)

        for epoch_id in range(self.max_epochs):
            self.epoch = epoch_id
            self._run_training_epoch(model, train_loader)
            if val_loader is not None:
                self._run_validation_epoch(model, val_loader)
            else:
                self.save_model(model)

        if self._best_state is not None:
            model.load_state_dict(self._best_state)
        elif val_loader is not None:
            raise RuntimeError("No epoch had a finite validation loss; no model was selected")
        if test_loader is not None:
            self.test_metrics_ = self._run_evaluation_epoch(model, test_loader)
        return self.test_metrics_

    # ── epoch runners ─────────────────────────────────────────────────────────

    def _run_training_epoch(
        self,
        model: torch.nn.Module,
        train_loader: torch_utils.DataLoader,
    ) -> None:
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
    ) -> Dict[str, float]:
        with torch.inference_mode():
            model.eval()
            for batch in val_loader:
                self.validation_step(model=model, batch=batch)

        metrics = self.validation_epoch_end()
        epoch_loss = metrics.get("loss", float("nan"))
        if not np.isfinite(epoch_loss):
            logger.warning("Epoch %d: non-finite validation loss; epoch not selectable",
                           self.epoch)
        elif epoch_loss <= self._best_loss:
            self._best_loss = epoch_loss
            self.best_epoch_ = self.epoch
            self.best_threshold_ = float(metrics["threshold"])
            self.best_val_metrics_ = dict(metrics)
            self._best_state = {k: v.detach().cpu().clone()
                                for k, v in self._unwrap(model).state_dict().items()}
            logger.info("Best epoch %d | val loss %.6f | threshold %.4f",
                        self.epoch, epoch_loss, self.best_threshold_)
            self.save_model(model)
        return metrics

    def _run_evaluation_epoch(
        self,
        model: torch.nn.Module,
        test_loader: torch_utils.DataLoader,
    ) -> Dict[str, float]:
        with torch.inference_mode():
            model.eval()
            for batch in test_loader:
                self.evaluation_step(model=model, batch=batch)
        return self.evaluation_epoch_end()

    # ── step implementations ──────────────────────────────────────────────────

    def training_step(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, ...],
    ) -> Dict[str, float]:
        """One gradient step."""
        inputs, labels, *_ = batch
        inputs = inputs.unsqueeze(1).to(self.device)
        labels = labels.unsqueeze(1).to(self.device).float()

        if self.scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x=inputs)
                loss = self._combo_loss(logits.float(), labels)
            self.scaler.scale(loss).backward()
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
        """Accumulate logits / labels for the epoch-end metrics."""
        inputs, labels, *rest = batch
        vol_ids = rest[0] if rest else ["unknown"] * inputs.shape[0]
        mask = rest[1] if len(rest) > 1 else torch.ones_like(labels)

        inputs = inputs.unsqueeze(1).to(self.device)
        labels = labels.unsqueeze(1).to(self.device).float()

        if self.use_tta:
            logits = self.tta_predict(model, inputs)
        else:
            logits = model(x=inputs)

        loss = self._combo_loss(logits, labels)

        self._val_logits.append(logits.detach().float().cpu())
        self._val_labels.append(labels.detach().cpu())
        self._val_volids.extend(list(vol_ids))
        self._val_masks.append(torch.as_tensor(mask).unsqueeze(1).bool().cpu())

        return {"loss": float(loss.item())}

    def evaluation_step(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, ...],
    ) -> Dict[str, float]:
        """Same as :meth:`validation_step` (test buffers are processed by
        :meth:`evaluation_epoch_end`)."""
        return self.validation_step(model=model, batch=batch)

    def validation_epoch_end(self) -> Dict[str, float]:
        """Select the threshold on the validation buffers, log ``Val/*`` metrics."""
        threshold = self.manual_threshold  # None -> F1-optimal on validation
        return self._epoch_metrics("Val", threshold)

    def evaluation_epoch_end(self) -> Dict[str, float]:
        """Log ``Test/*`` metrics at the best validation threshold (no tuning on test)."""
        return self._epoch_metrics("Test", self.best_threshold_)

    @staticmethod
    def _f1_optimal(p: torch.Tensor, y_true: torch.Tensor, n_bins: int = 10_000) -> float:
        """F1-optimal threshold (predict positive when ``p >= t``) from per-class
        histograms of all voxels: no subsampling, so rare defects are all used.
        The threshold is a bin edge (resolution ``1 / n_bins``)."""
        b = torch.clamp((p.double() * n_bins).long(), 0, n_bins - 1)
        pos = torch.bincount(b[y_true > 0], minlength=n_bins).double()
        neg = torch.bincount(b[y_true == 0], minlength=n_bins).double()
        tp = torch.flip(torch.cumsum(torch.flip(pos, [0]), 0), [0])   # p >= edge k
        fp = torch.flip(torch.cumsum(torch.flip(neg, [0]), 0), [0])
        fn = pos.sum() - tp
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
        return float(torch.argmax(f1).item()) / n_bins

    def _predict(self, probs: torch.Tensor, threshold: float) -> torch.Tensor:
        """Binary predictions, with per-patch small-component removal."""
        if self.min_size > 1:
            raw = (probs >= threshold).cpu().numpy()
            filtered = np.stack([remove_small_components(raw[i, 0], self.min_size)
                                 for i in range(raw.shape[0])])
            return torch.from_numpy(filtered).unsqueeze(1)
        return probs >= threshold

    def _epoch_metrics(self, prefix: str, threshold: Optional[float]) -> Dict[str, float]:
        if not self._val_logits:
            return {}

        logits = torch.cat(self._val_logits, dim=0).float()
        labels = torch.cat(self._val_labels, dim=0).float()
        owned = torch.cat(self._val_masks, dim=0)
        vol_ids = list(self._val_volids)

        self._val_logits.clear()
        self._val_labels.clear()
        self._val_volids.clear()
        self._val_masks.clear()

        bce_cpu = self._cpu_bce_loss()
        dice_cpu = SoftDiceLoss(smooth=self.dice.smooth)
        epoch_loss = float(
            self.w_bce * bce_cpu(logits, labels)
            + self.w_dice * dice_cpu(logits, labels)
        )

        probs = logits.sigmoid()
        y_all = labels > 0.5

        n_pos = int(y_all[owned].sum())
        if threshold is None and n_pos == 0:
            threshold = self.best_threshold_ if self.best_threshold_ is not None else 0.5
            logger.warning("%s split contains no defect voxel: no threshold can be selected; "
                           "using %.3f", prefix, threshold)
        elif threshold is None:
            if n_pos < 100:
                logger.warning("Threshold selected on only %d defect voxels", n_pos)
            threshold = self._f1_optimal(probs[owned], y_all[owned])
            if self.min_size > 1:
                # Re-select among candidates after small-component removal, so
                # the reported F1 is the quantity that was maximised.
                cands = sorted({threshold, *np.round(np.linspace(0.05, 0.95, 19), 2).tolist()})
                best = (-1.0, threshold)
                for t in cands:
                    pr = self._predict(probs, t)
                    best = max(best, (_f1_score(y_all[owned].float(), pr[owned].float()), t))
                threshold = best[1]

        preds = self._predict(probs, threshold)
        yo, po = y_all[owned].float(), preds[owned].float()
        metrics: Dict[str, float] = {
            "loss": epoch_loss,
            "threshold": float(threshold),
            "n_defect_voxels": float(n_pos),
            "precision": _precision(yo, po),
            "recall": _recall(yo, po),
            "f1_score": _f1_score(yo, po),
            "specificity": _specificity(yo, po),
            "accuracy": _accuracy(yo, po),
        }

        vol_to_idxs: Dict[str, List[int]] = collections.defaultdict(list)
        for i, v in enumerate(vol_ids):
            vol_to_idxs[str(v)].append(i)

        for vol, idxs in vol_to_idxs.items():
            m = owned[idxs]
            yt = y_all[idxs][m].int()
            yp = preds[idxs][m].int()
            tp_v = int((yp & yt).sum().item())
            fp_v = int((yp & (1 - yt)).sum().item())
            fn_v = int(((1 - yp) & yt).sum().item())
            prec_v = tp_v / (tp_v + fp_v + 1e-12)
            rec_v = tp_v / (tp_v + fn_v + 1e-12)
            metrics[f"{vol}/tp"] = float(tp_v)
            metrics[f"{vol}/fp"] = float(fp_v)
            metrics[f"{vol}/fn"] = float(fn_v)
            metrics[f"{vol}/pos_rate"] = float(yt.float().mean().item())
            metrics[f"{vol}/pred_pos_rate"] = float(yp.float().mean().item())
            metrics[f"{vol}/precision"] = prec_v
            metrics[f"{vol}/recall"] = rec_v
            metrics[f"{vol}/f1_score"] = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-12)

        for k, v in metrics.items():
            self._emit(f"{prefix}/{k}", v, self.global_step)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    # ── TTA ──────────────────────────────────────────────────────────────────

    def tta_predict(
        self, model: torch.nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        """Average the **probabilities** over the original and D/H/W-flipped
        inputs (as :func:`run_inference` does); returns the logit of the mean.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, D, H, W)
        """
        return flip_tta_logits(model, x) if self.tta_mode == "flip" else model(x=x)

    # ── checkpointing ─────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model

    def save_model(self, model: torch.nn.Module) -> None:
        """Write ``trained_model_epoch_<N>.bin`` to ``save_directory``.

        The file holds ``state_dict``, the validation ``threshold`` (``None``
        if none was selected), ``epoch``, ``val_metrics`` and :attr:`metadata`;
        :func:`~tomographia.segmentation.run_inference` reads its defaults
        from it.
        """
        if self.log_n_last_models <= 0 or not self.save_directory:
            return

        os.makedirs(self.save_directory, exist_ok=True)
        path = os.path.join(self.save_directory, f"trained_model_epoch_{self.epoch}.bin")
        torch.save({
            "state_dict": self._unwrap(model).state_dict(),
            "threshold": None if self.best_threshold_ is None else float(self.best_threshold_),
            "epoch": int(self.epoch),
            "val_metrics": {k: float(v) for k, v in self.best_val_metrics_.items()},
            **self.metadata,
        }, path)
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
        """CPU copy of the voxel-wise loss for epoch-end computation."""
        pw = getattr(self.bce, "pos_weight", None)
        pw = pw.detach().cpu() if pw is not None else None
        if isinstance(self.bce, FocalLoss):
            return FocalLoss(alpha=self.bce.alpha, gamma=self.bce.gamma, pos_weight=pw)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    def _emit(self, key: str, value: float, step: int) -> None:
        if self.on_log is not None:
            self.on_log(key, value, step)
