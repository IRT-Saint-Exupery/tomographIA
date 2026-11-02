"""3-D segmentation inference on a directory of TIF slices."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as torch_utils

from ._dataset import SingleVolumeBig
from ._io import remove_small_components, save_all_inputs, save_all_labels
from .training import flip_tta_logits
from ..models.utils import select_model

logger = logging.getLogger(__name__)

_PREFIXES = ("module.", "_orig_mod.")


def load_segmentation_checkpoint(
    model: torch.nn.Module,
    checkpoint: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Load weights into *model* and return the checkpoint metadata.

    Accepts the checkpoints written by
    :meth:`~tomographia.segmentation.SegmentationTrainer.save_model`
    (``{"state_dict": ..., "threshold": ..., ...}``), a ``{"model": ...}``
    dict, or a bare state dict.  ``module.`` (DataParallel) and ``_orig_mod.``
    (``torch.compile``) prefixes are stripped.  Loading is **strict**: missing
    or unexpected keys and shape mismatches raise.

    Raises
    ------
    FileNotFoundError : *checkpoint* does not exist.
    RuntimeError      : the weights do not match the architecture.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(str(path), map_location=map_location, weights_only=False)
    meta: Dict[str, Any] = {}
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
        meta = {k: v for k, v in obj.items() if k != "state_dict"}
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
        meta = {k: v for k, v in obj.items() if k != "model"}
    else:
        state = obj
    if not isinstance(state, dict):
        raise RuntimeError(f"{path} does not contain a state dict")
    clean = {}
    for k, v in state.items():
        for p in _PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
        clean[k] = v
    model.load_state_dict(clean, strict=True)
    return meta


def _save_heatmap_pngs(prob: np.ndarray, output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for i, slc in enumerate(prob):
        fig, ax = plt.subplots()
        im = ax.imshow(slc, cmap="gray", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="P(defect)")
        ax.set_title(f"Slice {i}")
        fig.savefig(os.path.join(output_dir, f"heatmap_slice_{i}.png"))
        plt.close(fig)


def run_inference(
    input_dir: Union[str, Path],
    checkpoint: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    num_channels: int = 1,
    patch_shape: Optional[Tuple[int, int, int]] = None,
    normalization: Optional[str] = None,
    threshold: Optional[float] = None,
    min_size: Optional[int] = None,
    use_tta: Optional[bool] = None,
    padding: int = 3,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cpu",
    min_file: Optional[int] = None,
    max_file: Optional[int] = None,
    save_heatmap: bool = False,
    save_heatmap_png: bool = False,
    save_inputs: bool = True,
    compile_model: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """
    Run 3-D segmentation inference on a directory of TIF slices.

    Overlapping patches are predicted, their sigmoid **probabilities** are
    averaged where they overlap, and the averaged probability map is
    thresholded once.

    Settings left to ``None`` are read from the checkpoint written by
    :class:`~tomographia.segmentation.SegmentationTrainer` (patch shape,
    normalisation, validation threshold, ``min_size``, flip TTA), so inference
    applies the model the way it was validated.  A value passed explicitly
    that differs from the checkpoint raises a warning.

    Outputs written to *output_dir*:

    ``labeled_volume.npy``   — binary prediction, ``uint8`` (0/1), shape ``(Z, Y, X)``.
    ``predictions_<z>.png``  — per-slice binary masks.
    ``inputs_<z>.png``       — per-slice normalised inputs (if *save_inputs*).
    ``labeled_heatmap.npy``  — ``float32`` probability volume (if *save_heatmap*).
    ``heatmap_slice_<z>.png``— per-slice probability images (if *save_heatmap_png*).

    Parameters
    ----------
    input_dir    : directory containing the input ``.tif`` slices.
    checkpoint   : model weights (see :func:`load_segmentation_checkpoint`).
    output_dir   : directory where outputs are written (created if missing).
    num_channels : number of input channels (1 for greyscale CT).
    patch_shape  : (D, H, W) patch size, multiples of 16.  Default: checkpoint,
                   else ``(16, 64, 64)`` with a warning.
    normalization: one of :data:`tomographia.segmentation.NORMALIZATIONS`.
                   Default: checkpoint, else ``"robust_minmax"`` with a warning.
                   Statistics are computed on the loaded slices
                   (``min_file:max_file``).
    threshold    : probability threshold.  Default: the validation threshold
                   stored in the checkpoint, else 0.5 with a warning.
    min_size     : remove predicted components smaller than this many voxels
                   (26-connectivity, whole volume).  Default: checkpoint, else 1.
    use_tta      : average probabilities over D/H/W flips (4 forward passes).
                   Default: checkpoint, else ``False``.
    padding      : patch overlap; the stride is ``patch_shape - 2 * padding``.
    batch_size   : DataLoader batch size.
    num_workers  : DataLoader worker processes.
    device       : ``"cpu"``, ``"cuda:0"``, ...
    min_file     : first slice index (0-based) to load; slice ``0`` of the
                   outputs is this slice.
    max_file     : one-past-last slice index (``None`` = all).
    save_heatmap : save the probability volume.
    save_heatmap_png : also save one PNG per slice of the probability volume.
    save_inputs  : save the normalised input slices as PNG.
    compile_model: ``torch.compile`` the model.
    progress_cb  : optional ``(batch_done, total_batches) -> None``.

    Returns
    -------
    ``Path`` to *output_dir*.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = select_model(name="unet3d", num_channels=num_channels, num_classes=1)
    meta = load_segmentation_checkpoint(model, checkpoint, map_location="cpu")

    def _setting(name, given, fallback, warn_missing):
        stored = meta.get(name)
        if given is None:
            if stored is not None:
                return stored
            if warn_missing:
                logger.warning("Checkpoint does not store %s; using %r", name, fallback)
            return fallback
        if stored is not None and (tuple(given) if isinstance(given, (list, tuple)) else given) != (
                tuple(stored) if isinstance(stored, (list, tuple)) else stored):
            logger.warning("%s=%r differs from the value the model was trained / validated "
                           "with (%r)", name, given, stored)
        return given

    patch_shape = tuple(int(p) for p in _setting("patch_shape", patch_shape, (16, 64, 64), True))
    normalization = _setting("normalization", normalization, "robust_minmax", True)
    threshold = float(_setting("threshold", threshold, 0.5, True))
    min_size = int(_setting("min_size", min_size, 1, False))
    use_tta = bool(_setting("use_tta", use_tta, False, False))
    from ..models.unet import DIVISOR
    if any(p % DIVISOR for p in patch_shape):
        raise ValueError(f"patch_shape {patch_shape} must be divisible by {DIVISOR}")

    dataset = SingleVolumeBig(
        directories={"volume": (str(input_dir), "")},
        patch_shape=patch_shape,
        normalization_name=normalization,
        padding=padding,
        min_file=min_file,
        max_file=max_file,
    )
    loader = torch_utils.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.to(device)
    model.eval()
    if compile_model:
        model = torch.compile(model)

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    num_patches = len(dataset)
    D, H, W = patch_shape
    memmap_path = str(output_dir / "_prob_patches.dat")
    prob_patches = np.memmap(memmap_path, mode="w+", dtype=np.float32,
                             shape=(num_patches, D, H, W))

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_cuda
        else torch.autocast(device_type="cpu", enabled=False)
    )
    total_batches = len(loader)
    global_idx = 0
    try:
        with torch.inference_mode():
            for batch_idx, inputs in enumerate(loader):
                if inputs.ndim == 4:
                    inputs = inputs.unsqueeze(1)  # (B, 1, D, H, W)
                inputs = inputs.to(device)
                with autocast_ctx:
                    logits = flip_tta_logits(model, inputs) if use_tta else model(inputs)
                prob = torch.sigmoid(logits.float()).cpu()[:, 0]
                b = prob.shape[0]
                prob_patches[global_idx: global_idx + b] = prob.numpy()
                global_idx += b
                if progress_cb:
                    progress_cb(batch_idx + 1, total_batches)
        prob_patches.flush()
        prob_vol = dataset.rebuild_volume_from_patches(prob_patches)
    finally:
        del prob_patches
        try:
            os.remove(memmap_path)
        except OSError:
            pass

    pred_vol = prob_vol >= threshold
    if min_size > 1:
        pred_vol = remove_small_components(pred_vol, min_size)
    pred_vol = pred_vol.astype(np.uint8)

    if save_heatmap:
        np.save(str(output_dir / "labeled_heatmap.npy"), prob_vol)
    if save_heatmap_png:
        _save_heatmap_pngs(prob_vol, str(output_dir))
    save_all_labels(str(output_dir), pred_vol, prefix="predictions")
    if save_inputs:
        save_all_inputs(str(output_dir), dataset.volume)
    logger.info("Segmentation written to %s (threshold %.4f, min_size %d, TTA %s)",
                output_dir, threshold, min_size, use_tta)
    return output_dir
