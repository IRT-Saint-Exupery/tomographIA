"""
3-D UNet inference on a directory of TIF slices.

Extracted from ``tomographIA.inference.predict_volume_big`` (which only ran
as a ``__main__`` script).  The core logic is now an importable function.

MLflow experiment tracking is intentionally omitted — that is deployment
infrastructure.  Torch-compile is available but off by default.

Subprocess note
---------------
``tomo_app.backend.pipeline.segmentation`` still calls the original script as
a subprocess to capture tqdm progress in real-time via stderr.  That approach
remains valid; :func:`run_inference` provides a direct Python API for callers
that do not need subprocess-level progress streaming.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as torch_utils

from ._dataset import SingleVolumeBig
from ._io import save_all_inputs, save_all_labels
from ..models.utils import select_model


def _check_bit_depth(input_dir: Path) -> Path:
    """
    Convert I;16B (big-endian 16-bit) TIFs to I;16 in a sibling ``_converted``
    directory.  Returns the original directory if no conversion is needed.

    Fixes a bug in the original ``check_mode.py``:
    - Added ``converted_dir.mkdir()`` before saving (was missing).
    - Uses a single ``needs_conversion`` flag instead of overwriting per file.
    """
    from PIL import Image

    converted_dir = Path(str(input_dir) + "_converted")
    needs_conversion = False
    for f in sorted(input_dir.iterdir()):
        if f.suffix.lower() not in (".tif", ".tiff"):
            continue
        try:
            with Image.open(f) as img:
                if img.mode == "I;16B":
                    converted_dir.mkdir(parents=True, exist_ok=True)
                    img.convert("I;16").save(converted_dir / f.name)
                    needs_conversion = True
        except OSError:
            pass
    return converted_dir if needs_conversion else input_dir


def _save_heatmap(logits_vol: np.ndarray, output_dir: str) -> None:
    """Save sigmoid probability map as ``labeled_heatmap.npy`` + per-slice PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    prob = 1.0 / (1.0 + np.exp(-logits_vol.astype(np.float64)))  # sigmoid
    np.save(os.path.join(output_dir, "labeled_heatmap.npy"), prob)
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
    model_name: str = "unet3d",
    num_channels: int = 1,
    num_classes: int = 1,
    patch_shape: Tuple[int, int, int] = (16, 64, 64),
    normalization: str = "robust_minmax",
    threshold: float = 0.5,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cpu",
    min_file: Optional[int] = None,
    max_file: Optional[int] = None,
    save_heatmap: bool = False,
    save_inputs: bool = True,
    image_format: str = "png",
    prefix: str = "slice",
    compile_model: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """
    Run 3-D UNet segmentation inference on a directory of TIF slices.

    Outputs written to *output_dir*:

    ``labeled_volume.npy``      — binary prediction (uint8, same shape as input).
    ``predictions_*.{fmt}``     — per-slice binary masks.
    ``inputs_*.{fmt}``          — per-slice input images (if *save_inputs*).
    ``predicted_patches.dat``   — memmap scratch file (may be deleted after).
    ``labeled_heatmap.npy``     — sigmoid probability volume (only if *save_heatmap*).

    Parameters
    ----------
    input_dir    : directory containing input ``.tif`` slices.
    checkpoint   : path to model weights (``.pt`` / ``.pth`` / ``.bin``).
    output_dir   : directory where outputs are written (created if missing).
    model_name   : architecture key passed to :func:`select_model`.
    num_channels : number of input channels (1 for greyscale CT).
    num_classes  : number of output classes (1 for binary segmentation).
    patch_shape  : (D, H, W) spatial patch size.
    normalization: normalisation applied to the volume before patching.
    threshold    : sigmoid threshold for binarisation.
    batch_size   : DataLoader batch size.
    num_workers  : DataLoader worker processes (0 = main-process loading).
    device       : ``"cpu"`` or ``"cuda:0"`` (or similar).
    min_file     : first TIF slice index to load (0-based).
    max_file     : one-past-last TIF slice index (``None`` = all).
    save_heatmap : also save the continuous sigmoid probability map.
    save_inputs  : also save the normalised input slices as PNG.
    image_format : output image format (``"png"`` or ``"tif"``).
    prefix       : label filename prefix for the training path (inference only).
    compile_model: call ``torch.compile(model)`` for potential GPU speedup.
    progress_cb  : optional ``(batch_done: int, total_batches: int) → None``.

    Returns
    -------
    ``Path`` to *output_dir*.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Big-endian TIF conversion
    input_dir = _check_bit_depth(input_dir)

    # Dataset & DataLoader
    dataset = SingleVolumeBig(
        directories={"volume": (str(input_dir), "")},
        patch_shape=patch_shape,
        normalization_name=normalization,
        prefix=prefix,
        padding=3,
        min_file=min_file,
        max_file=max_file,
    )
    loader = torch_utils.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Model
    # patch_shape[::-1]: the original script passed (W,H,D) to select_model;
    # callers that build the model from (D,H,W) should note this reversal.
    model = select_model(
        name=model_name,
        num_channels=num_channels,
        num_classes=num_classes,
        patch_shape=tuple(patch_shape[::-1]),
    )
    ckpt = Path(checkpoint)
    if ckpt.is_file():
        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    if compile_model:
        model = torch.compile(model)

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Preallocate disk-backed memmaps for patches
    num_patches = len(dataset)
    D, H, W = patch_shape
    memmap_path = str(output_dir / "predicted_patches.dat")
    pred_patches = np.memmap(
        memmap_path, mode="w+", dtype=np.uint8, shape=(num_patches, D, H, W)
    )
    logits_patches = None
    if save_heatmap:
        logits_patches = np.memmap(
            str(output_dir / "logits_patches.dat"),
            mode="w+", dtype=np.float32, shape=(num_patches, D, H, W),
        )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_cuda
        else torch.autocast(device_type="cpu", enabled=False)
    )
    total_batches = len(loader)
    global_idx = 0

    with torch.inference_mode():
        for batch_idx, inputs in enumerate(loader):
            if inputs.ndim == 4:
                inputs = inputs.unsqueeze(1)   # (B, 1, D, H, W)
            inputs = inputs.to(device)

            with autocast_ctx:
                if save_heatmap:
                    logits, predictions = model.predict(
                        x=inputs, threshold=threshold, return_logits=True
                    )
                else:
                    out = model.predict(x=inputs, threshold=threshold, return_logits=False)
                    predictions = out if not isinstance(out, tuple) else out[1]
                    logits = None

            predictions = predictions.detach().cpu()
            if predictions.ndim == 5:
                predictions = predictions.squeeze(1)
            preds_np = (predictions > threshold).numpy().astype(np.uint8)
            b = preds_np.shape[0]
            pred_patches[global_idx : global_idx + b] = preds_np

            if save_heatmap and logits is not None:
                logits = logits.detach().cpu()
                if logits.ndim == 5:
                    logits = logits.squeeze(1)
                logits_patches[global_idx : global_idx + b] = (
                    logits.numpy().astype(np.float32)
                )

            global_idx += b
            del inputs, predictions, logits
            if progress_cb:
                progress_cb(batch_idx + 1, total_batches)

    pred_patches.flush()
    if logits_patches is not None:
        logits_patches.flush()

    # Reopen read-only for volume rebuild
    pred_patches = np.memmap(
        memmap_path, mode="r", dtype=np.uint8, shape=(num_patches, D, H, W)
    )
    pred_vol = dataset.rebuild_volume_from_patches(patches=pred_patches)

    if save_heatmap and logits_patches is not None:
        logits_ro = np.memmap(
            str(output_dir / "logits_patches.dat"),
            mode="r", dtype=np.float32, shape=(num_patches, D, H, W),
        )
        logits_vol = dataset.rebuild_volume_from_patches(patches=logits_ro)
        _save_heatmap(logits_vol, str(output_dir))

    save_all_labels(
        str(output_dir), pred_vol,
        prefix="predictions", image_format=image_format,
    )
    if save_inputs:
        save_all_inputs(str(output_dir), dataset.volume, image_format=image_format)

    return output_dir
