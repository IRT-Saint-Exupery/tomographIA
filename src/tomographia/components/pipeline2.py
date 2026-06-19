"""
Pipeline 2 — connected-component analysis (skimage, default).

This is the production CC path used by tomo_app.  For the legacy cc3d path
see :mod:`tomographia.components.pipeline1`.

Public API:
    :func:`load_grayscale_volume`
    :func:`build_cc_mask`
    :func:`compute_descriptors`
"""
from __future__ import annotations

import multiprocessing as _mp
import os
from multiprocessing.pool import ThreadPool as _ThreadPool
from typing import Callable, Optional

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage.measure import label, marching_cubes, mesh_surface_area, regionprops_table
from skimage.morphology import ball, binary_closing, binary_dilation


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_tif(path: str) -> np.ndarray:
    try:
        return tifffile.imread(path)
    except ValueError as e:
        if "requires the 'imagecodecs' package" in str(e):
            with Image.open(path) as im:
                return np.array(im)
        raise


def load_grayscale_volume(data_path: str) -> np.ndarray:
    """Stack all .tif/.tiff slices in *data_path* into a (Z, Y, X) array."""
    slices = []
    for f in sorted(os.listdir(data_path)):
        if f.lower().endswith((".tif", ".tiff")):
            slices.append(_read_tif(os.path.join(data_path, f)))
    if not slices:
        raise FileNotFoundError(f"No TIFF files found in {data_path}")
    return np.stack(slices, axis=0)


# ---------------------------------------------------------------------------
# Connected-component mask
# ---------------------------------------------------------------------------

def build_cc_mask(
    labeled_volume: np.ndarray,
    bridge_radius: int = 1,
    dilation_radius: int = 1,
) -> np.ndarray:
    """
    Binary mask → morphological closing → dilation → label connected components.

    Parameters
    ----------
    labeled_volume : raw prediction array (any non-zero = defect).
    bridge_radius  : ball radius for ``binary_closing`` (bridges ~1-voxel gaps).
                     Set to 0 to skip.
    dilation_radius: ball radius for ``binary_dilation`` after closing.
                     Set to 0 to skip.

    Returns
    -------
    Integer-labelled CC array from ``skimage.measure.label``.
    """
    mask = labeled_volume > 0
    if bridge_radius > 0:
        mask = binary_closing(mask, ball(bridge_radius))
    if dilation_radius > 0:
        mask = binary_dilation(mask, ball(dilation_radius))
    return label(mask, connectivity=3)


# ---------------------------------------------------------------------------
# Descriptor computation helpers (thread-safe callables)
# ---------------------------------------------------------------------------

class _SurfaceAreaWorker:
    def __init__(self, cc: np.ndarray, voxel_size: tuple) -> None:
        self.cc = cc
        self.voxel_size = voxel_size

    def __call__(self, args: tuple) -> float:
        label_id, z0, z1, y0, y1, x0, x1 = args
        dz, dy, dx = self.voxel_size
        sub = (self.cc[z0:z1, y0:y1, x0:x1] == label_id).astype(np.uint8)
        if sub.sum() == 0:
            return float("nan")
        try:
            verts, faces, _, _ = marching_cubes(sub, level=0.5, spacing=(dz, dy, dx))
            return float(mesh_surface_area(verts, faces))
        except Exception:
            return float("nan")


class _IntensityWorker:
    def __init__(self, cc: np.ndarray, grayscale_volume: np.ndarray) -> None:
        self.cc = cc
        self.gs = grayscale_volume

    def __call__(self, args: tuple) -> tuple:
        label_id, z0, z1, y0, y1, x0, x1 = args
        mask = self.cc[z0:z1, y0:y1, x0:x1] == label_id
        vals = self.gs[z0:z1, y0:y1, x0:x1][mask]
        if vals.size:
            return float(vals.mean()), float(np.median(vals)), float(vals.std())
        return float("nan"), float("nan"), float("nan")


# ---------------------------------------------------------------------------
# compute_descriptors
# ---------------------------------------------------------------------------

def compute_descriptors(
    cc: np.ndarray,
    voxel_size: tuple,
    grayscale_volume: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Compute geometric + intensity descriptors for every connected component.

    Parameters
    ----------
    cc              : integer label array from ``skimage.measure.label``.
    voxel_size      : (dz, dy, dx) physical voxel size (same units, e.g. mm).
    grayscale_volume: optional raw CT volume for intensity stats.
    progress_cb     : optional ``(done: int, total: int) → None`` callback
                      fired after each component's surface area is computed.

    Returns
    -------
    DataFrame with one row per component (background label 0 excluded).

    Notes
    -----
    ``solidity`` (voxels / convex-hull voxels) is intentionally set to NaN —
    computing 3-D convex hulls for thousands of components is prohibitively slow.
    ``fill_frac`` (voxels / bbox_voxels) is a fast proxy.

    The 12 canonical classifier features produced here are::

        sphericity, anisotropy, flatness, elongation, fill_frac,
        r1, r2, ar_zx, ar_zy, ar_yx, int_std_rel, log_voxels
    """
    props = regionprops_table(
        cc,
        properties=(
            "label", "area", "bbox", "centroid", "inertia_tensor_eigvals",
        ),
    )
    df = pd.DataFrame(props).rename(columns={
        "label":                    "label_id",
        "area":                     "voxels",
        "centroid-0": "cz", "centroid-1": "cy", "centroid-2": "cx",
        "bbox-0": "zmin", "bbox-1": "ymin", "bbox-2": "xmin",
        "bbox-3": "zmax", "bbox-4": "ymax", "bbox-5": "xmax",
        "inertia_tensor_eigvals-0": "eig0",
        "inertia_tensor_eigvals-1": "eig1",
        "inertia_tensor_eigvals-2": "eig2",
    })

    df["dz"] = df["zmax"] - df["zmin"]
    df["dy"] = df["ymax"] - df["ymin"]
    df["dx"] = df["xmax"] - df["xmin"]

    dz_mm, dy_mm, dx_mm = voxel_size
    voxel_vol = dz_mm * dy_mm * dx_mm
    df["volume"] = df["voxels"] * voxel_vol

    if progress_cb:
        progress_cb(0, len(df))

    # Surface area via marching cubes (thread-parallel)
    sa_args = [
        (int(r["label_id"]),
         int(r["zmin"]), int(r["zmax"]),
         int(r["ymin"]), int(r["ymax"]),
         int(r["xmin"]), int(r["xmax"]))
        for _, r in df.iterrows()
    ]
    n_workers = min(len(df), max(1, _mp.cpu_count()), 32)
    chunksize = max(1, len(df) // (n_workers * 4))
    sa_func = _SurfaceAreaWorker(cc, voxel_size)
    areas: list = [None] * len(df)
    with _ThreadPool(n_workers) as pool:
        for i, res in enumerate(pool.imap(sa_func, sa_args, chunksize=chunksize)):
            areas[i] = res
            if progress_cb:
                progress_cb(i + 1, len(df))
    df["surface_area"] = areas

    # Sphericity: Ψ = π^(1/3) · (6V)^(2/3) / A
    df["sphericity"] = np.where(
        np.isfinite(df["surface_area"]) & (df["surface_area"] > 0),
        (np.pi ** (1 / 3)) * ((6 * df["volume"]) ** (2 / 3)) / df["surface_area"],
        np.nan,
    )

    # Inertia-tensor shape descriptors (eigvals sorted descending)
    eig = np.sort(df[["eig0", "eig1", "eig2"]].to_numpy())[:, ::-1]
    e0, e1, e2 = eig[:, 0], eig[:, 1], eig[:, 2]
    df["elongation"] = np.where(e1 > 0, e0 / e1, np.nan)
    df["flatness"]   = np.where(e2 > 0, e1 / e2, np.nan)
    df["anisotropy"] = np.where(e0 > 0, 1.0 - e2 / e0, np.nan)
    df["r1"]         = np.where(e0 > 0, e1 / e0, np.nan)  # λ2/λ1 ∈ [0,1]
    df["r2"]         = np.where(e0 > 0, e2 / e0, np.nan)  # λ3/λ1 ∈ [0,1]

    # Bounding-box aspect ratios
    df["ar_zx"] = np.where(df["dx"] > 0, df["dz"] / df["dx"], np.nan)
    df["ar_zy"] = np.where(df["dy"] > 0, df["dz"] / df["dy"], np.nan)
    df["ar_yx"] = np.where(df["dx"] > 0, df["dy"] / df["dx"], np.nan)

    # Intensity statistics (optional, thread-parallel)
    if grayscale_volume is not None:
        int_args = [
            (int(r["label_id"]),
             int(r["zmin"]), int(r["zmax"]),
             int(r["ymin"]), int(r["ymax"]),
             int(r["xmin"]), int(r["xmax"]))
            for _, r in df.iterrows()
        ]
        int_func = _IntensityWorker(cc, grayscale_volume)
        int_results: list = [None] * len(df)
        with _ThreadPool(n_workers) as pool:
            for i, res in enumerate(
                pool.imap(int_func, int_args, chunksize=chunksize)
            ):
                int_results[i] = res
                if progress_cb:
                    progress_cb(i + 1, len(df))
        df["int_mean"]    = [r[0] for r in int_results]
        df["int_median"]  = [r[1] for r in int_results]
        df["int_std"]     = [r[2] for r in int_results]
        df["int_std_rel"] = df["int_std"] / df["int_mean"]

    df["bbox_vox"]   = df["dz"] * df["dy"] * df["dx"]
    df["fill_frac"]  = np.where(df["bbox_vox"] > 0, df["voxels"] / df["bbox_vox"], np.nan)
    df["log_volume"] = np.log10(df["volume"] + 1e-12)
    # log_voxels = log(voxels) + log(voxel_vol) matches the training formula:
    #   np.log(voxels) + np.log(VOXEL_SIZES[i]**3)
    df["log_voxels"] = np.log(df["voxels"]) + np.log(voxel_vol)
    df["solidity"]   = np.nan  # intentionally omitted (3-D convex hull too slow)

    return df
