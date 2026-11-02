"""
Connected-component analysis and shape descriptors (scikit-image / SciPy).

Public API:
    :func:`load_grayscale_volume`
    :func:`build_cc_mask`
    :func:`compute_descriptors`

Descriptor definitions
----------------------
All geometric descriptors are computed in **physical units**: the voxel size
is given with its unit and every length, area and volume is stored in
**millimetres** (mm, mm², mm³), so tables from different scans can be pooled.

Principal axes
    ``C`` is the covariance matrix of the component, treated as a union of
    voxel boxes: the covariance of the voxel centres plus each voxel's own
    variance ``diag(dz², dy², dx²) / 12``.  Its eigenvalues
    ``λ1 ≥ λ2 ≥ λ3`` give the semi-axes of the ellipsoid with the same second
    moments, ``axis_major ≥ axis_mid ≥ axis_minor = sqrt(5 λ)``.

``elongation = axis_mid / axis_major``  ∈ (0, 1]
    IBSI-style definition (Zwanenburg et al., *Radiology* 2020): 1 for a body
    with no dominant axis, → 0 for a needle.  Unlike IBSI, the voxel-box
    variance is included, so a single voxel gets 1 instead of an undefined
    value; for components of a few voxels shape descriptors are dominated by
    discretisation.
``flatness = axis_minor / axis_major``  ∈ (0, 1]
    IBSI-style definition: 1 for a sphere, → 0 for a plate or a needle.
``sphericity``  ∈ (0, 1]
    ``π^(1/3) (6 V)^(2/3) / A`` with the area ``A`` **and** the enclosed
    volume ``V`` of the same marching-cubes mesh, so that the isoperimetric
    inequality guarantees ``≤ 1``.  The mesh of a binary mask over-estimates
    the area, so values are biased low and depend on size: a voxelised sphere
    of radius 20 voxels gives about 0.92, a 10-voxel cube 0.85 (exact 0.81).
``fill_frac``  ∈ (0, 1]
    voxels / bounding-box voxels.
``ar_zx``, ``ar_zy``, ``ar_yx``
    Ratios of the physical bounding-box extents along the scan axes.  They
    depend on how the part is oriented in the scanner (e.g. layer direction),
    unlike the rotation-invariant descriptors above.
``log_volume``
    ``log10`` of the physical volume in mm³ (voxel count × voxel volume).
``int_std_rel``
    Coefficient of variation ``std / mean`` of the raw grey values inside the
    component (needs non-negative raw intensities; NaN when the mean is ≤ 0).
    It depends on the scanner and reconstruction settings (and is not
    invariant to a grey-value offset): pool it across scans only if they are
    comparable.

Resolution
----------
Sizes in mm are comparable across voxel sizes; **shapes** of features only a
few voxels thick are not (``fill_frac``, ``flatness`` and ``sphericity`` of a
thin crack change with the voxel size).  Pool scans of similar resolution, or
restrict to components spanning enough voxels.
"""
from __future__ import annotations

import multiprocessing as _mp
import warnings
from multiprocessing.pool import ThreadPool as _ThreadPool
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import label, marching_cubes, mesh_surface_area, regionprops_table
from skimage.morphology import ball

from .._slices import list_slices, read_volume

#: Shape and intensity features produced by :func:`compute_descriptors` and used
#: by default by the classifiers.
CANONICAL_FEATURES = [
    "sphericity", "elongation", "flatness", "fill_frac",
    "ar_zx", "ar_zy", "ar_yx", "int_std_rel", "log_volume",
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_grayscale_volume(data_path: str) -> np.ndarray:
    """Stack the ``.tif``/``.tiff`` slices of *data_path* (natural order) into ``(Z, Y, X)``."""
    paths = list_slices(data_path)
    if not paths:
        raise FileNotFoundError(f"No TIFF files found in {data_path}")
    return read_volume(paths)


# ---------------------------------------------------------------------------
# Connected-component mask
# ---------------------------------------------------------------------------

def _morph(mask: np.ndarray, radius: int, op) -> np.ndarray:
    """Apply a ball-shaped binary morphology with zero padding (no border effect)."""
    padded = np.pad(mask, radius)
    out = op(padded, structure=ball(radius))
    return out[(slice(radius, -radius),) * mask.ndim]


def build_cc_mask(
    labeled_volume: np.ndarray,
    bridge_radius: int = 1,
    dilation_radius: int = 0,
    min_size: int = 1,
) -> np.ndarray:
    """
    Label the connected components (defects) of a binary prediction.

    Closing (and optionally dilation) only decides **which voxels belong to
    the same defect**: fragments separated by gaps narrower than the bridge
    are grouped.  The returned labels cover exactly the predicted voxels, so
    sizes and shapes are measured on the segmentation, not on the
    morphologically grown mask.

    Parameters
    ----------
    labeled_volume : prediction array (any non-zero voxel = defect).
    bridge_radius  : ball radius (in voxels) of the binary closing used for
                     grouping (0 disables it); with anisotropic voxels the
                     grouping distance is anisotropic too.
    dilation_radius: ball radius of an additional dilation used for grouping
                     (0, the default, disables it).  Larger values merge
                     defects further apart.
    min_size       : components with fewer predicted voxels are discarded.

    Returns
    -------
    ``int32`` label array (26-connectivity), 0 = background, labels ``1..N``.
    """
    mask = np.asarray(labeled_volume) > 0
    group = mask
    if bridge_radius > 0:
        group = _morph(group, bridge_radius, ndimage.binary_closing) | mask
    if dilation_radius > 0:
        group = _morph(group, dilation_radius, ndimage.binary_dilation)
    lab = label(group, connectivity=3)
    lab[~mask] = 0
    if min_size > 1:
        counts = np.bincount(lab.ravel())
        small = counts < min_size
        small[0] = False
        lab[small[lab]] = 0
    # Consecutive labels 1..N
    lab, _, _ = _relabel(lab)
    return lab.astype(np.int32, copy=False)


def _relabel(lab: np.ndarray):
    from skimage.segmentation import relabel_sequential
    return relabel_sequential(lab)


# ---------------------------------------------------------------------------
# Descriptor helpers (thread-safe callables)
# ---------------------------------------------------------------------------

class _MeshWorker:
    """Marching-cubes surface area and enclosed volume of one component."""

    def __init__(self, cc: np.ndarray, voxel_size: tuple) -> None:
        self.cc = cc
        self.voxel_size = voxel_size

    def __call__(self, args: tuple) -> tuple:
        label_id, z0, z1, y0, y1, x0, x1 = args
        sub = (self.cc[z0:z1, y0:y1, x0:x1] == label_id).astype(np.uint8)
        if sub.sum() == 0:
            return float("nan"), float("nan")
        # Zero-pad so that the mesh is closed on the bounding-box faces.
        sub = np.pad(sub, 1)
        try:
            verts, faces, _, _ = marching_cubes(sub, level=0.5, spacing=self.voxel_size)
        except (RuntimeError, ValueError):
            return float("nan"), float("nan")
        area = float(mesh_surface_area(verts, faces))
        tri = verts[faces]
        vol = float(abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)
        return area, vol


class _IntensityWorker:
    def __init__(self, cc: np.ndarray, grayscale_volume: np.ndarray) -> None:
        self.cc = cc
        self.gs = grayscale_volume

    def __call__(self, args: tuple) -> tuple:
        label_id, z0, z1, y0, y1, x0, x1 = args
        mask = self.cc[z0:z1, y0:y1, x0:x1] == label_id
        vals = self.gs[z0:z1, y0:y1, x0:x1][mask].astype(np.float64)
        if vals.size:
            return float(vals.mean()), float(np.median(vals)), float(vals.std())
        return float("nan"), float("nan"), float("nan")


_COLUMNS = [
    "label_id", "voxels", "volume", "cz", "cy", "cx",
    "zmin", "ymin", "xmin", "zmax", "ymax", "xmax", "ext_z", "ext_y", "ext_x", "touches_border",
    "axis_major", "axis_mid", "axis_minor", "surface_area", "mesh_volume",
] + CANONICAL_FEATURES + ["int_mean", "int_median", "int_std"]


# ---------------------------------------------------------------------------
# compute_descriptors
# ---------------------------------------------------------------------------

#: Length units accepted by :func:`compute_descriptors`, in millimetres.
LENGTH_UNITS = {"mm": 1.0, "um": 1e-3, "µm": 1e-3, "μm": 1e-3, "nm": 1e-6, "cm": 10.0, "m": 1e3}


def compute_descriptors(
    cc: np.ndarray,
    voxel_size: Union[float, Sequence[float]],
    *,
    unit: str,
    grayscale_volume: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Geometric and intensity descriptors of every connected component.

    Parameters
    ----------
    cc              : integer label array (e.g. from :func:`build_cc_mask`).
    voxel_size      : voxel size along ``(z, y, x)``, or one number for
                      isotropic voxels, in *unit*.
    unit            : length unit of *voxel_size*: ``"mm"``, ``"um"`` /
                      ``"µm"``, ``"nm"``, ``"cm"`` or ``"m"``.  Outputs are
                      always in mm.
    grayscale_volume: optional raw CT volume (same shape) for intensity
                      statistics; without it ``int_*`` columns are NaN.
    progress_cb     : optional ``(done, total) -> None`` callback.

    Returns
    -------
    One row per component.  Columns: ``label_id``, ``voxels``, ``volume``
    (mm³), centroid ``cz, cy, cx`` and bounding box ``zmin..xmax`` (voxel
    indices, **max exclusive**), bounding-box extents ``ext_z, ext_y, ext_x``
    (voxels), ``touches_border`` (the component reaches a face of the
    volume, so it may be truncated and its size and shape under-estimated), equivalent-ellipsoid semi-axes ``axis_major/mid/minor`` (mm),
    ``surface_area`` (mm²) and ``mesh_volume`` (mm³), the
    :data:`CANONICAL_FEATURES` (see the module docstring for definitions)
    and ``int_mean``, ``int_median``, ``int_std``.
    """
    if unit not in LENGTH_UNITS:
        raise ValueError(f"unit must be one of {sorted(LENGTH_UNITS)}, got {unit!r}")
    vs = np.broadcast_to(np.asarray(voxel_size, dtype=float), (3,)) * LENGTH_UNITS[unit]
    if np.any(vs <= 0):
        raise ValueError("voxel_size must be positive")
    sz, sy, sx = (float(v) for v in vs)
    cc = np.asarray(cc)
    if grayscale_volume is not None and grayscale_volume.shape != cc.shape:
        raise ValueError(f"grayscale_volume shape {grayscale_volume.shape} != cc shape {cc.shape}")

    props = regionprops_table(
        cc, properties=("label", "area", "bbox", "centroid", "inertia_tensor"),
    )
    if len(props["label"]) == 0:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in _COLUMNS})

    df = pd.DataFrame({
        "label_id": props["label"], "voxels": props["area"].astype(np.int64),
        "cz": props["centroid-0"], "cy": props["centroid-1"], "cx": props["centroid-2"],
        "zmin": props["bbox-0"], "ymin": props["bbox-1"], "xmin": props["bbox-2"],
        "zmax": props["bbox-3"], "ymax": props["bbox-4"], "xmax": props["bbox-5"],
    })
    Z, Y, X = cc.shape
    df["touches_border"] = ((df["zmin"] == 0) | (df["ymin"] == 0) | (df["xmin"] == 0)
                            | (df["zmax"] == Z) | (df["ymax"] == Y) | (df["xmax"] == X))
    df["ext_z"] = df["zmax"] - df["zmin"]
    df["ext_y"] = df["ymax"] - df["ymin"]
    df["ext_x"] = df["xmax"] - df["xmin"]
    df["volume"] = df["voxels"] * (sz * sy * sx)

    # Principal axes: covariance of the voxel boxes in physical units.
    inertia = np.stack([[props[f"inertia_tensor-{i}-{j}"] for j in range(3)]
                        for i in range(3)]).transpose(2, 0, 1)            # [N, 3, 3] voxel units
    cov = np.trace(inertia, axis1=1, axis2=2)[:, None, None] / 2 * np.eye(3) - inertia
    S = np.diag(vs)
    cov = S @ cov @ S + np.diag(vs ** 2 / 12.0)
    lam = np.clip(np.linalg.eigvalsh(cov)[:, ::-1], 0.0, None)             # λ1 ≥ λ2 ≥ λ3
    axes = np.sqrt(5.0 * lam)
    df["axis_major"], df["axis_mid"], df["axis_minor"] = axes[:, 0], axes[:, 1], axes[:, 2]
    df["elongation"] = axes[:, 1] / axes[:, 0]
    df["flatness"] = axes[:, 2] / axes[:, 0]

    # Bounding box
    df["fill_frac"] = df["voxels"] / (df["ext_z"] * df["ext_y"] * df["ext_x"])
    ez, ey, ex = df["ext_z"] * sz, df["ext_y"] * sy, df["ext_x"] * sx
    df["ar_zx"] = ez / ex
    df["ar_zy"] = ez / ey
    df["ar_yx"] = ey / ex
    df["log_volume"] = np.log10(df["volume"])

    # Surface area / sphericity (thread-parallel marching cubes)
    if progress_cb:
        progress_cb(0, len(df))
    bbox_args = [tuple(int(v) for v in r) for r in
                 df[["label_id", "zmin", "zmax", "ymin", "ymax", "xmin", "xmax"]].to_numpy()]
    n_workers = min(len(df), max(1, _mp.cpu_count()), 32)
    chunksize = max(1, len(df) // (n_workers * 4))
    mesh_func = _MeshWorker(cc, (sz, sy, sx))
    mesh = [None] * len(df)
    with _ThreadPool(n_workers) as pool:
        for i, res in enumerate(pool.imap(mesh_func, bbox_args, chunksize=chunksize)):
            mesh[i] = res
            if progress_cb:
                progress_cb(i + 1, len(df))
    df["surface_area"] = [m[0] for m in mesh]
    df["mesh_volume"] = [m[1] for m in mesh]
    area, mvol = df["surface_area"].to_numpy(float), df["mesh_volume"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["sphericity"] = np.where((area > 0) & (mvol > 0),
                                    np.pi ** (1 / 3) * (6 * mvol) ** (2 / 3) / area, np.nan)

    # Intensity statistics
    if grayscale_volume is not None:
        if np.issubdtype(grayscale_volume.dtype, np.floating) or np.issubdtype(
                grayscale_volume.dtype, np.signedinteger):
            if float(np.min(grayscale_volume)) < 0:
                warnings.warn("grayscale_volume has negative values: int_std_rel (std / mean) "
                              "assumes raw, non-negative CT intensities", stacklevel=2)
        int_func = _IntensityWorker(cc, grayscale_volume)
        res = [None] * len(df)
        with _ThreadPool(n_workers) as pool:
            for i, r in enumerate(pool.imap(int_func, bbox_args, chunksize=chunksize)):
                res[i] = r
        df["int_mean"] = [r[0] for r in res]
        df["int_median"] = [r[1] for r in res]
        df["int_std"] = [r[2] for r in res]
        mean = df["int_mean"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            df["int_std_rel"] = np.where(mean > 0, df["int_std"].to_numpy(float) / mean, np.nan)
    else:
        for c in ("int_mean", "int_median", "int_std", "int_std_rel"):
            df[c] = np.nan

    return df[_COLUMNS]
