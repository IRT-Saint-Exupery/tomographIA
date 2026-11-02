"""
PDF defect report: yield table and views of the largest defects.

Requires the ``report`` extra (``fpdf2``, ``matplotlib``).
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._slices import list_slices, read_slice

_BBOX = ("zmin", "zmax", "ymin", "ymax", "xmin", "xmax")


def _save_fig(fig) -> str:
    from matplotlib import pyplot as plt
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        name = tmp.name
    fig.savefig(name, bbox_inches="tight")
    plt.close(fig)
    return name


def generate_pdf_report(
    yield_report: dict,
    defects: pd.DataFrame,
    data_path: str,
    save_path: str,
    title: str,
    voxel_size: Union[float, Sequence[float]] = 1.0,
    unit: str = "µm",
    heatmap: Optional[np.ndarray] = None,
    classified_vol: Optional[np.ndarray] = None,
    n_largest: int = 10,
    slice_start: int = 0,
    ref_value: float = 0.0,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    logo_dir: Optional[str] = None,
) -> None:
    """
    Write a PDF report with the yield summary and the largest defects.

    For each of the *n_largest* defects (by voxel count) the report shows its
    class, its bounding-box dimensions, orthogonal mid-planes of the
    probability map (if *heatmap* is given) and its central slice with the
    matrix / pore / crack overlay.

    Parameters
    ----------
    yield_report  : output of :func:`compute_yield_report` (percentages).
    defects       : component table from ``compute_descriptors``, optionally
                    classified (``cls_pred``: 0 = pore, 1 = crack).  Bounding
                    boxes ``zmin..xmax`` are voxel indices with exclusive max.
    data_path     : directory of the raw slices (natural order).
    save_path     : output ``.pdf`` path.
    title         : volume name shown in the header.
    voxel_size    : ``(dz, dy, dx)`` or isotropic voxel size, in *unit*.
    unit          : length unit for the dimensions.
    heatmap       : optional probability volume ``(Z, Y, X)`` from
                    ``run_inference(save_heatmap=True)``.
    classified_vol: optional ``uint8`` volume (0 = background, 1 = pore,
                    2 = crack) from ``build_classified_volume``.
    n_largest     : number of defects shown.
    slice_start   : index in *data_path* of the first slice of the analysed
                    volume (``min_file`` of the segmentation).
    ref_value, lower, upper : matrix intensity window of the overlay; pass
                    the values used for :func:`compute_yield_report`.
                    ``ref_value=0`` (default) draws no matrix window (every
                    non-defect pixel is drawn as matrix).
    logo_dir      : optional directory with ``logo_IRT Saint Exupery_RVB.png``
                    and ``logo_tomoIA.png``.
    """
    import fpdf
    from fpdf.enums import XPos, YPos
    from matplotlib import pyplot as plt

    if ref_value > 0 and (lower is None or upper is None):
        raise ValueError("pass lower and upper with ref_value (same as compute_yield_report)")
    missing = [c for c in ("voxels",) + _BBOX if c not in defects.columns]
    if missing:
        raise KeyError(f"defects lacks columns {missing} (use compute_descriptors output)")
    vs = np.broadcast_to(np.asarray(voxel_size, dtype=float), (3,))
    slice_files = list_slices(data_path)
    nl = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf = fpdf.FPDF()
    pdf.add_page()
    if logo_dir:
        logo_irt = os.path.join(logo_dir, "logo_IRT Saint Exupery_RVB.png")
        logo_tomo = os.path.join(logo_dir, "logo_tomoIA.png")
        if os.path.isfile(logo_irt):
            pdf.image(logo_irt, x=155, y=20, w=40)
        if os.path.isfile(logo_tomo):
            pdf.image(logo_tomo, x=10, y=20, w=70)
        pdf.ln(40)
    else:
        pdf.ln(10)

    pdf.set_font("helvetica", style="B", size=14)
    safe_title = str(title).encode("latin-1", "replace").decode("latin-1")
    pdf.cell(0, 10, text=f"Defect report: {safe_title}", align="C", **nl)
    pdf.ln(5)
    pdf.set_font("helvetica", size=10)

    table_w = 90
    x_pos = (210 - table_w) / 2
    pdf.set_x(x_pos)
    pdf.cell(40, 10, text="Type", border=1)
    pdf.cell(50, 10, text="Defect fraction (%)", border=1, **nl)
    for label, key in (("All defects", "total_yield"), ("Pores", "pore_yield"),
                       ("Cracks", "crack_yield")):
        pdf.set_x(x_pos)
        pdf.cell(40, 10, text=label, border=1)
        pdf.cell(50, 10, text=f"{float(yield_report.get(key, float('nan'))):.4f}", border=1, **nl)

    pdf.ln(10)
    pdf.set_font("helvetica", size=14)
    pdf.cell(0, 10, text=f"{min(n_largest, len(defects))} largest defects", align="C", **nl)
    pdf.ln(5)
    pdf.set_font("helvetica", size=12)

    for _, item in defects.nlargest(n_largest, "voxels").iterrows():
        z0, z1, y0, y1, x0, x1 = (int(item[c]) for c in _BBOX)
        kind = "Defect"
        if "cls_pred" in item.index and pd.notna(item["cls_pred"]):
            kind = ["Pore", "Crack"][int(item["cls_pred"])]
        dims = [round(n * s, 1) for n, s in zip((x1 - x0, y1 - y0, z1 - z0), vs[::-1])]
        pdf.cell(0, 10, text=f"{kind} - label {int(item.get('label_id', -1))}", **nl)
        pdf.cell(0, 10, text=f"Bounding box (x, y, z) = {dims} {unit}", **nl)

        if heatmap is not None:
            hm = heatmap[z0:z1, y0:y1, x0:x1]
            fig, axs = plt.subplots(1, 3, figsize=(8, 3))
            axs[0].imshow(hm[:, :, hm.shape[2] // 2], vmin=0, vmax=1)
            axs[1].imshow(hm[:, hm.shape[1] // 2, :], vmin=0, vmax=1)
            axs[2].imshow(hm[hm.shape[0] // 2, :, :], vmin=0, vmax=1)
            for ax, name in zip(axs, ("z-y plane", "z-x plane", "y-x plane")):
                ax.set_title(name)
            tmp = _save_fig(fig)
            pdf.image(tmp, w=180)
            os.unlink(tmp)

        mid_z = (z0 + z1 - 1) // 2
        file_z = slice_start + mid_z
        if 0 <= file_z < len(slice_files):
            arr = read_slice(slice_files[file_z]).astype(np.float32)
            if ref_value > 0:
                in_range = (arr >= ref_value * (1.0 - lower)) & (arr <= ref_value * (1.0 + upper))
            else:
                in_range = np.ones(arr.shape, dtype=bool)
            if classified_vol is not None and 0 <= mid_z < classified_vol.shape[0]:
                cls_slc = np.asarray(classified_vol[mid_z])
            else:
                cls_slc = np.zeros(arr.shape, dtype=np.uint8)
            rgba = np.full((*arr.shape, 4), [15, 15, 15, 255], dtype=np.uint8)
            rgba[in_range & (cls_slc == 0)] = [0, 210, 77, 255]   # matrix
            rgba[cls_slc == 1] = [33, 150, 243, 255]              # pore
            rgba[cls_slc == 2] = [244, 67, 54, 255]               # crack

            fig2, axs2 = plt.subplots(1, 2, figsize=(8, 4))
            axs2[0].imshow(rgba)
            axs2[0].set_title(os.path.basename(slice_files[file_z]))
            axs2[1].imshow(rgba[y0:y1, x0:x1])
            axs2[1].set_title("Zoom on defect")
            tmp2 = _save_fig(fig2)
            pdf.image(tmp2, w=180)
            os.unlink(tmp2)
        pdf.ln(10)

    pdf.output(save_path)
