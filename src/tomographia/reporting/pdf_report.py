"""
Legacy PDF defect report (Pipeline 1 / old workflow).

Ported from ``tomographIA.evaluation_tools.report``.

Fixes applied
-------------
* ``report_generation()`` previously referenced ``opt`` (a global CLI
  argparse namespace) inside the function body (line 157 in the original),
  causing a ``NameError`` when called programmatically.  The ``data_path``
  and ``scale`` values are now proper parameters.
* Logo paths are made optional (``logo_dir=None`` skips logo insertion).

Requires: ``fpdf2``  (``pip install fpdf2``)
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd


def get_defect_contribution(
    reduced_limits: dict, report_df: pd.DataFrame
) -> pd.Series:
    """
    Compute each defect's fractional spatial contribution within *reduced_limits*.

    Parameters
    ----------
    reduced_limits : ``{"x": (lo, hi), "y": (lo, hi), "z": (lo, hi)}``
                     pixel bounds defining the region of interest.
    report_df      : DataFrame with ``bbox_x_min`` / ``bbox_x_max`` columns.

    Returns
    -------
    pd.Series of contribution fractions in [0, 1].
    """
    df = report_df.copy()
    for coord in ("x", "y", "z"):
        lo, hi = reduced_limits[coord]
        df[f"cont_{coord}"] = df.apply(
            lambda r, c=coord, lo=lo, hi=hi: max(
                0.0,
                (min(r[f"bbox_{c}_max"] + 1, hi) - max(r[f"bbox_{c}_min"], lo))
                / (r[f"bbox_{c}_max"] + 1 - r[f"bbox_{c}_min"]),
            ),
            axis=1,
        )
    df["contribution"] = df["cont_x"] * df["cont_y"] * df["cont_z"]
    return df["contribution"]


def generate_pdf_report(
    yield_dict: dict,
    largest_df: pd.DataFrame,
    heatmap: np.ndarray,
    data_path: str,
    save_path: str,
    filename: str,
    scale: float = 14.0,
    logo_dir: Optional[str] = None,
    classified_vol: Optional[np.ndarray] = None,
    ref_value: float = 0.0,
    lower: float = 0.4,
    upper: float = 0.2,
    slice_start: int = 0,
) -> None:
    """
    Write a PDF defect report with yield summary and orthogonal-projection overlays.

    Parameters
    ----------
    yield_dict  : Yield summary with exactly these keys (values are fractions,
                  range 0–1, NOT percentages)::

                      {
                          "defect_yield"   : float,  # total defect fraction
                          "porosity_yield" : float,  # pore fraction
                          "crack_yield"    : float,  # crack fraction
                      }

                  ``compute_yield_report()`` uses different key names.  Remap
                  before calling::

                      yr = compute_yield_report(...)
                      yield_dict = {
                          "defect_yield"   : yr["total_yield"] / 100,
                          "porosity_yield" : yr["pore_yield"]  / 100,
                          "crack_yield"    : yr["crack_yield"] / 100,
                      }

    largest_df  : DataFrame of the N largest defects to annotate (typically
                  ``df_classified.nlargest(10, "voxels")``).  Required
                  columns — note the naming differs from ``compute_descriptors``
                  output (``zmin``/``zmax``) and must be renamed::

                      bbox_z_min, bbox_z_max  (int, voxel index)
                      bbox_y_min, bbox_y_max  (int)
                      bbox_x_min, bbox_x_max  (int)
                      crack                   (int: 0=pore, 1=crack)

                  Renaming from Pipeline 2 output::

                      df_top = df_classified.nlargest(10, "voxels").rename(columns={
                          "zmin": "bbox_z_min", "zmax": "bbox_z_max",
                          "ymin": "bbox_y_min", "ymax": "bbox_y_max",
                          "xmin": "bbox_x_min", "xmax": "bbox_x_max",
                      })
                      df_top["crack"] = df_top["cls_pred"]

    heatmap     : 3-D float32 array (Z, Y, X), values in [0, 1].  Load from
                  the probability volume saved by ``run_inference(save_heatmap=True)``::

                      heatmap = np.load("outputs/labeled_heatmap.npy")

    data_path   : directory containing raw greyscale TIF or PNG slices, used
                  for orthogonal-projection background images.
    save_path   : output PDF file path (``*.pdf``).
    filename    : volume name shown in the report header.
    scale          : µm per voxel, used for bounding-box dimension annotation.
                     Default 14 (14 µm).
    logo_dir       : directory containing ``logo_IRT Saint Exupery_noir_RVB.png``
                     and ``logo_tomoIA.png``.  ``None`` → logos omitted.
    classified_vol : optional uint8 volume (Z, Y, X) with 0=background,
                     1=pore, 2=crack.  When provided, per-defect slice panels
                     show blue/red defect pixels.
    ref_value      : reference grey value for the matrix intensity window.
                     0 (default) → all in-range pixels treated as matrix.
    lower          : fractional lower half-width below *ref_value* (e.g. 0.4).
    upper          : fractional upper half-width above *ref_value* (e.g. 0.2).
    slice_start    : absolute TIF index corresponding to ``classified_vol[0]``.
    """
    import fpdf
    from matplotlib import pyplot as plt

    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    if logo_dir:
        logo_irt  = os.path.join(logo_dir, "logo_IRT Saint Exupery_noir_RVB.png")
        logo_tomo = os.path.join(logo_dir, "logo_tomoIA.png")
        pdf.ln(40)
        if os.path.isfile(logo_irt):
            pdf.image(logo_irt, x=155, y=20, w=40)
        if os.path.isfile(logo_tomo):
            pdf.image(logo_tomo, x=10, y=20, w=70)
    else:
        pdf.ln(10)

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, txt=f"Defect report on {filename}", align="C", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=10)

    table_w = 90
    x_pos = (210 - table_w) / 2
    pdf.set_xy(x_pos, pdf.get_y())
    pdf.cell(40, 10, txt="Type", border=1)
    pdf.cell(50, 10, txt="Value (%)", border=1)
    pdf.ln()

    for label_key in ("defect", "porosity", "crack"):
        pdf.set_xy(x_pos, pdf.get_y())
        pdf.cell(40, 10, txt=f"{label_key.capitalize()} Yield", border=1)
        value = yield_dict.get(f"{label_key}_yield", 0.0)
        pdf.cell(50, 10, txt=f"{value * 100:.4f}", border=1)
        pdf.ln()

    pdf.ln(10)
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, txt="Ten largest defects", align="C", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)

    slice_files = sorted(f for f in os.listdir(data_path) if not f.startswith("."))

    for i in range(len(largest_df)):
        item = largest_df.iloc[i]

        pdf.cell(0, 10, txt=["Pore", "Crack"][int(item["crack"])], ln=True)
        dims = [
            round((item[f"bbox_{c}_max"] - item[f"bbox_{c}_min"] + 1) * scale, 1)
            for c in ("x", "y", "z")
        ]
        pdf.cell(0, 10, txt=f"Dimensions = {dims} µm", ln=True)

        # Orthogonal projections of the heatmap bbox
        _hm = heatmap[
            int(item["bbox_z_min"]) : int(item["bbox_z_max"]) + 1,
            int(item["bbox_y_min"]) : int(item["bbox_y_max"]) + 1,
            int(item["bbox_x_min"]) : int(item["bbox_x_max"]) + 1,
        ]
        fig, axs = plt.subplots(1, 3, figsize=(8, 5))
        axs[0].imshow(_hm[:, :, _hm.shape[2] // 2])
        axs[1].imshow(_hm[:, _hm.shape[1] // 2, :])
        axs[2].imshow(_hm[_hm.shape[0] // 2, :, :])
        for j, coord in enumerate(["x", "y", "z"]):
            axs[j].set_title(f"{coord}-projection")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_name = tmp.name
        fig.savefig(tmp_name)
        plt.close(fig)
        pdf.image(tmp_name, w=180)
        os.unlink(tmp_name)

        # Masked composite slice (intensity + classification overlay)
        mid_z = (int(item["bbox_z_min"]) + int(item["bbox_z_max"]) + 1) // 2
        if mid_z < len(slice_files):
            from PIL import Image as _PILImage
            fig2, axs2 = plt.subplots(1, 2, figsize=(8, 5))
            arr = np.array(
                _PILImage.open(os.path.join(data_path, slice_files[mid_z])),
                dtype=np.float32,
            )

            # Intensity mask — same window logic as tomo_app masked-slice endpoint
            if ref_value > 0:
                in_range = (arr >= ref_value * (1.0 - lower)) & (arr <= ref_value * (1.0 + upper))
            else:
                in_range = np.ones(arr.shape, dtype=bool)

            # Classification masks from classified_vol when available
            rel_z = mid_z - slice_start
            if classified_vol is not None and 0 <= rel_z < classified_vol.shape[0]:
                cls_slc = np.array(classified_vol[rel_z])
                matrix_mask = in_range & (cls_slc == 0)
                pore_mask   = cls_slc == 1
                crack_mask  = cls_slc == 2
            else:
                matrix_mask = in_range
                pore_mask   = np.zeros(arr.shape, dtype=bool)
                crack_mask  = np.zeros(arr.shape, dtype=bool)

            # Build RGBA composite: near-black base → matrix → defects on top
            rgba = np.full((*arr.shape, 4), [15, 15, 15, 255], dtype=np.uint8)
            rgba[matrix_mask] = [0,   210,  77, 255]   # green — matrix
            rgba[pore_mask]   = [33,  150, 243, 255]   # blue  — pore
            rgba[crack_mask]  = [244,  67,  54, 255]   # red   — crack

            axs2[0].imshow(rgba)
            axs2[0].set_title(slice_files[mid_z])
            axs2[1].imshow(rgba[
                int(item["bbox_y_min"]) : int(item["bbox_y_max"]) + 1,
                int(item["bbox_x_min"]) : int(item["bbox_x_max"]) + 1,
            ])
            axs2[1].set_title("Zoom on defect")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
                tmp2_name = tmp2.name
            fig2.savefig(tmp2_name)
            plt.close(fig2)
            pdf.image(tmp2_name, w=180)
            os.unlink(tmp2_name)

        pdf.ln(20)

    pdf.output(save_path)
