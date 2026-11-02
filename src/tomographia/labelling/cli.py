"""
Terminal labelling session.

Usage::

    tomographia-label --data-path /path/to/volume --identifier 123456 --threshold 50
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from .config import LabelingConfig
from .io import load_label_df, save_label_df
from .selection import apply_threshold_labels, df_recap
from .session import LabelingSession


def _recap(df, col) -> str:
    r = df_recap(df, col)
    return (f"unlabelled: {r['unlabelled']} | 0: {r['label_0']} | "
            f"1: {r['label_1']} | 2: {r['label_2']}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Label 2-D CT patches (0: no defect, 1: defect, 2: unsure).")
    p.add_argument("--data-path", required=True, help="Folder with the patch CSV and the .tif slices.")
    p.add_argument("--identifier", required=True, help="Substring of the CSV file name.")
    p.add_argument("--threshold", type=float, default=50.0, help="Auto-label 0 if intensity < threshold.")
    p.add_argument("--label-col", default="defect")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--focus-unzoom", type=int, default=256)
    p.add_argument("--scale", type=float, default=1.0, help="Display contrast scale.")
    p.add_argument("--offset", type=float, default=0.0, help="Display contrast offset.")
    a = p.parse_args(argv)

    import matplotlib.pyplot as plt

    cfg = LabelingConfig(data_path=a.data_path, identifier=a.identifier, label_col=a.label_col,
                         threshold=a.threshold, save_after_x_labels=a.save_every,
                         focus_unzoom=a.focus_unzoom, scale=a.scale, offset=a.offset)
    df = load_label_df(cfg.data_path, cfg.identifier)
    apply_threshold_labels(df, cfg.label_col, cfg.intensity_col, cfg.threshold)
    print(_recap(df, cfg.label_col))
    if df[cfg.label_col].isna().sum() == 0:
        save_label_df(df, cfg.data_path, cfg.identifier)
        print("Nothing left to label.")
        return

    session = LabelingSession(df, cfg)
    for patch in session.iter_patches():
        fig, _ = session.figure_for_patch(session.load_slice(patch), patch)
        hint = "" if patch.prediction is None else f" | model P(defect)={patch.prediction:.2f}"
        print(f"patch {patch.id_global}  intensity={patch.intensity:.1f}{hint}")
        plt.show(block=False)
        key = input("label 0/1/2, 'n' skip, 'k' stop: ").strip().lower()
        plt.close(fig)
        if key == "k":
            break
        if key in ("0", "1", "2"):
            session.assign_label(patch, int(key))
        elif key != "n":
            print("Invalid input, skipped.")

    session.save()
    print("Labels saved. " + _recap(session.df, cfg.label_col))


if __name__ == "__main__":
    main()
