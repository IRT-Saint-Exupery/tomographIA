"""Tests for the patch-labelling tool (no GUI)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tomographia.labelling import (
    LabelingConfig,
    LabelingSession,
    apply_threshold_labels,
    df_recap,
    extract_patch,
    load_label_df,
)


@pytest.fixture
def volume(tmp_path):
    for z in range(2):
        Image.fromarray(np.full((128, 128), 100, np.uint8)).save(tmp_path / f"s_{z:03d}.tif")
    df = pd.DataFrame({
        "id_global": ["a", "b", "c", "d"],
        "x": [32, 64, 96, 64], "y": [32, 64, 96, 64], "z": [0, 0, 1, 5],
        "patch_size": 32, "intensity": [10.0, 80.0, 90.0, 70.0],
    })
    df.to_csv(tmp_path / "patches_12345.csv", index=False)
    return tmp_path


def test_threshold_and_recap(volume):
    df = load_label_df(str(volume), "12345")
    apply_threshold_labels(df, "defect", "intensity", 50)
    assert df_recap(df, "defect") == {"unlabelled": 3, "label_0": 1, "label_1": 0, "label_2": 0}


def test_session_labels_and_saves(volume):
    cfg = LabelingConfig(data_path=str(volume), identifier="12345", save_after_x_labels=1)
    df = apply_threshold_labels(load_label_df(cfg.data_path, cfg.identifier), "defect", "intensity", 50)
    sess = LabelingSession(df, cfg)
    patches = list(sess.iter_patches())
    assert {p.id_global for p in patches} == {"b", "c"}  # "d" has no slice z=5
    img = sess.load_slice(patches[0])
    assert img.shape == (128, 128) and img.dtype == np.uint8
    sess.assign_label(patches[0], 1)  # triggers autosave
    saved = load_label_df(cfg.data_path, cfg.identifier)
    assert saved.loc[saved.id_global == patches[0].id_global, "defect"].item() == 1
    with pytest.raises(ValueError):
        sess.assign_label(patches[1], 7)


def test_extract_patch_clamps_borders():
    img = np.arange(100).reshape(10, 10)
    p, bounds = extract_patch(img, 1, 1, 4)
    assert p.shape == (3, 3) and bounds == (-1, -1, 3, 3)
