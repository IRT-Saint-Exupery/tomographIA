"""Regression tests for the 2-D pipeline."""
import os

import numpy as np
import pandas as pd
import pytest

from tomographia.evaluation import f1_optimal_threshold, point_metrics
from tomographia.labelling import apply_threshold_labels
from tomographia.patches import build_patch_index, split_patches


def test_f1_threshold_keeps_saturated_optimum():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.0, 0.05, 0.9, 1.0, 1.0, 1.0])  # positives saturated at 1.0
    thr = f1_optimal_threshold(y, p)
    assert point_metrics(y, p, thr)["f1"] == 1.0


def test_split_requires_unique_index(patch_folders):
    df = build_patch_index(patch_folders)
    dup = pd.concat([df, df])  # duplicated index
    with pytest.raises(ValueError, match="unique"):
        split_patches(dup)


def test_split_warns_on_single_class_split(patch_folders):
    df = build_patch_index(patch_folders)
    df["group"] = df["label"]  # each class is one group -> pathological split
    with pytest.warns(UserWarning, match="single class"):
        split_patches(df, ratios=(0.5, 0.25, 0.25), group_col="group")


def test_threshold_labels_never_overwrite():
    df = pd.DataFrame({"intensity": [10.0, 10.0, 10.0, 90.0],
                       "defect": [1.0, 2.0, np.nan, np.nan]})
    apply_threshold_labels(df, "defect", "intensity", 50)
    assert df.defect.tolist()[:3] == [1.0, 2.0, 0.0] and np.isnan(df.defect.iloc[3])
    df2 = pd.DataFrame({"defect": [np.nan]})  # no intensity column: no-op
    assert np.isnan(apply_threshold_labels(df2, "defect", "intensity", 50).defect.iloc[0])


def test_natural_slice_order(tmp_path):
    from tomographia.patches.dataset import natural_key
    names = ["s_10.tif", "s_2.tif", "s_1.tif"]
    assert sorted(names, key=natural_key) == ["s_1.tif", "s_2.tif", "s_10.tif"]


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_frozen_batchnorm_stats_do_not_drift():
    from tomographia.classification import ResNetPatchClassifier
    from tomographia.prototypes import build_prototype_network

    clf = ResNetPatchClassifier("resnet18")
    D = clf.embed_dim
    pnet = build_prototype_network(clf, {0: np.zeros((1, D)), 1: np.ones((1, D))},
                                   np.zeros(D), np.ones(D))
    pnet.set_trainable("last_block")
    pnet.train()
    before = pnet.backbone.model.bn1.running_mean.clone()
    l4_before = pnet.backbone.model.layer4[0].bn1.running_mean.clone()
    pnet(torch.randn(4, 1, 32, 32) * 3 + 1)
    assert torch.equal(before, pnet.backbone.model.bn1.running_mean)        # frozen
    assert not torch.equal(l4_before, pnet.backbone.model.layer4[0].bn1.running_mean)  # trainable


def test_anchor_index_validation():
    from tomographia.prototypes import prototypes_from_anchors

    z = np.random.default_rng(0).normal(size=(4, 3))
    anchors = pd.DataFrame({"index": [0, 3], "cls": [0, 1], "type": [0, 0]})
    with pytest.raises(ValueError, match="n_total"):
        prototypes_from_anchors(z, np.array([0, 1, 2, 3]), anchors, n_total=3)
    with pytest.raises(ValueError, match="every class"):
        prototypes_from_anchors(z, np.array([0, 1, 2, 3]), anchors.iloc[:1], n_total=4)


def test_model_assist_pads_clipped_side():
    from tomographia.labelling.model_assist import suggest_for_patch
    from tomographia.labelling.session import PatchRecord

    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, x):
            seen["x"] = x.clone()
            return torch.zeros(x.shape[0], 1)

    img = np.zeros((32, 32), np.float32)
    img[0:4, 0:4] = 255.0  # bright block in the image corner
    suggest_for_patch(img, PatchRecord("a", x=2, y=2, z=0, intensity=0.0, patch_size=8), Probe())
    window = seen["x"][0, 0]
    # patch centred at (2, 2) spans rows/cols -2..5: the block belongs at window rows/cols 2..5
    assert float(window[3, 3]) == pytest.approx(1.0) and float(window[7, 7]) == pytest.approx(-1.0)
