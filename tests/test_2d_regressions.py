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


def test_cluster_bootstrap_guards():
    from tomographia.evaluation import bootstrap_ci
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 200), rng.random(200)
    with pytest.raises(ValueError, match="two groups"):
        bootstrap_ci(y, p, 0.5, groups=np.zeros(200), n_bootstrap=20)
    with pytest.warns(UserWarning, match="only 3 groups"):
        bootstrap_ci(y, p, 0.5, groups=np.arange(200) % 3, n_bootstrap=20)
    with pytest.warns(UserWarning, match="undefined"):       # single-class groups
        bootstrap_ci(np.r_[np.zeros(100), np.ones(100)], p, 0.5,
                     groups=np.r_[np.zeros(100), np.ones(100)] + (np.arange(200) % 2) * 2,
                     n_bootstrap=50)


def test_grouped_split_on_folders_warns(patch_folders):
    df = build_patch_index(patch_folders)
    with pytest.warns(UserWarning, match="source folder"):
        split_patches(df, group_col="group")


def test_extract_patch_has_requested_size():
    from tomographia.labelling.visualize import extract_patch
    patch, _ = extract_patch(np.zeros((50, 50)), 25, 25, 7)
    assert patch.shape == (7, 7)


torch = pytest.importorskip("torch")


def test_temperature_separable_validation_warns():
    from tomographia.evaluation import fit_temperature
    z = np.r_[-np.full(50, 3.0), np.full(50, 3.0)]
    y = np.r_[np.zeros(50), np.ones(50)]
    with pytest.warns(UserWarning, match="separate the classes"):
        t = fit_temperature(z, y)
    assert t == 1.0                                            # calibration skipped
    y[0] = 1                                                   # one error: finite optimum
    assert 0.05 < fit_temperature(z, y) < 20


def test_predict_patches_applies_temperature():
    from tomographia.slice_inference import predict_patches

    class Const(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1), 2.0)

    x = torch.zeros(3, 1, 8, 8)
    p1 = predict_patches(Const(), x)
    p2 = predict_patches(Const(), x, temperature=2.0)
    assert p1[0] == pytest.approx(1 / (1 + np.exp(-2))) and p2[0] == pytest.approx(1 / (1 + np.exp(-1)))


def test_tau_outside_bounds_raises():
    from tomographia.classification import ResNetPatchClassifier
    from tomographia.prototypes import build_prototype_network
    clf = ResNetPatchClassifier("resnet18")
    D = clf.embed_dim
    cents = {0: np.zeros((1, D)), 1: np.ones((1, D))}
    with pytest.raises(ValueError, match="tau_bounds"):
        build_prototype_network(clf, cents, np.zeros(D), np.ones(D), init_tau=0.1)
    build_prototype_network(clf, cents, np.zeros(D), np.ones(D), init_tau=0.1,
                            tau_bounds=(0.05, 5.0))


def test_conflicting_anchor_types_raise():
    from tomographia.prototypes import prototypes_from_anchors
    z = np.random.default_rng(0).normal(size=(4, 3))
    anchors = pd.DataFrame({"index": [0, 0, 3], "cls": [0, 0, 1], "type": [0, 1, 0]})
    with pytest.raises(ValueError, match="different classes or types"):
        prototypes_from_anchors(z, np.arange(4), anchors, n_total=4)
    dup = pd.DataFrame({"index": [0, 0, 3], "cls": [0, 0, 1], "type": [0, 0, 0]})
    assert prototypes_from_anchors(z, np.arange(4), dup, n_total=4).k_per_class == [1, 1]
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


def test_calibrated_evaluation_uses_logits_on_separable_validation():
    """Separable validation: no calibration, threshold between the classes, and
    test AUC computed on logits (no ties from saturated probabilities)."""
    from tomographia.evaluation import evaluate_calibrated
    rng = np.random.default_rng(0)
    zv = np.r_[rng.uniform(-8, -2, 50), rng.uniform(2, 40, 50)]
    yv = np.r_[np.zeros(50), np.ones(50)]
    zt = np.r_[rng.uniform(-8, 3, 60), rng.uniform(-1, 40, 60)]
    yt = np.r_[np.zeros(60), np.ones(60)]
    with pytest.warns(UserWarning, match="separate"):
        rep = evaluate_calibrated(zv, yv, zt, yt, n_bootstrap=0)
    from sklearn.metrics import roc_auc_score
    assert rep["temperature"] == 1.0
    assert zv[yv == 0].max() < rep["threshold_logit"] <= zv[yv == 1].min()
    assert rep["test"]["roc_auc"] == pytest.approx(roc_auc_score(yt, zt))


def test_per_patch_split_over_folders_warns(patch_folders):
    with pytest.warns(UserWarning, match="Per-patch split"):
        split_patches(build_patch_index(patch_folders))


def test_predict_logits_rejects_two_class_models():
    from torch.utils.data import DataLoader, TensorDataset
    from tomographia.classification import predict_logits

    class TwoLogits(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 2)

    dl = DataLoader(TensorDataset(torch.zeros(4, 1, 8, 8), torch.zeros(4)), batch_size=2)
    with pytest.raises(ValueError, match="single-logit"):
        predict_logits(TwoLogits(), dl)
