"""Tests for ResNet patch classifiers, prototype networks and slice inference."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch.utils.data import DataLoader  # noqa: E402

from tomographia.classification import (  # noqa: E402
    PatchClassifierTrainer,
    PatchTrainConfig,
    ResNetPatchClassifier,
    predict_proba,
)
from tomographia.patches import PatchDataset, build_patch_index, split_patches  # noqa: E402
from tomographia.prototypes import (  # noqa: E402
    ProtoTrainConfig,
    build_prototype_network,
    coalescence_stats,
    extract_embeddings,
    fit_embedding_scaler,
    nearest_examples,
    per_prototype_confusion,
    predict_prototype_network,
    prototype_loss,
    prototypes_from_anchors,
    train_prototype_network,
)


@pytest.fixture
def splits(patch_folders):
    return split_patches(build_patch_index(patch_folders), ratios=(0.6, 0.2, 0.2), seed=0)


def _loader(df, **kw):
    return DataLoader(PatchDataset(df, **kw), batch_size=16, shuffle=False)


# ─── ResNet classifier ──────────────────────────────────────────────────────

@pytest.mark.parametrize("arch,dim", [("resnet18", 512), ("resnet50", 2048)])
def test_resnet_shapes(arch, dim):
    m = ResNetPatchClassifier(arch).eval()
    x = torch.randn(2, 1, 32, 32)
    assert m(x).shape == (2, 1)
    assert m.features(x).shape == (2, dim) == (2, m.embed_dim)
    assert torch.allclose(m(x), m.model.fc(m.extract_features(x)))


def test_resnet_state_dict_compatible_with_original_layout():
    keys = ResNetPatchClassifier("resnet18").state_dict().keys()
    assert "model.conv1.weight" in keys and "model.fc.weight" in keys


def test_patch_trainer_runs(splits):
    torch.manual_seed(0)
    model = ResNetPatchClassifier("resnet18")
    cfg = PatchTrainConfig(max_epochs=2, lr=1e-3)
    res = PatchClassifierTrainer(model, cfg).fit(
        DataLoader(PatchDataset(splits[splits.split == "train"]), batch_size=8, shuffle=True),
        _loader(splits[splits.split == "val"]))
    assert len(res.history) == 2 and 0 < res.threshold < 1
    prob, y, idx = predict_proba(model, _loader(splits[splits.split == "test"], return_index=True))
    assert prob.shape == y.shape == idx.shape
    assert set(idx) <= set(splits.index[splits.split == "test"])


# ─── Prototype network ──────────────────────────────────────────────────────

def _anchor_setup(splits):
    torch.manual_seed(0)
    clf = ResNetPatchClassifier("resnet18").eval()
    train = splits[splits.split == "train"]
    Z, y, idx = extract_embeddings(clf, _loader(train, return_index=True))
    scaler = fit_embedding_scaler(Z)
    # two expert types per class, three anchors per type
    rows = []
    for c in (0, 1):
        ids = train.index[train.label == c][:6]
        rows += [{"index": i, "cls": c, "type": k // 3} for k, i in enumerate(ids)]
    anchors = pd.DataFrame(rows)
    aset = prototypes_from_anchors(scaler.transform(Z), idx, anchors, n_total=len(splits),
                                   labels=splits.label)
    return clf, scaler, aset, Z, y


def test_anchor_medoids_are_real_samples(splits):
    _, scaler, aset, Z, _ = _anchor_setup(splits)
    assert aset.k_per_class == [2, 2]
    zs = scaler.transform(Z)
    for c, rows in aset.medoid_rows.items():
        assert np.allclose(aset.centroids_by_class[c], zs[rows])
    assert int((aset.anchor_class >= 0).sum()) == 12


def test_prototype_head_uses_embedding_not_logit(splits):
    """Regression test: the head must receive the embedding, not the fc logit."""
    clf, scaler, aset, _, _ = _anchor_setup(splits)
    pnet = build_prototype_network(clf, aset.centroids_by_class, scaler.mean_, scaler.scale_).eval()
    x = torch.randn(3, 1, 32, 32)
    out = pnet(x)
    assert out["embedding"].shape == (3, 512)
    # changing the classifier's fc layer must not change the prototype output
    with torch.no_grad():
        pnet.backbone.model.fc.weight.mul_(-5.0)
    assert torch.allclose(out["class_probs"], pnet(x)["class_probs"])
    # a logit-shaped input is rejected instead of silently broadcast
    with pytest.raises(ValueError):
        pnet.head(torch.randn(3, 1))


def test_set_trainable_last_block(splits):
    clf, scaler, aset, _, _ = _anchor_setup(splits)
    pnet = build_prototype_network(clf, aset.centroids_by_class, scaler.mean_, scaler.scale_)
    pnet.set_trainable("last_block")
    trainable = {n.split(".")[1] for n, p in pnet.backbone.named_parameters() if p.requires_grad}
    assert trainable == {"layer4"}
    assert all(p.requires_grad for p in pnet.head.parameters())


def test_prototype_loss_terms_and_gradients(splits):
    clf, scaler, aset, _, _ = _anchor_setup(splits)
    pnet = build_prototype_network(clf, aset.centroids_by_class, scaler.mean_, scaler.scale_)
    x, y, idx = next(iter(_loader(splits[splits.split == "train"], return_index=True)))
    terms = prototype_loss(pnet(x), y, pnet.head, sample_indices=torch.as_tensor(idx),
                           anchor_class=aset.anchor_class, anchor_proto=aset.anchor_proto,
                           medoids=torch.as_tensor(aset.medoids))
    assert torch.isfinite(terms["total"])
    assert float(terms["anchor"]) > 0  # first batch contains anchors
    assert float(terms["medoid"]) == pytest.approx(0.0, abs=1e-10)  # init at medoids
    terms["total"].backward()
    assert pnet.head.prototypes.grad is not None


def test_train_prototype_network_end_to_end(splits):
    clf, scaler, aset, Z, y = _anchor_setup(splits)
    pnet = build_prototype_network(clf, aset.centroids_by_class, scaler.mean_, scaler.scale_)
    train = DataLoader(PatchDataset(splits[splits.split == "train"], return_index=True),
                       batch_size=8, shuffle=True)
    val = _loader(splits[splits.split == "val"])
    res = train_prototype_network(pnet, train, val, medoids=aset.medoids,
                                  anchor_class=aset.anchor_class, anchor_proto=aset.anchor_proto,
                                  config=ProtoTrainConfig(max_epochs=2))
    assert len(res.history) == 2 and np.isfinite(res.best_val_loss)
    pred = predict_prototype_network(pnet, _loader(splits[splits.split == "test"]))
    assert pred["proto_probs"].shape[1] == 4
    assert np.allclose(1 / (1 + np.exp(-pred["logit"])), pred["prob"], atol=1e-5)
    stats = coalescence_stats(pnet.head)
    assert stats[0]["k"] == 2 and "min" in stats[0]
    nn_rows = nearest_examples(pnet.head, Z, k=3, restrict_to_class=y)
    assert all(len(r) == 3 for r in nn_rows.values())
    conf = per_prototype_confusion(pred["proto_probs"].argmax(1), pred["label"],
                                   (pred["prob"] > res.threshold).astype(int), [2, 2])
    assert conf.n.sum() == len(pred["label"])


# ─── Slice inference ────────────────────────────────────────────────────────

def test_slice_inference_grid_and_volume(tmp_path):
    from PIL import Image
    from tomographia.slice_inference import (
        predict_volume, predictions_to_volume, remove_sparse_voxels, tile_slice)

    img = np.random.default_rng(0).integers(0, 255, (100, 130)).astype(np.uint8)
    patches, rows, cols = tile_slice(img, 32)
    assert patches.shape == (len(rows) * len(cols), 1, 32, 32) and (len(rows), len(cols)) == (3, 4)
    assert np.allclose(patches[1, 0].numpy(), (img[0:32, 32:64] / 255.0 - 0.5) / 0.5, atol=1e-6)
    _, r2, c2 = tile_slice(img, 32, stride=16)
    assert (len(r2), len(c2)) == (5, 7)

    for z in range(3):
        Image.fromarray(img).save(tmp_path / f"slice_{z:03d}.tif")
    model = ResNetPatchClassifier("resnet18")
    preds = predict_volume(model, str(tmp_path), patch_size=32, batch_size=5)
    vol = predictions_to_volume(preds, threshold=0.5)
    assert vol.shape == (3, 3, 4) and set(np.unique(vol)) <= {0, 1}
    assert remove_sparse_voxels(vol).shape == vol.shape


def test_isosurface_mesh_empty_and_cube():
    from tomographia.slice_inference import isosurface_mesh

    v, f = isosurface_mesh(np.zeros((4, 4, 4)))
    assert v.shape == (0, 3) and f.shape == (0, 3)
    cube = np.zeros((6, 6, 6)); cube[2:4, 2:4, 2:4] = 1
    v, f = isosurface_mesh(cube, spacing=(2.0, 1.0, 1.0))
    assert len(f) > 0 and v[:, 0].max() > v[:, 1].max()  # z scaled by 2


def test_remove_sparse_voxels_isolated():
    from tomographia.slice_inference import remove_sparse_voxels

    v = np.zeros((5, 5, 5), np.uint8)
    v[2, 2, 2] = 1            # isolated
    v[0, 0, 0] = v[0, 0, 1] = 1  # pair
    out = remove_sparse_voxels(v, min_neighbors=1)
    assert out[2, 2, 2] == 0 and out[0, 0, 0] == 1 and out[0, 0, 1] == 1
