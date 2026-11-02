"""Regression tests for the 3-D pipeline: checkpoints, stitching, slice order,
patch splits and reporting."""
import os

import numpy as np
import pandas as pd
import pytest
import tifffile


def _write_slices(volume, directory, names=None):
    os.makedirs(directory, exist_ok=True)
    for i, slc in enumerate(volume):
        name = names[i] if names else f"slice_{i}.tif"
        tifffile.imwrite(os.path.join(directory, name), slc)


# ---------------------------------------------------------------------------
# Slice order (shared by every reader)
# ---------------------------------------------------------------------------

def test_every_reader_uses_natural_slice_order(tmp_path):
    from tomographia._slices import list_slices
    from tomographia.components import load_grayscale_volume
    from tomographia.reporting import compute_yield_report

    vol = np.stack([np.full((4, 4), z, np.uint16) for z in range(12)])
    _write_slices(vol, tmp_path)                       # slice_0 ... slice_11 (unpadded)
    names = [os.path.basename(p) for p in list_slices(str(tmp_path))]
    assert names[:3] == ["slice_0.tif", "slice_1.tif", "slice_2.tif"]
    np.testing.assert_array_equal(load_grayscale_volume(str(tmp_path))[:, 0, 0], np.arange(12))

    # yield: slice z is matrix only where its grey value is in the window
    cls = np.zeros_like(vol, dtype=np.uint8)
    rep = compute_yield_report(cls, tmp_path, ref_value=10, lower=0.05, upper=0.05)
    assert rep["matrix_voxels"] == 16                 # only slice 10 (value 10) is matrix


def test_read_big_endian_tiff(tmp_path):
    from tomographia._slices import read_slice
    a = (np.arange(16, dtype=np.uint16) * 1000).reshape(4, 4)
    tifffile.imwrite(tmp_path / "be.tif", a.astype(">u2"), byteorder=">")
    np.testing.assert_array_equal(read_slice(str(tmp_path / "be.tif")), a)


# ---------------------------------------------------------------------------
# Segmentation inference helpers (torch)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_checkpoint_missing_or_mismatched_raises(tmp_path):
    from tomographia.models import Unet3D
    from tomographia.segmentation import load_segmentation_checkpoint

    model = Unet3D(1, 1)
    with pytest.raises(FileNotFoundError):
        load_segmentation_checkpoint(model, tmp_path / "nope.bin")
    torch.save({"state_dict": Unet3D(1, 2).state_dict()}, tmp_path / "bad.bin")
    with pytest.raises(RuntimeError):
        load_segmentation_checkpoint(model, tmp_path / "bad.bin")


def test_checkpoint_formats_and_prefixes(tmp_path):
    from tomographia.models import Unet3D
    from tomographia.segmentation import load_segmentation_checkpoint

    src = Unet3D(1, 1)
    state = {"module." + k: v for k, v in src.state_dict().items()}
    torch.save({"state_dict": state, "threshold": 0.37}, tmp_path / "a.bin")
    torch.save(src.state_dict(), tmp_path / "b.bin")
    dst = Unet3D(1, 1)
    assert load_segmentation_checkpoint(dst, tmp_path / "a.bin")["threshold"] == 0.37
    assert load_segmentation_checkpoint(dst, tmp_path / "b.bin") == {}
    k = "segmenter.weight"
    assert torch.equal(dst.state_dict()[k], src.state_dict()[k])


def test_unet_rejects_indivisible_shapes():
    from tomographia.models import Unet3D
    with pytest.raises(ValueError, match="divisible by 16"):
        Unet3D(1, 1)(torch.zeros(1, 1, 16, 24, 32))


def test_stitching_averages_probabilities_before_thresholding():
    """Overlap of p = 0.9 and p = 0.2 must give the mean probability 0.55
    (binarising each patch first would give 0.5 and lose the information)."""
    from tomographia.segmentation._dataset import PatchedVolumeBig

    pv = PatchedVolumeBig(np.zeros((4, 4, 6), np.float32), (4, 4, 4), stride=(4, 4, 2))
    assert len(pv) == 2
    out = pv.rebuild_volume_from_patches([np.full((4, 4, 4), 0.9), np.full((4, 4, 4), 0.2)])
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], [0.9, 0.9, 0.55, 0.55, 0.2, 0.2], rtol=1e-6)


def test_small_volume_is_padded_and_cropped():
    from tomographia.segmentation._dataset import PatchedVolumeBig

    vol = np.arange(3 * 5 * 6, dtype=np.float32).reshape(3, 5, 6)
    pv = PatchedVolumeBig(vol, (4, 8, 8))
    assert len(pv) == 1 and tuple(pv[0].shape) == (4, 8, 8)
    np.testing.assert_array_equal(pv.rebuild_volume_from_patches([pv[0].numpy()]), vol)


# ---------------------------------------------------------------------------
# Segmentation losses
# ---------------------------------------------------------------------------

def test_focal_alpha_is_class_balanced():
    from tomographia.segmentation import FocalLoss

    logits = torch.zeros(1, 1, 2)
    targets = torch.tensor([[[1.0, 0.0]]])
    per_voxel = FocalLoss(alpha=0.75, gamma=0.0, reduction="none")(logits, targets)
    # same BCE for both voxels; weight 0.75 for the positive, 0.25 for the negative
    assert per_voxel[0, 0, 0] / per_voxel[0, 0, 1] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# PatchGenerator (h5py)
# ---------------------------------------------------------------------------

def _two_volumes(tmp_path):
    rng = np.random.default_rng(0)
    dirs = {}
    for name in ("A", "B"):
        raw, lab = tmp_path / name / "raw", tmp_path / name / "labels"
        _write_slices(rng.random((20, 20, 20)).astype(np.float32), raw)
        _write_slices((rng.random((20, 20, 20)) > 0.9).astype(np.uint8), lab)
        dirs[name] = (str(raw), str(lab))
    return dirs


def test_patch_generator_volume_split_and_cache(tmp_path):
    pytest.importorskip("h5py")
    from tomographia.segmentation import PatchGenerator

    dirs = _two_volumes(tmp_path)
    gen = PatchGenerator(dirs, (16, 16, 16), str(tmp_path / "p"),
                         split_volumes={"A": "train", "B": "test"})
    s = gen.get_splits()
    assert set(s["train"].volume) == {"A"} and set(s["test"].volume) == {"B"}
    assert len(s["val"]) == 0
    # 20 = 16 + 4: patches at 0 and 4 along each axis, no mirrored padding
    assert len(s["train"]) == 8
    assert {os.path.basename(p).split("_patch_raw_")[1] for p in s["train"].inputs} >= {"4_4_4.h5"}
    # the shifted patch owns only the voxels not covered by the first one
    from tomographia.segmentation import SegmentationDataset
    ds = SegmentationDataset(s["train"])
    owned_total = sum(ds[i][3].sum() for i in range(len(ds)))
    assert owned_total == 20 ** 3
    assert gen.metadata == {"normalization": "no_norm", "patch_shape": (16, 16, 16)}

    # cache is reused for identical settings, rebuilt when a setting changes
    again = PatchGenerator(dirs, (16, 16, 16), str(tmp_path / "p"),
                           split_volumes={"A": "train", "B": "test"})
    pd.testing.assert_frame_equal(again.get_splits()["train"], s["train"])
    other = PatchGenerator(dirs, (16, 16, 16), str(tmp_path / "p"),
                           split_volumes={"A": "test", "B": "train"})
    assert set(other.get_splits()["train"].volume) == {"B"}


def test_patch_generator_validates_split_volumes(tmp_path):
    from tomographia.segmentation import PatchGenerator
    with pytest.raises(ValueError, match="split_volumes"):
        PatchGenerator({"A": ("x", "y")}, (16, 16, 16), str(tmp_path),
                       split_volumes={"A": "holdout"})


# ---------------------------------------------------------------------------
# PDF report (fpdf2 + matplotlib)
# ---------------------------------------------------------------------------

def test_pdf_report_from_pipeline_outputs(tmp_path):
    pytest.importorskip("fpdf")
    pytest.importorskip("matplotlib")
    from tomographia.classification import build_classified_volume
    from tomographia.components import build_cc_mask, compute_descriptors
    from tomographia.reporting import compute_yield_report, generate_pdf_report

    vol = np.full((12, 24, 24), 1000, np.uint16)
    pred = np.zeros(vol.shape, np.uint8)
    pred[3:7, 4:10, 4:10] = 1
    pred[5:9, 14:16, 12:22] = 1
    vol[pred > 0] = 100
    _write_slices(vol, tmp_path / "raw")

    cc = build_cc_mask(pred)
    df = compute_descriptors(cc, 14.0, unit="µm", grayscale_volume=vol)
    df["cls_pred"] = [0, 1]
    cls = build_classified_volume(cc, df)
    yr = compute_yield_report(cls, tmp_path / "raw", ref_value=1000, lower=0.1, upper=0.1)
    out = tmp_path / "r.pdf"
    generate_pdf_report(yr, df, str(tmp_path / "raw"), str(out), "test",
                        voxel_size=14.0, unit="μm",      # Greek mu: must not crash
                        heatmap=pred.astype(np.float32), classified_vol=cls,
                        ref_value=1000, lower=0.1, upper=0.1)
    assert out.stat().st_size > 1000


def test_patch_generator_detects_edited_labels(tmp_path):
    pytest.importorskip("h5py")
    from tomographia.segmentation import PatchGenerator

    dirs = _two_volumes(tmp_path)
    kw = dict(split_volumes={"A": "train", "B": "test"})
    PatchGenerator(dirs, (16, 16, 16), str(tmp_path / "p"), **kw)
    lab = os.path.join(dirs["A"][1], "slice_0.tif")
    tifffile.imwrite(lab, np.ones((20, 20), np.uint8))
    os.utime(lab, (1e9, 1e9))                       # make sure the signature changes
    gen = PatchGenerator(dirs, (16, 16, 16), str(tmp_path / "p"), **kw)
    from tomographia.segmentation import SegmentationDataset
    y = SegmentationDataset(gen.get_splits()["train"])[0][1]
    assert y[0].min() == 1.0                        # re-extracted with the new label


# ---------------------------------------------------------------------------
# Label pairing
# ---------------------------------------------------------------------------

def _label_for(directory, raw_name):
    from tomographia.segmentation._io import get_tif_labels
    for i in range(3):
        tifffile.imwrite(os.path.join(directory, f"slice_{i:04d}.tif"), np.zeros((2, 2), np.uint8))
    return get_tif_labels(str(directory), [os.path.join(str(directory), raw_name)])[0]


def test_labels_matched_on_slice_number(tmp_path):
    from tomographia.segmentation._io import get_tif_labels
    assert os.path.basename(_label_for(tmp_path, "scan_2.tif")) == "slice_0002.tif"
    raw = [str(tmp_path / f"scan_{i:04d}_rec.tif") for i in range(3)]
    for i in range(3):
        tifffile.imwrite(tmp_path / f"slice_{i:04d}.tif", np.zeros((2, 2), np.uint8))
    out = get_tif_labels(str(tmp_path), raw)
    assert [os.path.basename(p) for p in out] == ["slice_0000.tif", "slice_0001.tif",
                                                  "slice_0002.tif"]
    with pytest.raises(ValueError, match="same label"):
        get_tif_labels(str(tmp_path), [raw[0], str(tmp_path / "other_0000.tif")])


# ---------------------------------------------------------------------------
# Yield / KMeans / cc3d edge cases
# ---------------------------------------------------------------------------

def test_yield_rejects_bad_inputs(tmp_path):
    from tomographia.reporting import compute_yield_report
    _write_slices(np.ones((3, 4, 4), np.uint16), tmp_path)
    with pytest.raises(ValueError, match="ref_value"):
        compute_yield_report(np.zeros((3, 4, 4), np.uint8), tmp_path, 0, 0.1, 0.1)
    with pytest.raises(ValueError, match="slice_start"):
        compute_yield_report(np.zeros((3, 4, 4), np.uint8), tmp_path, 1, 0.1, 0.1, slice_start=1)


def test_cluster_volume_ignores_unknown_labels():
    from tomographia.classification import make_cluster_volume
    cc = np.array([[[0, 1, 2]]])
    df = pd.DataFrame({"label_id": [1, 2, 7], "cluster": [0, 1, 1]})
    np.testing.assert_array_equal(make_cluster_volume(cc, df), [[[0, 1, 2]]])


def test_cc3d_pipeline_keeps_border_defects():
    pytest.importorskip("cc3d")
    from tomographia.components import cc_report, close_image, dict_to_df
    vol = np.zeros((12, 12, 12), np.uint8)
    vol[0:3, 0:5, 0:5] = 1                          # touches the border
    vol[5, 8, 8] = vol[6, 9, 9] = 1                 # diagonal neighbours
    labels, n = close_image(vol, structuring_dim=3)
    assert (labels > 0).sum() == vol.sum()          # nothing eroded, nothing added
    df = dict_to_df(cc_report(labels, min_pixels=1))
    assert sorted(df["size"]) == [2, 75]
    big = df.loc[df["size"] == 75].iloc[0]
    assert big.bbox_z_max == 3 and big.centroid_z == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Trainer threshold and TTA metadata
# ---------------------------------------------------------------------------

def test_threshold_uses_every_rare_defect_voxel():
    from tomographia.segmentation import SegmentationTrainer
    rng = np.random.default_rng(0)
    n = 2_000_000
    y = torch.zeros(n, dtype=torch.int32)
    y[:30] = 1
    p = torch.from_numpy(rng.uniform(0, 0.6, n)).float()
    p[:30] = torch.from_numpy(rng.uniform(0.5, 1.0, 30)).float()
    t = SegmentationTrainer._f1_optimal(p, y)
    pred = p >= t
    tp = int((pred & (y > 0)).sum()); fp = int((pred & (y == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + (30 - tp))
    # brute force over the distinct scores of the defect voxels
    best = max(2 * int(((p >= c) & (y > 0)).sum())
               / (int((p >= c).sum()) + 30) for c in p[:30].tolist())
    assert f1 == pytest.approx(best, abs=0.02)


def test_validation_without_defects_keeps_threshold(caplog):
    from tomographia.segmentation import SegmentationTrainer
    tr = SegmentationTrainer(device="cpu", metadata={"normalization": "no_norm",
                                                    "patch_shape": (16, 16, 16)})
    tr._val_logits.append(torch.randn(2, 1, 4, 4, 4) - 3)
    tr._val_labels.append(torch.zeros(2, 1, 4, 4, 4))
    tr._val_masks.append(torch.ones(2, 1, 4, 4, 4, dtype=torch.bool))
    tr._val_volids.extend(["a", "a"])
    with caplog.at_level("WARNING"):
        m = tr.validation_epoch_end()
    assert m["threshold"] == 0.5 and "no defect voxel" in caplog.text


def test_tta_mode_validated_and_stored():
    from tomographia.segmentation import SegmentationTrainer
    meta = {"normalization": "no_norm", "patch_shape": (16, 16, 16)}
    assert SegmentationTrainer(device="cpu", use_tta=True, tta_mode="FLIP",
                               metadata=meta).metadata["use_tta"] is True
    with pytest.raises(ValueError, match="tta_mode"):
        SegmentationTrainer(device="cpu", tta_mode="flips", metadata=meta)


def test_run_inference_requires_unrecorded_settings(tmp_path):
    from tomographia.models import Unet3D
    from tomographia.segmentation import run_inference
    _write_slices(np.zeros((16, 16, 16), np.uint16), tmp_path / "raw")
    torch.save(Unet3D(1, 1).state_dict(), tmp_path / "bare.bin")
    with pytest.raises(ValueError, match="normalization"):
        run_inference(tmp_path / "raw", tmp_path / "bare.bin", tmp_path / "out",
                      patch_shape=(16, 16, 16))
    with pytest.raises(ValueError, match="padding"):
        run_inference(tmp_path / "raw", tmp_path / "bare.bin", tmp_path / "out",
                      patch_shape=(16, 16, 16), normalization="minmax", padding=8)


# ---------------------------------------------------------------------------
# Descriptors / classification / reporting
# ---------------------------------------------------------------------------

def test_touches_border_flag():
    from tomographia.components import build_cc_mask, compute_descriptors
    vol = np.zeros((10, 10, 10), np.uint8)
    vol[0:2, 4:6, 4:6] = 1
    vol[5:7, 4:6, 4:6] = 1
    df = compute_descriptors(build_cc_mask(vol, bridge_radius=0), 1.0, unit="mm")
    assert df.sort_values("zmin")["touches_border"].tolist() == [True, False]


def _tables():
    from tests.test_classification import _make_df
    pore_only = _make_df(n_pore=60, n_crack=0, seed=9)
    return [_make_df(seed=s) for s in range(3)] + [pore_only]


@pytest.mark.parametrize("seed", range(6))
def test_volume_holdout_with_pore_only_specimen(seed):
    from tomographia.classification import fit_classifier
    b = fit_classifier(_tables(), holdout="volume", test_size=0.3, random_state=seed)
    ho = b["holdout_results"]
    assert ho["n_crack"] > 0 and ho["n_pore"] > 0 and np.isfinite(ho["roc_auc"])


def test_feature_absent_at_inference_raises():
    from tests.test_classification import _make_df
    from tomographia.classification import classify_defects_logistic, fit_classifier
    b = fit_classifier([_make_df()])
    with pytest.raises(ValueError, match="int_std_rel"):
        classify_defects_logistic(_make_df().assign(int_std_rel=np.nan), b)


def test_yield_empty_slice_is_nan(tmp_path):
    from tomographia.reporting import compute_yield_report
    vol = np.full((2, 4, 4), 100, np.uint16)
    vol[1] = 0                                        # slice 1: neither matrix nor defect
    _write_slices(vol, tmp_path)
    rep = compute_yield_report(np.zeros((2, 4, 4), np.uint8), tmp_path, 100, 0.1, 0.1)
    assert rep["per_slice"]["total"][0] == 0.0 and np.isnan(rep["per_slice"]["total"][1])
