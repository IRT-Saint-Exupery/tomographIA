"""Tests for classification: logistic fit, dispatch, and volume encoding."""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared synthetic-data factory
# ---------------------------------------------------------------------------

def _make_df(n_pore: int = 60, n_crack: int = 20, seed: int = 42) -> pd.DataFrame:
    """Synthetic descriptor DataFrame — all 11 geometric features + int_std_rel."""
    rng = np.random.default_rng(seed)
    n = n_pore + n_crack
    return pd.DataFrame({
        "label_id":    np.arange(1, n + 1),
        "voxels":      rng.integers(10, 5000, n),
        # 11 geometric canonical features (log_voxels derived in _preprocess_for_training)
        "sphericity":  rng.uniform(0.3, 1.0, n),
        "anisotropy":  rng.uniform(0.0, 0.9, n),
        "flatness":    rng.uniform(1.0, 5.0, n),
        "elongation":  rng.uniform(1.0, 8.0, n),
        "fill_frac":   rng.uniform(0.1, 1.0, n),
        "r1":          rng.uniform(0.1, 1.0, n),
        "r2":          rng.uniform(0.0, 1.0, n),
        "ar_zx":       rng.uniform(0.5, 5.0, n),
        "ar_zy":       rng.uniform(0.5, 5.0, n),
        "ar_yx":       rng.uniform(0.5, 5.0, n),
        "int_std_rel": rng.uniform(0.0, 0.5, n),
        # Labels
        "cls_label":   ["pore"] * n_pore + ["crack"] * n_crack,
    })


# ---------------------------------------------------------------------------
# _optimal_f1_threshold
# ---------------------------------------------------------------------------

class TestOptimalF1Threshold:
    def test_perfectly_separable_data(self):
        """Threshold should land between the two score clusters."""
        from tomographia.classification.logistic import _optimal_f1_threshold
        # Positives > 0.65, negatives < 0.65
        y_true  = np.array([1, 1, 0, 0, 1, 0])
        y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.4])
        thr = _optimal_f1_threshold(y_true, y_score)
        preds = (y_score >= thr).astype(int)
        assert (preds == y_true).all(), (
            f"Threshold {thr:.4f} failed to separate classes (preds={preds.tolist()})"
        )

    def test_threshold_in_unit_interval(self):
        rng = np.random.default_rng(1)
        from tomographia.classification.logistic import _optimal_f1_threshold
        y_true  = rng.integers(0, 2, 100)
        y_score = rng.uniform(0.0, 1.0, 100)
        thr = _optimal_f1_threshold(y_true, y_score)
        assert 0.0 <= thr <= 1.0, f"Threshold {thr} out of [0,1]"

    def test_returns_float(self):
        from tomographia.classification.logistic import _optimal_f1_threshold
        thr = _optimal_f1_threshold(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8]),
        )
        assert isinstance(thr, float)


# ---------------------------------------------------------------------------
# fit_classifier
# ---------------------------------------------------------------------------

class TestFitClassifier:
    @pytest.fixture(scope="class")
    def bundle(self, tmp_path_factory):
        from tomographia.classification.logistic import fit_classifier
        tmp = tmp_path_factory.mktemp("fit_cls")
        csv = tmp / "descriptors.csv"
        _make_df(n_pore=60, n_crack=20).to_csv(csv, index=False)
        return fit_classifier([csv], voxel_sizes=[0.014], random_state=42)

    def test_bundle_keys(self, bundle):
        expected = {"model", "scaler", "features", "threshold",
                    "holdout_results", "volumes", "voxel_sizes"}
        assert set(bundle.keys()) == expected

    def test_threshold_in_unit_interval(self, bundle):
        assert 0.0 <= bundle["threshold"] <= 1.0

    def test_all_features_in_known_columns(self, bundle):
        for feat in bundle["features"]:
            assert feat in [
                "sphericity", "anisotropy", "flatness", "elongation", "fill_frac",
                "r1", "r2", "ar_zx", "ar_zy", "ar_yx", "int_std_rel", "log_voxels",
            ], f"Unexpected feature in bundle: {feat!r}"

    def test_holdout_metric_keys(self, bundle):
        ho = bundle["holdout_results"]
        for key in ("accuracy", "f1_crack", "f1_pore", "roc_auc"):
            assert key in ho, f"Missing holdout metric: {key!r}"

    def test_holdout_metrics_in_range(self, bundle):
        ho = bundle["holdout_results"]
        for key in ("accuracy", "f1_crack", "f1_pore", "f1_macro",
                    "precision_crack", "recall_crack"):
            v = ho[key]
            assert 0.0 <= v <= 1.0, f"Holdout metric {key}={v} out of [0,1]"

    def test_feats_avail_excludes_missing_column(self, tmp_path):
        """Bundle must store the actual usable subset, not the full features param."""
        from tomographia.classification.logistic import fit_classifier
        df = _make_df()
        df = df.drop(columns=["int_std_rel"])  # intentionally drop one canonical feature
        csv = tmp_path / "d.csv"
        df.to_csv(csv, index=False)
        bundle = fit_classifier([csv], voxel_sizes=[0.014], random_state=42)
        assert "int_std_rel" not in bundle["features"]

    def test_len_mismatch_raises(self, tmp_path):
        from tomographia.classification.logistic import fit_classifier
        csv = tmp_path / "d.csv"
        _make_df().to_csv(csv, index=False)
        with pytest.raises(ValueError, match="same length"):
            fit_classifier([csv, csv], voxel_sizes=[0.014])  # 2 CSVs, 1 voxel size

    def test_log_voxels_computed_from_raw_voxels(self, tmp_path):
        """_preprocess_for_training must recompute log_voxels from voxels column."""
        from tomographia.classification.logistic import fit_classifier
        df = _make_df()
        csv = tmp_path / "d.csv"
        df.to_csv(csv, index=False)
        bundle = fit_classifier([csv], voxel_sizes=[0.014], random_state=42)
        # log_voxels should be in feats_avail when voxels column is present
        assert "log_voxels" in bundle["features"]

    def test_save_and_reload(self, bundle, tmp_path):
        import joblib
        from tomographia.classification.logistic import fit_classifier
        df = _make_df()
        csv = tmp_path / "d.csv"
        df.to_csv(csv, index=False)
        save_path = tmp_path / "bundle.joblib"
        fit_classifier([csv], voxel_sizes=[0.014], save_path=save_path, random_state=42)
        assert save_path.exists()
        loaded = joblib.load(str(save_path))
        assert set(loaded.keys()) == set(bundle.keys())


# ---------------------------------------------------------------------------
# classify_defects_logistic
# ---------------------------------------------------------------------------

class TestClassifyDefectsLogistic:
    @pytest.fixture(scope="class")
    def bundle(self, tmp_path_factory):
        from tomographia.classification.logistic import fit_classifier
        tmp = tmp_path_factory.mktemp("infer_bundle")
        csv = tmp / "d.csv"
        _make_df(n_pore=60, n_crack=20).to_csv(csv, index=False)
        return fit_classifier([csv], voxel_sizes=[0.014], random_state=42)

    def test_adds_three_columns(self, bundle):
        from tomographia.classification.logistic import classify_defects_logistic
        result = classify_defects_logistic(_make_df(), bundle)
        for col in ("cls_proba", "cls_pred", "cls_label"):
            assert col in result.columns, f"Missing output column: {col!r}"

    def test_cls_pred_is_binary(self, bundle):
        from tomographia.classification.logistic import classify_defects_logistic
        result = classify_defects_logistic(_make_df(), bundle)
        assert set(result["cls_pred"].unique()).issubset({0, 1})

    def test_cls_label_consistent_with_cls_pred(self, bundle):
        from tomographia.classification.logistic import classify_defects_logistic
        result = classify_defects_logistic(_make_df(), bundle)
        for _, row in result.iterrows():
            expected = "crack" if row["cls_pred"] == 1 else "pore"
            assert row["cls_label"] == expected

    def test_cls_proba_in_unit_interval(self, bundle):
        from tomographia.classification.logistic import classify_defects_logistic
        result = classify_defects_logistic(_make_df(), bundle)
        assert (result["cls_proba"] >= 0.0).all()
        assert (result["cls_proba"] <= 1.0).all()

    def test_threshold_override(self, bundle):
        """Override threshold to 0.0 → everything labelled crack."""
        from tomographia.classification.logistic import classify_defects_logistic
        result = classify_defects_logistic(_make_df(), bundle, threshold=0.0)
        assert (result["cls_pred"] == 1).all()

    def test_threshold_override_one(self, bundle):
        """Override threshold to 1.0 → everything labelled pore."""
        from tomographia.classification.logistic import classify_defects_logistic
        result = classify_defects_logistic(_make_df(), bundle, threshold=1.0)
        assert (result["cls_pred"] == 0).all()


# ---------------------------------------------------------------------------
# classify_defects dispatch
# ---------------------------------------------------------------------------

class TestClassifyDefectsDispatch:
    @pytest.fixture(scope="class")
    def bundle(self, tmp_path_factory):
        from tomographia.classification.logistic import fit_classifier
        tmp = tmp_path_factory.mktemp("dispatch")
        csv = tmp / "d.csv"
        _make_df(n_pore=60, n_crack=20).to_csv(csv, index=False)
        return fit_classifier([csv], voxel_sizes=[0.014], random_state=42)

    def test_logistic_dispatch_adds_columns(self, bundle):
        from tomographia.classification import classify_defects
        result = classify_defects(_make_df(), method="logistic", bundle=bundle)
        assert "cls_label" in result.columns

    def test_kmeans_dispatch_adds_cluster_column(self):
        from tomographia.classification import classify_defects
        features = ["sphericity", "elongation", "flatness", "fill_frac"]
        result = classify_defects(_make_df(), method="kmeans", features=features)
        assert "cluster" in result.columns

    def test_unknown_method_raises(self):
        from tomographia.classification import classify_defects
        with pytest.raises(ValueError, match="Unknown method"):
            classify_defects(_make_df(), method="svm")

    def test_logistic_without_bundle_raises(self):
        from tomographia.classification import classify_defects
        with pytest.raises(ValueError, match="bundle="):
            classify_defects(_make_df(), method="logistic")


# ---------------------------------------------------------------------------
# build_classified_volume
# ---------------------------------------------------------------------------

class TestBuildClassifiedVolume:
    def test_background_is_zero(self):
        from tomographia.classification.logistic import build_classified_volume
        cc  = np.zeros((5, 5, 5), dtype=np.int32)
        df  = pd.DataFrame({"label_id": [], "cls_pred": []})
        out = build_classified_volume(cc, df)
        assert (out == 0).all()

    def test_pore_encoded_as_one(self):
        from tomographia.classification.logistic import build_classified_volume
        cc  = np.array([[[0, 1]]], dtype=np.int32)
        df  = pd.DataFrame({"label_id": [1], "cls_pred": [0]})
        out = build_classified_volume(cc, df)
        assert out[0, 0, 0] == 0   # background
        assert out[0, 0, 1] == 1   # pore → code 1

    def test_crack_encoded_as_two(self):
        from tomographia.classification.logistic import build_classified_volume
        cc  = np.array([[[0, 1]]], dtype=np.int32)
        df  = pd.DataFrame({"label_id": [1], "cls_pred": [1]})
        out = build_classified_volume(cc, df)
        assert out[0, 0, 0] == 0   # background
        assert out[0, 0, 1] == 2   # crack → code 2

    def test_mixed_labels(self):
        from tomographia.classification.logistic import build_classified_volume
        cc  = np.array([[[0, 1, 2, 3]]], dtype=np.int32)
        df  = pd.DataFrame({
            "label_id": [1, 2, 3],
            "cls_pred": [0, 1, 0],   # pore, crack, pore
        })
        out = build_classified_volume(cc, df)
        assert out[0, 0, 0] == 0   # background
        assert out[0, 0, 1] == 1   # pore
        assert out[0, 0, 2] == 2   # crack
        assert out[0, 0, 3] == 1   # pore

    def test_output_dtype_is_uint8(self):
        from tomographia.classification.logistic import build_classified_volume
        cc  = np.array([[[0, 1]]], dtype=np.int32)
        df  = pd.DataFrame({"label_id": [1], "cls_pred": [0]})
        out = build_classified_volume(cc, df)
        assert out.dtype == np.uint8

    def test_output_shape_matches_input(self):
        from tomographia.classification.logistic import build_classified_volume
        shape = (10, 8, 6)
        cc = np.zeros(shape, dtype=np.int32)
        df = pd.DataFrame({"label_id": [], "cls_pred": []})
        assert build_classified_volume(cc, df).shape == shape
