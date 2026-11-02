"""Tests for classification: logistic fit, dispatch, and volume encoding."""
import numpy as np
import pandas as pd
import pytest

from tomographia.components import CANONICAL_FEATURES


# ---------------------------------------------------------------------------
# Shared synthetic-data factory
# ---------------------------------------------------------------------------

def _make_df(n_pore: int = 60, n_crack: int = 20, seed: int = 42) -> pd.DataFrame:
    """Synthetic descriptor table: cracks are flatter and less spherical."""
    rng = np.random.default_rng(seed)
    n = n_pore + n_crack
    crack = np.r_[np.zeros(n_pore), np.ones(n_crack)].astype(bool)
    return pd.DataFrame({
        "label_id":    np.arange(1, n + 1),
        "voxels":      rng.integers(10, 5000, n),
        "sphericity":  np.where(crack, rng.uniform(0.2, 0.6, n), rng.uniform(0.5, 1.0, n)),
        "elongation":  rng.uniform(0.2, 1.0, n),
        "flatness":    np.where(crack, rng.uniform(0.05, 0.4, n), rng.uniform(0.3, 1.0, n)),
        "fill_frac":   rng.uniform(0.1, 1.0, n),
        "ar_zx":       rng.uniform(0.5, 5.0, n),
        "ar_zy":       rng.uniform(0.5, 5.0, n),
        "ar_yx":       rng.uniform(0.5, 5.0, n),
        "int_std_rel": rng.uniform(0.0, 0.5, n),
        "log_volume":  rng.uniform(-5, -2, n),
        "label":       ["pore"] * n_pore + ["crack"] * n_crack,
    })


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
        return fit_classifier([csv], random_state=42)

    def test_bundle_keys(self, bundle):
        assert {"model", "features", "threshold", "holdout", "holdout_results",
                "volumes", "n_train", "n_test", "tomographia_version"} <= set(bundle)

    def test_features_default_to_canonical(self, bundle):
        assert bundle["features"] == list(CANONICAL_FEATURES)

    def test_threshold_in_unit_interval(self, bundle):
        assert 0.0 <= bundle["threshold"] <= 1.0

    def test_holdout_metrics_in_range(self, bundle):
        ho = bundle["holdout_results"]
        for key in ("accuracy", "f1_crack", "f1_pore", "f1_macro",
                    "precision_crack", "recall_crack", "roc_auc"):
            assert 0.0 <= ho[key] <= 1.0, f"Holdout metric {key}={ho[key]} out of [0,1]"
        assert ho["n_pore"] + ho["n_crack"] == bundle["n_test"]

    def test_learns_separable_signal(self, bundle):
        assert bundle["holdout_results"]["roc_auc"] > 0.9

    def test_accepts_dataframes(self):
        from tomographia.classification.logistic import fit_classifier
        b = fit_classifier([_make_df(seed=1), _make_df(seed=2)])
        assert b["n_train"] + b["n_test"] == 160

    def test_missing_feature_raises(self):
        from tomographia.classification.logistic import fit_classifier
        with pytest.raises(KeyError, match="int_std_rel"):
            fit_classifier([_make_df().drop(columns=["int_std_rel"])])

    def test_all_nan_feature_raises(self):
        from tomographia.classification.logistic import fit_classifier
        df = _make_df().assign(int_std_rel=np.nan)
        with pytest.raises(ValueError, match="no finite value"):
            fit_classifier([df])

    def test_unlabelled_rows_ignored_unknown_labels_raise(self):
        from tomographia.classification.logistic import fit_classifier
        df = _make_df()
        df.loc[:9, "label"] = np.nan
        assert fit_classifier([df])["n_train"] + fit_classifier([df])["n_test"] == 70
        df.loc[0, "label"] = "bubble"
        with pytest.raises(ValueError, match="bubble"):
            fit_classifier([df])

    def test_refuses_to_train_on_predictions(self, bundle):
        from tomographia.classification.logistic import classify_defects_logistic, fit_classifier
        classified = classify_defects_logistic(_make_df(), bundle)
        with pytest.raises(ValueError, match="predictions"):
            fit_classifier([classified], label_col="cls_label")

    def test_nan_counts_reported(self):
        from tomographia.classification.logistic import fit_classifier
        df = _make_df().assign(int_std_rel=np.nan)
        b = fit_classifier([_make_df(), df])
        assert b["nan_counts"]["int_std_rel"] == {"<DataFrame 1>": 80}

    def test_volume_holdout_keeps_volumes_apart(self):
        from tomographia.classification.logistic import fit_classifier
        tables = [_make_df(seed=s) for s in range(5)]
        b = fit_classifier(tables, holdout="volume", test_size=0.2)
        assert b["holdout"] == "volume" and b["n_test"] == 80     # one whole volume
        assert len(b["test_volumes"]) == 1
        with pytest.raises(ValueError, match="two"):
            fit_classifier(tables[:1], holdout="volume")

    def test_save_and_reload(self, tmp_path):
        import joblib
        from tomographia.classification.logistic import fit_classifier
        save_path = tmp_path / "bundle.joblib"
        b = fit_classifier([_make_df()], save_path=save_path, random_state=42)
        loaded = joblib.load(str(save_path))
        assert set(loaded) == set(b)


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
        return fit_classifier([csv], random_state=42)

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

    def test_missing_feature_raises(self, bundle):
        from tomographia.classification.logistic import classify_defects_logistic
        with pytest.raises(KeyError, match="int_std_rel"):
            classify_defects_logistic(_make_df().drop(columns=["int_std_rel"]), bundle)

    def test_nan_feature_imputed_with_training_median(self, bundle):
        """A NaN is replaced by the training median, not by 0."""
        from tomographia.classification.logistic import classify_defects_logistic
        df = _make_df().iloc[:1].copy()
        med = bundle["model"][0].statistics_[bundle["features"].index("log_volume")]
        p_nan = classify_defects_logistic(df.assign(log_volume=np.nan), bundle)["cls_proba"].iloc[0]
        p_med = classify_defects_logistic(df.assign(log_volume=med), bundle)["cls_proba"].iloc[0]
        assert p_nan == pytest.approx(p_med)

    def test_out_of_range_features_warn(self, bundle, caplog):
        """log_volume in µm³ instead of mm³ (+9 decades) must not pass silently."""
        from tomographia.classification.logistic import classify_defects_logistic
        df = _make_df()
        df["log_volume"] += 9
        with caplog.at_level("WARNING"):
            classify_defects_logistic(df, bundle)
        assert "log_volume" in caplog.text and "training range" in caplog.text

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
        return fit_classifier([csv], random_state=42)

    def test_logistic_dispatch_adds_columns(self, bundle):
        from tomographia.classification import classify_defects
        result = classify_defects(_make_df(), method="logistic", bundle=bundle)
        assert "cls_label" in result.columns

    def test_kmeans_dispatch_adds_cluster_column(self):
        from tomographia.classification import classify_defects
        features = ["sphericity", "elongation", "flatness", "fill_frac"]
        result = classify_defects(_make_df(), method="kmeans", features=features)
        assert "cluster" in result.columns

    def test_kmeans_default_features(self):
        from tomographia.classification import classify_defects
        result = classify_defects(_make_df(), method="kmeans", n_clusters=3)
        assert result["cluster"].notna().all()

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
