"""Tests for PatchedVolumeBig and normalization utilities."""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")


# ---------------------------------------------------------------------------
# PatchedVolumeBig
# ---------------------------------------------------------------------------

class TestPatchedVolumeBig:
    def test_round_trip_zero_error(self):
        """Rebuild a volume from patches and verify it matches the original."""
        from tomographia.segmentation._dataset import PatchedVolumeBig
        rng  = np.random.default_rng(0)
        vol  = rng.random((20, 32, 32)).astype(np.float32)
        ds   = PatchedVolumeBig(vol, patch_shape=(8, 16, 16), padding=2)
        patches = np.stack([ds[i].numpy() for i in range(len(ds))])
        rebuilt = ds.rebuild_volume_from_patches(patches)
        np.testing.assert_allclose(
            rebuilt, vol, atol=1e-5,
            err_msg=f"Max rebuild error: {np.abs(rebuilt - vol).max():.2e}",
        )

    def test_patch_count_is_positive(self):
        from tomographia.segmentation._dataset import PatchedVolumeBig
        vol = np.ones((10, 16, 16), dtype=np.float32)
        ds  = PatchedVolumeBig(vol, patch_shape=(4, 8, 8), padding=1)
        assert len(ds) > 0

    def test_patch_shape_correct(self):
        from tomographia.segmentation._dataset import PatchedVolumeBig
        patch_shape = (8, 16, 16)
        vol = np.ones((20, 32, 32), dtype=np.float32)
        ds  = PatchedVolumeBig(vol, patch_shape=patch_shape, padding=2)
        for i in range(len(ds)):
            assert ds[i].shape == torch.Size(patch_shape), (
                f"Patch {i} has shape {tuple(ds[i].shape)}, expected {patch_shape}"
            )

    def test_small_volume_uses_single_patch(self):
        """When volume fits inside one patch, exactly one patch is produced."""
        from tomographia.segmentation._dataset import PatchedVolumeBig
        vol = np.ones((4, 4, 4), dtype=np.float32)
        ds  = PatchedVolumeBig(vol, patch_shape=(8, 8, 8), padding=0)
        assert len(ds) == 1

    def test_rebuild_wrong_count_raises(self):
        from tomographia.segmentation._dataset import PatchedVolumeBig
        vol  = np.ones((10, 10, 10), dtype=np.float32)
        ds   = PatchedVolumeBig(vol, patch_shape=(4, 4, 4), padding=1)
        fake = np.ones((1, 4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            ds.rebuild_volume_from_patches(fake)

    def test_item_is_float_tensor(self):
        from tomographia.segmentation._dataset import PatchedVolumeBig
        vol   = np.ones((8, 8, 8), dtype=np.float32)
        ds    = PatchedVolumeBig(vol, patch_shape=(4, 4, 4), padding=0)
        patch = ds[0]
        assert isinstance(patch, torch.Tensor)
        assert patch.dtype == torch.float32

    def test_non_3d_volume_raises(self):
        from tomographia.segmentation._dataset import PatchedVolumeBig
        with pytest.raises(ValueError, match="3-D"):
            PatchedVolumeBig(np.ones((8, 8)), patch_shape=(4, 4, 4), padding=0)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_robust_minmax_output_in_minus1_to_1(self):
        from tomographia.segmentation._io import apply_normalization_by_name
        rng = np.random.default_rng(0)
        vol = rng.random((10, 10, 10)).astype(np.float32)
        out = apply_normalization_by_name("robust_minmax", vol)
        assert float(out.min()) >= -1.0 - 1e-5
        assert float(out.max()) <= 1.0 + 1e-5

    def test_no_norm_returns_original_array(self):
        from tomographia.segmentation._io import apply_normalization_by_name
        vol = np.random.default_rng(1).random((5, 5, 5)).astype(np.float32)
        out = apply_normalization_by_name("no_norm", vol)
        np.testing.assert_array_equal(vol, out)

    def test_unknown_name_raises(self):
        from tomographia.segmentation._io import apply_normalization_by_name
        vol = np.random.default_rng(2).random((5, 5, 5)).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown normalization"):
            apply_normalization_by_name("not_a_real_norm", vol)

    def test_minmax_is_volume_wide(self):
        """A dark slice must stay dark: statistics are global, not per slice."""
        from tomographia.segmentation._io import apply_normalization_by_name
        vol = np.stack([np.full((4, 4), 10.0), np.full((4, 4), 100.0)]).astype(np.float32)
        vol[0, 0, 0] = 0.0
        out = apply_normalization_by_name("minmax", vol)
        assert float(out[0].max()) < -0.7 and float(out[1].min()) > 0.99

    def test_standardization_approximately_zero_mean(self):
        from tomographia.segmentation._io import apply_normalization_by_name
        rng = np.random.default_rng(3)
        vol = rng.uniform(100, 200, (20, 20, 20)).astype(np.float32)
        out = apply_normalization_by_name("standardization", vol)
        assert abs(float(out.mean())) < 0.1

    def test_minmax_output_in_minus1_to_1(self):
        from tomographia.segmentation._io import apply_normalization_by_name
        rng = np.random.default_rng(4)
        vol = rng.random((10, 10, 10)).astype(np.float32)
        out = apply_normalization_by_name("minmax", vol)
        assert float(out.min()) >= -1.0 - 1e-5
        assert float(out.max()) <= 1.0 + 1e-5
