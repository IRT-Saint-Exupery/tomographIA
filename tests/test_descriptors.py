"""Tests for compute_descriptors geometric formulas and build_cc_mask."""
import numpy as np
import pytest
from skimage.measure import label as sk_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_ball(radius: int = 7, padding: int = 3) -> np.ndarray:
    d = 2 * (radius + padding) + 1
    arr = np.zeros((d, d, d), dtype=np.uint8)
    c = radius + padding
    z, y, x = np.ogrid[:d, :d, :d]
    arr[(z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius ** 2] = 1
    return arr


def _solid_box(dz: int, dy: int, dx: int, padding: int = 2) -> np.ndarray:
    arr = np.zeros((dz + 2 * padding, dy + 2 * padding, dx + 2 * padding), dtype=np.uint8)
    arr[padding:padding + dz, padding:padding + dy, padding:padding + dx] = 1
    return arr


def _solid_ellipsoid(rz: int, ry: int, rx: int, padding: int = 3) -> np.ndarray:
    """Solid ellipsoid with semi-axes (rz, ry, rx). Corners of bbox are empty."""
    d = (2 * (rz + padding) + 1, 2 * (ry + padding) + 1, 2 * (rx + padding) + 1)
    arr = np.zeros(d, dtype=np.uint8)
    cz, cy, cx = rz + padding, ry + padding, rx + padding
    z, y, x = np.ogrid[:d[0], :d[1], :d[2]]
    arr[((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0] = 1
    return arr


# ---------------------------------------------------------------------------
# Fixtures — compute once per module (marching cubes is slow)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ball_df():
    from tomographia.components.pipeline2 import compute_descriptors
    cc = sk_label(_solid_ball(), connectivity=3)
    return compute_descriptors(cc, voxel_size=(1.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def flat_disk_df():
    """Flat disk ellipsoid (rz=1, ry=10, rx=10): bbox corners empty → marching cubes works.
    Flat shapes have elongation ≈ 2.0 under the physical inertia tensor convention."""
    from tomographia.components.pipeline2 import compute_descriptors
    cc = sk_label(_solid_ellipsoid(rz=1, ry=10, rx=10), connectivity=3)
    return compute_descriptors(cc, voxel_size=(1.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def cube_df():
    from tomographia.components.pipeline2 import compute_descriptors
    cc = sk_label(_solid_box(dz=8, dy=8, dx=8), connectivity=3)
    return compute_descriptors(cc, voxel_size=(1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Column coverage
# ---------------------------------------------------------------------------

class TestDescriptorColumns:
    CANONICAL = [
        "sphericity", "anisotropy", "flatness", "elongation",
        "fill_frac", "r1", "r2", "ar_zx", "ar_zy", "ar_yx",
        "log_voxels",
    ]

    def test_all_canonical_features_present(self, ball_df):
        for col in self.CANONICAL:
            assert col in ball_df.columns, f"Missing column: {col}"

    def test_single_component_returned(self, ball_df):
        assert len(ball_df) == 1

    def test_label_id_starts_at_one(self, ball_df):
        assert ball_df.iloc[0]["label_id"] == 1


# ---------------------------------------------------------------------------
# Sphericity
# ---------------------------------------------------------------------------

class TestSphericity:
    def test_ball_sphericity_close_to_one(self, ball_df):
        s = float(ball_df.iloc[0]["sphericity"])
        assert 0.85 <= s <= 1.05, f"Ball sphericity expected ≈1.0, got {s:.4f}"

    def test_flat_disk_sphericity_less_than_ball(self, ball_df, flat_disk_df):
        ball_s = float(ball_df.iloc[0]["sphericity"])
        disk_s = float(flat_disk_df.iloc[0]["sphericity"])
        assert np.isfinite(disk_s), "Flat disk sphericity should be finite"
        assert disk_s < ball_s, (
            f"Flat disk (s={disk_s:.3f}) should have lower sphericity "
            f"than ball (s={ball_s:.3f})"
        )


# ---------------------------------------------------------------------------
# fill_frac
# ---------------------------------------------------------------------------

class TestFillFrac:
    def test_solid_cube_fill_frac_is_one(self, cube_df):
        ff = float(cube_df.iloc[0]["fill_frac"])
        assert abs(ff - 1.0) < 1e-9, f"Solid cube fill_frac expected 1.0, got {ff}"

    def test_solid_box_fill_frac_is_one(self):
        from tomographia.components.pipeline2 import compute_descriptors
        cc = sk_label(_solid_box(dz=6, dy=6, dx=6), connectivity=3)
        df = compute_descriptors(cc, voxel_size=(1.0, 1.0, 1.0))
        ff = float(df.iloc[0]["fill_frac"])
        assert abs(ff - 1.0) < 1e-9, f"Solid box fill_frac expected 1.0, got {ff}"


# ---------------------------------------------------------------------------
# log_voxels formula
# ---------------------------------------------------------------------------

class TestLogVoxels:
    def test_unit_voxel_formula(self, ball_df):
        row = ball_df.iloc[0]
        # voxel_vol = 1.0 * 1.0 * 1.0 = 1.0, so log_voxels = log(voxels) + log(1.0)
        expected = np.log(row["voxels"]) + np.log(1.0)
        assert abs(float(row["log_voxels"]) - expected) < 1e-10

    def test_physical_voxel_size(self):
        from tomographia.components.pipeline2 import compute_descriptors
        vs = 0.014
        cc = sk_label(_solid_box(dz=4, dy=4, dx=4), connectivity=3)
        df = compute_descriptors(cc, voxel_size=(vs, vs, vs))
        row = df.iloc[0]
        # voxel_vol = vs * vs * vs
        expected = np.log(row["voxels"]) + np.log(vs ** 3)
        assert abs(float(row["log_voxels"]) - expected) < 1e-10

    def test_formula_matches_physical_volume(self):
        from tomographia.components.pipeline2 import compute_descriptors
        vs = 0.020
        cc = sk_label(_solid_box(dz=4, dy=4, dx=4), connectivity=3)
        df = compute_descriptors(cc, voxel_size=(vs, vs, vs))
        row = df.iloc[0]
        expected = np.log(row["voxels"]) + np.log(vs ** 3)
        assert abs(float(row["log_voxels"]) - expected) < 1e-10


# ---------------------------------------------------------------------------
# Elongation
# ---------------------------------------------------------------------------

class TestElongation:
    def test_flat_disk_has_elongation_near_two(self, flat_disk_df):
        """A flat disk has e0≈2*e1 under the physical inertia tensor convention."""
        e = float(flat_disk_df.iloc[0]["elongation"])
        assert 1.5 <= e <= 2.5, (
            f"Flat disk elongation expected ≈2.0 (inertia convention), got {e:.3f}"
        )

    def test_ball_elongation_near_one(self, ball_df):
        e = float(ball_df.iloc[0]["elongation"])
        assert 0.9 <= e <= 1.1, f"Ball elongation expected ≈1.0, got {e:.3f}"


# ---------------------------------------------------------------------------
# Analytical cross-check (inertia tensor formula vs. compute_descriptors)
# ---------------------------------------------------------------------------

class TestAnalyticalCrossCheck:
    """
    Cross-check compute_descriptors against the closed-form solid-ellipsoid
    formula for a scalene ellipsoid where all three axes are distinct.

    For a solid uniform ellipsoid with semi-axes a(z), b(y), c(x), the
    per-voxel principal inertia moments are (derivation: standard rigid-body
    inertia tensor, normalised by voxel count N):

        I_z = (c² + b²) / 5        [rotation around z — sum of x²+y²]
        I_y = (c² + a²) / 5        [rotation around y — sum of x²+z²]
        I_x = (b² + a²) / 5        [rotation around x — sum of y²+z²]

    Shape chosen: rz=3, ry=6, rx=9 — all axes distinct, no degenerate
    symmetry, axis ratio not discussed earlier in the session.

    Hand-computed theoretical values:
        I_z = (81+36)/5 = 23.40   I_y = (81+9)/5 = 18.00   I_x = (36+9)/5 = 9.00
        sorted descending → e0=23.40, e1=18.00, e2=9.00
        elongation  = 23.40/18.00 = 1.300
        flatness    = 18.00/9.00  = 2.000
        r1          = 18.00/23.40 = 0.769
        r2          =  9.00/23.40 = 0.385
        anisotropy  = 1 - 0.385   = 0.615
    Tolerance: 5 % relative to cover voxelisation noise.
    """

    THEORY = {
        "elongation": 1.300,
        "flatness":   2.000,
        "r1":         0.769,
        "r2":         0.385,
        "anisotropy": 0.615,
    }

    @pytest.fixture(scope="class")
    def scalene_df(self):
        from tomographia.components.pipeline2 import compute_descriptors
        cc = sk_label(_solid_ellipsoid(rz=3, ry=6, rx=9), connectivity=3)
        return compute_descriptors(cc, voxel_size=(1.0, 1.0, 1.0))

    @pytest.mark.parametrize("descriptor,theory", [
        ("elongation", 1.300),
        ("flatness",   2.000),
        ("r1",         0.769),
        ("r2",         0.385),
        ("anisotropy", 0.615),
    ])
    def test_descriptor_within_5pct_of_theory(self, scalene_df, descriptor, theory):
        observed = float(scalene_df.iloc[0][descriptor])
        rel_error = abs(observed - theory) / theory
        assert rel_error <= 0.05, (
            f"{descriptor}: observed={observed:.4f}, theory={theory:.4f}, "
            f"relative error={rel_error:.3f} (> 5 % tolerance)"
        )


# ---------------------------------------------------------------------------
# build_cc_mask
# ---------------------------------------------------------------------------

class TestBuildCcMask:
    def test_output_is_integer_labeled(self):
        from tomographia.components.pipeline2 import build_cc_mask
        vol = np.zeros((10, 10, 10), dtype=np.uint8)
        vol[3:7, 3:7, 3:7] = 1
        cc = build_cc_mask(vol, bridge_radius=0, dilation_radius=0)
        assert cc.dtype.kind in ("i", "u"), "Output should be integer dtype"
        assert cc.max() >= 1, "Should have at least one labeled component"

    def test_background_stays_zero(self):
        from tomographia.components.pipeline2 import build_cc_mask
        vol = np.zeros((10, 10, 10), dtype=np.uint8)
        cc = build_cc_mask(vol, bridge_radius=0, dilation_radius=0)
        assert cc.max() == 0, "Empty volume should have no components"

    def test_two_separate_objects_get_different_labels(self):
        from tomographia.components.pipeline2 import build_cc_mask
        vol = np.zeros((20, 10, 10), dtype=np.uint8)
        vol[1:4, 3:6, 3:6] = 1   # first object
        vol[16:19, 3:6, 3:6] = 1  # second object, well separated
        cc = build_cc_mask(vol, bridge_radius=0, dilation_radius=0)
        assert cc.max() >= 2, "Two separate objects should get distinct labels"
