"""Tests for compute_descriptors (definitions checked on analytical shapes) and build_cc_mask."""
import numpy as np
import pytest
from skimage.measure import label as sk_label

from tomographia.components import CANONICAL_FEATURES, build_cc_mask, compute_descriptors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_ellipsoid(rz: float, ry: float, rx: float, padding: int = 3) -> np.ndarray:
    """Solid ellipsoid with semi-axes (rz, ry, rx) in voxels."""
    d = tuple(int(2 * (np.ceil(r) + padding) + 1) for r in (rz, ry, rx))
    c = [(n - 1) / 2 for n in d]
    z, y, x = np.ogrid[:d[0], :d[1], :d[2]]
    return ((((z - c[0]) / rz) ** 2 + ((y - c[1]) / ry) ** 2 + ((x - c[2]) / rx) ** 2) <= 1.0
            ).astype(np.uint8)


def _solid_box(dz: int, dy: int, dx: int, padding: int = 2) -> np.ndarray:
    arr = np.zeros((dz + 2 * padding, dy + 2 * padding, dx + 2 * padding), dtype=np.uint8)
    arr[padding:padding + dz, padding:padding + dy, padding:padding + dx] = 1
    return arr


def _row(mask, voxel_size=1.0, **kw):
    return compute_descriptors(sk_label(mask, connectivity=3), voxel_size, **kw).iloc[0]


@pytest.fixture(scope="module")
def ball():
    return _row(_solid_ellipsoid(10, 10, 10))


@pytest.fixture(scope="module")
def scalene():
    return _row(_solid_ellipsoid(3, 6, 9))


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def test_canonical_features_present(ball):
    for col in CANONICAL_FEATURES:
        assert col in ball.index, f"Missing column: {col}"


def test_empty_volume_returns_empty_table():
    df = compute_descriptors(np.zeros((6, 6, 6), dtype=np.int32), 1.0)
    assert len(df) == 0 and set(CANONICAL_FEATURES) <= set(df.columns)


# ---------------------------------------------------------------------------
# Principal axes, elongation, flatness (IBSI definitions)
# ---------------------------------------------------------------------------

def test_semi_axes_recover_ellipsoid(scalene):
    np.testing.assert_allclose([scalene.axis_minor, scalene.axis_mid, scalene.axis_major],
                               [3, 6, 9], rtol=0.05)


def test_elongation_flatness_definitions(scalene):
    assert scalene.elongation == pytest.approx(6 / 9, rel=0.05)
    assert scalene.flatness == pytest.approx(3 / 9, rel=0.05)


def test_sphere_is_neither_elongated_nor_flat(ball):
    assert ball.elongation == pytest.approx(1.0, abs=0.01)
    assert ball.flatness == pytest.approx(1.0, abs=0.01)


def test_needle_and_plate_are_distinguished():
    needle = _row(_solid_ellipsoid(2, 2, 15))
    plate = _row(_solid_ellipsoid(2, 15, 15))
    assert needle.elongation < 0.2 and needle.flatness < 0.2      # one dominant axis
    assert plate.elongation > 0.95 and plate.flatness < 0.2       # two dominant axes


def test_single_voxel_is_defined():
    one = _row(np.pad(np.ones((1, 1, 1), np.uint8), 2))
    assert one.elongation == pytest.approx(1.0) and one.flatness == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Physical voxel size
# ---------------------------------------------------------------------------

def test_anisotropic_voxels_use_physical_shape():
    """A sphere of radius 8 sampled with dz = 2 (4 slices radius) stays a sphere."""
    r = _row(_solid_ellipsoid(4, 8, 8), voxel_size=(2.0, 1.0, 1.0))
    assert r.flatness > 0.95 and r.elongation > 0.95
    assert r.ar_zx == pytest.approx(1.0, rel=0.15)


def test_scalar_and_tuple_voxel_size_agree():
    m = _solid_ellipsoid(3, 5, 7)
    a, b = _row(m, 0.014), _row(m, (0.014, 0.014, 0.014))
    for c in CANONICAL_FEATURES:
        if c != "int_std_rel":
            assert a[c] == pytest.approx(b[c])


def test_volume_and_log_volume():
    vs = 0.02
    r = _row(_solid_box(4, 4, 4), voxel_size=vs)
    assert r.voxels == 64
    assert r.volume == pytest.approx(64 * vs ** 3)
    assert r.log_volume == pytest.approx(np.log10(64 * vs ** 3))


def test_shape_features_scale_invariant():
    m = _solid_ellipsoid(3, 6, 9)
    a, b = _row(m, 1.0), _row(m, 0.014)
    for c in ("sphericity", "elongation", "flatness", "fill_frac", "ar_zx", "ar_zy", "ar_yx"):
        assert a[c] == pytest.approx(b[c], rel=1e-6)


# ---------------------------------------------------------------------------
# Sphericity
# ---------------------------------------------------------------------------

def test_ball_sphericity_close_to_one(ball):
    assert 0.88 <= ball.sphericity <= 1.0


@pytest.mark.parametrize("mask", [
    _solid_ellipsoid(1, 1, 1),            # 7 voxels
    np.pad(np.ones((1, 1, 1), np.uint8), 2),
    _solid_box(2, 20, 20),                # thin plate
    _solid_box(1, 1, 12),                 # one-voxel rod
])
def test_sphericity_never_exceeds_one(mask):
    assert 0 < _row(mask).sphericity <= 1.0


def test_plate_less_spherical_than_ball(ball):
    assert _row(_solid_box(2, 20, 20)).sphericity < ball.sphericity


# ---------------------------------------------------------------------------
# fill_frac, bounding box
# ---------------------------------------------------------------------------

def test_solid_box_fill_frac_is_one():
    assert _row(_solid_box(6, 5, 4)).fill_frac == pytest.approx(1.0)


def test_bbox_max_is_exclusive():
    r = _row(_solid_box(3, 4, 5, padding=2))
    assert (r.zmin, r.zmax, r.dz, r.dy, r.dx) == (2, 5, 3, 4, 5)


# ---------------------------------------------------------------------------
# Intensity
# ---------------------------------------------------------------------------

def test_intensity_statistics():
    m = _solid_box(4, 4, 4)
    gs = np.where(m > 0, 100.0, 0.0)
    gs[2, 2, 2] = 200.0
    r = _row(m, grayscale_volume=gs)
    vals = np.full(64, 100.0); vals[0] = 200.0
    assert r.int_mean == pytest.approx(vals.mean())
    assert r.int_std_rel == pytest.approx(vals.std() / vals.mean())


def test_intensity_nan_without_grayscale(ball):
    assert np.isnan(ball.int_std_rel)


def test_negative_intensities_warn():
    m = _solid_box(3, 3, 3)
    with pytest.warns(UserWarning, match="non-negative"):
        _row(m, grayscale_volume=np.where(m > 0, -1.0, 0.0))


# ---------------------------------------------------------------------------
# build_cc_mask
# ---------------------------------------------------------------------------

def test_cc_mask_does_not_grow_defects():
    """Voxel counts are those of the prediction, whatever the grouping."""
    vol = _solid_ellipsoid(4, 4, 4)
    for kw in ({}, {"bridge_radius": 2}, {"dilation_radius": 2}):
        cc = build_cc_mask(vol, **kw)
        assert (cc > 0).sum() == vol.sum()


def test_bridge_groups_close_fragments_without_filling_the_gap():
    vol = np.zeros((9, 9, 20), np.uint8)
    vol[3:6, 3:6, 2:9] = 1
    vol[3:6, 3:6, 10:17] = 1          # one-voxel gap at x = 9
    separate = build_cc_mask(vol, bridge_radius=0)
    grouped = build_cc_mask(vol, bridge_radius=1)
    assert separate.max() == 2 and grouped.max() == 1
    assert grouped[4, 4, 9] == 0      # the gap is not counted as defect


def test_far_objects_stay_separate():
    vol = np.zeros((20, 10, 10), dtype=np.uint8)
    vol[1:4, 3:6, 3:6] = 1
    vol[16:19, 3:6, 3:6] = 1
    assert build_cc_mask(vol).max() == 2


def test_min_size_and_consecutive_labels():
    vol = np.zeros((10, 10, 20), np.uint8)
    vol[1, 1, 1] = 1                  # 1 voxel
    vol[4:7, 4:7, 4:7] = 1            # 27 voxels
    vol[4:7, 4:7, 12:15] = 1          # 27 voxels
    cc = build_cc_mask(vol, min_size=5)
    assert cc[1, 1, 1] == 0 and sorted(np.unique(cc)) == [0, 1, 2]


def test_empty_prediction():
    assert build_cc_mask(np.zeros((5, 5, 5), np.uint8)).max() == 0
