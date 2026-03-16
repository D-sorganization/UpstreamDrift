import numpy as np
import pytest

from src.shared.python.spatial_algebra.spatial_vectors import (
    crf,
    crm,
    cross_force,
    cross_force_fast,
    cross_motion,
    cross_motion_axis,
    cross_motion_fast,
    skew,
    spatial_cross,
)


class TestSpatialVectors:
    def test_skew(self):
        v = np.array([1.0, 2.0, 3.0])
        res = skew(v)
        expected = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]], dtype=float)
        np.testing.assert_array_equal(res, expected)

        with pytest.raises(ValueError, match="Input must be 3x1"):
            skew(np.array([1, 2]))

    def test_crm(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res = crm(v)
        assert res.shape == (6, 6)

        # simple check on a few elements
        assert res[0, 1] == -3.0
        assert res[0, 2] == 2.0
        assert res[3, 1] == -6.0
        assert res[3, 2] == 5.0

        with pytest.raises(ValueError, match="Input must be 6x1"):
            crm(np.array([1, 2]))

    def test_crf(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res = crf(v)
        assert res.shape == (6, 6)

        assert res[0, 4] == -6.0
        assert res[0, 5] == 5.0

        with pytest.raises(ValueError, match="Input must be 6x1"):
            crf(np.array([1, 2]))

    def test_cross_motion(self):
        v = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        m = np.array([6, 5, 4, 3, 2, 1], dtype=float)

        res = cross_motion(v, m)
        assert res.shape == (6,)

        with pytest.raises(ValueError, match="v must be 6x1"):
            cross_motion(np.array([1.0]), m)
        with pytest.raises(ValueError, match="m must be 6x1"):
            cross_motion(v, np.array([1.0]))

    def test_cross_force(self):
        v = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        f = np.array([6, 5, 4, 3, 2, 1], dtype=float)

        res = cross_force(v, f)
        assert res.shape == (6,)

        with pytest.raises(ValueError, match="v must be 6x1"):
            cross_force(np.array([1.0]), f)
        with pytest.raises(ValueError, match="f must be 6x1"):
            cross_force(v, np.array([1.0]))

    def test_cross_motion_fast(self):
        v = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        m = np.array([6, 5, 4, 3, 2, 1], dtype=float)
        out = np.zeros(6, dtype=float)
        cross_motion_fast(v, m, out)

        expected = cross_motion(v, m)
        np.testing.assert_array_equal(out, expected)

    def test_cross_force_fast(self):
        v = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        f = np.array([6, 5, 4, 3, 2, 1], dtype=float)
        out = np.zeros(6, dtype=float)
        cross_force_fast(v, f, out)

        expected = cross_force(v, f)
        np.testing.assert_array_equal(out, expected)

    def test_cross_motion_axis(self):
        v = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        val = 2.0

        for axis_idx in range(6):
            out = np.zeros(6, dtype=float)
            cross_motion_axis(v, axis_idx, val, out)

            # Compare with standard cross_motion
            m = np.zeros(6, dtype=float)
            m[axis_idx] = val
            expected = cross_motion(v, m)

            np.testing.assert_array_equal(out, expected)

    def test_spatial_cross(self):
        v = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        u = np.array([6, 5, 4, 3, 2, 1], dtype=float)

        res_motion = spatial_cross(v, u, cross_type="motion")
        np.testing.assert_array_equal(res_motion, cross_motion(v, u))

        res_force = spatial_cross(v, u, cross_type="force")
        np.testing.assert_array_equal(res_force, cross_force(v, u))

        with pytest.raises(ValueError, match="cross_type must be"):
            spatial_cross(v, u, cross_type="invalid")  # type: ignore
