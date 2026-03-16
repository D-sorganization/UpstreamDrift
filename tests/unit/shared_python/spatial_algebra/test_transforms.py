import numpy as np
import pytest

from src.shared.python.spatial_algebra.transforms import inv_xtrans, xlt, xrot, xtrans


class TestTransforms:
    def test_xrot(self):
        e_rot = np.eye(3, dtype=float)
        res = xrot(e_rot)
        assert res.shape == (6, 6)
        np.testing.assert_array_equal(res, np.eye(6, dtype=float))

        # Test 90 deg rotation around z
        e_rot_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

        res = xrot(e_rot_z)
        np.testing.assert_array_equal(res[0:3, 0:3], e_rot_z)
        np.testing.assert_array_equal(res[3:6, 3:6], e_rot_z)

        with pytest.raises(ValueError, match="E must be 3x3"):
            xrot(np.eye(2))

        with pytest.raises(ValueError, match="E may not be a valid rotation"):
            xrot(np.zeros((3, 3)))

    def test_xlt(self):
        r = np.array([1, 2, 3], dtype=float)
        res = xlt(r)
        assert res.shape == (6, 6)

        np.testing.assert_array_equal(res[0:3, 0:3], np.eye(3, dtype=float))
        np.testing.assert_array_equal(res[3:6, 3:6], np.eye(3, dtype=float))

        # Check cross product in bottom-left
        expected_skew = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]], dtype=float)
        np.testing.assert_array_equal(res[3:6, 0:3], -expected_skew)

        with pytest.raises(ValueError, match="r must be 3x1"):
            xlt(np.array([1, 2]))

    def test_xtrans(self):
        e_rot = np.eye(3, dtype=float)
        r = np.array([1, 2, 3], dtype=float)

        res = xtrans(e_rot, r)
        assert res.shape == (6, 6)

        np.testing.assert_array_equal(res[0:3, 0:3], e_rot)
        np.testing.assert_array_equal(res[3:6, 3:6], e_rot)

        expected_skew = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]], dtype=float)
        np.testing.assert_array_equal(res[3:6, 0:3], -expected_skew)

        with pytest.raises(ValueError, match="E must be 3x3"):
            xtrans(np.eye(2), r)
        with pytest.raises(ValueError, match="r must be 3x1"):
            xtrans(e_rot, np.array([1, 2]))

    def test_inv_xtrans(self):
        e_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        r = np.array([1, 2, 3], dtype=float)

        res_xtrans = xtrans(e_rot, r)
        res_inv = inv_xtrans(e_rot, r)

        assert res_inv.shape == (6, 6)

        identity = res_xtrans @ res_inv
        np.testing.assert_array_almost_equal(identity, np.eye(6, dtype=float))

        with pytest.raises(ValueError, match="E must be 3x3"):
            inv_xtrans(np.eye(2), r)
        with pytest.raises(ValueError, match="r must be 3x1"):
            inv_xtrans(e_rot, np.array([1, 2]))
