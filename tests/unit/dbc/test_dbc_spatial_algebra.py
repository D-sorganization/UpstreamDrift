"""DbC tests for spatial algebra modules (spatial_vectors, transforms, inertia).

Validates that shape checks, rotation matrix validation, mass positivity,
and matrix dimension assertions fire on invalid inputs, and that valid
computations produce correct algebraic properties (skew-symmetry,
transform invertibility, inertia symmetry).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ["DBC_LEVEL"] = "enforce"

# Spatial algebra modules raise ValueError directly for shape checks,
# which is the correct DbC behavior for these low-level math functions.
_CONTRACT_EXC = (ValueError,)


class TestSkewPreconditions(unittest.TestCase):
    """Preconditions on skew() - must receive a 3-element vector."""

    def test_wrong_shape_2d_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import skew

        with self.assertRaises(_CONTRACT_EXC):
            skew(np.array([1.0, 2.0]))

    def test_wrong_shape_4d_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import skew

        with self.assertRaises(_CONTRACT_EXC):
            skew(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_wrong_shape_matrix_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import skew

        with self.assertRaises(_CONTRACT_EXC):
            skew(np.eye(3))

    def test_empty_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import skew

        with self.assertRaises(_CONTRACT_EXC):
            skew(np.array([]))


class TestSkewPostconditions(unittest.TestCase):
    """Postconditions: skew matrix must be antisymmetric."""

    def test_skew_is_antisymmetric(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import skew

        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)
        # S^T = -S
        np.testing.assert_array_almost_equal(S.T, -S)

    def test_skew_shape_is_3x3(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import skew

        S = skew(np.array([1.0, 0.0, 0.0]))
        self.assertEqual(S.shape, (3, 3))

    def test_skew_diagonal_is_zero(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import skew

        S = skew(np.array([5.0, -3.0, 7.0]))
        np.testing.assert_array_almost_equal(np.diag(S), [0.0, 0.0, 0.0])

    def test_skew_cross_product_identity(self) -> None:
        """skew(v) @ u == cross(v, u)"""
        from src.shared.python.spatial_algebra.spatial_vectors import skew

        v = np.array([1.0, 2.0, 3.0])
        u = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(skew(v) @ u, np.cross(v, u))


class TestCrmPreconditions(unittest.TestCase):
    """Preconditions on crm() - must receive a 6-element spatial vector."""

    def test_3d_vector_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import crm

        with self.assertRaises(_CONTRACT_EXC):
            crm(np.array([1.0, 2.0, 3.0]))

    def test_empty_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import crm

        with self.assertRaises(_CONTRACT_EXC):
            crm(np.array([]))

    def test_7d_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import crm

        with self.assertRaises(_CONTRACT_EXC):
            crm(np.ones(7))


class TestCrmPostconditions(unittest.TestCase):
    """crm() must return a 6x6 matrix."""

    def test_crm_shape(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import crm

        result = crm(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        self.assertEqual(result.shape, (6, 6))

    def test_crm_zero_input_gives_zero(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import crm

        result = crm(np.zeros(6))
        np.testing.assert_array_almost_equal(result, np.zeros((6, 6)))


class TestCrfPreconditions(unittest.TestCase):
    """Preconditions on crf() - must receive a 6-element spatial vector."""

    def test_3d_vector_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import crf

        with self.assertRaises(_CONTRACT_EXC):
            crf(np.array([1.0, 2.0, 3.0]))


class TestCrfPostconditions(unittest.TestCase):
    """crf() = -crm()^T (dual relationship)."""

    def test_crf_is_negative_transpose_of_crm(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import crf, crm

        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(crf(v), -crm(v).T)


class TestCrossMotionPreconditions(unittest.TestCase):
    """cross_motion must receive two 6-element vectors."""

    def test_3d_first_arg_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import cross_motion

        with self.assertRaises(_CONTRACT_EXC):
            cross_motion(np.ones(3), np.ones(6))

    def test_3d_second_arg_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import cross_motion

        with self.assertRaises(_CONTRACT_EXC):
            cross_motion(np.ones(6), np.ones(3))


class TestCrossMotionPostconditions(unittest.TestCase):
    """cross_motion must return 6-element vector, consistent with crm()."""

    def test_output_shape(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import cross_motion

        result = cross_motion(np.ones(6), np.ones(6))
        self.assertEqual(result.shape, (6,))

    def test_consistent_with_crm(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import (
            crm,
            cross_motion,
        )

        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        m = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        np.testing.assert_array_almost_equal(cross_motion(v, m), crm(v) @ m)

    def test_zero_cross_zero(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import cross_motion

        result = cross_motion(np.zeros(6), np.ones(6))
        np.testing.assert_array_almost_equal(result, np.zeros(6))


class TestCrossForcePreconditions(unittest.TestCase):
    """cross_force must receive two 6-element vectors."""

    def test_wrong_shape_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import cross_force

        with self.assertRaises(_CONTRACT_EXC):
            cross_force(np.ones(4), np.ones(6))


class TestCrossForcePostconditions(unittest.TestCase):
    """cross_force must be consistent with crf()."""

    def test_consistent_with_crf(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import (
            crf,
            cross_force,
        )

        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        f = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        np.testing.assert_array_almost_equal(cross_force(v, f), crf(v) @ f)


class TestSpatialCrossPreconditions(unittest.TestCase):
    """spatial_cross must accept valid cross_type."""

    def test_invalid_type_raises(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import spatial_cross

        with self.assertRaises(_CONTRACT_EXC):
            spatial_cross(np.ones(6), np.ones(6), cross_type="invalid")


# ─── Transforms Tests ──────────────────────────────────────────


class TestXrotPreconditions(unittest.TestCase):
    """xrot() requires a valid 3x3 rotation matrix."""

    def test_wrong_shape_raises(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xrot

        with self.assertRaises(_CONTRACT_EXC):
            xrot(np.eye(4))

    def test_non_rotation_raises(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xrot

        with self.assertRaises(_CONTRACT_EXC):
            xrot(np.zeros((3, 3)))  # det=0, not rotation

    def test_scaling_matrix_raises(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xrot

        with self.assertRaises(_CONTRACT_EXC):
            xrot(2.0 * np.eye(3))  # det=8, not rotation


class TestXrotPostconditions(unittest.TestCase):
    """xrot() must produce 6x6 block-diagonal matrix."""

    def test_identity_rotation(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xrot

        result = xrot(np.eye(3))
        np.testing.assert_array_almost_equal(result, np.eye(6))

    def test_output_shape(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xrot

        result = xrot(np.eye(3))
        self.assertEqual(result.shape, (6, 6))


class TestXltPreconditions(unittest.TestCase):
    """xlt() requires a 3-element vector."""

    def test_2d_raises(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xlt

        with self.assertRaises(_CONTRACT_EXC):
            xlt(np.array([1.0, 2.0]))

    def test_6d_raises(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xlt

        with self.assertRaises(_CONTRACT_EXC):
            xlt(np.ones(6))


class TestXltPostconditions(unittest.TestCase):
    """xlt() zero translation gives identity."""

    def test_zero_translation(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xlt

        result = xlt(np.zeros(3))
        np.testing.assert_array_almost_equal(result, np.eye(6))


class TestXtransPreconditions(unittest.TestCase):
    """xtrans() requires matching shapes."""

    def test_wrong_rotation_shape_raises(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xtrans

        with self.assertRaises(_CONTRACT_EXC):
            xtrans(np.eye(4), np.zeros(3))

    def test_wrong_translation_shape_raises(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xtrans

        with self.assertRaises(_CONTRACT_EXC):
            xtrans(np.eye(3), np.zeros(4))


class TestXtransPostconditions(unittest.TestCase):
    """xtrans with identity rotation and zero translation gives I6."""

    def test_identity_transform(self) -> None:
        from src.shared.python.spatial_algebra.transforms import xtrans

        result = xtrans(np.eye(3), np.zeros(3))
        np.testing.assert_array_almost_equal(result, np.eye(6))


class TestInvXtransPreconditions(unittest.TestCase):
    """inv_xtrans() requires matching shapes."""

    def test_wrong_rotation_shape_raises(self) -> None:
        from src.shared.python.spatial_algebra.transforms import inv_xtrans

        with self.assertRaises(_CONTRACT_EXC):
            inv_xtrans(np.eye(2), np.zeros(3))


class TestTransformInverse(unittest.TestCase):
    """xtrans() and inv_xtrans() must be inverses."""

    def test_inverse_property(self) -> None:
        from src.shared.python.spatial_algebra.transforms import inv_xtrans, xtrans

        E = np.eye(3)
        r = np.array([1.0, 2.0, 3.0])
        X = xtrans(E, r)
        X_inv = inv_xtrans(E, r)
        np.testing.assert_array_almost_equal(X @ X_inv, np.eye(6), decimal=10)


# ─── Spatial Inertia Tests ─────────────────────────────────────


class TestMcIPreconditions(unittest.TestCase):
    """mcI() requires positive mass, 3-vector COM, 3x3 inertia."""

    def test_negative_mass_raises(self) -> None:
        from src.shared.python.spatial_algebra.inertia import mcI

        with self.assertRaises(_CONTRACT_EXC):
            mcI(-1.0, np.zeros(3), np.eye(3))

    def test_zero_mass_raises(self) -> None:
        from src.shared.python.spatial_algebra.inertia import mcI

        with self.assertRaises(_CONTRACT_EXC):
            mcI(0.0, np.zeros(3), np.eye(3))

    def test_wrong_com_shape_raises(self) -> None:
        from src.shared.python.spatial_algebra.inertia import mcI

        with self.assertRaises(_CONTRACT_EXC):
            mcI(1.0, np.zeros(2), np.eye(3))

    def test_wrong_inertia_shape_raises(self) -> None:
        from src.shared.python.spatial_algebra.inertia import mcI

        with self.assertRaises(_CONTRACT_EXC):
            mcI(1.0, np.zeros(3), np.eye(4))

    def test_string_mass_raises(self) -> None:
        from src.shared.python.spatial_algebra.inertia import mcI

        with self.assertRaises((ValueError, TypeError)):
            mcI("heavy", np.zeros(3), np.eye(3))  # type: ignore[arg-type]


class TestMcIPostconditions(unittest.TestCase):
    """mcI() output must be 6x6, symmetric, and positive semi-definite (for physical inertias)."""

    def test_output_shape(self) -> None:
        from src.shared.python.spatial_algebra.inertia import mcI

        result = mcI(2.0, np.zeros(3), np.eye(3))
        self.assertEqual(result.shape, (6, 6))

    def test_output_symmetric(self) -> None:
        from src.shared.python.spatial_algebra.inertia import mcI

        result = mcI(2.0, np.array([0.1, 0.2, 0.3]), np.diag([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(result, result.T)

    def test_point_mass_at_origin(self) -> None:
        """Point mass at origin: off-diag zero, lower-right block = m*I."""
        from src.shared.python.spatial_algebra.inertia import mcI

        m = 5.0
        result = mcI(m, np.zeros(3), np.zeros((3, 3)))
        # Lower-right 3x3 should be m*I
        np.testing.assert_array_almost_equal(result[3:6, 3:6], m * np.eye(3))
        # Upper-left 3x3 should be zero (no rotational inertia at origin)
        np.testing.assert_array_almost_equal(result[0:3, 0:3], np.zeros((3, 3)))

    def test_eigenvalues_non_negative(self) -> None:
        from src.shared.python.spatial_algebra.inertia import mcI

        result = mcI(3.0, np.array([0.1, 0.0, 0.0]), np.diag([0.5, 0.5, 0.5]))
        eigenvalues = np.linalg.eigvalsh(result)
        self.assertTrue(np.all(eigenvalues >= -1e-10))


class TestTransformSpatialInertiaPreconditions(unittest.TestCase):
    """transform_spatial_inertia() requires 6x6 matrices."""

    def test_wrong_I_shape_raises(self) -> None:
        from src.shared.python.spatial_algebra.inertia import transform_spatial_inertia

        with self.assertRaises(_CONTRACT_EXC):
            transform_spatial_inertia(np.eye(3), np.eye(6))

    def test_wrong_X_shape_raises(self) -> None:
        from src.shared.python.spatial_algebra.inertia import transform_spatial_inertia

        with self.assertRaises(_CONTRACT_EXC):
            transform_spatial_inertia(np.eye(6), np.eye(3))


class TestTransformSpatialInertiaPostconditions(unittest.TestCase):
    """Transformed inertia must remain symmetric."""

    def test_output_symmetric(self) -> None:
        from src.shared.python.spatial_algebra.inertia import (
            mcI,
            transform_spatial_inertia,
        )

        inertia = mcI(2.0, np.zeros(3), np.diag([1.0, 2.0, 3.0]))
        X = np.eye(6)
        result = transform_spatial_inertia(inertia, X)
        np.testing.assert_array_almost_equal(result, result.T)

    def test_identity_transform_preserves(self) -> None:
        from src.shared.python.spatial_algebra.inertia import (
            mcI,
            transform_spatial_inertia,
        )

        inertia = mcI(2.0, np.zeros(3), np.diag([1.0, 2.0, 3.0]))
        result = transform_spatial_inertia(inertia, np.eye(6))
        np.testing.assert_array_almost_equal(result, inertia)


class TestCrossMotionAxisPreconditions(unittest.TestCase):
    """cross_motion_axis with invalid axis_idx should produce zeros for unhandled axes."""

    def test_valid_axes_produce_output(self) -> None:
        from src.shared.python.spatial_algebra.spatial_vectors import cross_motion_axis

        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        for axis_idx in range(6):
            out = np.zeros(6)
            cross_motion_axis(v, axis_idx, 1.0, out)
            # Should produce finite output
            self.assertTrue(np.all(np.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
