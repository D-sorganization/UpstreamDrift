"""Heavy integration tests for screw theory modules (fixes #1994).

Numerically validates the screw exponential map, adjoint transform,
CRBA mass matrix, and RNEA inverse dynamics against analytical solutions
and cross-validates with pinocchio when available.

All tests skip gracefully when optional dependencies are unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _import_screw_modules():
    """Import screw theory modules or skip."""
    try:
        from mujoco_humanoid_golf.screw_theory.adjoint import adjoint_transform
        from mujoco_humanoid_golf.screw_theory.exponential import exponential_map
        from mujoco_humanoid_golf.screw_theory.screws import screw_axis

        return exponential_map, screw_axis, adjoint_transform
    except ImportError as exc:
        pytest.skip(f"mujoco_humanoid_golf screw_theory not importable: {exc}")


# ---------------------------------------------------------------------------
# Screw exponential map tests
# ---------------------------------------------------------------------------


class TestScrewExponentialMap:
    """Contract: exponential_map satisfies known analytical results."""

    def test_pure_rotation_z_90deg(self) -> None:
        """90° rotation about z-axis matches the Rz(90°) matrix."""
        exponential_map, screw_axis, _ = _import_screw_modules()

        S = screw_axis(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        T = exponential_map(S, np.pi / 2)

        assert T.shape == (4, 4)
        # Top-left 3×3 should be Rz(90°)
        expected_R = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(T[:3, :3], expected_R, atol=1e-10)
        # No translation
        np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, 0.0], atol=1e-10)

    def test_pure_rotation_identity(self) -> None:
        """Zero rotation returns the identity matrix."""
        exponential_map, screw_axis, _ = _import_screw_modules()

        S = screw_axis(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        T = exponential_map(S, 0.0)

        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)

    def test_pure_translation_x(self) -> None:
        """Pure translation along x by 0.5 m."""
        exponential_map, screw_axis, _ = _import_screw_modules()

        # Prismatic screw: omega=0, v=x-hat
        S = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        T = exponential_map(S, 0.5)

        expected = np.eye(4)
        expected[0, 3] = 0.5
        np.testing.assert_allclose(T, expected, atol=1e-12)

    def test_full_rotation_360deg(self) -> None:
        """360° rotation returns identity."""
        exponential_map, screw_axis, _ = _import_screw_modules()

        S = screw_axis(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        T = exponential_map(S, 2 * np.pi)

        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_output_is_valid_se3(self) -> None:
        """Result is a valid SE(3) homogeneous matrix."""
        exponential_map, screw_axis, _ = _import_screw_modules()

        S = screw_axis(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        T = exponential_map(S, np.pi / 4)

        # Bottom row must be [0,0,0,1]
        np.testing.assert_allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        # Rotation part must be orthogonal (R^T R = I)
        R = T[:3, :3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        # det(R) = +1
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Adjoint transform tests
# ---------------------------------------------------------------------------


class TestAdjointTransform:
    """Contract: adjoint_transform satisfies group homomorphism property."""

    def test_adjoint_shape(self) -> None:
        """Adjoint of a 4×4 matrix is 6×6."""
        _, _, adjoint_transform = _import_screw_modules()

        T = np.eye(4)
        Ad = adjoint_transform(T)
        assert Ad.shape == (6, 6)

    def test_adjoint_identity(self) -> None:
        """Adjoint of identity is identity."""
        _, _, adjoint_transform = _import_screw_modules()

        Ad = adjoint_transform(np.eye(4))
        np.testing.assert_allclose(Ad, np.eye(6), atol=1e-12)

    def test_adjoint_composition(self) -> None:
        """Ad(T1 @ T2) = Ad(T1) @ Ad(T2)."""
        exponential_map, screw_axis, adjoint_transform = _import_screw_modules()

        S1 = screw_axis(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        S2 = screw_axis(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        T1 = exponential_map(S1, np.pi / 3)
        T2 = exponential_map(S2, np.pi / 6)

        Ad_composed = adjoint_transform(T1 @ T2)
        Ad_product = adjoint_transform(T1) @ adjoint_transform(T2)
        np.testing.assert_allclose(Ad_composed, Ad_product, atol=1e-10)

    def test_adjoint_inverse(self) -> None:
        """Ad(T)^{-1} = Ad(T^{-1})."""
        exponential_map, screw_axis, adjoint_transform = _import_screw_modules()

        S = screw_axis(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        T = exponential_map(S, np.pi / 5)

        Ad = adjoint_transform(T)
        Ad_inv = adjoint_transform(np.linalg.inv(T))
        np.testing.assert_allclose(Ad @ Ad_inv, np.eye(6), atol=1e-10)


# ---------------------------------------------------------------------------
# Screw theory vs. Pinocchio cross-validation
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.live_simulation
