"""Tests for Reference Frame Transformations.

Guideline E4 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.reference_frames import (
    ReferenceFrame,
    ReferenceFrameTransformer,
    SwingPlaneFrame,
    WrenchInFrame,
    compute_rotation_matrix_from_axes,
    decompose_wrench_in_swing_plane,
    fit_functional_swing_plane,
    fit_instantaneous_swing_plane,
    transform_wrench_to_frame,
)


class TestRotationMatrix:
    """Tests for rotation matrix computation."""

    def test_identity_from_standard_axes(self) -> None:
        """Standard axes should produce identity matrix."""
        R = compute_rotation_matrix_from_axes(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_matrix_is_orthogonal(self) -> None:
        """Rotation matrix should be orthogonal (R^T @ R = I)."""
        # Arbitrary rotation (45 degrees about Z)
        angle = np.pi / 4
        x = np.array([np.cos(angle), np.sin(angle), 0.0])
        y = np.array([-np.sin(angle), np.cos(angle), 0.0])
        z = np.array([0.0, 0.0, 1.0])

        R = compute_rotation_matrix_from_axes(x, y, z)

        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestWrenchTransformation:
    """Tests for wrench transformation between frames."""

    def test_identity_transform_preserves_wrench(self) -> None:
        """Identity rotation should preserve wrench values."""
        wrench = WrenchInFrame(
            force=np.array([1.0, 2.0, 3.0]),
            torque=np.array([0.5, 1.0, 1.5]),
            frame=ReferenceFrame.GLOBAL,
        )

        result = transform_wrench_to_frame(wrench, ReferenceFrame.LOCAL, np.eye(3))

        np.testing.assert_allclose(result.force, wrench.force, atol=1e-10)
        np.testing.assert_allclose(result.torque, wrench.torque, atol=1e-10)
        assert result.frame == ReferenceFrame.LOCAL

    def test_rotation_transforms_force_correctly(self) -> None:
        """90-degree rotation should swap components."""
        wrench = WrenchInFrame(
            force=np.array([1.0, 0.0, 0.0]),
            torque=np.array([0.0, 1.0, 0.0]),
            frame=ReferenceFrame.GLOBAL,
        )

        # 90 degree rotation about Z
        R = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        result = transform_wrench_to_frame(wrench, ReferenceFrame.LOCAL, R)

        np.testing.assert_allclose(result.force, np.array([0.0, 1.0, 0.0]), atol=1e-10)


class TestInstantaneousSwingPlane:
    """Tests for instantaneous swing plane fitting."""

    def test_horizontal_swing_plane(self) -> None:
        """Horizontal swing should have vertical normal."""
        velocity = np.array([1.0, 0.0, 0.0])  # Moving in X direction
        grip = np.array([0.0, 0.0, 1.0])  # Grip above
        clubhead = np.array([0.0, 0.0, 0.0])  # Clubhead at origin

        plane = fit_instantaneous_swing_plane(velocity, grip, clubhead)

        # Normal should be perpendicular to both velocity and grip axis
        assert np.abs(np.dot(plane.normal, velocity)) < 0.1
        assert plane.in_plane_x is not None
        assert plane.in_plane_y is not None

    def test_swing_plane_axes_orthogonal(self) -> None:
        """Swing plane axes should be mutually orthogonal."""
        velocity = np.array([1.0, 0.5, 0.0])
        grip = np.array([0.0, 0.0, 1.0])
        clubhead = np.array([1.0, 0.0, 0.0])

        plane = fit_instantaneous_swing_plane(velocity, grip, clubhead)

        # Check orthogonality
        np.testing.assert_allclose(
            np.dot(plane.normal, plane.in_plane_x), 0.0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.dot(plane.normal, plane.in_plane_y), 0.0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.dot(plane.in_plane_x, plane.in_plane_y), 0.0, atol=1e-10
        )


class TestFunctionalSwingPlane:
    """Tests for Functional Swing Plane fitting."""

    def test_planar_trajectory_gives_low_rmse(self) -> None:
        """Points exactly on a plane should give zero RMSE."""
        # Create perfectly planar trajectory
        t = np.linspace(0, 1, 100)
        trajectory = np.column_stack(
            [
                t,
                np.zeros_like(t),
                np.zeros_like(t),
            ]
        )

        fsp = fit_functional_swing_plane(
            trajectory, t, impact_time=0.5, window_ms=1000.0
        )

        assert fsp.fitting_rmse < 1e-10

    def test_curved_trajectory_gives_nonzero_rmse(self) -> None:
        """Out-of-plane trajectory should have measurable RMSE."""
        t = np.linspace(0, 1, 100)
        # Create a helix - truly 3D trajectory that doesn't lie on any plane
        radius = 0.1
        trajectory = np.column_stack(
            [
                radius * np.cos(4 * np.pi * t),  # X
                radius * np.sin(4 * np.pi * t),  # Y (circular motion)
                t * 0.5,  # Z rises linearly - creates helix
            ]
        )

        fsp = fit_functional_swing_plane(
            trajectory, t, impact_time=0.5, window_ms=1000.0
        )

        # Helix will have non-zero deviation from best-fit plane
        assert fsp.fitting_rmse > 0.001  # Should be measurable

    def test_window_parameter_filters_points(self) -> None:
        """Fitting window should limit which points are used."""
        t = np.linspace(0, 1, 100)
        trajectory = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])

        # Very narrow window
        fsp = fit_functional_swing_plane(trajectory, t, impact_time=0.5, window_ms=50.0)

        assert fsp.fitting_window_ms == 50.0


class TestWrenchDecomposition:
    """Tests for wrench decomposition in swing plane."""

    @pytest.fixture
    def horizontal_swing_plane(self) -> SwingPlaneFrame:
        """Create a horizontal swing plane for testing."""
        return SwingPlaneFrame(
            origin=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),  # Vertical normal
            in_plane_x=np.array([1.0, 0.0, 0.0]),
            in_plane_y=np.array([0.0, 1.0, 0.0]),
            grip_axis=np.array([1.0, 0.0, 0.0]),
        )

    def test_vertical_force_is_out_of_plane(
        self, horizontal_swing_plane: SwingPlaneFrame
    ) -> None:
        """Vertical force should be entirely out-of-plane for horizontal plane."""
        wrench = WrenchInFrame(
            force=np.array([0.0, 0.0, 100.0]),
            torque=np.array([0.0, 0.0, 0.0]),
            frame=ReferenceFrame.GLOBAL,
        )

        decomp = decompose_wrench_in_swing_plane(wrench, horizontal_swing_plane)

        np.testing.assert_allclose(decomp["force_out_of_plane"], 100.0, atol=1e-10)
        np.testing.assert_allclose(decomp["force_in_plane"], 0.0, atol=1e-10)

    def test_horizontal_force_is_in_plane(
        self, horizontal_swing_plane: SwingPlaneFrame
    ) -> None:
        """Horizontal force should be entirely in-plane for horizontal plane."""
        wrench = WrenchInFrame(
            force=np.array([50.0, 50.0, 0.0]),
            torque=np.array([0.0, 0.0, 0.0]),
            frame=ReferenceFrame.GLOBAL,
        )

        decomp = decompose_wrench_in_swing_plane(wrench, horizontal_swing_plane)

        np.testing.assert_allclose(decomp["force_out_of_plane"], 0.0, atol=1e-10)
        expected_in_plane = np.sqrt(50**2 + 50**2)
        np.testing.assert_allclose(
            decomp["force_in_plane"], expected_in_plane, atol=1e-10
        )


class TestReferenceFrameTransformer:
    """Tests for the ReferenceFrameTransformer class."""

    def test_global_to_swing_plane_transform(self) -> None:
        """Transform should correctly convert to swing plane frame."""
        transformer = ReferenceFrameTransformer()

        swing_plane = SwingPlaneFrame(
            origin=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            in_plane_x=np.array([1.0, 0.0, 0.0]),
            in_plane_y=np.array([0.0, 1.0, 0.0]),
            grip_axis=np.array([1.0, 0.0, 0.0]),
        )
        transformer.set_swing_plane(swing_plane)

        wrench = WrenchInFrame(
            force=np.array([1.0, 2.0, 3.0]),
            torque=np.array([0.5, 1.0, 1.5]),
            frame=ReferenceFrame.GLOBAL,
        )

        result = transformer.global_to_swing_plane(wrench)

        assert result.frame == ReferenceFrame.SWING_PLANE
        # For identity-like swing plane, values should be similar
        np.testing.assert_allclose(result.force, wrench.force, atol=1e-10)

    def test_missing_swing_plane_raises_error(self) -> None:
        """Transforming without swing plane should raise error."""
        transformer = ReferenceFrameTransformer()
        wrench = WrenchInFrame(
            force=np.array([1.0, 2.0, 3.0]),
            torque=np.array([0.5, 1.0, 1.5]),
            frame=ReferenceFrame.GLOBAL,
        )

        with pytest.raises(ValueError, match="Swing plane not set"):
            transformer.global_to_swing_plane(wrench)
