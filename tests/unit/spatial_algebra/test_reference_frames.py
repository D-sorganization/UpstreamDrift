"""Tests for src.shared.python.spatial_algebra.reference_frames (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.spatial_algebra.reference_frames import (
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


def _identity_wrench() -> WrenchInFrame:
    return WrenchInFrame(
        force=np.array([1.0, 0.0, 0.0]),
        torque=np.array([0.0, 1.0, 0.0]),
        frame=ReferenceFrame.GLOBAL,
    )


class TestComputeRotationMatrixFromAxes:
    def test_identity_axes_give_identity(self) -> None:
        R = compute_rotation_matrix_from_axes(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_returns_3x3(self) -> None:
        R = compute_rotation_matrix_from_axes(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        assert R.shape == (3, 3)

    def test_columns_are_input_axes(self) -> None:
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 1.0])
        z = np.array([0.0, 1.0, 0.0])
        R = compute_rotation_matrix_from_axes(x, y, z)
        np.testing.assert_array_equal(R[:, 0], x)
        np.testing.assert_array_equal(R[:, 1], y)
        np.testing.assert_array_equal(R[:, 2], z)


class TestTransformWrenchToFrame:
    def test_identity_rotation_preserves_wrench(self) -> None:
        w = _identity_wrench()
        result = transform_wrench_to_frame(w, ReferenceFrame.LOCAL, np.eye(3))
        np.testing.assert_array_equal(result.force, w.force)
        np.testing.assert_array_equal(result.torque, w.torque)

    def test_target_frame_is_set(self) -> None:
        w = _identity_wrench()
        result = transform_wrench_to_frame(w, ReferenceFrame.LOCAL, np.eye(3))
        assert result.frame == ReferenceFrame.LOCAL

    def test_180_degree_rotation_negates_force(self) -> None:
        w = WrenchInFrame(
            force=np.array([1.0, 0.0, 0.0]),
            torque=np.array([0.0, 0.0, 0.0]),
            frame=ReferenceFrame.GLOBAL,
        )
        R = np.diag([-1.0, 1.0, 1.0])  # Flip x
        result = transform_wrench_to_frame(w, ReferenceFrame.LOCAL, R)
        np.testing.assert_allclose(result.force, np.array([-1.0, 0.0, 0.0]), atol=1e-12)

    def test_point_is_transformed_when_present(self) -> None:
        w = WrenchInFrame(
            force=np.array([1.0, 0.0, 0.0]),
            torque=np.array([0.0, 0.0, 0.0]),
            frame=ReferenceFrame.GLOBAL,
            point=np.array([1.0, 2.0, 3.0]),
        )
        result = transform_wrench_to_frame(w, ReferenceFrame.LOCAL, np.eye(3))
        assert result.point is not None
        np.testing.assert_array_equal(result.point, w.point)

    def test_point_is_none_when_not_set(self) -> None:
        w = _identity_wrench()
        result = transform_wrench_to_frame(w, ReferenceFrame.LOCAL, np.eye(3))
        assert result.point is None

    def test_body_name_preserved(self) -> None:
        w = WrenchInFrame(
            force=np.zeros(3),
            torque=np.zeros(3),
            frame=ReferenceFrame.GLOBAL,
            body_name="arm",
        )
        result = transform_wrench_to_frame(w, ReferenceFrame.LOCAL, np.eye(3))
        assert result.body_name == "arm"


class TestFitInstantaneousSwingPlane:
    def test_returns_swing_plane_frame(self) -> None:
        result = fit_instantaneous_swing_plane(
            clubhead_velocity=np.array([10.0, 0.0, 0.0]),
            grip_position=np.array([0.0, 0.0, 0.0]),
            clubhead_position=np.array([1.0, 0.0, 0.0]),
        )
        assert isinstance(result, SwingPlaneFrame)

    def test_reference_frames_normal_is_unit_vector(self) -> None:
        result = fit_instantaneous_swing_plane(
            clubhead_velocity=np.array([10.0, 0.0, 5.0]),
            grip_position=np.array([0.0, 0.0, 0.0]),
            clubhead_position=np.array([1.5, 0.0, 0.5]),
        )
        assert np.linalg.norm(result.normal) == pytest.approx(1.0, abs=1e-9)

    def test_zero_velocity_uses_default(self) -> None:
        # Should not raise; uses default x-axis for in_plane_x
        result = fit_instantaneous_swing_plane(
            clubhead_velocity=np.array([0.0, 0.0, 0.0]),
            grip_position=np.array([0.0, 0.0, 0.0]),
            clubhead_position=np.array([1.0, 0.0, 0.0]),
        )
        assert isinstance(result, SwingPlaneFrame)

    def test_origin_is_clubhead_position(self) -> None:
        ch_pos = np.array([2.0, 3.0, 4.0])
        result = fit_instantaneous_swing_plane(
            clubhead_velocity=np.array([5.0, 0.0, 0.0]),
            grip_position=np.array([0.0, 0.0, 0.0]),
            clubhead_position=ch_pos,
        )
        np.testing.assert_array_equal(result.origin, ch_pos)


class TestFitFunctionalSwingPlane:
    def _circular_traj(self, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, 1.0, n)
        traj = np.column_stack(
            [
                np.cos(2 * np.pi * t),
                np.sin(2 * np.pi * t),
                np.zeros(n),
            ]
        )
        return traj, t

    def test_returns_swing_plane_frame(self) -> None:
        traj, t = self._circular_traj()
        result = fit_functional_swing_plane(traj, t, impact_time=0.5)
        assert isinstance(result, SwingPlaneFrame)

    def test_reference_frames_normal_is_unit_vector(self) -> None:
        traj, t = self._circular_traj()
        result = fit_functional_swing_plane(traj, t, impact_time=0.5)
        assert np.linalg.norm(result.normal) == pytest.approx(1.0, abs=1e-9)

    def test_rmse_is_non_negative(self) -> None:
        traj, t = self._circular_traj()
        result = fit_functional_swing_plane(traj, t, impact_time=0.5)
        assert result.fitting_rmse >= 0.0

    def test_window_ms_stored(self) -> None:
        traj, t = self._circular_traj()
        result = fit_functional_swing_plane(traj, t, impact_time=0.5, window_ms=200.0)
        assert result.fitting_window_ms == pytest.approx(200.0)

    def test_few_points_fallback(self) -> None:
        # Only 2 points — falls back to all points
        traj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        t = np.array([0.0, 1.0])
        result = fit_functional_swing_plane(traj, t, impact_time=0.5)
        assert isinstance(result, SwingPlaneFrame)


class TestDecomposeWrenchInSwingPlane:
    def _make_xy_plane(self) -> SwingPlaneFrame:
        return SwingPlaneFrame(
            origin=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            in_plane_x=np.array([1.0, 0.0, 0.0]),
            in_plane_y=np.array([0.0, 1.0, 0.0]),
            grip_axis=np.array([1.0, 0.0, 0.0]),
        )

    def test_reference_frames_returns_dict(self) -> None:
        w = _identity_wrench()
        result = decompose_wrench_in_swing_plane(w, self._make_xy_plane())
        assert isinstance(result, dict)

    def test_expected_keys(self) -> None:
        w = _identity_wrench()
        result = decompose_wrench_in_swing_plane(w, self._make_xy_plane())
        for key in [
            "force_in_plane",
            "force_out_of_plane",
            "force_along_grip",
            "torque_in_plane",
            "torque_out_of_plane",
            "torque_about_grip",
        ]:
            assert key in result

    def test_x_force_is_in_plane(self) -> None:
        w = WrenchInFrame(
            force=np.array([5.0, 0.0, 0.0]),
            torque=np.zeros(3),
            frame=ReferenceFrame.GLOBAL,
        )
        result = decompose_wrench_in_swing_plane(w, self._make_xy_plane())
        assert result["force_in_plane"] == pytest.approx(5.0, abs=1e-10)
        assert result["force_out_of_plane"] == pytest.approx(0.0, abs=1e-10)

    def test_z_force_is_out_of_plane(self) -> None:
        w = WrenchInFrame(
            force=np.array([0.0, 0.0, 3.0]),
            torque=np.zeros(3),
            frame=ReferenceFrame.GLOBAL,
        )
        result = decompose_wrench_in_swing_plane(w, self._make_xy_plane())
        assert result["force_out_of_plane"] == pytest.approx(3.0, abs=1e-10)
        assert result["force_in_plane"] == pytest.approx(0.0, abs=1e-10)


class TestReferenceFrameTransformer:
    def test_global_to_local_raises_without_rotation(self) -> None:
        xfm = ReferenceFrameTransformer()
        w = _identity_wrench()
        with pytest.raises(ValueError):
            xfm.global_to_local(w, "missing_body")

    def test_global_to_swing_plane_raises_without_plane(self) -> None:
        xfm = ReferenceFrameTransformer()
        w = _identity_wrench()
        with pytest.raises(ValueError):
            xfm.global_to_swing_plane(w)

    def test_global_to_local_identity(self) -> None:
        xfm = ReferenceFrameTransformer()
        xfm.set_body_rotation("arm", np.eye(3))
        w = _identity_wrench()
        result = xfm.global_to_local(w, "arm")
        np.testing.assert_array_equal(result.force, w.force)

    def test_global_to_swing_plane_identity(self) -> None:
        xfm = ReferenceFrameTransformer()
        plane = SwingPlaneFrame(
            origin=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            in_plane_x=np.array([1.0, 0.0, 0.0]),
            in_plane_y=np.array([0.0, 1.0, 0.0]),
            grip_axis=np.array([1.0, 0.0, 0.0]),
        )
        xfm.set_swing_plane(plane)
        w = _identity_wrench()
        result = xfm.global_to_swing_plane(w)
        assert result.frame == ReferenceFrame.SWING_PLANE

    def test_get_swing_plane_decomposition_raises_without_plane(self) -> None:
        xfm = ReferenceFrameTransformer()
        with pytest.raises(ValueError):
            xfm.get_swing_plane_decomposition(_identity_wrench())
