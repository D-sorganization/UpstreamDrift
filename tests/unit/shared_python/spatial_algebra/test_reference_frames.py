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


class TestReferenceFrames:
    def test_compute_rotation_matrix_from_axes(self):
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        z = np.array([0, 0, 1])
        R = compute_rotation_matrix_from_axes(x, y, z)
        np.testing.assert_array_equal(R, np.eye(3))

    def test_transform_wrench_to_frame(self):
        wrench = WrenchInFrame(
            force=np.array([1, 2, 3], dtype=float),
            torque=np.array([4, 5, 6], dtype=float),
            frame=ReferenceFrame.LOCAL,
            body_name="torso",
            point=np.array([0, 0, 1], dtype=float),
        )

        # 90 deg rotation around z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

        transformed = transform_wrench_to_frame(wrench, ReferenceFrame.GLOBAL, R)
        assert transformed.frame == ReferenceFrame.GLOBAL
        assert transformed.body_name == "torso"

        np.testing.assert_array_almost_equal(transformed.force, np.array([-2, 1, 3]))
        np.testing.assert_array_almost_equal(transformed.torque, np.array([-5, 4, 6]))
        assert transformed.point is not None
        np.testing.assert_array_almost_equal(transformed.point, np.array([0, 0, 1]))

    def test_fit_instantaneous_swing_plane(self):
        grip_pos = np.array([0, 0, 1], dtype=float)
        club_pos = np.array([1, 0, 0], dtype=float)
        club_vel = np.array([0, 1, 0], dtype=float)  # velocity in y direction

        plane = fit_instantaneous_swing_plane(club_vel, grip_pos, club_pos)

        # Velocity is the in-plane X
        np.testing.assert_array_almost_equal(plane.in_plane_x, np.array([0, 1, 0]))

        # Grip axis is club_pos - grip_pos: [1, 0, -1] normalized
        expected_grip = np.array([1, 0, -1]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(plane.grip_axis, expected_grip)

        # Normal goes in z-ish direction (cross of grip and in-plane_x)
        expected_normal = np.cross(expected_grip, plane.in_plane_x)
        expected_normal /= np.linalg.norm(expected_normal)
        np.testing.assert_array_almost_equal(plane.normal, expected_normal)

        # Edge cases
        fit_instantaneous_swing_plane(np.zeros(3), grip_pos, club_pos)
        fit_instantaneous_swing_plane(club_vel, np.array([1, 0, 0]), np.array([1, 0, 0]))
        fit_instantaneous_swing_plane(
            np.array([1, 0, 0]), np.array([0, 0, -1]), np.array([1, 0, 0])
        )

    def test_fit_functional_swing_plane(self):
        t = np.linspace(0, 1, 100)
        # points in x-y plane
        traj = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])

        plane = fit_functional_swing_plane(traj, t, impact_time=0.5, window_ms=1000)

        assert plane.fitting_rmse >= 0

        # very small window
        fit_functional_swing_plane(traj, t, impact_time=0.5, window_ms=0.1)

    def test_decompose_wrench_in_swing_plane(self):
        wrench = WrenchInFrame(
            force=np.array([1, 0, 0], dtype=float),
            torque=np.array([0, 1, 0], dtype=float),
            frame=ReferenceFrame.GLOBAL,
        )

        plane = SwingPlaneFrame(
            origin=np.zeros(3),
            normal=np.array([0, 0, 1], dtype=float),
            in_plane_x=np.array([1, 0, 0], dtype=float),
            in_plane_y=np.array([0, 1, 0], dtype=float),
            grip_axis=np.array([0, 0, 1], dtype=float),
        )

        res = decompose_wrench_in_swing_plane(wrench, plane)
        assert res["force_in_plane"] == 1.0
        assert res["force_out_of_plane"] == 0.0
        assert res["torque_in_plane"] == 1.0
        assert res["torque_out_of_plane"] == 0.0

    def test_transformer(self):
        transformer = ReferenceFrameTransformer()

        plane = SwingPlaneFrame(
            origin=np.zeros(3),
            normal=np.array([0, 0, 1], dtype=float),
            in_plane_x=np.array([1, 0, 0], dtype=float),
            in_plane_y=np.array([0, 1, 0], dtype=float),
            grip_axis=np.array([0, 0, 1], dtype=float),
        )

        with pytest.raises(ValueError):
            transformer.global_to_swing_plane(
                WrenchInFrame(np.zeros(3), np.zeros(3), ReferenceFrame.GLOBAL)
            )

        with pytest.raises(ValueError):
            transformer.get_swing_plane_decomposition(
                WrenchInFrame(np.zeros(3), np.zeros(3), ReferenceFrame.GLOBAL)
            )

        transformer.set_swing_plane(plane)

        wrench = WrenchInFrame(
            force=np.array([1, 0, 0], dtype=float),
            torque=np.array([0, 1, 0], dtype=float),
            frame=ReferenceFrame.GLOBAL,
        )

        decomp = transformer.get_swing_plane_decomposition(wrench)
        assert decomp["force_in_plane"] == 1.0
        transformed_sp = transformer.global_to_swing_plane(wrench)
        assert transformed_sp.frame == ReferenceFrame.SWING_PLANE

        R = np.eye(3)
        transformer.set_body_rotation("pelvis", R)

        transformed_body = transformer.global_to_local(wrench, "pelvis")
        assert transformed_body.frame == ReferenceFrame.LOCAL

        with pytest.raises(ValueError):
            transformer.global_to_local(wrench, "unknown")
