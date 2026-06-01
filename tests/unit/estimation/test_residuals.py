from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.residuals import (
    anthropometric_prior_residual,
    dynamics_residual,
    finite_difference_jacobian,
    project_pinhole,
    reprojection_residual,
    reprojection_residual_from_points,
    residual_jacobian,
    smoothness_residual,
)


def test_project_pinhole_identity_camera() -> None:
    points = np.array([[2.0, 4.0, 2.0], [3.0, -6.0, 3.0]])
    camera = np.eye(3)

    projected = project_pinhole(points, camera)

    np.testing.assert_allclose(projected, [[1.0, 2.0], [1.0, -2.0]])


def test_reprojection_residual_weights_confidence_and_offsets() -> None:
    q = np.array([1.0, 2.0, 4.0, 2.0, 2.0, 2.0])
    observed = np.array([[0.5, 0.5], [1.0, 1.0]])
    confidence = np.array([1.0, 0.0])
    offsets = np.array([[1.0, 0.0, 0.0], [10.0, 10.0, 0.0]])

    residual = reprojection_residual(
        q,
        observed,
        lambda value: value.reshape(2, 3),
        np.eye(3),
        confidence,
        keypoint_offsets_m=offsets,
    )

    np.testing.assert_allclose(residual, [0.0, 0.0, 0.0, 0.0])


def test_reprojection_jacobian_matches_hand_derived_projection_jacobian() -> None:
    q = np.array([1.2, 0.4, 3.0, -0.8, 0.7, 2.5])
    observed = np.zeros((2, 2))
    confidence = np.array([0.25, 1.0])
    camera = np.array([[500.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])

    def residual(value: np.ndarray) -> np.ndarray:
        return reprojection_residual(
            value,
            observed,
            lambda q_value: q_value.reshape(2, 3),
            camera,
            confidence,
        )

    jacobian = finite_difference_jacobian(residual, q)

    expected = np.zeros((4, 6))
    for idx in range(2):
        x, y, z = q.reshape(2, 3)[idx]
        weight = np.sqrt(confidence[idx])
        row = 2 * idx
        col = 3 * idx
        expected[row, col] = weight * camera[0, 0] / z
        expected[row, col + 2] = -weight * camera[0, 0] * x / z**2
        expected[row + 1, col + 1] = weight * camera[1, 1] / z
        expected[row + 1, col + 2] = -weight * camera[1, 1] * y / z**2

    np.testing.assert_allclose(jacobian, expected, rtol=1e-6, atol=1e-6)


def test_dynamics_residual_penalizes_selected_rnea_torques() -> None:
    q = np.array([1.0, 2.0, 3.0])
    v = np.array([0.5, -1.0])
    a = np.array([0.25, 0.75])
    weights = np.array([1.0, 4.0])

    def rnea(
        q_value: np.ndarray, v_value: np.ndarray, a_value: np.ndarray
    ) -> np.ndarray:
        return np.array(
            [
                q_value[0] + v_value[0],
                q_value[1] - v_value[1] + 2.0 * a_value[1],
            ]
        )

    residual = dynamics_residual(
        q,
        v,
        a,
        rnea,
        torque_target=np.array([1.0, 0.5]),
        torque_weights=weights,
    )

    np.testing.assert_allclose(residual, [0.5, 8.0])


def test_dynamics_jacobian_matches_linear_rnea_model() -> None:
    q = np.array([0.2, -0.3, 0.4])
    v = np.array([1.0, -1.5])
    a = np.array([0.7, -0.2])
    x = np.concatenate([q, v, a])
    q_matrix = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]])
    v_matrix = np.array([[0.5, 0.0], [2.0, -1.0]])
    a_matrix = np.array([[0.0, 4.0], [-2.0, 0.25]])
    weights = np.array([4.0, 9.0])

    def rnea(
        q_value: np.ndarray, v_value: np.ndarray, a_value: np.ndarray
    ) -> np.ndarray:
        return q_matrix @ q_value + v_matrix @ v_value + a_matrix @ a_value

    def residual(value: np.ndarray) -> np.ndarray:
        return dynamics_residual(
            value[:3],
            value[3:5],
            value[5:7],
            rnea,
            torque_weights=weights,
        )

    jacobian = finite_difference_jacobian(residual, x)
    scale = np.sqrt(weights)[:, None]
    expected = scale * np.hstack([q_matrix, v_matrix, a_matrix])

    np.testing.assert_allclose(jacobian, expected, rtol=1e-8, atol=1e-8)


def test_anthropometric_prior_jacobian_matches_finite_difference() -> None:
    parameters = np.array([1.8, 82.0, 0.43])
    nominal = np.array([1.75, 80.0, 0.40])
    sigma = np.array([0.05, 5.0, 0.02])
    weights = np.array([1.0, 4.0, 0.25])

    def residual(value: np.ndarray) -> np.ndarray:
        return anthropometric_prior_residual(value, nominal, sigma, weights=weights)

    jacobian = finite_difference_jacobian(residual, parameters)
    expected = np.diag(np.sqrt(weights) / sigma)

    np.testing.assert_allclose(jacobian, expected, rtol=1e-8, atol=1e-8)


def test_smoothness_jacobian_matches_second_difference_operator() -> None:
    trajectory = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ]
    )
    x = trajectory.reshape(-1)
    weights = np.array([1.0, 4.0])
    dt = 0.5

    def residual(value: np.ndarray) -> np.ndarray:
        return smoothness_residual(
            value.reshape(4, 2),
            dt=dt,
            order=2,
            weights=weights,
        )

    jacobian = finite_difference_jacobian(residual, x)
    expected = np.zeros((4, 8))
    stencil = np.array([1.0, -2.0, 1.0]) / dt**2
    row = 0
    for frame in range(2):
        for dof in range(2):
            scale = np.sqrt(weights[dof])
            for offset, coeff in enumerate(stencil):
                col = (frame + offset) * 2 + dof
                expected[row, col] = scale * coeff
            row += 1

    np.testing.assert_allclose(jacobian, expected, rtol=1e-8, atol=1e-8)


def test_residual_jacobian_uses_finite_method_explicitly() -> None:
    x = np.array([1.0, 2.0, 3.0])

    jacobian = residual_jacobian(lambda value: value**2, x, method="finite")

    np.testing.assert_allclose(jacobian, np.diag(2.0 * x), rtol=1e-6, atol=1e-6)


def test_autodiff_jacobian_matches_finite_difference_when_jax_available() -> None:
    pytest.importorskip("jax")
    x = np.array([0.2, -0.1, 0.5])

    auto = residual_jacobian(lambda value: value * value + 2.0 * value, x, method="jax")
    finite = residual_jacobian(
        lambda value: value * value + 2.0 * value,
        x,
        method="finite",
    )

    np.testing.assert_allclose(auto, finite, rtol=1e-5, atol=1e-5)


def test_project_pinhole_matches_brown_conrady_rig(monkeypatch=None) -> None:
    """project_pinhole with distortion must match the synthetic rig's
    Brown-Conrady projection (issue #6892)."""
    from src.shared.python.estimation.synthetic_ground_truth import (
        SyntheticCamera,
        project_world_point,
    )
    from src.shared.python.motion_pipeline import (
        CameraExtrinsics,
        CameraIntrinsics,
    )

    intr = CameraIntrinsics(
        fx=600.0,
        fy=550.0,
        cx=320.0,
        cy=240.0,
        k1=-0.21,
        k2=0.08,
        p1=0.001,
        p2=-0.002,
    )
    rotation = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=float
    )
    translation = np.array([0.1, -0.2, 4.0])
    extr = CameraExtrinsics(
        rotation=rotation.tolist(), translation=translation.tolist()
    )
    camera = SyntheticCamera("cam", intr, extr)

    points = np.array([[0.3, 0.1, 0.2], [-0.4, 0.25, -0.1]])
    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]]
    )

    projected = project_pinhole(
        points,
        camera_matrix,
        rotation_world_to_camera=rotation,
        translation_world_to_camera=translation,
        distortion=(intr.k1, intr.k2, intr.p1, intr.p2),
    )

    expected = np.array([project_world_point(camera, p)[:2] for p in points])
    np.testing.assert_allclose(projected, expected, rtol=1e-9, atol=1e-9)


def test_project_pinhole_zero_distortion_is_pinhole() -> None:
    """Zero distortion must reproduce the plain pinhole projection."""
    points = np.array([[2.0, 4.0, 2.0], [3.0, -6.0, 3.0]])
    camera = np.eye(3)
    plain = project_pinhole(points, camera)
    with_zero = project_pinhole(points, camera, distortion=(0.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(plain, with_zero)


def test_project_pinhole_accepts_five_term_distortion() -> None:
    """A 5-term distortion array is accepted and correctly computed."""
    points = np.array([[1.0, 1.0, 2.0]])
    camera = np.eye(3)
    distortion_4 = (0.1, 0.05, 0.01, 0.02)
    distortion_5 = (0.1, 0.05, 0.01, 0.02, 0.03)

    proj_4 = project_pinhole(points, camera, distortion=distortion_4)
    proj_5 = project_pinhole(points, camera, distortion=distortion_5)

    assert not np.allclose(proj_4, proj_5)

    # Let's verify that a bad shape still raises a ValueError
    with pytest.raises(ValueError, match="distortion must have shape"):
        project_pinhole(points, camera, distortion=(0.1, 0.2, 0.3))


def test_reprojection_residual_threads_distortion() -> None:
    """A fit with nonzero distortion recovers truth when residual threads
    the distortion coefficients (issue #6892)."""
    points = np.array([[0.2, -0.1, 0.05]])
    camera_matrix = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    distortion = (-0.15, 0.05, 0.0, 0.0)
    observed = project_pinhole(points, camera_matrix, distortion=distortion)

    residual = reprojection_residual_from_points(
        points,
        observed,
        camera_matrix,
        np.array([1.0]),
        distortion=distortion,
    )
    np.testing.assert_allclose(residual, np.zeros(2), atol=1e-9)


def test_invalid_confidence_rejected() -> None:
    with pytest.raises(ValueError, match="confidence values"):
        reprojection_residual(
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 2.0]]),
            lambda value: value.reshape(1, 3),
            np.eye(3),
            np.array([1.5]),
        )
