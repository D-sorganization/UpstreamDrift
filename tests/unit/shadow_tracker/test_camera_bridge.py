"""Unit tests for Shadow Tracker camera direction conversion (Packet C, #10139)."""

from __future__ import annotations

import numpy as np
import pytest

from motion_pipeline.contracts import (
    CameraExtrinsics as PipelineExtrinsics,
    CameraIntrinsics as PipelineIntrinsics,
)
from pose_estimation.observations import (
    CameraCalibration,
    CameraExtrinsics as ObsExtrinsics,
    CameraIntrinsics as ObsIntrinsics,
)
from shared.python.shadow_tracker.camera_bridge import (
    from_pipeline_camera,
    to_pipeline_camera,
)

pytestmark = pytest.mark.unit


def _make_obs_camera(
    *,
    camera_id: str = "cam-main",
    image_size_px: tuple[int, int] = (640, 480),
    matrix: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
    rotation_wc: np.ndarray | None = None,
    translation_wc: np.ndarray | None = None,
) -> CameraCalibration:
    if matrix is None:
        matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if rotation_wc is None:
        rotation_wc = np.eye(3, dtype=np.float64)
    if translation_wc is None:
        translation_wc = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    return CameraCalibration(
        camera_id=camera_id,
        image_size_px=image_size_px,
        intrinsics=ObsIntrinsics(matrix=matrix, distortion=distortion),
        extrinsics=ObsExtrinsics(
            rotation_world_from_camera=rotation_wc,
            translation_world_from_camera_m=translation_wc,
        ),
    )


# --------------------------------------------------------------------------- #
#  Acceptance Cases (Table in CONTRACT_FREEZE.md)                             #
# --------------------------------------------------------------------------- #


def test_acceptance_identity_rotation_with_translation() -> None:
    """R_wc=I, t_wc=(1,2,3) -> R_cw=I, t_cw=(-1,-2,-3)."""
    camera = _make_obs_camera(
        rotation_wc=np.eye(3),
        translation_wc=np.array([1.0, 2.0, 3.0]),
    )
    p_intrinsics, p_extrinsics = to_pipeline_camera(camera)

    assert np.allclose(p_extrinsics.rotation, np.eye(3))
    assert np.allclose(p_extrinsics.translation, [-1.0, -2.0, -3.0])

    # Round trip
    restored = from_pipeline_camera(
        camera_id=camera.camera_id,
        image_size_px=camera.image_size_px,
        intrinsics=p_intrinsics,
        extrinsics=p_extrinsics,
    )
    assert np.allclose(
        restored.extrinsics.rotation_world_from_camera,
        camera.extrinsics.rotation_world_from_camera,
    )
    assert np.allclose(
        restored.extrinsics.translation_world_from_camera_m,
        camera.extrinsics.translation_world_from_camera_m,
    )


def test_acceptance_90_degree_z_rotation_with_translation() -> None:
    """90-degree Z rotation with nonzero translation round trips properly."""
    # 90 degrees around Z: cos(90)=0, sin(90)=1
    # [ [0, -1, 0], [1, 0, 0], [0, 0, 1] ]
    r_wc = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t_wc = np.array([2.0, 3.0, 5.0])

    camera = _make_obs_camera(rotation_wc=r_wc, translation_wc=t_wc)
    p_intrinsics, p_extrinsics = to_pipeline_camera(camera)

    expected_r_cw = r_wc.T
    expected_t_cw = -r_wc.T @ t_wc
    assert np.allclose(p_extrinsics.rotation, expected_r_cw)
    assert np.allclose(p_extrinsics.translation, expected_t_cw)

    # Round trip
    restored = from_pipeline_camera(
        camera_id="cam-rot",
        image_size_px=(1920, 1080),
        intrinsics=p_intrinsics,
        extrinsics=p_extrinsics,
    )
    assert np.allclose(restored.extrinsics.rotation_world_from_camera, r_wc)
    assert np.allclose(restored.extrinsics.translation_world_from_camera_m, t_wc)


def test_acceptance_rejection_nonzero_skew() -> None:
    """Nonzero skew in K matrix must be rejected."""
    bad_k = np.array([[500.0, 0.5, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    camera = _make_obs_camera(matrix=bad_k)
    with pytest.raises(ValueError, match="skew"):
        to_pipeline_camera(camera)


def test_acceptance_rejection_reflection() -> None:
    """Reflection (det == -1) must be rejected."""
    # det of this matrix is -1
    bad_r = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    # ObsExtrinsics constructor itself enforces det == +1
    with pytest.raises(ValueError, match="determinant"):
        ObsExtrinsics(
            rotation_world_from_camera=bad_r,
            translation_world_from_camera_m=np.zeros(3),
        )


def test_acceptance_rejection_nan() -> None:
    """NaN or infinite values must fail explicitly."""
    bad_k = np.array([[float("nan"), 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        _make_obs_camera(matrix=bad_k)


def test_acceptance_unsupported_distortion_lengths() -> None:
    """Distortion lengths other than 0, 4, 5 fail."""
    for length in [1, 2, 3, 6, 8]:
        dist = np.ones(length)
        camera = _make_obs_camera(distortion=dist)
        with pytest.raises(ValueError, match="distortion"):
            to_pipeline_camera(camera)


def test_acceptance_world_point_projection_consistency() -> None:
    """World point obtained from X_camera=(0,0,2), K center=(320,240) projects to (320,240)."""
    r_wc = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t_wc = np.array([1.5, -2.5, 0.5])
    camera = _make_obs_camera(
        matrix=np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
        rotation_wc=r_wc,
        translation_wc=t_wc,
    )
    p_intrinsics, p_extrinsics = to_pipeline_camera(camera)

    x_cam = np.array([0.0, 0.0, 2.0])
    x_world = r_wc @ x_cam + t_wc

    # Pipeline world-to-camera projection
    r_cw = np.asarray(p_extrinsics.rotation)
    t_cw = np.asarray(p_extrinsics.translation)
    x_cam_pipeline = r_cw @ x_world + t_cw
    assert np.allclose(x_cam_pipeline, x_cam)

    u_pipe = p_intrinsics.fx * (x_cam_pipeline[0] / x_cam_pipeline[2]) + p_intrinsics.cx
    v_pipe = p_intrinsics.fy * (x_cam_pipeline[1] / x_cam_pipeline[2]) + p_intrinsics.cy
    assert np.isclose(u_pipe, 320.0)
    assert np.isclose(v_pipe, 240.0)


# --------------------------------------------------------------------------- #
#  Distortion Canonicalization Tests                                          #
# --------------------------------------------------------------------------- #


def test_distortion_none_canonicalizes_to_five_zeros() -> None:
    camera = _make_obs_camera(distortion=None)
    p_intrinsics, p_extrinsics = to_pipeline_camera(camera)
    assert p_intrinsics.k1 == 0.0
    assert p_intrinsics.k2 == 0.0
    assert p_intrinsics.p1 == 0.0
    assert p_intrinsics.p2 == 0.0
    assert p_intrinsics.k3 == 0.0

    restored = from_pipeline_camera(
        camera_id=camera.camera_id,
        image_size_px=camera.image_size_px,
        intrinsics=p_intrinsics,
        extrinsics=p_extrinsics,
    )
    assert restored.intrinsics.distortion is not None
    assert np.allclose(restored.intrinsics.distortion, [0.0, 0.0, 0.0, 0.0, 0.0])


def test_distortion_four_coeffs_canonicalizes_with_k3_zero() -> None:
    dist_4 = np.array([0.1, -0.05, 0.001, -0.002])
    camera = _make_obs_camera(distortion=dist_4)
    p_intrinsics, p_extrinsics = to_pipeline_camera(camera)
    assert p_intrinsics.k1 == 0.1
    assert p_intrinsics.k2 == -0.05
    assert p_intrinsics.p1 == 0.001
    assert p_intrinsics.p2 == -0.002
    assert p_intrinsics.k3 == 0.0

    restored = from_pipeline_camera(
        camera_id=camera.camera_id,
        image_size_px=camera.image_size_px,
        intrinsics=p_intrinsics,
        extrinsics=p_extrinsics,
    )
    assert restored.intrinsics.distortion is not None
    assert np.allclose(restored.intrinsics.distortion, [0.1, -0.05, 0.001, -0.002, 0.0])


def test_distortion_five_coeffs_preserved() -> None:
    dist_5 = np.array([0.1, -0.05, 0.001, -0.002, 0.0003])
    camera = _make_obs_camera(distortion=dist_5)
    p_intrinsics, p_extrinsics = to_pipeline_camera(camera)
    assert p_intrinsics.k3 == 0.0003

    restored = from_pipeline_camera(
        camera_id=camera.camera_id,
        image_size_px=camera.image_size_px,
        intrinsics=p_intrinsics,
        extrinsics=p_extrinsics,
    )
    assert restored.intrinsics.distortion is not None
    assert np.allclose(restored.intrinsics.distortion, dist_5)


# --------------------------------------------------------------------------- #
#  Precondition and Type Validation                                           #
# --------------------------------------------------------------------------- #


def test_to_pipeline_camera_type_error() -> None:
    with pytest.raises(TypeError, match="camera"):
        to_pipeline_camera("not-a-camera")  # type: ignore[arg-type]


def test_from_pipeline_camera_type_errors() -> None:
    p_intrinsics = PipelineIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    p_extrinsics = PipelineExtrinsics()

    with pytest.raises(TypeError, match="camera_id"):
        from_pipeline_camera(
            camera_id=123,  # type: ignore[arg-type]
            image_size_px=(640, 480),
            intrinsics=p_intrinsics,
            extrinsics=p_extrinsics,
        )

    with pytest.raises(TypeError, match="image_size_px"):
        from_pipeline_camera(
            camera_id="cam-1",
            image_size_px="640x480",  # type: ignore[arg-type]
            intrinsics=p_intrinsics,
            extrinsics=p_extrinsics,
        )

    with pytest.raises(TypeError, match="intrinsics"):
        from_pipeline_camera(
            camera_id="cam-1",
            image_size_px=(640, 480),
            intrinsics="bad",  # type: ignore[arg-type]
            extrinsics=p_extrinsics,
        )

    with pytest.raises(TypeError, match="extrinsics"):
        from_pipeline_camera(
            camera_id="cam-1",
            image_size_px=(640, 480),
            intrinsics=p_intrinsics,
            extrinsics="bad",  # type: ignore[arg-type]
        )


def test_to_pipeline_camera_negative_focal_length_rejected() -> None:
    camera = _make_obs_camera()
    bad_k = np.array([[-500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    object.__setattr__(camera.intrinsics, "matrix", bad_k)
    with pytest.raises(ValueError, match="Focal lengths must be positive"):
        to_pipeline_camera(camera)
