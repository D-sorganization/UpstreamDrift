"""Camera direction conversion between pose_estimation and motion_pipeline contracts (Packet C)."""

from __future__ import annotations

import numpy as np

from motion_pipeline.contracts import (
    CameraExtrinsics as PipelineExtrinsics,
    CameraIntrinsics as PipelineIntrinsics,
)
from pose_estimation.observations import (
    CameraCalibration,
    CameraExtrinsics as ObsExtrinsics,
    CameraIntrinsics as ObsIntrinsics,
)
from spatial_algebra.pose6dof.transform import Transform6DOF

from ._validation import check_id, check_pos_int


def to_pipeline_camera(
    camera: CameraCalibration,
) -> tuple[PipelineIntrinsics, PipelineExtrinsics]:
    """Convert observation camera-to-world calibration to pipeline world-to-camera models.

    Preconditions:
        - camera must be an instance of pose_estimation.observations.CameraCalibration.
        - intrinsics matrix K must have zero skew, bottom row [0, 0, 1], and positive focal lengths.
        - distortion must be None, 0, 4, or 5 coefficients.
        - extrinsics must be a valid SE(3) transform (camera-to-world).

    Postconditions:
        - Returns a tuple of (CameraIntrinsics, CameraExtrinsics).
        - R_cw = R_wc.T and t_cw = -R_wc.T @ t_wc.
        - Distortion canonicalizes: None or empty to all zeros; 4 coeffs to 5 with k3=0.
    """
    if not isinstance(camera, CameraCalibration):
        raise TypeError(
            f"camera must be a CameraCalibration, got {type(camera).__name__}"
        )

    matrix = np.asarray(camera.intrinsics.matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"Camera intrinsics matrix must be finite 3x3, got shape {matrix.shape}"
        )

    if not np.allclose(matrix[2, :], [0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(
            f"Bottom row of intrinsics matrix must be [0, 0, 1], got {matrix[2, :]}"
        )

    if not np.isclose(matrix[0, 1], 0.0, atol=1e-9) or not np.isclose(
        matrix[1, 0], 0.0, atol=1e-9
    ):
        raise ValueError("Nonzero skew in camera intrinsics matrix is unsupported")

    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Focal lengths must be positive, got fx={fx}, fy={fy}")

    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])

    distortion = camera.intrinsics.distortion
    if distortion is None:
        k1 = k2 = p1 = p2 = k3 = 0.0
    else:
        dist_arr = np.asarray(distortion, dtype=np.float64).ravel()
        if not np.all(np.isfinite(dist_arr)):
            raise ValueError("Distortion coefficients must be finite")
        if len(dist_arr) == 0:
            k1 = k2 = p1 = p2 = k3 = 0.0
        elif len(dist_arr) == 4:
            k1, k2, p1, p2 = (
                float(dist_arr[0]),
                float(dist_arr[1]),
                float(dist_arr[2]),
                float(dist_arr[3]),
            )
            k3 = 0.0
        elif len(dist_arr) == 5:
            k1, k2, p1, p2, k3 = (
                float(dist_arr[0]),
                float(dist_arr[1]),
                float(dist_arr[2]),
                float(dist_arr[3]),
                float(dist_arr[4]),
            )
        else:
            raise ValueError(
                f"Unsupported distortion length {len(dist_arr)}; expected 0, 4, or 5 coefficients"
            )

    p_intrinsics = PipelineIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3
    )

    r_wc = np.asarray(camera.extrinsics.rotation_world_from_camera, dtype=np.float64)
    t_wc = np.asarray(
        camera.extrinsics.translation_world_from_camera_m, dtype=np.float64
    )
    tf = Transform6DOF(rotation=r_wc, translation=t_wc)
    tf_inv = tf.inverse()
    r_cw = tf_inv.rotation_matrix
    t_cw = tf_inv.translation

    p_extrinsics = PipelineExtrinsics(rotation=r_cw.tolist(), translation=t_cw.tolist())

    return p_intrinsics, p_extrinsics


def from_pipeline_camera(
    *,
    camera_id: str,
    image_size_px: tuple[int, int],
    intrinsics: PipelineIntrinsics,
    extrinsics: PipelineExtrinsics,
) -> CameraCalibration:
    """Convert pipeline world-to-camera models back to observation CameraCalibration.

    Preconditions:
        - camera_id must be a non-empty trimmed string.
        - image_size_px must be a tuple of 2 positive integers (width, height).
        - intrinsics must be motion_pipeline.contracts.CameraIntrinsics.
        - extrinsics must be motion_pipeline.contracts.CameraExtrinsics.

    Postconditions:
        - Returns a CameraCalibration with camera-to-world extrinsics.
        - Distortion is canonicalized to 5 coefficients.
    """
    check_id(camera_id, "camera_id")

    if not isinstance(image_size_px, tuple) or len(image_size_px) != 2:
        raise TypeError(
            f"image_size_px must be a tuple of 2 ints, got {type(image_size_px).__name__}"
        )
    width = check_pos_int(image_size_px[0], "image_size_px[0]")
    height = check_pos_int(image_size_px[1], "image_size_px[1]")

    if not isinstance(intrinsics, PipelineIntrinsics):
        raise TypeError(
            f"intrinsics must be a PipelineIntrinsics, got {type(intrinsics).__name__}"
        )
    if not isinstance(extrinsics, PipelineExtrinsics):
        raise TypeError(
            f"extrinsics must be a PipelineExtrinsics, got {type(extrinsics).__name__}"
        )

    matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.array(
        [
            intrinsics.k1,
            intrinsics.k2,
            intrinsics.p1,
            intrinsics.p2,
            intrinsics.k3,
        ],
        dtype=np.float64,
    )
    obs_intrinsics = ObsIntrinsics(matrix=matrix, distortion=distortion)

    r_cw = np.asarray(extrinsics.rotation, dtype=np.float64)
    t_cw = np.asarray(extrinsics.translation, dtype=np.float64)
    tf = Transform6DOF(rotation=r_cw, translation=t_cw)
    tf_inv = tf.inverse()
    r_wc = tf_inv.rotation_matrix
    t_wc = tf_inv.translation

    obs_extrinsics = ObsExtrinsics(
        rotation_world_from_camera=r_wc,
        translation_world_from_camera_m=t_wc,
    )

    return CameraCalibration(
        camera_id=camera_id,
        image_size_px=(width, height),
        intrinsics=obs_intrinsics,
        extrinsics=obs_extrinsics,
    )
