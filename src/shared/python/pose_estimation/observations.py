"""Canonical observation schema for multi-camera markerless pose estimation.

CC-12 (#6785) — input-side contract for the physics-matched estimator (CC-18).

Observations capture per-camera 2D keypoints, camera calibration, per-keypoint
confidence, and optionally triangulated 3D positions.  Confidence is preserved
end-to-end so downstream estimators (CC-18) can weight residuals per keypoint.

Typical usage::

    obs = CanonicalObservations(
        cameras=(cam0_calib, cam1_calib),
        keypoints_2d=(kps_cam0, kps_cam1),   # each shape (N_kp, 2) [px]
        confidences=(conf_cam0, conf_cam1),   # each shape (N_kp,) ∈ [0, 1]
        detector_layout=("nose", "left_hip", ...),
        timestamp=0.033,
    )
    d = observations_to_dict(obs)   # JSON-serialisable
    obs2 = observations_from_dict(d)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CanonicalObservations",
    "CameraCalibration",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "observations_from_dict",
    "observations_to_dict",
]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters.

    Attributes:
        fx: Horizontal focal length [px].
        fy: Vertical focal length [px].
        cx: Principal point x [px].
        cy: Principal point y [px].
        width: Image width [px].
        height: Image height [px].
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                f"focal lengths must be positive; got fx={self.fx}, fy={self.fy}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"image dimensions must be positive; "
                f"got width={self.width}, height={self.height}"
            )

    def as_matrix(self) -> np.ndarray:
        """Return the 3×3 intrinsic matrix **K**.

        Returns:
            Array of shape ``(3, 3)``::

                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        """
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    """Camera extrinsic parameters (world-to-camera rigid transform).

    Attributes:
        rotation: 3×3 rotation matrix **R** (world-to-camera).
        translation: 3-vector **t** (camera origin in world coordinates) [m].
    """

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def __post_init__(self) -> None:
        r = np.asarray(self.rotation, dtype=float)
        t = np.asarray(self.translation, dtype=float)
        if r.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {r.shape}")
        if t.shape != (3,):
            raise ValueError(f"translation must have shape (3,), got {t.shape}")
        # Re-assign as float arrays (dataclass is frozen, so use __setattr__).
        object.__setattr__(self, "rotation", r)
        object.__setattr__(self, "translation", t)


@dataclass(frozen=True)
class CameraCalibration:
    """Combined intrinsic and extrinsic calibration for one camera.

    Attributes:
        camera_id: Unique identifier string (e.g. ``"cam0"``, ``"left"``).
        intrinsics: Pinhole intrinsic parameters.
        extrinsics: World-to-camera rigid transform.
    """

    camera_id: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")


@dataclass(frozen=True)
class CanonicalObservations:
    """Canonical multi-camera observation for a single frame.

    Captures the raw 2D keypoint observations from all cameras together with
    per-keypoint confidence weights and camera calibration.  Optionally
    includes triangulated 3D positions.

    Preconditions (enforced in ``__post_init__``):

    * ``len(cameras) == len(keypoints_2d) == len(confidences)``
    * ``keypoints_2d[i].shape == (n_keypoints, 2)`` for every *i*
    * ``confidences[i].shape == (n_keypoints,)`` with all values in ``[0, 1]``
    * ``len(detector_layout) == n_keypoints > 0``
    * ``keypoints_3d`` (if given) has shape ``(n_keypoints, 3)``

    Attributes:
        cameras: Per-camera calibration list.
        keypoints_2d: Per-camera 2D keypoints, each shape ``(n_kp, 2)`` [px].
        confidences: Per-camera per-keypoint confidence scores,
            each shape ``(n_kp,)`` with values in ``[0, 1]``.
        detector_layout: Ordered keypoint names matching the detector output.
        timestamp: Frame timestamp [s].
        keypoints_3d: Optional triangulated 3D positions,
            shape ``(n_kp, 3)`` [m].
    """

    cameras: tuple[CameraCalibration, ...]
    keypoints_2d: tuple[np.ndarray, ...]
    confidences: tuple[np.ndarray, ...]
    detector_layout: tuple[str, ...]
    timestamp: float
    keypoints_3d: np.ndarray | None = None

    def __post_init__(self) -> None:  # noqa: C901
        n_cams = len(self.cameras)
        if len(self.keypoints_2d) != n_cams:
            raise ValueError(
                f"keypoints_2d length {len(self.keypoints_2d)} != "
                f"number of cameras {n_cams}"
            )
        if len(self.confidences) != n_cams:
            raise ValueError(
                f"confidences length {len(self.confidences)} != "
                f"number of cameras {n_cams}"
            )
        n_kp = len(self.detector_layout)
        if n_kp == 0:
            raise ValueError("detector_layout must be non-empty")
        for i, kp in enumerate(self.keypoints_2d):
            kp_arr = np.asarray(kp, dtype=float)
            if kp_arr.shape != (n_kp, 2):
                raise ValueError(
                    f"keypoints_2d[{i}] must have shape ({n_kp}, 2), got {kp_arr.shape}"
                )
        for i, conf in enumerate(self.confidences):
            c_arr = np.asarray(conf, dtype=float)
            if c_arr.shape != (n_kp,):
                raise ValueError(
                    f"confidences[{i}] must have shape ({n_kp},), got {c_arr.shape}"
                )
            if not (np.all(c_arr >= 0.0) and np.all(c_arr <= 1.0)):
                raise ValueError(
                    f"confidence values must be in [0, 1]; "
                    f"confidences[{i}] has min={float(c_arr.min()):.4f}, "
                    f"max={float(c_arr.max()):.4f}"
                )
        if self.keypoints_3d is not None:
            kp3 = np.asarray(self.keypoints_3d, dtype=float)
            if kp3.shape != (n_kp, 3):
                raise ValueError(
                    f"keypoints_3d must have shape ({n_kp}, 3), got {kp3.shape}"
                )
            object.__setattr__(self, "keypoints_3d", kp3)

    @property
    def n_cameras(self) -> int:
        """Number of cameras in this observation."""
        return len(self.cameras)

    @property
    def n_keypoints(self) -> int:
        """Number of keypoints declared in ``detector_layout``."""
        return len(self.detector_layout)


# ---------------------------------------------------------------------------
# Serialisation helpers (JSON-compatible dict ↔ CanonicalObservations)
# ---------------------------------------------------------------------------


def observations_to_dict(obs: CanonicalObservations) -> dict[str, Any]:
    """Serialise a :class:`CanonicalObservations` to a JSON-native dict.

    All NumPy arrays are converted to nested Python lists so the result can
    be passed to ``json.dumps`` or included in a CC-4 provenance record.

    Args:
        obs: Observation to serialise.

    Returns:
        A dict with only JSON-native scalars, lists, and strings.
    """
    cameras_out = []
    for cal in obs.cameras:
        cameras_out.append(
            {
                "camera_id": cal.camera_id,
                "intrinsics": {
                    "fx": cal.intrinsics.fx,
                    "fy": cal.intrinsics.fy,
                    "cx": cal.intrinsics.cx,
                    "cy": cal.intrinsics.cy,
                    "width": cal.intrinsics.width,
                    "height": cal.intrinsics.height,
                },
                "extrinsics": {
                    "rotation": cal.extrinsics.rotation.tolist(),
                    "translation": cal.extrinsics.translation.tolist(),
                },
            }
        )
    return {
        "cameras": cameras_out,
        "keypoints_2d": [kp.tolist() for kp in obs.keypoints_2d],
        "confidences": [c.tolist() for c in obs.confidences],
        "detector_layout": list(obs.detector_layout),
        "timestamp": float(obs.timestamp),
        "keypoints_3d": (
            obs.keypoints_3d.tolist() if obs.keypoints_3d is not None else None
        ),
    }


def observations_from_dict(d: dict[str, Any]) -> CanonicalObservations:
    """Deserialise a dict produced by :func:`observations_to_dict`.

    Args:
        d: Dict previously produced by :func:`observations_to_dict`.

    Returns:
        A :class:`CanonicalObservations` instance with all validation applied.

    Raises:
        KeyError: If a required key is missing from *d*.
        ValueError: If the data violates the ``CanonicalObservations``
            preconditions.
    """
    cameras = tuple(
        CameraCalibration(
            camera_id=c["camera_id"],
            intrinsics=CameraIntrinsics(
                fx=c["intrinsics"]["fx"],
                fy=c["intrinsics"]["fy"],
                cx=c["intrinsics"]["cx"],
                cy=c["intrinsics"]["cy"],
                width=int(c["intrinsics"]["width"]),
                height=int(c["intrinsics"]["height"]),
            ),
            extrinsics=CameraExtrinsics(
                rotation=np.array(c["extrinsics"]["rotation"]),
                translation=np.array(c["extrinsics"]["translation"]),
            ),
        )
        for c in d["cameras"]
    )
    keypoints_2d = tuple(np.array(kp) for kp in d["keypoints_2d"])
    confidences = tuple(np.array(c) for c in d["confidences"])
    kp3d_raw = d.get("keypoints_3d")
    keypoints_3d = np.array(kp3d_raw) if kp3d_raw is not None else None
    return CanonicalObservations(
        cameras=cameras,
        keypoints_2d=keypoints_2d,
        confidences=confidences,
        detector_layout=tuple(d["detector_layout"]),
        timestamp=float(d["timestamp"]),
        keypoints_3d=keypoints_3d,
    )
