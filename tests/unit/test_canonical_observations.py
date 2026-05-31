"""Unit tests for pose_estimation.observations (CC-12, #6785).

TDD: these tests were written before the implementation and drove the design.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_estimation.observations import (
    CanonicalObservations,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    observations_from_dict,
    observations_to_dict,
)

pytestmark = pytest.mark.unit

_N_KP = 5
_LAYOUT = tuple(f"kp{i}" for i in range(_N_KP))
_IMG_W, _IMG_H = 1920, 1080


def _intrinsics(fx: float = 800.0, fy: float = 800.0) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=_IMG_W / 2,
        cy=_IMG_H / 2,
        width=_IMG_W,
        height=_IMG_H,
    )


def _extrinsics(offset: float = 0.0) -> CameraExtrinsics:
    return CameraExtrinsics(
        rotation=np.eye(3),
        translation=np.array([offset, 0.0, 5.0]),
    )


def _calib(cam_id: str = "cam0", offset: float = 0.0) -> CameraCalibration:
    return CameraCalibration(
        camera_id=cam_id,
        intrinsics=_intrinsics(),
        extrinsics=_extrinsics(offset),
    )


def _make_obs(
    n_cams: int = 2,
    include_3d: bool = False,
) -> CanonicalObservations:
    rng = np.random.default_rng(42)
    cameras = tuple(_calib(f"cam{i}", float(i)) for i in range(n_cams))
    kps = tuple(rng.uniform(0, 1920, size=(_N_KP, 2)) for _ in range(n_cams))
    confs = tuple(rng.uniform(0.5, 1.0, size=(_N_KP,)) for _ in range(n_cams))
    kp3d = rng.standard_normal((_N_KP, 3)) if include_3d else None
    return CanonicalObservations(
        cameras=cameras,
        keypoints_2d=kps,
        confidences=confs,
        detector_layout=_LAYOUT,
        timestamp=1.23,
        keypoints_3d=kp3d,
    )


# ---------------------------------------------------------------------------
# CameraIntrinsics
# ---------------------------------------------------------------------------


class TestCameraIntrinsics:
    def test_construction(self) -> None:
        k = _intrinsics()
        assert k.fx == 800.0
        assert k.width == _IMG_W

    def test_as_matrix(self) -> None:
        k = _intrinsics(fx=600.0, fy=700.0)
        mat = k.as_matrix()
        assert mat.shape == (3, 3)
        assert mat[0, 0] == pytest.approx(600.0)
        assert mat[1, 1] == pytest.approx(700.0)
        assert mat[2, 2] == pytest.approx(1.0)

    def test_invalid_focal_length(self) -> None:
        with pytest.raises(ValueError, match="focal"):
            CameraIntrinsics(
                fx=0.0, fy=800.0, cx=960.0, cy=540.0, width=_IMG_W, height=_IMG_H
            )

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            CameraIntrinsics(
                fx=800.0, fy=800.0, cx=960.0, cy=540.0, width=0, height=_IMG_H
            )


# ---------------------------------------------------------------------------
# CameraExtrinsics
# ---------------------------------------------------------------------------


class TestCameraExtrinsics:
    def test_construction(self) -> None:
        ext = _extrinsics()
        assert ext.rotation.shape == (3, 3)
        assert ext.translation.shape == (3,)

    def test_invalid_rotation_shape(self) -> None:
        with pytest.raises(ValueError, match="rotation"):
            CameraExtrinsics(rotation=np.eye(4), translation=np.zeros(3))

    def test_invalid_translation_shape(self) -> None:
        with pytest.raises(ValueError, match="translation"):
            CameraExtrinsics(rotation=np.eye(3), translation=np.zeros(4))


# ---------------------------------------------------------------------------
# CameraCalibration
# ---------------------------------------------------------------------------


class TestCameraCalibration:
    def test_construction(self) -> None:
        cal = _calib("left")
        assert cal.camera_id == "left"

    def test_empty_camera_id(self) -> None:
        with pytest.raises(ValueError, match="camera_id"):
            CameraCalibration(
                camera_id="",
                intrinsics=_intrinsics(),
                extrinsics=_extrinsics(),
            )


# ---------------------------------------------------------------------------
# CanonicalObservations
# ---------------------------------------------------------------------------


class TestCanonicalObservations:
    def test_construction_2cam(self) -> None:
        obs = _make_obs(n_cams=2)
        assert obs.n_cameras == 2
        assert obs.n_keypoints == _N_KP
        assert obs.timestamp == pytest.approx(1.23)
        assert obs.keypoints_3d is None

    def test_construction_with_3d(self) -> None:
        obs = _make_obs(include_3d=True)
        assert obs.keypoints_3d is not None
        assert obs.keypoints_3d.shape == (_N_KP, 3)

    def test_mismatched_camera_and_keypoints(self) -> None:
        rng = np.random.default_rng(0)
        cameras = tuple(_calib(f"cam{i}") for i in range(2))
        kps = (rng.uniform(0, 1920, (_N_KP, 2)),)  # only 1, not 2
        confs = tuple(np.ones(_N_KP) for _ in range(2))
        with pytest.raises(ValueError, match="keypoints_2d"):
            CanonicalObservations(
                cameras=cameras,
                keypoints_2d=kps,
                confidences=confs,
                detector_layout=_LAYOUT,
                timestamp=0.0,
            )

    def test_confidence_out_of_range(self) -> None:
        rng = np.random.default_rng(0)
        cameras = (_calib(),)
        kps = (rng.uniform(0, 1920, (_N_KP, 2)),)
        bad_conf = (np.full(_N_KP, 1.5),)  # > 1 → invalid
        with pytest.raises(ValueError, match="confidence"):
            CanonicalObservations(
                cameras=cameras,
                keypoints_2d=kps,
                confidences=bad_conf,
                detector_layout=_LAYOUT,
                timestamp=0.0,
            )

    def test_confidence_negative(self) -> None:
        rng = np.random.default_rng(0)
        cameras = (_calib(),)
        kps = (rng.uniform(0, 1920, (_N_KP, 2)),)
        bad_conf = (np.full(_N_KP, -0.1),)
        with pytest.raises(ValueError, match="confidence"):
            CanonicalObservations(
                cameras=cameras,
                keypoints_2d=kps,
                confidences=bad_conf,
                detector_layout=_LAYOUT,
                timestamp=0.0,
            )

    def test_keypoints_wrong_shape(self) -> None:
        rng = np.random.default_rng(0)
        cameras = (_calib(),)
        # Wrong: (N_KP, 3) instead of (N_KP, 2)
        kps = (rng.uniform(0, 1920, (_N_KP, 3)),)
        confs = (np.ones(_N_KP),)
        with pytest.raises(ValueError, match="keypoints_2d"):
            CanonicalObservations(
                cameras=cameras,
                keypoints_2d=kps,
                confidences=confs,
                detector_layout=_LAYOUT,
                timestamp=0.0,
            )

    def test_3d_wrong_shape(self) -> None:
        rng = np.random.default_rng(0)
        cameras = (_calib(),)
        kps = (rng.uniform(0, 1920, (_N_KP, 2)),)
        confs = (np.ones(_N_KP),)
        with pytest.raises(ValueError, match="keypoints_3d"):
            CanonicalObservations(
                cameras=cameras,
                keypoints_2d=kps,
                confidences=confs,
                detector_layout=_LAYOUT,
                timestamp=0.0,
                keypoints_3d=np.ones((_N_KP, 2)),  # should be (N_KP, 3)
            )

    def test_empty_detector_layout(self) -> None:
        cameras = (_calib(),)
        with pytest.raises(ValueError, match="detector_layout"):
            CanonicalObservations(
                cameras=cameras,
                keypoints_2d=(np.zeros((0, 2)),),
                confidences=(np.zeros(0),),
                detector_layout=(),
                timestamp=0.0,
            )


# ---------------------------------------------------------------------------
# Round-trip serialisation
# ---------------------------------------------------------------------------


class TestObservationsRoundTrip:
    def test_round_trip_no_3d(self) -> None:
        obs = _make_obs(n_cams=2, include_3d=False)
        d = observations_to_dict(obs)
        obs2 = observations_from_dict(d)

        assert obs2.n_cameras == obs.n_cameras
        assert obs2.n_keypoints == obs.n_keypoints
        assert obs2.timestamp == pytest.approx(obs.timestamp)
        assert obs2.detector_layout == obs.detector_layout
        assert obs2.keypoints_3d is None

        for i in range(obs.n_cameras):
            assert np.allclose(obs2.keypoints_2d[i], obs.keypoints_2d[i])
            assert np.allclose(obs2.confidences[i], obs.confidences[i])
            assert obs2.cameras[i].camera_id == obs.cameras[i].camera_id

    def test_round_trip_with_3d(self) -> None:
        obs = _make_obs(n_cams=1, include_3d=True)
        d = observations_to_dict(obs)
        obs2 = observations_from_dict(d)
        assert obs2.keypoints_3d is not None
        assert np.allclose(obs2.keypoints_3d, obs.keypoints_3d)

    def test_confidence_preserved(self) -> None:
        """Confidence values must survive the round-trip exactly."""
        rng = np.random.default_rng(7)
        cameras = (_calib("c0"),)
        kps = (rng.uniform(0, 1920, (_N_KP, 2)),)
        confs = (rng.uniform(0.0, 1.0, _N_KP),)
        obs = CanonicalObservations(
            cameras=cameras,
            keypoints_2d=kps,
            confidences=confs,
            detector_layout=_LAYOUT,
            timestamp=0.5,
        )
        obs2 = observations_from_dict(observations_to_dict(obs))
        assert np.allclose(obs2.confidences[0], obs.confidences[0])

    def test_dict_is_json_serialisable(self) -> None:
        """observations_to_dict must return only JSON-native types."""
        import json

        obs = _make_obs(n_cams=1)
        d = observations_to_dict(obs)
        # Should not raise
        json.dumps(d)
