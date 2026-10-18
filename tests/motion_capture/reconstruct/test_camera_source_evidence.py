"""Camera-source bytes, rather than a filename, identify a calibration revision."""

from pathlib import Path

import pytest

from src.motion_capture.reconstruct.camera_source import CameraSourceEvidence

pytestmark = pytest.mark.unit


def test_camera_evidence_detects_same_path_changes(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text('{"cameras": []}', encoding="utf-8")
    evidence = CameraSourceEvidence.capture(path)
    evidence.verify()
    path.write_text('{"cameras": [1]}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        evidence.verify()


def test_camera_evidence_detects_removed_source(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{}", encoding="utf-8")
    evidence = CameraSourceEvidence.capture(path)
    path.unlink()
    with pytest.raises(ValueError, match="unavailable"):
        evidence.verify()


def test_validate_camera_baselines() -> None:
    import numpy as np

    from src.motion_capture.reconstruct import PinholeCamera, look_at
    from src.motion_capture.reconstruct.cameras import intrinsics_from_fov
    from src.motion_capture.reconstruct.camera_source import validate_camera_baselines

    k = intrinsics_from_fov(1920, 1200, 70.0)
    target = np.array([0.0, 1.0, 0.0])
    pos_a = np.array([0.0, 1.2, 4.0])
    pos_b = np.array([-4.0, 1.2, 0.0])
    cam_a = PinholeCamera("a", k, look_at(pos_a, target), pos_a, (1920, 1200))
    cam_b = PinholeCamera("b", k, look_at(pos_b, target), pos_b, (1920, 1200))

    # Distinct camera poses pass cleanly
    validate_camera_baselines([cam_a, cam_b])

    # Coincident camera poses raise ValueError
    cam_c = PinholeCamera("c", k, look_at(pos_a, target), pos_a, (1920, 1200))
    with pytest.raises(ValueError, match="degenerate baseline"):
        validate_camera_baselines([cam_a, cam_c])

    # Contract checks
    with pytest.raises(Exception, match="at least two cameras"):
        validate_camera_baselines([cam_a])
    with pytest.raises(Exception, match="positive"):
        validate_camera_baselines([cam_a, cam_b], min_baseline_m=0.0)

