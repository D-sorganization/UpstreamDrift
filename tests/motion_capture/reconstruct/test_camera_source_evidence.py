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
