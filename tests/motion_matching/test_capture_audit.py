"""Audit actual C3D validity through the canonical ingestion adapter."""

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.capture_audit import audit_capture
from src.shared.python.motion_matching import capture_audit

pytestmark = pytest.mark.unit


def test_audit_preserves_occlusion_and_source_units(tmp_path: Path) -> None:
    ezc3d = pytest.importorskip("ezc3d")
    capture = ezc3d.c3d()
    capture["parameters"]["POINT"]["RATE"]["value"] = [100.0]
    capture["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
    capture["parameters"]["POINT"]["LABELS"]["value"] = ["One", "Two"]
    points = np.ones((4, 2, 4)) * 100
    capture["data"]["points"] = points
    residuals = np.zeros((1, 2, 4))
    residuals[0, 1, 2] = -1
    capture["data"]["meta_points"]["residuals"] = residuals
    path = tmp_path / "target.c3d"
    capture.write(str(path))
    report = audit_capture(path)
    assert report["frame_count"] == 4
    assert report["duration_s"] == pytest.approx(0.03)
    assert report["source_units"] == "mm"
    assert report["markers"]["Two"]["valid_samples"] == 3
    assert report["markers"]["One"]["first_position_m"] == pytest.approx([0.1] * 3)
    assert len(report["source_sha256"]) == 64
    assert report["qualification"] == "capture-audit-only"


def test_audit_rejects_non_c3d(tmp_path: Path) -> None:
    path = tmp_path / "fake.txt"
    path.write_text("not a capture")
    with pytest.raises(ValueError, match="C3D"):
        audit_capture(path)


def test_cli_preserves_existing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit.json"
    monkeypatch.setattr("sys.argv", ["audit", "input.c3d", str(output)])
    monkeypatch.setattr(capture_audit, "audit_capture", lambda path: {"checked": True})
    capture_audit.main()
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        capture_audit.main()
    assert output.read_bytes() == original
