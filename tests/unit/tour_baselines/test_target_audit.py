"""Unit tests for Tour Baselines target audit and content verification (TB-01 #10586)."""

from pathlib import Path
import pytest

from src.shared.python.motion_matching.tour_capture_contract import (
    TOUR_CAPTURE,
    TOUR_CAPTURE_IRON,
    load_tour_capture,
    verify_capture_content,
)
from src.shared.python.tour_baselines.audit import audit_tour_target

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_C3D = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"
IRON_C3D = REPO_ROOT / "data" / "C3D_TA_Iron.c3d"


def test_verify_capture_content_by_hash_not_path(tmp_path: Path) -> None:
    # Copy driver file to an arbitrary different filename
    renamed = tmp_path / "some_arbitrary_unrelated_name.c3d"
    renamed.write_bytes(DRIVER_C3D.read_bytes())

    kind, spec = verify_capture_content(renamed)
    assert kind == "driver"
    assert spec.sha256 == TOUR_CAPTURE.sha256
    assert spec.rate_hz == 360.0


def test_verify_capture_content_mismatch_raises(tmp_path: Path) -> None:
    bad = tmp_path / "fake.c3d"
    bad.write_bytes(b"fake data not matching canonical tour capture sha")
    with pytest.raises(ValueError, match="not a canonical tour-average capture"):
        verify_capture_content(bad)


def test_verify_capture_content_expected_kind_mismatch() -> None:
    with pytest.raises(ValueError, match="Expected capture kind iron"):
        verify_capture_content(DRIVER_C3D, expected_kind="iron")


def test_audit_driver_target() -> None:
    cap = load_tour_capture(DRIVER_C3D)
    audit = audit_tour_target(cap, "driver")

    assert audit.source_sha256 == TOUR_CAPTURE.sha256
    assert audit.frames == 654
    assert audit.rate_hz == 360.0
    assert audit.duration_s == pytest.approx(653 / 360.0)
    assert audit.total_valid_samples == 24135
    assert audit.total_missing_samples == 24852 - 24135
    assert audit.handedness == "right"
    assert audit.vertical_axis == "y"
    assert audit.units == "m"

    # Specific missing spans
    r_shoulder = audit.markers["RShoulderTop"]
    assert r_shoulder.missing_samples == 526
    assert r_shoulder.missing_spans == ((0, 525),)

    grip = audit.markers["Marker_2:2:1"]
    assert grip.missing_samples == 36
    assert (518, 550) in grip.missing_spans


def test_audit_iron_target() -> None:
    cap = load_tour_capture(IRON_C3D)
    audit = audit_tour_target(cap, "iron")

    assert audit.source_sha256 == TOUR_CAPTURE_IRON.sha256
    assert audit.frames == 657
    assert audit.rate_hz == 359.0
    assert audit.total_valid_samples == 24219
    assert audit.total_missing_samples == 24966 - 24219
    assert audit.handedness == "right"

    # LShoulderTop has missing frames in Iron
    l_shoulder = audit.markers["LShoulderTop"]
    assert l_shoulder.missing_samples == 110
    assert l_shoulder.missing_spans == ((0, 109),)

    # Pelvis exists in Iron and has 8 missing samples
    pelvis = audit.markers["pelvis"]
    assert pelvis.missing_samples == 8
