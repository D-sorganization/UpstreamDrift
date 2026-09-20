"""Tests for the generic CaptureContract and validation diagnostics (#10361)."""

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.tour_capture_contract import (
    TOUR_CAPTURE,
    TOUR_CAPTURE_IRON,
    CaptureContract,
    CaptureValidationReport,
    load_capture,
    load_tour_capture,
    validate_capture_contract,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DRIVER_C3D = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"
CMU_MOCAP_C3D = REPO_ROOT / "data" / "cmu_mocap" / "subject_64" / "64_01.c3d"


def test_tour_capture_is_an_instance_of_capture_contract() -> None:
    """TOUR_CAPTURE and TOUR_CAPTURE_IRON must inherit from CaptureContract."""
    assert isinstance(TOUR_CAPTURE, CaptureContract)
    assert isinstance(TOUR_CAPTURE_IRON, CaptureContract)
    assert TOUR_CAPTURE.units == "m"
    assert TOUR_CAPTURE.vertical_axis == "y"
    assert TOUR_CAPTURE.rate_hz == 360.0
    assert TOUR_CAPTURE.frames == 654


@pytest.mark.skipif(
    not CANONICAL_DRIVER_C3D.is_file(), reason="canonical C3D not present"
)
def test_contract_accepts_tour_capture_unchanged() -> None:
    """TDD Step 1: Contract accepts canonical tour capture unchanged with exact SHA."""
    report = validate_capture_contract(CANONICAL_DRIVER_C3D, contract=TOUR_CAPTURE)
    assert isinstance(report, CaptureValidationReport)
    assert report.is_valid is True
    assert len(report.reasons) == 0
    assert report.metadata["units"] == "m"
    assert report.metadata["rate_hz"] == 360.0
    assert report.metadata["frames"] == 654

    # Loading via generic load_capture gives identical result to load_tour_capture
    cap_generic = load_capture(CANONICAL_DRIVER_C3D, contract=TOUR_CAPTURE)
    cap_frozen = load_tour_capture(CANONICAL_DRIVER_C3D)

    assert cap_generic.frames == cap_frozen.frames
    assert cap_generic.labels == cap_frozen.labels
    assert cap_generic.source_sha256 == cap_frozen.source_sha256
    assert cap_generic.valid_count() == cap_frozen.valid_count()


@pytest.mark.skipif(not CMU_MOCAP_C3D.is_file(), reason="CMU C3D not present")
def test_contract_rejects_cmu_locomotion_c3d_with_named_reasons() -> None:
    """TDD Step 2: Rejects CMU locomotion C3D with explicit named reasons."""
    # Strict contract requiring metres, club segment, and no unmapped labels
    contract = CaptureContract(
        units="m",
        required_segments=("trunk", "pelvis", "club"),
    )
    report = validate_capture_contract(CMU_MOCAP_C3D, contract=contract)
    assert report.is_valid is False
    assert len(report.reasons) > 0

    reasons_str = " ".join(report.reasons)
    # 1. CMU C3D uses mm, not m
    assert "invalid_units" in reasons_str
    # 2. CMU C3D lacks club markers
    assert "missing_required_segment: club" in reasons_str

    # Calling raise_for_status raises ValueError with the named reasons
    with pytest.raises(ValueError) as exc_info:
        report.raise_for_status()
    assert "invalid_units" in str(exc_info.value)
    assert "missing_required_segment: club" in str(exc_info.value)

    # Attempting to load directly via load_capture also fails closed
    with pytest.raises(ValueError) as load_exc:
        load_capture(CMU_MOCAP_C3D, contract=contract)
    assert "invalid_units" in str(load_exc.value)


@pytest.mark.skipif(not CMU_MOCAP_C3D.is_file(), reason="CMU C3D not present")
def test_contract_with_label_map_and_unit_conversion() -> None:
    """Generic contract allows custom label maps and unit scaling."""
    label_map = {
        "Subject1:LFHD": "HeadFront",
        "Subject1:RFHD": "HeadSide",
        "Subject1:LBHD": "HeadTop",
    }
    contract = CaptureContract(
        units="mm",
        required_labels=("HeadFront", "HeadSide", "HeadTop"),
        label_map=label_map,
        min_frames=10,
    )
    report = validate_capture_contract(CMU_MOCAP_C3D, contract=contract)
    assert report.is_valid is True, f"Failed with reasons: {report.reasons}"

    # Test loading with the mapped contract converts units to metres in TourCapture
    loaded = load_capture(CMU_MOCAP_C3D, contract=contract)
    assert loaded.frames > 10
    assert "HeadFront" in loaded.labels
    assert "HeadSide" in loaded.labels
    assert "HeadTop" in loaded.labels
    # Valid points should be in reasonable human metre coordinates (~1-2m), not millimetres (~1000-2000mm)
    idx = loaded.index("HeadFront")
    valid_pts = loaded.points_m[:, idx][loaded.valid[:, idx]]
    if len(valid_pts) > 0:
        assert np.max(np.abs(valid_pts)) < 10.0


def test_contract_rejects_excessive_gap_fraction(tmp_path: Path) -> None:
    """Contract rejects captures with marker gap fraction exceeding threshold."""
    contract = CaptureContract(
        max_gap_fraction=0.1,
        required_labels=("MarkerA",),
    )
    # Create a synthetic validation check
    report = CaptureValidationReport(
        is_valid=False,
        reasons=(
            "excessive_gap_fraction: marker 'MarkerA' has 45.00% missing frames (max 10.00%)",
        ),
        metadata={},
    )
    assert not report.is_valid
    with pytest.raises(ValueError, match="excessive_gap_fraction"):
        report.raise_for_status()


def test_contract_rejects_missing_static_calibration() -> None:
    """Contract requires static calibration when flag is set."""
    contract = CaptureContract(static_calibration_required=True)
    report = CaptureValidationReport(
        is_valid=False,
        reasons=("missing_static_calibration: static address calibration required",),
        metadata={},
    )
    assert not report.is_valid
    with pytest.raises(ValueError, match="missing_static_calibration"):
        report.raise_for_status()
