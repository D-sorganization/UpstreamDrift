"""Tests for C3D capture copy hash-locking and marker validity policy (MS-04)."""

import hashlib
import json
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.tour_matching import marker_map
from src.shared.python.motion_matching import tour_capture_contract as contract
from src.shared.python.motion_matching.pipeline import reference

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_JSON = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "driver_capture_audit.json"
)

DRIVER_COPIES = (
    REPO_ROOT / "data" / "C3D_TA_Driver.c3d",
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "data"
    / "tour_average_mocap"
    / "C3DExport Tour average.c3d",
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "Data"
    / "Mocap C3D Files"
    / "C3DExport Tour average.c3d",
)

IRON_COPIES = (
    REPO_ROOT / "data" / "C3D_TA_Iron.c3d",
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "data"
    / "tour_average_mocap"
    / "C3DExport tour average iron.c3d",
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "Data"
    / "Mocap C3D Files"
    / "C3DExport tour average iron.c3d",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.unit
def test_capture_copies_are_identical() -> None:
    """Every copy of driver and iron C3D matches its canonical contract SHA-256."""
    for copy_path in DRIVER_COPIES:
        assert copy_path.is_file(), f"Driver copy missing: {copy_path}"
        assert _sha256(copy_path) == contract.TOUR_CAPTURE.sha256, (
            f"Hash mismatch for {copy_path}: expected {contract.TOUR_CAPTURE.sha256}"
        )

    for copy_path in IRON_COPIES:
        assert copy_path.is_file(), f"Iron copy missing: {copy_path}"
        assert _sha256(copy_path) == contract.TOUR_CAPTURE_IRON.sha256, (
            f"Hash mismatch for {copy_path}: expected {contract.TOUR_CAPTURE_IRON.sha256}"
        )


@pytest.mark.unit
def test_validity_policy_matches_audit() -> None:
    """Marker validity policy valid-sample counts equal driver_capture_audit.json."""
    assert AUDIT_JSON.is_file(), f"Audit file not found: {AUDIT_JSON}"
    audit_data = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    audit_markers = audit_data["markers"]

    policy = contract.MARKER_VALIDITY_POLICY
    assert len(policy) == len(contract.TOUR_CAPTURE.labels)

    for label in contract.TOUR_CAPTURE.labels:
        assert label in policy, f"Missing {label} in MARKER_VALIDITY_POLICY"
        entry = policy[label]
        audit_entry = audit_markers[label]
        assert entry.valid_samples == audit_entry["valid_samples"], (
            f"Mismatch for {label} valid_samples: {entry.valid_samples} != {audit_entry['valid_samples']}"
        )
        assert entry.missing_samples == audit_entry["missing_samples"], (
            f"Mismatch for {label} missing_samples: {entry.missing_samples} != {audit_entry['missing_samples']}"
        )


@pytest.mark.unit
def test_reference_uses_policy_weights() -> None:
    """RShoulderTop weight is 0 when invalid, 1 otherwise; unassigned labels 0."""
    assert reference.marker_weight("RShoulderTop", is_valid=False) == 0.0
    assert reference.marker_weight("RShoulderTop", is_valid=True) == 1.0

    # Normal tracked label
    assert reference.marker_weight("WaistLeft", is_valid=False) == 0.0
    assert reference.marker_weight("WaistLeft", is_valid=True) == 1.0

    # Unassigned label excluded by policy
    assert reference.marker_weight("Marker_0:0:0", is_valid=True) == 0.0
    assert reference.marker_weight("Marker_0:0:0", is_valid=False) == 0.0

    # OpenSim marker_map also consumes policy weights
    assert marker_map.marker_weight("RShoulderTop", is_valid=False) == 0.0
    assert marker_map.marker_weight("RShoulderTop", is_valid=True) == 1.0
    assert marker_map.marker_weight("Marker_0:0:0", is_valid=True) == 0.0
