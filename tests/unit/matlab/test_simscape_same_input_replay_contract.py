"""Contract tests for MATLAB R2025b Simscape same-input replay (#11613, #11605).

Verifies that scripts/matlab/run_simscape_same_input_replay.m strictly enforces
MATLAB R2025b, consumes same-input-bundle/v1 torque bundles, evaluates L0/L1/L2
parity metrics, and emits the simscape-same-input-replay/v1 receipt schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
REPLAY_SCRIPT = REPO_ROOT / "scripts/matlab/run_simscape_same_input_replay.m"

SCHEMA = "simscape-same-input-replay/v1"
BUNDLE_SCHEMA = "same-input-bundle/v1"
CANONICAL_DRIVER_SHA256 = (
    "d7aeb9a886c6e2cf414e99af9d0e33cf62d6060271b1a9d23c36678cacfeb678"
)
CANONICAL_IRON_SHA256 = (
    "9ef8e23bb4eb4df13e7f3f96030cd9a7301760f429d70b369e9cb3c2e751e576"
)


def _load_matlab_script() -> str:
    """Read scripts/matlab/run_simscape_same_input_replay.m text."""
    assert REPLAY_SCRIPT.is_file(), f"Expected MATLAB script at {REPLAY_SCRIPT}"
    return REPLAY_SCRIPT.read_text(encoding="utf-8")


def _make_valid_receipt() -> dict[str, Any]:
    """Return a valid synthetic Simscape same-input replay receipt."""
    return {
        "schema": SCHEMA,
        "bundle_schema": BUNDLE_SCHEMA,
        "issue": 11613,
        "epic": 11605,
        "host": "DeskComputer",
        "matlab_release": "2025b",
        "matlab_version": "25.2.0.0000000 (R2025b)",
        "effort_hold_rate_hz": 1000.0,
        "segment_duration_ms": 50.0,
        "clubs": {
            "driver": {
                "club": "driver",
                "bundle_sha256": CANONICAL_DRIVER_SHA256,
                "bundle_path": "evidence/driver.npz",
                "urdf_path": "models/golfer.urdf",
                "urdf_sha256": "0" * 64,
                "effort_hold_rate_hz": 1000.0,
                "actuation_mode": "joint_torque_zoh",
                "l0": {
                    "smimport_ok": True,
                    "n_coordinates": 44,
                    "total_mass_kg": 79.358,
                    "mass_tolerance_rel": 1e-9,
                    "passed": True,
                },
                "l1": {
                    "reference_engine": "pinocchio_rnea",
                    "tolerance_nm": 1e-6,
                    "max_torque_difference_nm": 3.8e-8,
                    "rms_torque_difference_nm": 5.2e-9,
                    "passed": True,
                },
                "l2": {
                    "segment_duration_ms": 50.0,
                    "coordinate_bound_rad": 1e-6,
                    "frame_bound_m": 1e-3,
                    "segmented": {
                        "worst_coordinate_error_rad": 8.5e-11,
                        "worst_frame_error_m": 2.1e-11,
                        "passed": True,
                    },
                    "open_loop": {
                        "agreement_horizon_s": 0.58,
                        "growth_rate_per_s": 39.4,
                        "full_horizon_duration_s": 1.81,
                        "divergence_termination": "graceful_exit_within_horizon",
                    },
                    "passed": True,
                },
            },
            "iron7": {
                "club": "iron7",
                "bundle_sha256": CANONICAL_IRON_SHA256,
                "bundle_path": "evidence/iron.npz",
                "urdf_path": "models/golfer.urdf",
                "urdf_sha256": "0" * 64,
                "effort_hold_rate_hz": 1000.0,
                "actuation_mode": "joint_torque_zoh",
                "l0": {
                    "smimport_ok": True,
                    "n_coordinates": 44,
                    "total_mass_kg": 79.358,
                    "mass_tolerance_rel": 1e-9,
                    "passed": True,
                },
                "l1": {
                    "reference_engine": "pinocchio_rnea",
                    "tolerance_nm": 1e-6,
                    "max_torque_difference_nm": 4.1e-8,
                    "rms_torque_difference_nm": 6.0e-9,
                    "passed": True,
                },
                "l2": {
                    "segment_duration_ms": 50.0,
                    "coordinate_bound_rad": 1e-6,
                    "frame_bound_m": 1e-3,
                    "segmented": {
                        "worst_coordinate_error_rad": 1.2e-9,
                        "worst_frame_error_m": 3.4e-10,
                        "passed": True,
                    },
                    "open_loop": {
                        "agreement_horizon_s": 0.52,
                        "growth_rate_per_s": 40.2,
                        "full_horizon_duration_s": 1.83,
                        "divergence_termination": "graceful_exit_within_horizon",
                    },
                    "passed": True,
                },
            },
        },
        "parity_summary": {
            "l0_passed": True,
            "l1_passed": True,
            "l2_passed": True,
        },
    }


def validate_receipt_schema(receipt: dict[str, Any]) -> None:
    """Validate that receipt adheres strictly to simscape-same-input-replay/v1 contract."""
    assert receipt["schema"] == SCHEMA, f"Invalid schema: {receipt.get('schema')}"
    assert receipt["bundle_schema"] == BUNDLE_SCHEMA
    assert receipt["issue"] == 11613
    assert receipt["epic"] == 11605

    # Strict MATLAB release check - fails closed if not 2025b!
    release = receipt.get("matlab_release")
    if release != "2025b":
        raise ValueError(
            f"Prohibited MATLAB release {release!r}: only '2025b' is accepted (SIP P-8, #11613)"
        )

    assert (
        "clubs" in receipt
        and "driver" in receipt["clubs"]
        and "iron7" in receipt["clubs"]
    )
    for club_key in ("driver", "iron7"):
        club_data = receipt["clubs"][club_key]
        assert club_data["club"] == club_key
        assert len(club_data["bundle_sha256"]) == 64
        assert club_data["actuation_mode"] == "joint_torque_zoh"

        # L0 validation
        l0 = club_data["l0"]
        assert l0["n_coordinates"] > 0
        assert l0["total_mass_kg"] > 0.0
        assert isinstance(l0["passed"], bool)

        # L1 validation
        l1 = club_data["l1"]
        assert l1["reference_engine"] == "pinocchio_rnea"
        assert l1["max_torque_difference_nm"] <= l1["tolerance_nm"]
        assert l1["passed"] is True

        # L2 validation
        l2 = club_data["l2"]
        assert l2["segment_duration_ms"] == 50.0
        assert (
            l2["segmented"]["worst_coordinate_error_rad"] <= l2["coordinate_bound_rad"]
        )
        assert l2["segmented"]["worst_frame_error_m"] <= l2["frame_bound_m"]
        assert l2["open_loop"]["agreement_horizon_s"] >= 0.50
        assert 35.0 <= l2["open_loop"]["growth_rate_per_s"] <= 45.0
        assert l2["passed"] is True

    summary = receipt["parity_summary"]
    assert summary["l0_passed"] is True
    assert summary["l1_passed"] is True
    assert summary["l2_passed"] is True


def test_script_exists_and_declares_r2025b_entry_point() -> None:
    """The MATLAB entry point must exist and have the documented function signature."""
    script_text = _load_matlab_script()
    assert "function receipt = run_simscape_same_input_replay(" in script_text
    assert "SIP P-8, #11613" in script_text


def test_script_enforces_strict_r2025b_release_and_fails_closed_on_r2026a() -> None:
    """The MATLAB harness must enforce MATLAB R2025b and explicitly prohibit R2026a."""
    script_text = _load_matlab_script()

    # Must query release and assert '2025b'
    assert "version('-release')" in script_text
    assert "strcmp(rel, '2025b')" in script_text

    # Must explicitly mention R2026a prohibition / error identifier
    assert "R2026a" in script_text
    assert "R2025bRequired" in script_text


def test_script_consumes_same_input_bundle_v1_and_smimport() -> None:
    """Harness must consume same-input-bundle/v1 for driver and iron and import URDF."""
    script_text = _load_matlab_script()

    # Consumes same-input-bundle/v1
    assert "same-input-bundle/v1" in script_text
    assert "smimport" in script_text
    assert "golfer.urdf" in script_text

    # Canonical bundle hashes
    assert CANONICAL_DRIVER_SHA256 in script_text
    assert CANONICAL_IRON_SHA256 in script_text

    # Joint torque actuation with ZOH from (q0, v0)
    assert "effort_hold_rate_hz" in script_text
    assert "joint_torque_zoh" in script_text


def test_script_evaluates_l0_l1_l2_parity_levels() -> None:
    """Harness must evaluate L0 (topology), L1 (RNEA inverse dynamics), and L2 (restarts/horizon)."""
    script_text = _load_matlab_script()

    # L0 mass and coordinate inventory
    assert "simscape_model_inventory" in script_text
    assert "n_coordinates" in script_text
    assert "total_mass_kg" in script_text

    # L1 Pinocchio RNEA comparison
    assert "pinocchio_rnea" in script_text
    assert "max_torque_difference_nm" in script_text

    # L2 50 ms windowed restarts and open-loop divergence
    assert "segment_duration_ms" in script_text
    assert "worst_coordinate_error_rad" in script_text
    assert "agreement_horizon_s" in script_text
    assert "growth_rate_per_s" in script_text


def test_receipt_schema_validation_passes_on_valid_payload(tmp_path: Path) -> None:
    """A conforming receipt dictionary passes strict validation and round-trips to JSON."""
    receipt = _make_valid_receipt()
    validate_receipt_schema(receipt)

    json_path = tmp_path / "simscape_same_input_replay_receipt.json"
    json_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    validate_receipt_schema(loaded)


def test_receipt_validation_fails_closed_on_non_2025b_release() -> None:
    """Receipt validation must fail closed if recorded release is R2026a or any non-2025b."""
    receipt = _make_valid_receipt()
    receipt["matlab_release"] = "2026a"
    receipt["matlab_version"] = "26.1.0.0000000 (R2026a)"

    with pytest.raises(ValueError, match="Prohibited MATLAB release '2026a'"):
        validate_receipt_schema(receipt)

    receipt["matlab_release"] = "2024b"
    with pytest.raises(ValueError, match="Prohibited MATLAB release '2024b'"):
        validate_receipt_schema(receipt)


def test_receipt_validation_fails_on_schema_or_issue_mismatch() -> None:
    """Receipt validation must reject unknown schema versions or wrong issue references."""
    receipt = _make_valid_receipt()
    receipt["schema"] = "simscape-same-input-replay/v2"
    with pytest.raises(AssertionError):
        validate_receipt_schema(receipt)

    receipt = _make_valid_receipt()
    receipt["issue"] = 99999
    with pytest.raises(AssertionError):
        validate_receipt_schema(receipt)
