"""Unit tests for MyoSuite native dual-club qualification and replay verification (MMR-10M, #11096)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.myosuite.python.native_qualification import (
    MYOSUITE_ENGINE_LIMITATIONS,
    MyoSuiteQualificationReceipt,
    MyoSuiteQualificationStatus,
    assess_myosuite_qualification,
    validate_myosuite_candidate_replay,
)

pytestmark = pytest.mark.unit

DRIVER_MODEL_HASH = "8e941be1c5a7d4b2a32a6fb7949ff527200eda069fee00a05664a81850fbc39b"
IRON_MODEL_HASH = "ddd93a125771f5712c1f8e62d676510b6df138290a58b20c2555817906f31086"
EVIDENCE_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "development"
    / "matched_swing_program"
    / "evidence"
    / "myosuite"
)


def _synthetic_candidate(
    club: str = "driver", model_hash: str = DRIVER_MODEL_HASH
) -> dict[str, Any]:
    return {
        "club": club,
        "source_sha256": "cand_hash_1234567890abcdef",
        "model_sha256": model_hash,
        "capture_sha256": "cap_hash_abcdef1234567890",
    }


def _synthetic_valid_replay() -> dict[str, Any]:
    t = np.linspace(0.0, 1.0, 51)
    q = np.zeros((51, 8))
    # free-joint root position
    q[:, 0] = 0.1 * np.sin(2 * np.pi * t)
    q[:, 1] = 0.2 * np.cos(2 * np.pi * t)
    q[:, 2] = 1.0
    # free-joint unit quaternion [w, x, y, z] = [1, 0, 0, 0]
    q[:, 3] = 1.0
    q[:, 4] = 0.0
    q[:, 5] = 0.0
    q[:, 6] = 0.0
    # joint coordinate
    q[:, 7] = np.sin(t)

    dt = float(t[1] - t[0])
    v = np.gradient(q, dt, axis=0)

    native_state = np.hstack([q, v])
    activations = 0.2 + 0.5 * np.sin(np.pi * t)
    markers = np.zeros((51, 10, 3))
    target = markers + 0.01

    return {
        "is_fresh_simulation": True,
        "actuation_applied": True,
        "time_s": t,
        "native_state": native_state,
        "activations": activations,
        "markers_m": markers,
        "target_m": target,
    }


def test_engine_specific_limitations_declared() -> None:
    assert len(MYOSUITE_ENGINE_LIMITATIONS) >= 5
    assert any(
        "excitation-activation" in s.lower() for s in MYOSUITE_ENGINE_LIMITATIONS
    )
    assert any("hill-type" in s.lower() for s in MYOSUITE_ENGINE_LIMITATIONS)
    assert any("quaternion" in s.lower() for s in MYOSUITE_ENGINE_LIMITATIONS)
    assert any("weld" in s.lower() for s in MYOSUITE_ENGINE_LIMITATIONS)
    assert any("hunt-crossley" in s.lower() for s in MYOSUITE_ENGINE_LIMITATIONS)


def test_committed_dual_club_evidence_receipts_are_fail_closed() -> None:
    """Committed club receipts must honestly record unavailable engines.

    No native MyoSuite execution exists for these clubs on any available host,
    so the only honest receipt is UNAVAILABLE with empty evidence fields,
    recorded missing evidence, and a resolvable remedy. Placeholder shas or
    invented metrics (previous content of these files) must never return.
    """
    driver_path = EVIDENCE_DIR / "driver_receipt.json"
    iron_path = EVIDENCE_DIR / "iron_receipt.json"

    assert driver_path.is_file(), f"Missing committed evidence: {driver_path}"
    assert iron_path.is_file(), f"Missing committed evidence: {iron_path}"

    driver_receipt = MyoSuiteQualificationReceipt.load(driver_path)
    assert driver_receipt.engine == "myosuite"
    assert driver_receipt.club == "driver"
    assert driver_receipt.status == MyoSuiteQualificationStatus.UNAVAILABLE
    assert driver_receipt.runtime_available is False
    assert driver_receipt.candidate_sha256 == ""
    assert driver_receipt.model_sha256 == ""
    assert driver_receipt.capture_sha256 == ""
    assert driver_receipt.is_fresh_simulation is False
    assert driver_receipt.derivatives_consistent is False
    assert driver_receipt.marker_metrics == {}
    assert driver_receipt.muscle_metrics == {}
    assert driver_receipt.missing_evidence
    assert any("marker" in ev for ev in driver_receipt.missing_evidence)
    assert "myosuite" in driver_receipt.remedy.lower()

    iron_receipt = MyoSuiteQualificationReceipt.load(iron_path)
    assert iron_receipt.engine == "myosuite"
    assert iron_receipt.club == "iron_7"
    assert iron_receipt.status == MyoSuiteQualificationStatus.UNAVAILABLE
    assert iron_receipt.runtime_available is False
    assert iron_receipt.candidate_sha256 == ""
    assert iron_receipt.missing_evidence
    assert iron_receipt.remedy


def test_missing_native_test_count_does_not_qualify() -> None:
    """An unrecorded native test count cannot be assumed nonzero (fail-closed)."""
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    receipt = validate_myosuite_candidate_replay(cand, replay)
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert any("native test count" in r.lower() for r in receipt.rejection_reasons)
    assert any("native test" in ev for ev in receipt.missing_evidence)
    assert receipt.remedy


def test_unavailable_runtime_fails_closed_even_with_replay_payload() -> None:
    """A replay payload cannot produce QUALIFIED while myosuite is unavailable."""
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    receipt = validate_myosuite_candidate_replay(cand, replay, myosuite_available=False)
    assert receipt.status == MyoSuiteQualificationStatus.UNAVAILABLE
    assert receipt.runtime_available is False
    assert receipt.missing_evidence
    assert receipt.remedy


def test_missing_marker_observations_do_not_qualify() -> None:
    """Replays without aligned marker observations must not qualify."""
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    replay.pop("markers_m")
    replay.pop("target_m")
    receipt = validate_myosuite_candidate_replay(
        cand, replay, native_tests_executed=10, myosuite_available=True
    )
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert receipt.marker_metrics == {}
    assert any("marker" in ev for ev in receipt.missing_evidence)


def test_missing_rollout_data_does_not_qualify() -> None:
    """Replays without native state/time data must not qualify."""
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    replay.pop("native_state")
    receipt = validate_myosuite_candidate_replay(
        cand, replay, native_tests_executed=10, myosuite_available=True
    )
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert receipt.derivatives_consistent is False
    assert receipt.energy_balance_checked is False
    assert "native_state/time_s dynamic rollout" in receipt.missing_evidence


def test_derivatives_mismatch_rejects() -> None:
    """Joint velocities inconsistent with dq/dt must not qualify."""
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    state = np.asarray(replay["native_state"])
    half = state.shape[1] // 2
    state[:, half:] = state[:, half:] + 1.0
    receipt = validate_myosuite_candidate_replay(
        cand, replay, native_tests_executed=10, myosuite_available=True
    )
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert receipt.derivatives_consistent is False


def test_marker_metrics_are_computed_not_invented() -> None:
    """Only metrics derived from the recorded observations may appear."""
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    receipt = validate_myosuite_candidate_replay(
        cand, replay, native_tests_executed=10, myosuite_available=True
    )
    assert set(receipt.marker_metrics) <= {"whole_rms_m"}
    assert "pelvis_yaw_error_pct" not in receipt.marker_metrics


def test_dual_club_support_driver_and_7iron() -> None:
    cand_driver = _synthetic_candidate("driver", DRIVER_MODEL_HASH)
    replay_driver = _synthetic_valid_replay()
    receipt_driver = validate_myosuite_candidate_replay(
        cand_driver, replay_driver, native_tests_executed=10
    )
    assert receipt_driver.status == MyoSuiteQualificationStatus.QUALIFIED
    assert receipt_driver.club == "driver"

    cand_iron = _synthetic_candidate("iron_7", IRON_MODEL_HASH)
    replay_iron = _synthetic_valid_replay()
    receipt_iron = validate_myosuite_candidate_replay(
        cand_iron, replay_iron, native_tests_executed=10
    )
    assert receipt_iron.status == MyoSuiteQualificationStatus.QUALIFIED
    assert receipt_iron.club == "iron_7"


def test_reject_copied_state_trajectory() -> None:
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    replay["copied_from_reference"] = True
    replay["is_fresh_simulation"] = False

    receipt = validate_myosuite_candidate_replay(cand, replay)
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert any(
        "Copied state trajectory detected" in r for r in receipt.rejection_reasons
    )


def test_reject_fk_only_playback_without_dynamic_actuation() -> None:
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    replay["actuation_applied"] = False

    receipt = validate_myosuite_candidate_replay(cand, replay)
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert any("FK-only playback detected" in r for r in receipt.rejection_reasons)


def test_reject_zero_collected_native_tests() -> None:
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()

    receipt = validate_myosuite_candidate_replay(cand, replay, native_tests_executed=0)
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert any("Zero collected native tests" in r for r in receipt.rejection_reasons)


def test_muscle_physiological_limit_violation_rejects() -> None:
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    replay["activations"] = np.array([0.1, 0.5, 1.45, 0.8])  # exceeds 1.0

    receipt = validate_myosuite_candidate_replay(cand, replay)
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert any(
        "Muscle activation exceeds physiological bounds" in r
        for r in receipt.rejection_reasons
    )


def test_free_joint_quaternion_normalization_rejects() -> None:
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    state = replay["native_state"].copy()
    state[:, 3] = 2.0  # norm = 2.0, not 1.0
    replay["native_state"] = state

    receipt = validate_myosuite_candidate_replay(cand, replay)
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert any("quaternion is not normalized" in r for r in receipt.rejection_reasons)


def test_non_finite_values_rejects() -> None:
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    state = replay["native_state"].copy()
    state[10, 0] = np.nan
    replay["native_state"] = state

    receipt = validate_myosuite_candidate_replay(cand, replay)
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert any("Non-finite values" in r for r in receipt.rejection_reasons)


def test_stale_or_mismatched_model_hash_rejects() -> None:
    cand = _synthetic_candidate(model_hash="wrong_hash")
    replay = _synthetic_valid_replay()

    receipt = validate_myosuite_candidate_replay(
        cand, replay, expected_model_sha=DRIVER_MODEL_HASH
    )
    assert receipt.status == MyoSuiteQualificationStatus.REJECTED
    assert any("model_sha256 mismatch" in r for r in receipt.rejection_reasons)


def test_derivative_and_energy_balance_checks() -> None:
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()

    receipt = validate_myosuite_candidate_replay(cand, replay, native_tests_executed=10)
    assert receipt.status == MyoSuiteQualificationStatus.QUALIFIED
    assert receipt.derivatives_consistent is True
    assert receipt.energy_balance_checked is True
    assert "kinetic_energy_j" in receipt.energy_summary


def test_missing_myosuite_runtime_reports_unavailable() -> None:
    cand = _synthetic_candidate()
    receipt = assess_myosuite_qualification(cand, myosuite_available=False)
    assert receipt.status == MyoSuiteQualificationStatus.UNAVAILABLE
    assert receipt.runtime_available is False
    assert "live simulation unavailable" in receipt.diagnostic_message


def test_receipt_serialization_round_trip(tmp_path: Path) -> None:
    cand = _synthetic_candidate()
    replay = _synthetic_valid_replay()
    receipt = validate_myosuite_candidate_replay(cand, replay, native_tests_executed=10)

    out_file = tmp_path / "receipt.json"
    receipt.save(out_file)

    loaded = MyoSuiteQualificationReceipt.load(out_file)
    assert loaded == receipt
    assert loaded.status == MyoSuiteQualificationStatus.QUALIFIED
    assert loaded.as_dict() == receipt.as_dict()
