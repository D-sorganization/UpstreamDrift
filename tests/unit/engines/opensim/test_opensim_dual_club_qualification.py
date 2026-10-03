"""Unit tests for OpenSim native dual-club dynamics and replay qualification (MMR-10O #11095).

Tests:
1. Rejection of copied state trajectories (fresh simulation required).
2. Rejection of FK-only playback without dynamic actuation.
3. Rejection of zero collected native tests on pinned host.
4. Detection of absent host opensim runtime returning UNAVAILABLE.
5. Dual-club validation for Driver and 7-Iron.
6. Rejection of stale/mismatched model hash.
7. Independent derivative consistency and energy/work balance verification.
8. Muscle activation dynamics and physiological limits checks.
9. Disclosure of OpenSim-specific engine limitations.
10. JSON serialization round-trip.
11. Verification of committed dual-club evidence receipts.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.native_qualification import (
    OPENSIM_ENGINE_LIMITATIONS,
    OpenSimQualificationReceipt,
    OpenSimQualificationStatus,
    assess_opensim_qualification,
    validate_opensim_candidate_replay,
)

pytestmark = pytest.mark.unit

EVIDENCE_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "development"
    / "matched_swing_program"
    / "evidence"
    / "opensim"
)


def _make_dummy_candidate(
    club: str = "driver",
    model_sha: str = "6fa980a3ccda49b1051c55fb025da596c9a7b66e1687a4b3254df19d49536a21",
) -> dict:
    return {
        "club": club,
        "source_sha256": "abcdef1234567890" * 4,
        "model_sha256": model_sha,
        "capture_sha256": "1234567890abcdef" * 4,
    }


def _make_dummy_replay(
    n_steps: int = 50,
    n_coords: int = 15,
    is_fresh: bool = True,
    actuation_applied: bool = True,
    activation_violation: bool = False,
) -> dict:
    t = np.linspace(0.0, 1.0, n_steps)
    dt = t[1] - t[0]
    q = np.sin(2 * np.pi * t[:, None] * np.linspace(0.5, 1.5, n_coords))
    v = np.gradient(q, dt, axis=0)
    native_state = np.hstack([q, v])

    # Synthesize marker positions
    target = np.zeros((n_steps, 10, 3))
    markers = target + 0.005 * np.cos(t[:, None, None])

    activations = np.full((n_steps, 8), 0.45)
    if activation_violation:
        activations[10, 2] = 1.85  # Exceeds physiological 1.0 bound!

    return {
        "time_s": t,
        "native_state": native_state,
        "is_fresh_simulation": is_fresh,
        "copied_from_reference": not is_fresh,
        "actuation_applied": actuation_applied,
        "markers_m": markers,
        "target_m": target,
        "activations": activations,
    }


def test_reject_copied_state_trajectory() -> None:
    """Copied state trajectories must fail qualification."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay(is_fresh=False)
    receipt = validate_opensim_candidate_replay(cand, replay)
    assert receipt.status == OpenSimQualificationStatus.REJECTED
    assert any("copied" in r.lower() for r in receipt.rejection_reasons)


def test_reject_fk_only_playback_without_dynamic_actuation() -> None:
    """Kinematic-only playback without dynamic actuation must fail qualification."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay(actuation_applied=False)
    receipt = validate_opensim_candidate_replay(cand, replay)
    assert receipt.status == OpenSimQualificationStatus.REJECTED
    assert any(
        "fk-only" in r.lower() or "dynamic" in r.lower()
        for r in receipt.rejection_reasons
    )


def test_reject_zero_collected_native_tests() -> None:
    """Execution reporting zero native tests on host must be rejected."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    receipt = validate_opensim_candidate_replay(cand, replay, native_tests_executed=0)
    assert receipt.status == OpenSimQualificationStatus.REJECTED
    assert any("zero" in r.lower() for r in receipt.rejection_reasons)


def test_missing_opensim_runtime_reports_unavailable() -> None:
    """When opensim module is absent, qualification reports UNAVAILABLE, never green."""
    cand = _make_dummy_candidate(club="driver")
    receipt = assess_opensim_qualification(cand, replay=None, opensim_available=False)
    assert receipt.status == OpenSimQualificationStatus.UNAVAILABLE
    assert receipt.runtime_available is False
    assert "opensim" in receipt.diagnostic_message.lower()


def test_dual_club_support_driver_and_7iron() -> None:
    """Both driver and 7-iron clubs must be supported with distinct receipts."""
    driver_cand = _make_dummy_candidate(club="driver")
    iron_cand = _make_dummy_candidate(
        club="7-iron",
        model_sha="3e7be1486821578efd6474916f21a25e794c353bf698b61b5cac28d725778688",
    )
    driver_replay = _make_dummy_replay()
    iron_replay = _make_dummy_replay()

    r_driver = validate_opensim_candidate_replay(
        driver_cand, driver_replay, native_tests_executed=10, opensim_available=True
    )
    r_iron = validate_opensim_candidate_replay(
        iron_cand, iron_replay, native_tests_executed=10, opensim_available=True
    )

    assert r_driver.club == "driver"
    assert r_iron.club == "7-iron"
    assert r_driver.status == OpenSimQualificationStatus.QUALIFIED
    assert r_iron.status == OpenSimQualificationStatus.QUALIFIED


def test_stale_or_mismatched_model_hash_rejects() -> None:
    """Mismatched model hash must be rejected."""
    cand = _make_dummy_candidate(model_sha="bad_hash" * 8)
    replay = _make_dummy_replay()
    receipt = validate_opensim_candidate_replay(
        cand,
        replay,
        expected_model_sha="6fa980a3ccda49b1051c55fb025da596c9a7b66e1687a4b3254df19d49536a21",
    )
    assert receipt.status == OpenSimQualificationStatus.REJECTED
    assert any("model_sha256" in r for r in receipt.rejection_reasons)


def test_derivative_and_energy_balance_checks() -> None:
    """Receipt verifies independent derivative consistency and energy accounting."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    receipt = validate_opensim_candidate_replay(
        cand, replay, native_tests_executed=10, opensim_available=True
    )
    assert receipt.derivatives_consistent is True
    assert receipt.energy_balance_checked is True
    assert "kinetic_energy_j" in receipt.energy_summary
    assert "potential_energy_j" in receipt.energy_summary


def test_muscle_physiological_limit_violation_rejects() -> None:
    """Physiological activation bound violations (> 1.0) must be rejected."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay(activation_violation=True)
    receipt = validate_opensim_candidate_replay(cand, replay)
    assert receipt.status == OpenSimQualificationStatus.REJECTED
    assert any(
        "activation" in r.lower() or "muscle" in r.lower()
        for r in receipt.rejection_reasons
    )


def test_engine_specific_limitations_declared() -> None:
    """Engine-specific musculoskeletal limitations must be disclosed."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    receipt = validate_opensim_candidate_replay(
        cand, replay, native_tests_executed=10, opensim_available=True
    )
    assert len(receipt.declared_limitations) >= 3
    assert any(
        "hill" in lim.lower() or "activation" in lim.lower()
        for lim in receipt.declared_limitations
    )
    assert any("tendon" in lim.lower() for lim in receipt.declared_limitations)


def test_receipt_serialization_round_trip(tmp_path: Path) -> None:
    """Receipt can be saved to JSON and loaded back preserving all fields."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    receipt = validate_opensim_candidate_replay(
        cand, replay, native_tests_executed=10, opensim_available=True
    )

    out_file = tmp_path / "test_opensim_receipt.json"
    receipt.save(out_file)
    assert out_file.exists()

    loaded = OpenSimQualificationReceipt.load(out_file)
    assert loaded.schema_version == receipt.schema_version
    assert loaded.engine == receipt.engine
    assert loaded.club == receipt.club
    assert loaded.status == receipt.status
    assert loaded.candidate_sha256 == receipt.candidate_sha256


def test_committed_dual_club_evidence_receipts_are_fail_closed() -> None:
    """Committed club receipts must honestly record unavailable engines.

    No native OpenSim execution exists for these clubs on any available host,
    so the only honest receipt is UNAVAILABLE with empty evidence fields,
    recorded missing evidence, and a resolvable remedy. Placeholder shas or
    invented metrics (previous content of these files) must never return.
    """
    driver_file = EVIDENCE_DIR / "driver_receipt.json"
    iron_file = EVIDENCE_DIR / "iron_receipt.json"

    assert driver_file.is_file(), f"Missing driver receipt: {driver_file}"
    assert iron_file.is_file(), f"Missing 7-iron receipt: {iron_file}"

    driver_rcpt = OpenSimQualificationReceipt.load(driver_file)
    assert driver_rcpt.status == OpenSimQualificationStatus.UNAVAILABLE
    assert driver_rcpt.club == "driver"
    assert driver_rcpt.runtime_available is False
    assert driver_rcpt.candidate_sha256 == ""
    assert driver_rcpt.model_sha256 == ""
    assert driver_rcpt.capture_sha256 == ""
    assert driver_rcpt.marker_metrics == {}
    assert driver_rcpt.muscle_metrics == {}
    assert driver_rcpt.missing_evidence
    assert any("marker" in ev for ev in driver_rcpt.missing_evidence)
    assert "opensim" in driver_rcpt.remedy.lower()

    iron_rcpt = OpenSimQualificationReceipt.load(iron_file)
    assert iron_rcpt.status == OpenSimQualificationStatus.UNAVAILABLE
    assert iron_rcpt.club == "7-iron"
    assert iron_rcpt.runtime_available is False
    assert iron_rcpt.candidate_sha256 == ""
    assert iron_rcpt.missing_evidence
    assert iron_rcpt.remedy


def test_missing_native_test_count_does_not_qualify() -> None:
    """An unrecorded native test count cannot be assumed nonzero (fail-closed)."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    receipt = validate_opensim_candidate_replay(cand, replay)
    assert receipt.status == OpenSimQualificationStatus.REJECTED
    assert any("native test count" in r.lower() for r in receipt.rejection_reasons)
    assert any("native test" in ev for ev in receipt.missing_evidence)
    assert receipt.remedy


def test_unavailable_runtime_fails_closed_even_with_replay_payload() -> None:
    """A replay payload cannot produce QUALIFIED while opensim is unavailable."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    receipt = validate_opensim_candidate_replay(cand, replay, opensim_available=False)
    assert receipt.status == OpenSimQualificationStatus.UNAVAILABLE
    assert receipt.runtime_available is False
    assert receipt.missing_evidence
    assert receipt.remedy


def test_missing_marker_observations_do_not_qualify() -> None:
    """Replays without aligned marker observations must not qualify."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    replay.pop("markers_m")
    replay.pop("target_m")
    receipt = validate_opensim_candidate_replay(
        cand, replay, native_tests_executed=10, opensim_available=True
    )
    assert receipt.status == OpenSimQualificationStatus.REJECTED
    assert receipt.marker_metrics == {}
    assert any("marker" in ev for ev in receipt.missing_evidence)


def test_missing_rollout_data_does_not_qualify() -> None:
    """Replays without native state/time data must not qualify."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    replay.pop("native_state")
    receipt = validate_opensim_candidate_replay(
        cand, replay, native_tests_executed=10, opensim_available=True
    )
    assert receipt.status == OpenSimQualificationStatus.REJECTED
    assert receipt.derivatives_consistent is False
    assert receipt.energy_balance_checked is False
    assert "native_state/time_s dynamic rollout" in receipt.missing_evidence


def test_derivatives_mismatch_rejects() -> None:
    """Joint velocities inconsistent with dq/dt must not qualify."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    state = np.asarray(replay["native_state"])
    half = state.shape[1] // 2
    state[:, half:] = state[:, half:] + 1.0
    receipt = validate_opensim_candidate_replay(
        cand, replay, native_tests_executed=10, opensim_available=True
    )
    assert receipt.status == OpenSimQualificationStatus.REJECTED
    assert receipt.derivatives_consistent is False


def test_marker_metrics_are_computed_not_invented() -> None:
    """Only metrics derived from the recorded observations may appear."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    receipt = validate_opensim_candidate_replay(
        cand, replay, native_tests_executed=10, opensim_available=True
    )
    assert set(receipt.marker_metrics) <= {"whole_rms_m"}
    assert "pelvis_yaw_error_pct" not in receipt.marker_metrics
