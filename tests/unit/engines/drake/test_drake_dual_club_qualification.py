"""Unit tests for Drake native dual-club dynamics qualification (MMR-10D #11094).

Acceptance Criteria:
- Nonzero native test count on pinned host (rejects zero-test runs).
- Saved controls drive a fresh simulation, not a loaded state trajectory (rejects FK-only / copied states).
- Derivative/energy/contact/constraint checks use independent references.
- Aligned common-marker metrics plus all engine-specific limitations.
- Unsupported features produce unavailable/rejected states.
- Portable reproduction and failure receipts for both clubs (driver and 7-iron).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.native_qualification import (
    DrakeQualificationReceipt,
    DrakeQualificationStatus,
    assess_drake_qualification,
    validate_drake_candidate_replay,
)

pytestmark = pytest.mark.unit


def _make_dummy_candidate(
    *,
    club: str = "driver",
    model_sha: str = "b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248",
    capture_sha: str = "cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d",
) -> dict[str, Any]:
    """Create a minimal valid candidate dictionary with 27-DOF upper-body controls."""
    np.random.seed(42)
    q0 = np.zeros(27).tolist()
    qd0 = np.zeros(27).tolist()
    # 27 joints x 7 polynomial coefficients
    coeffs = np.zeros((27, 7)).tolist()
    names = [f"coord_{i}" for i in range(27)]
    return {
        "schema_version": 1,
        "source_sha256": "cand_sha_" + "0" * 56,
        "model_sha256": model_sha,
        "capture_sha256": capture_sha,
        "club": club,
        "coordinate_names": names,
        "q0": q0,
        "qd0": qd0,
        "coefficients": coeffs,
        "coefficient_order": "highest-power-first",
        "time_basis": "absolute-seconds",
        "marker_labels": ["ClubHead", "Grip", "Sternum"],
        "marker_bodies": ["club", "grip", "torso"],
        "marker_offsets_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    }


def _make_dummy_replay(n_frames: int = 50, n_coords: int = 27) -> dict[str, Any]:
    """Create a synthetic replay bundle."""
    time_s = np.linspace(0.0, 1.0, n_frames)
    q = np.sin(np.outer(time_s, np.linspace(0.1, 1.0, n_coords)))
    v = np.gradient(q, time_s, axis=0)
    native_state = np.hstack([q, v])
    markers = np.zeros((n_frames, 3, 3))
    target = np.zeros((n_frames, 3, 3))
    valid = np.ones((n_frames, 3), dtype=bool)
    return {
        "time_s": time_s,
        "native_state": native_state,
        "markers_m": markers,
        "target_m": target,
        "valid": valid,
        "is_fresh_simulation": True,
        "copied_from_reference": False,
        "actuation_applied": True,
    }


def test_reject_copied_state_trajectory() -> None:
    """A replay whose state is copied from another engine/trajectory must be rejected."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    replay["copied_from_reference"] = True
    replay["is_fresh_simulation"] = False

    receipt = validate_drake_candidate_replay(cand, replay)
    assert receipt.status == DrakeQualificationStatus.REJECTED
    assert receipt.is_fresh_simulation is False
    assert any(
        "copied state" in r.lower() or "fresh" in r.lower()
        for r in receipt.rejection_reasons
    )


def test_reject_fk_only_playback_without_dynamic_simulation() -> None:
    """FK-only marker projection without dynamic integration fails qualification."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    replay["actuation_applied"] = False
    replay["is_fresh_simulation"] = False

    receipt = validate_drake_candidate_replay(cand, replay)
    assert receipt.status == DrakeQualificationStatus.REJECTED
    assert any(
        "fk-only" in r.lower() or "actuation" in r.lower()
        for r in receipt.rejection_reasons
    )


def test_reject_zero_test_collected() -> None:
    """Zero collected native tests must not qualify as a native dynamic match."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()

    # Pass 0 executed native tests
    receipt = validate_drake_candidate_replay(cand, replay, native_tests_executed=0)
    assert receipt.status == DrakeQualificationStatus.REJECTED
    assert any(
        "zero" in r.lower() and "test" in r.lower() for r in receipt.rejection_reasons
    )


def test_missing_pydrake_runtime_reports_unavailable() -> None:
    """When pydrake runtime is absent, qualification must report UNAVAILABLE, not pass."""
    cand = _make_dummy_candidate(club="driver")
    receipt = assess_drake_qualification(cand, replay=None, drake_available=False)
    assert receipt.status == DrakeQualificationStatus.UNAVAILABLE
    assert receipt.runtime_available is False
    assert "pydrake" in receipt.diagnostic_message.lower()


def test_dual_club_support_driver_and_7iron() -> None:
    """Both driver and 7-iron clubs must be supported with distinct receipts."""
    driver_cand = _make_dummy_candidate(club="driver")
    iron_cand = _make_dummy_candidate(club="7-iron")

    driver_replay = _make_dummy_replay()
    iron_replay = _make_dummy_replay()

    r_driver = validate_drake_candidate_replay(
        driver_cand, driver_replay, native_tests_executed=5, drake_available=True
    )
    r_iron = validate_drake_candidate_replay(
        iron_cand, iron_replay, native_tests_executed=5, drake_available=True
    )

    assert r_driver.club == "driver"
    assert r_iron.club == "7-iron"
    assert r_driver.status == DrakeQualificationStatus.QUALIFIED
    assert r_iron.status == DrakeQualificationStatus.QUALIFIED


def test_stale_or_mismatched_model_hash_rejects() -> None:
    """Mismatched model or capture hash must be rejected."""
    cand = _make_dummy_candidate(model_sha="invalid_hash" * 4)
    replay = _make_dummy_replay()
    receipt = validate_drake_candidate_replay(
        cand,
        replay,
        expected_model_sha="b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248",
    )
    assert receipt.status == DrakeQualificationStatus.REJECTED
    assert any("model_sha256" in r or "hash" in r for r in receipt.rejection_reasons)


def test_derivative_and_energy_balance_checks() -> None:
    """Receipt records independent derivative and energy balance verification."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    receipt = validate_drake_candidate_replay(
        cand, replay, native_tests_executed=5, drake_available=True
    )
    assert receipt.derivatives_consistent is True
    assert receipt.energy_balance_checked is True
    assert "kinetic_energy_j" in receipt.energy_summary
    assert "potential_energy_j" in receipt.energy_summary


def test_engine_specific_limitations_declared() -> None:
    """Engine-specific limitations must be honestly disclosed in the receipt."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    receipt = validate_drake_candidate_replay(
        cand, replay, native_tests_executed=5, drake_available=True
    )
    assert len(receipt.declared_limitations) >= 2
    assert any("upper_body" in lim.lower() for lim in receipt.declared_limitations)
    assert any(
        "ground" in lim.lower() or "contact" in lim.lower()
        for lim in receipt.declared_limitations
    )


def test_receipt_serialization_round_trip(tmp_path: Path) -> None:
    """Receipt can be saved to JSON and loaded back preserving all fields."""
    cand = _make_dummy_candidate(club="driver")
    replay = _make_dummy_replay()
    receipt = validate_drake_candidate_replay(
        cand, replay, native_tests_executed=5, drake_available=True
    )

    out_file = tmp_path / "test_drake_receipt.json"
    receipt.save(out_file)
    assert out_file.exists()

    loaded = DrakeQualificationReceipt.load(out_file)
    assert loaded.schema_version == receipt.schema_version
    assert loaded.engine == receipt.engine
    assert loaded.club == receipt.club
    assert loaded.status == receipt.status
    assert loaded.candidate_sha256 == receipt.candidate_sha256
    assert loaded.declared_limitations == receipt.declared_limitations


def test_committed_dual_club_evidence_receipts_load_and_validate() -> None:
    """Committed driver and 7-iron receipts must load cleanly and report qualified."""
    repo_root = Path(__file__).resolve().parents[4]
    evidence_dir = (
        repo_root
        / "docs"
        / "development"
        / "matched_swing_program"
        / "evidence"
        / "drake"
    )
    driver_file = evidence_dir / "driver_receipt.json"
    iron_file = evidence_dir / "iron_receipt.json"

    assert driver_file.is_file(), f"Missing driver receipt: {driver_file}"
    assert iron_file.is_file(), f"Missing 7-iron receipt: {iron_file}"

    driver_rcpt = DrakeQualificationReceipt.load(driver_file)
    assert driver_rcpt.status == DrakeQualificationStatus.QUALIFIED
    assert driver_rcpt.club == "driver"
    assert driver_rcpt.derivatives_consistent is True
    assert driver_rcpt.energy_balance_checked is True

    iron_rcpt = DrakeQualificationReceipt.load(iron_file)
    assert iron_rcpt.status == DrakeQualificationStatus.QUALIFIED
    assert iron_rcpt.club == "7-iron"
    assert iron_rcpt.derivatives_consistent is True
    assert iron_rcpt.energy_balance_checked is True
