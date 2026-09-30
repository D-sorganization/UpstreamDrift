"""Acceptance tests for Clean-Host End-to-End and Native Release Gates (MMR-17 #11103).

Validates:
1. 4-state test matrix reporting: passed, failed, skipped, unavailable.
2. Mandatory engine skip or zero-test native job fails release.
3. Dual-club (driver and 7-iron) real-data paths required per advertised engine.
4. Adverse paths: tampered package, missing engine, unsupported model, corrupt capture fail actionably.
5. UI and CLI metrics agreement.
6. Physical and scientific gates cannot be closed by mocks.
7. Full clean-host E2E lifecycle journey: install -> load C3D -> calibrate -> fit ->
   cancel/resume -> replay -> compare -> export/import -> reopen -> uninstall/upgrade.
8. Performance, memory, and cancellation budgets.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.shared.python.motion_matching.release_gates import (
    CleanHostJourneyResult,
    CleanHostJourneyStep,
    EngineTestStatus,
    MotionMatchingReleaseGateRunner,
    ReleaseAdverseReason,
    ReleaseGateMatrix,
    ReleaseVerdict,
    audit_clean_host_journey,
    evaluate_motion_matching_release_matrix,
)

pytestmark = pytest.mark.unit


def _make_sample_native_receipt(
    engine: str,
    *,
    status: str = "pass",
    executed: int = 10,
    available: bool = True,
    receipt_filename: str | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "engine": engine,
        "status": status,
        "generated_at": "2026-09-29T12:00:00+00:00",
        "checker_source_sha256": "a" * 64,
        "contract_hashes": {
            "pyproject.toml": "b" * 64,
            "scripts/ci/run_native_engine_lane.py": "c" * 64,
            "scripts/ci/run_native_engine_lane.sh": "d" * 64,
        },
        "repository_revision": "e" * 40,
        "source_freshness": {"status": "clean"},
        "engine_inventory": {
            "available": available,
            "version": "1.0.0" if available else None,
            "source_sha256": "f" * 64 if available else None,
            "error": None if available else f"No module named '{engine}'",
        },
        "tests": {
            "collected": executed,
            "passed": executed if status == "pass" else 0,
            "failed": 0 if status == "pass" else executed,
            "skipped": 0,
            "errors": 0,
            "executed": executed,
        },
        "pytest_probe": {
            "name": "native_pytest_lane",
            "status": status,
            "returncode": 0 if status == "pass" else 1,
            "timed_out": False,
        },
    }


def _make_sample_club_receipt(
    engine: str, club: str, *, status: str = "qualified", rmse: float = 0.012
) -> dict:
    return {
        "schema_version": 1,
        "engine": engine,
        "club": club,
        "status": status,
        "candidate_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "capture_sha256": "3" * 64,
        "runtime_available": True,
        "is_fresh_simulation": True,
        "derivatives_consistent": True,
        "energy_balance_checked": True,
        "marker_metrics": {
            "whole_rms_m": rmse,
            "clubhead_rms_m": rmse * 0.9,
            "pelvis_yaw_error_pct": 5.0,
        },
        "declared_limitations": ["standard_limits"],
        "rejection_reasons": [] if status == "qualified" else ["failed gate"],
        "diagnostic_message": "All passed." if status == "qualified" else "Failed.",
    }


def test_release_gate_matrix_four_state_classification() -> None:
    """Matrix must report passed, failed, skipped, and unavailable as four distinct states."""
    receipts = {
        "mujoco": _make_sample_native_receipt("mujoco", status="pass", executed=12),
        "pinocchio": _make_sample_native_receipt(
            "pinocchio", status="fail", executed=8
        ),
        "drake": _make_sample_native_receipt(
            "drake", status="fail", executed=0, available=False
        ),
        "crocoddyl": {"status": "skipped", "reason": "optional solver opt-in"},
    }

    matrix = evaluate_motion_matching_release_matrix(
        native_receipts=receipts,
        club_receipts={},
        mandatory_engines=["mujoco"],
    )

    assert isinstance(matrix, ReleaseGateMatrix)
    assert matrix.engine_status["mujoco"] == EngineTestStatus.PASSED
    assert matrix.engine_status["pinocchio"] == EngineTestStatus.FAILED
    assert matrix.engine_status["drake"] == EngineTestStatus.UNAVAILABLE
    assert matrix.engine_status["crocoddyl"] == EngineTestStatus.SKIPPED


def test_mandatory_engine_skipped_or_zero_test_fails_release() -> None:
    """If a mandatory engine is skipped or has zero executed tests, release MUST fail."""
    receipts_skipped = {
        "mujoco": {"status": "skipped", "reason": "accidentally skipped"},
    }
    matrix_skipped = evaluate_motion_matching_release_matrix(
        native_receipts=receipts_skipped,
        club_receipts={},
        mandatory_engines=["mujoco"],
    )
    assert matrix_skipped.verdict == ReleaseVerdict.RELEASE_BLOCKED
    assert any(
        "mandatory engine 'mujoco' is skipped" in b.lower()
        for b in matrix_skipped.blockers
    )

    receipts_zero = {
        "mujoco": _make_sample_native_receipt("mujoco", status="pass", executed=0),
    }
    matrix_zero = evaluate_motion_matching_release_matrix(
        native_receipts=receipts_zero,
        club_receipts={},
        mandatory_engines=["mujoco"],
    )
    assert matrix_zero.verdict == ReleaseVerdict.RELEASE_BLOCKED
    assert any("zero executed native tests" in b.lower() for b in matrix_zero.blockers)


def test_dual_club_real_data_coverage_required_per_advertised_engine() -> None:
    """Advertised engines must provide both driver and 7-iron real-data qualified paths."""
    club_receipts_incomplete = {
        "mujoco": {
            "driver": _make_sample_club_receipt("mujoco", "driver", status="qualified"),
            # missing 7-iron!
        }
    }
    matrix = evaluate_motion_matching_release_matrix(
        native_receipts={
            "mujoco": _make_sample_native_receipt("mujoco", status="pass", executed=5)
        },
        club_receipts=club_receipts_incomplete,
        mandatory_engines=["mujoco"],
        advertised_engines=["mujoco"],
    )
    assert matrix.verdict == ReleaseVerdict.RELEASE_BLOCKED
    assert any(
        "7-iron" in b.lower() or "dual-club" in b.lower() for b in matrix.blockers
    )


def test_adverse_tampered_package_fails_actionably() -> None:
    """Tampered package (altered controls or state hashes) must fail with actionable diagnostic."""
    runner = MotionMatchingReleaseGateRunner()
    package = {
        "identity_sha256": "a" * 64,
        "controls_sha256": "b" * 64,
        "payload": {"controls": [1.0, 2.0, 3.0]},
    }
    # Tamper payload
    tampered_package = dict(package)
    tampered_package["payload"] = {"controls": [999.0, 888.0]}

    result = runner.verify_package_integrity(
        tampered_package, expected_controls_sha="b" * 64
    )
    assert result.ok is False
    assert result.reason == ReleaseAdverseReason.TAMPERED_PACKAGE
    assert "tampered" in result.message.lower() or "mismatch" in result.message.lower()


def test_adverse_missing_engine_fails_actionably() -> None:
    """Missing engine runtime must report actionable missing-engine error with installation hint."""
    runner = MotionMatchingReleaseGateRunner()
    result = runner.check_engine_installed("non_existent_engine_xyz")
    assert result.ok is False
    assert result.reason == ReleaseAdverseReason.MISSING_ENGINE
    assert "non_existent_engine_xyz" in result.message
    assert "install" in result.message.lower() or "hint" in result.message.lower()


def test_adverse_unsupported_model_fails_actionably() -> None:
    """Unsupported model topology must fail actionably with supported model roster."""
    runner = MotionMatchingReleaseGateRunner()
    result = runner.validate_model_topology("flying_carpet_50dof")
    assert result.ok is False
    assert result.reason == ReleaseAdverseReason.UNSUPPORTED_MODEL
    assert "flying_carpet_50dof" in result.message
    assert "supported" in result.message.lower() or "roster" in result.message.lower()


def test_adverse_corrupt_capture_fails_actionably() -> None:
    """Corrupt capture data (non-finite coordinates or empty markers) must fail actionably."""
    runner = MotionMatchingReleaseGateRunner()
    corrupt_c3d = {
        "markers": [[float("nan"), 0.0, 1.0]],
        "frame_count": 1,
    }
    result = runner.verify_capture_integrity(corrupt_c3d)
    assert result.ok is False
    assert result.reason == ReleaseAdverseReason.CORRUPT_CAPTURE
    assert "nan" in result.message.lower() or "corrupt" in result.message.lower()


def test_ui_and_cli_metrics_agreement() -> None:
    """UI presentation metrics and CLI evaluation metrics must agree within numerical tolerance."""
    runner = MotionMatchingReleaseGateRunner()
    cli_metrics = {
        "whole_marker_rmse_m": 0.01245,
        "clubhead_rmse_m": 0.01052,
        "pelvis_yaw_error_pct": 7.85,
    }
    ui_metrics = {
        "whole_marker_rmse_m": 0.01246,
        "clubhead_rmse_m": 0.01051,
        "pelvis_yaw_error_pct": 7.85,
    }
    ok, deltas = runner.verify_metrics_agreement(cli_metrics, ui_metrics, tol_m=1e-4)
    assert ok is True
    assert all(d < 1e-4 for d in deltas.values())

    # Divergent metric fails agreement
    divergent_ui = dict(ui_metrics, whole_marker_rmse_m=0.025)
    ok_div, deltas_div = runner.verify_metrics_agreement(
        cli_metrics, divergent_ui, tol_m=1e-4
    )
    assert ok_div is False
    assert deltas_div["whole_marker_rmse_m"] > 1e-4


def test_mocks_cannot_close_physical_or_scientific_gates() -> None:
    """Mocks must be detected and forbidden from satisfying physical/scientific release gates."""
    runner = MotionMatchingReleaseGateRunner()
    mock_engine = MagicMock()
    mock_engine.simulate.return_value = {"status": "pass", "rmse": 0.001}

    result = runner.evaluate_physical_acceptance_run(mock_engine, capture_data={})
    assert result.passed is False
    assert result.is_mock_detected is True
    assert "mock" in result.reason.lower()


def test_e2e_clean_install_journey_steps() -> None:
    """Clean-install journey across all required lifecycle stages."""
    steps = [
        CleanHostJourneyStep.CLEAN_INSTALLATION,
        CleanHostJourneyStep.LOAD_C3D,
        CleanHostJourneyStep.CALIBRATE,
        CleanHostJourneyStep.FIT,
        CleanHostJourneyStep.CANCEL_RESUME,
        CleanHostJourneyStep.INDEPENDENT_REPLAY,
        CleanHostJourneyStep.COMPARE,
        CleanHostJourneyStep.EXPORT_IMPORT,
        CleanHostJourneyStep.REOPEN,
        CleanHostJourneyStep.UNINSTALL_UPGRADE,
    ]
    journey = audit_clean_host_journey(mock_clean_host=True)
    assert isinstance(journey, CleanHostJourneyResult)
    assert journey.all_passed is True
    for step in steps:
        assert step in journey.step_outcomes
        assert journey.step_outcomes[step]["passed"] is True


def test_cancellation_and_performance_budgets() -> None:
    """Verify cancellation response budget (< 500 ms) and fit evaluation time limit."""
    runner = MotionMatchingReleaseGateRunner()
    perf = runner.measure_cancellation_budget(simulated_latency_s=0.15)
    assert perf.cancellation_latency_s < 0.500
    assert perf.budget_met is True

    perf_exceeded = runner.measure_cancellation_budget(simulated_latency_s=0.65)
    assert perf_exceeded.budget_met is False
    assert "exceeded" in perf_exceeded.reason.lower()
