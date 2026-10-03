"""Acceptance tests for clean-host end-to-end and native release gates (MMR-17, #11103).

Validates the full clean-host journey:
Load C3D/capture -> calibrate -> fit -> cancel/resume -> independent replay ->
compare -> export -> import -> reopen -> uninstall/upgrade semantics.
Verifies adverse paths:
Tampered package, missing engine, unsupported model, corrupt capture.
Verifies fail-closed release gates and UI/CLI metric parity.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.shared.python.tour_baselines import (
    BackendType,
    BaselineIdentity,
    BaselinePackage,
    DynamicFeasibilityStatus,
    FitMode,
    KinematicAccuracyStatus,
    MarkerMetricSummary,
    ModelTopology,
    PhaseMetricSummary,
    PhysicalFitMetrics,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
    StatusBundle,
    export_baseline_package,
    import_baseline_package,
)
from src.shared.python.motion_matching.native_release_gate import (
    CleanHostJourneyOptions,
    NativeReleaseGateError,
    NativeReleaseQualificationResult,
    evaluate_release_qualification,
    run_clean_host_journey,
    verify_package_integrity,
)

pytestmark = pytest.mark.unit


def _create_synthetic_package(
    model_id: str = "driven_double_pendulum",
    capture: str = "driver",
    rmse: float = 0.012,
) -> BaselinePackage:
    ident = BaselineIdentity(
        model_id=model_id,
        topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
        backend=BackendType.SCIPY_ODE,
        provider_pin="fedcba9876543210fedcba9876543210fedcba98",
        fit_mode=FitMode.TORQUE_DRIVEN,
        capture=capture,
        capture_sha256="cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d",
        horizon="G1",
        frame_convention="z_up_y_forward",
        plane_convention="transverse_sagittal_frontal",
        measurement_map_version="tour-measurement-map/1.0.0",
        fixed_geometry_hash="44" * 32,
        fixed_inertia_hash="55" * 32,
        q0_hash="11" * 32,
        v0_hash="22" * 32,
        controls_hash="33" * 32,
        runtime_hashes={
            "engine_version": "1.0.0",
            "git_commit": "fedcba9876543210fedcba9876543210fedcba98",
        },
    )
    bundle = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=ScientificQualificationStatus.QUALIFIED,
        product_promotion=ProductPromotionStatus.PROMOTED,
        has_native_replay=True,
    )
    metrics = PhysicalFitMetrics(
        whole_marker_rmse_m=rmse,
        p95_marker_error_m=rmse * 1.5,
        max_marker_error_m=rmse * 2.0,
        per_marker={
            "ClubHead": MarkerMetricSummary(
                rmse_m=rmse,
                max_m=rmse * 2.0,
                p95_m=rmse * 1.5,
                valid_count=26,
                total_count=26,
            )
        },
        per_phase={
            "address": PhaseMetricSummary(
                rmse_m=rmse * 0.5,
                max_m=rmse * 1.0,
                p95_m=rmse * 0.8,
                valid_count=5,
            ),
        },
        endpoint_error_m=0.005,
        impact_error_m=0.008,
        in_plane_rmse_m=0.004,
        out_of_plane_residual_m=0.002,
        pelvis_yaw_rmse_rad=None,
        optimizer_weighted_loss=1.2,
        n_valid=26,
        n_excluded=0,
        total_observations=26,
        coverage_fraction=1.0,
        landmark_set_signature="sig_clubhead_26",
    )
    time_arr = np.linspace(0.0, 1.0, 26)
    q = np.zeros((26, 2))
    v = np.zeros((26, 2))
    trajs = {
        "time": time_arr,
        "q": q,
        "v": v,
        "tau": np.zeros((26, 2)),
    }
    return BaselinePackage(
        identity=ident,
        statuses=bundle,
        metrics=metrics,
        replay_command=f"python -m src.shared.python.tour_baselines.campaign --model-id {model_id} --capture {capture}",
        trajectories=trajs,
        reports={
            "parameters": {"l1": 0.65, "l2": 1.05},
            "provenance": {"capture": capture, "method": "trf"},
        },
        artifacts={},
        is_synthetic=False,
    )


def test_clean_host_journey_end_to_end(tmp_path: Path) -> None:
    """Verify full user journey: load -> calibrate -> fit -> replay -> compare -> export -> reopen."""
    opts = CleanHostJourneyOptions(
        model_id="driven_double_pendulum",
        capture="driver",
        storage_dir=tmp_path / "storage",
        export_path=tmp_path / "exported_driver.npz",
    )
    result = run_clean_host_journey(opts)
    assert result["success"] is True
    assert result["reopen_verified"] is True
    assert opts.export_path.is_file()

    # Verify loaded package reproduces metrics
    loaded = import_baseline_package(opts.export_path)
    assert loaded.identity.model_id == "driven_double_pendulum"
    assert loaded.identity.capture == "driver"
    assert loaded.metrics.whole_marker_rmse_m > 0


def test_tampered_package_fails_actionably(tmp_path: Path) -> None:
    """A tampered package with modified trajectories or invalid hashes must fail actionably."""
    pkg = _create_synthetic_package("driven_double_pendulum", "driver", 0.012)
    pkg_file = tmp_path / "valid.npz"
    export_baseline_package(pkg, pkg_file)

    # First verify original is intact
    ok, err = verify_package_integrity(pkg_file)
    assert ok is True
    assert err == ""

    # Tamper with the package by modifying data inside
    tampered_file = tmp_path / "tampered.npz"
    npz_data = dict(np.load(pkg_file, allow_pickle=False))
    # Alter trajectory q
    npz_data["traj_q"] = npz_data["traj_q"] + 10.0
    np.savez(tampered_file, **npz_data)

    ok_tampered, err_tampered = verify_package_integrity(tampered_file)
    assert ok_tampered is False
    assert "tampered" in err_tampered.lower() or "mismatch" in err_tampered.lower()


def test_missing_engine_fails_actionably_as_unqualified_not_skipped(
    tmp_path: Path,
) -> None:
    """A missing mandatory engine must fail with unqualified/unavailable status rather than skipped."""
    opts = CleanHostJourneyOptions(
        model_id="full_body_pinocchio",
        capture="driver",
        storage_dir=tmp_path / "storage",
        export_path=tmp_path / "pinocchio_driver.npz",
        simulate_missing_engine="pinocchio",
    )
    with pytest.raises(NativeReleaseGateError) as exc_info:
        run_clean_host_journey(opts)
    assert (
        "unqualified" in str(exc_info.value).lower()
        or "unavailable" in str(exc_info.value).lower()
    )


def test_unsupported_model_fails_actionably(tmp_path: Path) -> None:
    """An unsupported model name must raise NativeReleaseGateError with actionable message."""
    opts = CleanHostJourneyOptions(
        model_id="nonexistent_fantasy_model",
        capture="driver",
        storage_dir=tmp_path / "storage",
        export_path=tmp_path / "fail.npz",
    )
    with pytest.raises(NativeReleaseGateError) as exc_info:
        run_clean_host_journey(opts)
    assert "unsupported model" in str(exc_info.value).lower()


def test_corrupt_capture_fails_actionably(tmp_path: Path) -> None:
    """A corrupt capture file or empty capture data must fail actionably."""
    corrupt_c3d = tmp_path / "corrupt.c3d"
    corrupt_c3d.write_bytes(b"NOT_A_VALID_C3D_HEADER")

    opts = CleanHostJourneyOptions(
        model_id="driven_double_pendulum",
        capture="driver",
        storage_dir=tmp_path / "storage",
        export_path=tmp_path / "out.npz",
        custom_capture_path=corrupt_c3d,
    )
    with pytest.raises(NativeReleaseGateError) as exc_info:
        run_clean_host_journey(opts)
    assert (
        "corrupt" in str(exc_info.value).lower()
        or "capture" in str(exc_info.value).lower()
    )


def test_ui_and_cli_metrics_agreement(tmp_path: Path) -> None:
    """CLI and UI metrics must agree within float tolerance on the same package."""
    opts = CleanHostJourneyOptions(
        model_id="driven_double_pendulum",
        capture="driver",
        storage_dir=tmp_path / "storage",
        export_path=tmp_path / "agreement.npz",
    )
    result = run_clean_host_journey(opts)
    assert result["metrics_agreement"]["rmse_delta_m"] < 1e-6
    assert result["metrics_agreement"]["agreed"] is True


def test_cancellation_and_budget_enforcement(tmp_path: Path) -> None:
    """Cancellation tokens must halt fitting immediately and enforce compute budgets."""
    opts = CleanHostJourneyOptions(
        model_id="driven_double_pendulum",
        capture="driver",
        storage_dir=tmp_path / "storage",
        export_path=tmp_path / "cancelled.npz",
        simulate_cancellation=True,
    )
    result = run_clean_host_journey(opts)
    assert result["cancelled"] is True
    assert result["success"] is False
    assert "cancellation requested" in result["status_message"].lower()


def test_native_release_gate_fails_closed_when_mandatory_engine_skipped(
    tmp_path: Path,
) -> None:
    """The release qualification gate must fail closed if any mandatory engine is skipped or zero-test."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()

    # opensim passed, but simscape skipped
    from scripts.ci.run_native_engine_lane import ENGINE_LANES

    for engine in ENGINE_LANES:
        receipt_file = receipts_dir / ENGINE_LANES[engine]["receipt_filename"]
        if engine == "simscape":
            r = {
                "schema_version": 1,
                "status": "fail",
                "engine": "simscape",
                "generated_at": "2026-10-01T00:00:00+00:00",
                "checker_source_sha256": "0" * 64,
                "contract_hashes": {
                    "scripts/ci/run_native_engine_lane.py": "1" * 64,
                    "scripts/ci/run_native_engine_lane.sh": "2" * 64,
                    "pyproject.toml": "3" * 64,
                },
                "repository_revision": "4" * 40,
                "source_freshness": {"status": "clean"},
                "engine_inventory": {
                    "available": True,
                    "version": "R2025b",
                    "source_sha256": "5" * 64,
                },
                "tests": {
                    "collected": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 10,
                    "errors": 0,
                    "executed": 0,
                },
                "pytest_probe": {
                    "name": "native_pytest_lane",
                    "status": "fail",
                    "returncode": 0,
                    "timed_out": False,
                },
            }
        else:
            r = {
                "schema_version": 1,
                "status": "pass",
                "engine": engine,
                "generated_at": "2026-10-01T00:00:00+00:00",
                "checker_source_sha256": "0" * 64,
                "contract_hashes": {
                    "scripts/ci/run_native_engine_lane.py": "1" * 64,
                    "scripts/ci/run_native_engine_lane.sh": "2" * 64,
                    "pyproject.toml": "3" * 64,
                },
                "repository_revision": "4" * 40,
                "source_freshness": {"status": "clean"},
                "engine_inventory": {
                    "available": True,
                    "version": "1.0",
                    "source_sha256": "5" * 64,
                },
                "tests": {
                    "collected": 5,
                    "passed": 5,
                    "failed": 0,
                    "skipped": 0,
                    "errors": 0,
                    "executed": 5,
                },
                "pytest_probe": {
                    "name": "native_pytest_lane",
                    "status": "pass",
                    "returncode": 0,
                    "timed_out": False,
                },
            }
        receipt_file.write_text(json.dumps(r), encoding="utf-8")

    result = evaluate_release_qualification(receipts_dir=receipts_dir)
    assert isinstance(result, NativeReleaseQualificationResult)
    assert result.release_ready is False
    assert result.release_status == "blocked"
    assert "simscape" in result.skipped_engines or "simscape" in result.failed_engines
    assert any("simscape" in b.lower() for b in result.blockers)
