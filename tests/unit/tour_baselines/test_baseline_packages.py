"""Tests for BaselineIdentity, StatusBundle, and BaselinePackage serialization/import (TB-02 #10587)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.shared.python.tour_baselines.baseline_package import (
    BASELINE_SCHEMA_VERSION,
    BaselineIdentity,
    BaselinePackage,
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
    StatusBundle,
    export_baseline_package,
    import_baseline_package,
)
from src.shared.python.tour_baselines.fit_metrics import (
    MarkerMetricSummary,
    PhysicalFitMetrics,
)
from src.shared.python.tour_baselines.models import (
    BackendType,
    FitMode,
    ModelTopology,
)


def _sample_identity() -> BaselineIdentity:
    return BaselineIdentity(
        model_id="full_body_golf_g1",
        topology=ModelTopology.FULL_BODY_MULTIBODY,
        backend=BackendType.MUJOCO,
        provider_pin="df8f6eeb3",
        fit_mode=FitMode.TORQUE_DRIVEN,
        capture="driver",
        capture_sha256="4d1a0c8b2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b",
        horizon="G1",
        frame_convention="z_up_y_forward",
        plane_convention="transverse_sagittal_frontal",
        measurement_map_version="tour-measurement-map/1.0.0",
        fixed_geometry_hash="1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
        fixed_inertia_hash="aaaabbbbccccddddeeeeffff1111222233334444555566667777888899990000",
        solver_name="ipopt",
        solver_config={"max_iter": 500, "tol": 1e-4},
        integrator="implicit_euler",
        seed=42,
        wall_clock_budget_s=60.0,
        max_evaluations_budget=1000,
        candidate_ancestry=("ancestor_hash_1",),
        runtime_hashes={"python": "3.11.9", "numpy": "1.26.4"},
        file_hashes={
            "model_xml": "ffffaaaabbbbccccddddeeeeffff111122223333444455556666777788889999"
        },
    )


def _sample_metrics() -> PhysicalFitMetrics:
    return PhysicalFitMetrics(
        whole_marker_rmse_m=0.022,
        p95_marker_error_m=0.035,
        max_marker_error_m=0.045,
        per_marker={
            "HeadFront": MarkerMetricSummary(
                rmse_m=0.015, max_m=0.025, p95_m=0.020, valid_count=100, total_count=100
            )
        },
        per_phase={},
        endpoint_error_m=0.020,
        impact_error_m=None,
        in_plane_rmse_m=0.018,
        out_of_plane_residual_m=0.012,
        pelvis_yaw_rmse_rad=0.035,
        optimizer_weighted_loss=4.12,
        n_valid=100,
        n_excluded=0,
        total_observations=100,
        coverage_fraction=1.0,
        landmark_set_signature="sig_headfront_100",
    )


def test_baseline_identity_hash_and_immutability() -> None:
    """Baseline identity is immutable and computes deterministic hash."""
    ident = _sample_identity()
    ident_hash = ident.compute_hash()
    assert len(ident_hash) == 64
    # Recomputing produces identical hash
    assert ident.compute_hash() == ident_hash

    from dataclasses import FrozenInstanceError

    # Frozen
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ident.model_id = "other"  # type: ignore[misc]


def test_status_bundle_orthogonality() -> None:
    """Status bundle keeps solver, kinematic, dynamic, scientific, product statuses separate."""
    statuses = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=ScientificQualificationStatus.QUALIFIED,
        product_promotion=ProductPromotionStatus.PROMOTED,
        has_native_replay=True,
    )
    assert statuses.solver_convergence == SolverConvergenceStatus.CONVERGED
    assert statuses.scientific_qualification == ScientificQualificationStatus.QUALIFIED


def test_missing_native_replay_cannot_scientifically_qualify() -> None:
    """A manifest with missing native replay CANNOT be scientifically qualified."""
    with pytest.raises(ValueError, match="cannot be QUALIFIED without native replay"):
        StatusBundle(
            solver_convergence=SolverConvergenceStatus.CONVERGED,
            kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
            dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
            scientific_qualification=ScientificQualificationStatus.QUALIFIED,
            product_promotion=ProductPromotionStatus.EXPLORATORY,
            has_native_replay=False,  # MISSING NATIVE REPLAY!
        )


def test_synthetic_package_cannot_be_promoted() -> None:
    """Synthetic test package cannot be promoted as product tour fit."""
    ident = _sample_identity()
    statuses = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=ScientificQualificationStatus.QUALIFIED,
        product_promotion=ProductPromotionStatus.PROMOTED,  # FORBIDDEN FOR SYNTHETIC
        has_native_replay=True,
    )
    metrics = _sample_metrics()

    with pytest.raises(
        ValueError, match="Synthetic test package cannot have product_promotion"
    ):
        BaselinePackage(
            identity=ident,
            statuses=statuses,
            metrics=metrics,
            replay_command="python -m simulate",
            is_synthetic=True,  # Synthetic test data!
        )


def test_package_export_and_clean_machine_import() -> None:
    """Exporting a package to disk and importing in a fresh directory preserves all contents and checksums."""
    ident = _sample_identity()
    statuses = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=ScientificQualificationStatus.QUALIFIED,
        product_promotion=ProductPromotionStatus.EXPLORATORY,
        has_native_replay=True,
    )
    metrics = _sample_metrics()

    q_arr = np.linspace(0.0, 1.0, 30).reshape(10, 3)
    tau_arr = np.ones((10, 3)) * 5.0
    trajectories = {"q": q_arr, "tau": tau_arr}

    package = BaselinePackage(
        identity=ident,
        statuses=statuses,
        metrics=metrics,
        replay_command="python -m simulate --run",
        trajectories=trajectories,
        is_synthetic=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "baseline_pkg.npz"
        export_baseline_package(package, archive_path)

        assert archive_path.is_file()

        # Import in fresh clean environment
        imported = import_baseline_package(archive_path)

        assert imported.identity.model_id == ident.model_id
        assert imported.identity.compute_hash() == ident.compute_hash()
        assert imported.statuses.solver_convergence == statuses.solver_convergence
        assert math.isclose(
            imported.metrics.whole_marker_rmse_m, metrics.whole_marker_rmse_m
        )
        assert imported.is_synthetic is True

        # Check trajectory array equality
        assert np.allclose(imported.trajectories["q"], q_arr)
        assert np.allclose(imported.trajectories["tau"], tau_arr)


def test_tampered_archive_raises_validation_error() -> None:
    """Tampering with archive arrays or manifest must be detected and rejected."""
    ident = _sample_identity()
    statuses = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=ScientificQualificationStatus.QUALIFIED,
        product_promotion=ProductPromotionStatus.EXPLORATORY,
        has_native_replay=True,
    )
    metrics = _sample_metrics()
    trajectories = {"q": np.zeros((5, 2))}

    package = BaselinePackage(
        identity=ident,
        statuses=statuses,
        metrics=metrics,
        replay_command="test",
        trajectories=trajectories,
        is_synthetic=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "tampered.npz"
        export_baseline_package(package, archive_path)

        # Tamper with archive: load raw, modify array, re-save
        raw = dict(np.load(archive_path))
        raw["traj_q"] = np.ones((5, 2)) * 999.0  # Modified array
        np.savez(archive_path, **raw)

        with pytest.raises(ValueError, match="Checksum mismatch"):
            import_baseline_package(archive_path)


def test_evaluate_baseline_package_acceptance_integration() -> None:
    """Acceptance service evaluates baseline package via evaluate_baseline_package_acceptance."""
    from src.shared.python.motion_matching.acceptance import (
        evaluate_baseline_package_acceptance,
    )

    ident = _sample_identity()
    statuses = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=ScientificQualificationStatus.QUALIFIED,
        product_promotion=ProductPromotionStatus.EXPLORATORY,
        has_native_replay=True,
    )
    metrics = _sample_metrics()
    package = BaselinePackage(
        identity=ident,
        statuses=statuses,
        metrics=metrics,
        replay_command="python -m simulate",
        is_synthetic=True,
    )

    verdict = evaluate_baseline_package_acceptance(package)
    assert verdict.horizon.value == "G1"
