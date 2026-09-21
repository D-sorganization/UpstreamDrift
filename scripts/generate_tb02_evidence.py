"""Generate synthetic valid and invalid example baseline packages for TB-02 evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.shared.python.tour_baselines.metrics import (
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverStatus,
    TourFitMetrics,
)
from src.shared.python.tour_baselines.models import FitMode, ModelClass
from src.shared.python.tour_baselines.packages import (
    BaselinePackage,
    BaselinePackageManifest,
    CoordinatePlane,
    NativeReplayEvidence,
    export_baseline_package,
)
from src.shared.python.tour_baselines.qualification import (
    DoublePendulumPlanarProfile,
    evaluate_qualification,
)


def main() -> None:
    evidence_dir = Path("docs/plans/tour_baselines/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic Valid Package
    prof = DoublePendulumPlanarProfile()
    metrics_valid = TourFitMetrics(
        observed_valid_denominator=1000,
        excluded_sample_count=20,
        coverage_fraction=0.98,
        whole_marker_rmse_m=0.045,
        p95_marker_error_m=0.060,
        max_marker_error_m=0.075,
        per_marker_rmse_m={"ClubHead": 0.055, "Grip": 0.035},
        per_phase_rmse_m={"downswing": 0.045, "impact": 0.050},
        impact_marker_error_m=0.050,
        endpoint_clubhead_rmse_m=0.055,
        optimizer_weighted_loss=0.082,
        original_frame_rmse_m=0.045,
        in_plane_rmse_m=0.040,
        out_of_plane_residual_m=0.015,
        landmarks_hash="0" * 64,
    )
    replay_valid = NativeReplayEvidence(
        engine="mujoco",
        replay_command="python -m src.engines.physics_engines.mujoco.replay --candidate synthetic_valid",
        replay_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        verified_reproduced=True,
    )
    verdict_valid = evaluate_qualification(
        prof, metrics_valid, replay_valid, is_synthetic_test_data=True
    )

    manifest_valid = BaselinePackageManifest(
        schema_version="tour-baseline-package/1.0.0",
        package_id="967eac5b-2e78-4207-a99f-d57437296d70",
        capture="driver",
        capture_sha256="545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba",
        target_hash="a" * 64,
        model_id="double_pendulum_planar",
        model_class=ModelClass.DOUBLE_PENDULUM_PLANAR,
        backend="mujoco",
        backend_version="3.6.0",
        fit_mode=FitMode.TORQUE_DRIVEN,
        horizon="G3",
        coordinate_plane=CoordinatePlane.TWO_DIMENSIONAL_SWING_PLANE,
        measurement_map_version="tour-measurement-map/1.0.0",
        fixed_geometry={
            "arm_length_m": 0.60,
            "club_length_m": 1.15,
            "arm_mass_kg": 3.5,
            "club_mass_kg": 0.35,
        },
        q0=[0.0, 0.0],
        v0=[0.0, 0.0],
        controls_parameterization={"type": "spline_torques", "knots": 12},
        solver_config={
            "optimizer": "scipy_slsqp",
            "ftol": 1e-6,
            "max_iter": 500,
        },
        budgets={"max_iterations": 500, "wall_clock_limit_s": 60.0},
        candidate_ancestry=["b" * 64],
        runtime_hashes={"python": "3.12.0"},
        statuses={
            "solver": SolverStatus.CONVERGED.value,
            "kinematic": KinematicAccuracyStatus.ACCURATE.value,
            "dynamic": DynamicFeasibilityStatus.FEASIBLE.value,
            "scientific": verdict_valid.scientific_status.value,
            "product": verdict_valid.product_status.value,
        },
        metrics=metrics_valid,
        qualification_gate_version=prof.gate_version,
        qualification_verdict=verdict_valid.as_dict(),
        replay=replay_valid,
        is_synthetic_test_data=True,
    )
    t_arr = np.linspace(0.0, 1.814, 200)
    pkg_valid = BaselinePackage(
        manifest=manifest_valid,
        trajectories={
            "time_s": t_arr,
            "q": np.zeros((200, 2)),
            "v": np.zeros((200, 2)),
        },
    )
    export_baseline_package(
        pkg_valid, evidence_dir / "synthetic_valid_baseline_package"
    )

    # 2. Synthetic Invalid Package
    metrics_invalid = TourFitMetrics(
        observed_valid_denominator=1000,
        excluded_sample_count=20,
        coverage_fraction=0.98,
        whole_marker_rmse_m=0.160,
        p95_marker_error_m=0.190,
        max_marker_error_m=0.220,
        per_marker_rmse_m={"ClubHead": 0.175, "Grip": 0.080},
        per_phase_rmse_m={"downswing": 0.160, "impact": 0.175},
        impact_marker_error_m=0.175,
        endpoint_clubhead_rmse_m=0.175,
        optimizer_weighted_loss=0.850,
        original_frame_rmse_m=0.160,
        in_plane_rmse_m=0.140,
        out_of_plane_residual_m=0.035,
        landmarks_hash="0" * 64,
    )
    replay_invalid = NativeReplayEvidence(
        engine="mujoco",
        replay_command="",
        replay_sha256="",
        verified_reproduced=False,
    )
    verdict_invalid = evaluate_qualification(
        prof, metrics_invalid, replay_invalid, is_synthetic_test_data=True
    )

    manifest_invalid = BaselinePackageManifest(
        schema_version="tour-baseline-package/1.0.0",
        package_id="11111111-2222-3333-4444-555555555555",
        capture="driver",
        capture_sha256="545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba",
        target_hash="a" * 64,
        model_id="double_pendulum_planar",
        model_class=ModelClass.DOUBLE_PENDULUM_PLANAR,
        backend="mujoco",
        backend_version="3.6.0",
        fit_mode=FitMode.TORQUE_DRIVEN,
        horizon="G3",
        coordinate_plane=CoordinatePlane.TWO_DIMENSIONAL_SWING_PLANE,
        measurement_map_version="tour-measurement-map/1.0.0",
        fixed_geometry={"arm_length_m": 0.60, "club_length_m": 1.15},
        q0=[0.0, 0.0],
        v0=[0.0, 0.0],
        controls_parameterization={"type": "spline_torques"},
        solver_config={"optimizer": "scipy_slsqp"},
        budgets={"max_iterations": 500, "wall_clock_limit_s": 60.0},
        candidate_ancestry=[],
        runtime_hashes={"python": "3.12.0"},
        statuses={
            "solver": SolverStatus.REACHED_MAX_ITER.value,
            "kinematic": KinematicAccuracyStatus.INACCURATE.value,
            "dynamic": DynamicFeasibilityStatus.INFEASIBLE.value,
            "scientific": verdict_invalid.scientific_status.value,
            "product": verdict_invalid.product_status.value,
        },
        metrics=metrics_invalid,
        qualification_gate_version=prof.gate_version,
        qualification_verdict=verdict_invalid.as_dict(),
        replay=replay_invalid,
        is_synthetic_test_data=True,
    )
    pkg_invalid = BaselinePackage(
        manifest=manifest_invalid,
        trajectories={"time_s": t_arr, "q": np.zeros((200, 2))},
    )
    export_baseline_package(
        pkg_invalid, evidence_dir / "synthetic_invalid_baseline_package"
    )

    receipt = {
        "schema": "baseline-package-evidence/1.0.0",
        "example_packages": [
            {
                "name": "synthetic_valid_baseline_package",
                "package_id": manifest_valid.package_id,
                "scientific_status": verdict_valid.scientific_status.value,
                "product_status": verdict_valid.product_status.value,
                "is_qualified": verdict_valid.is_qualified,
                "is_promoted": False,
                "rationale": verdict_valid.reason,
            },
            {
                "name": "synthetic_invalid_baseline_package",
                "package_id": manifest_invalid.package_id,
                "scientific_status": verdict_invalid.scientific_status.value,
                "product_status": verdict_invalid.product_status.value,
                "is_qualified": verdict_invalid.is_qualified,
                "is_promoted": False,
                "rationale": verdict_invalid.reason,
            },
        ],
    }
    (evidence_dir / "baseline_package_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
