"""Tests for frozen qualification profiles and baseline gate evaluation (TB-02 #10587)."""

from __future__ import annotations

import math
import pytest

from src.shared.python.tour_baselines.baseline_package import (
    BaselineIdentity,
    BaselinePackage,
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
    StatusBundle,
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
from src.shared.python.tour_baselines.qualification_profiles import (
    QUALIFICATION_PROFILE_VERSION,
    AuthoritativeFullBodyProfile,
    PlanarDrivenPendulumProfile,
    TriplePendulumProfile,
    UpperBodyGolferProfile,
    evaluate_baseline_qualification,
    get_qualification_profile,
)


def _make_package(
    *,
    model_id: str,
    topology: ModelTopology,
    horizon: str = "G1",
    rmse: float = 0.020,
    club_rmse: float = 0.040,
    out_of_plane: float = 0.010,
    closure_m: float = 0.001,
    has_native_replay: bool = True,
    tracked_markers: tuple[str, ...] = (
        "HeadFront",
        "WaistLeft",
        "WaistRight",
        "Marker_2",
        "Marker_3",
    ),
) -> BaselinePackage:
    ident = BaselineIdentity(
        model_id=model_id,
        topology=topology,
        backend=BackendType.MUJOCO,
        provider_pin="df8f6eeb3",
        fit_mode=FitMode.TORQUE_DRIVEN,
        capture="driver",
        capture_sha256="4d1a0c8b2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b",
        horizon=horizon,
        frame_convention="z_up_y_forward",
        plane_convention="transverse_sagittal_frontal",
        measurement_map_version="tour-measurement-map/1.0.0",
        fixed_geometry_hash="1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
        fixed_inertia_hash="aaaabbbbccccddddeeeeffff1111222233334444555566667777888899990000",
        solver_name="ipopt",
        solver_config={},
        integrator="implicit_euler",
        seed=0,
        wall_clock_budget_s=60.0,
        max_evaluations_budget=1000,
        candidate_ancestry=(),
        runtime_hashes={"python": "3.11.9"},
        file_hashes={},
    )
    statuses = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.UNEVALUATED,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=ScientificQualificationStatus.UNVERIFIED,
        product_promotion=ProductPromotionStatus.EXPLORATORY,
        has_native_replay=has_native_replay,
    )
    per_marker = {
        m: MarkerMetricSummary(
            rmse_m=club_rmse if "Marker_" in m else rmse,
            max_m=0.05,
            p95_m=0.04,
            valid_count=100,
            total_count=100,
        )
        for m in tracked_markers
    }
    metrics = PhysicalFitMetrics(
        whole_marker_rmse_m=rmse,
        p95_marker_error_m=rmse * 1.5,
        max_marker_error_m=rmse * 2.0,
        per_marker=per_marker,
        per_phase={},
        endpoint_error_m=rmse,
        impact_error_m=rmse,
        in_plane_rmse_m=rmse,
        out_of_plane_residual_m=out_of_plane,
        pelvis_yaw_rmse_rad=0.030,
        optimizer_weighted_loss=1.0,
        n_valid=100 * len(tracked_markers),
        n_excluded=0,
        total_observations=100 * len(tracked_markers),
        coverage_fraction=1.0,
        landmark_set_signature="sig",
    )
    return BaselinePackage(
        identity=ident,
        statuses=statuses,
        metrics=metrics,
        replay_command="python -m simulate",
        reports={"closure_residual_m": closure_m},
        is_synthetic=True,
    )


def test_authoritative_full_body_profile_g1_pass() -> None:
    """Full-body G1 package meeting 25 mm whole RMSE qualifies."""
    pkg = _make_package(
        model_id="full_body_g1",
        topology=ModelTopology.FULL_BODY_MULTIBODY,
        horizon="G1",
        rmse=0.022,  # <= 25 mm
        club_rmse=0.045,  # <= 60 mm
    )
    profile = AuthoritativeFullBodyProfile()
    verdict = evaluate_baseline_qualification(pkg, profile)

    assert (
        verdict.statuses.kinematic_accuracy == KinematicAccuracyStatus.WITHIN_TOLERANCE
    )
    assert (
        verdict.statuses.scientific_qualification
        == ScientificQualificationStatus.QUALIFIED
    )
    assert verdict.passed is True


def test_authoritative_full_body_profile_g1_fail_on_rmse() -> None:
    """Full-body G1 package violating 25 mm whole RMSE fails qualification."""
    pkg = _make_package(
        model_id="full_body_g1",
        topology=ModelTopology.FULL_BODY_MULTIBODY,
        horizon="G1",
        rmse=0.035,  # > 25 mm
        club_rmse=0.045,
    )
    profile = AuthoritativeFullBodyProfile()
    verdict = evaluate_baseline_qualification(pkg, profile)

    assert (
        verdict.statuses.kinematic_accuracy == KinematicAccuracyStatus.EXCEEDS_THRESHOLD
    )
    assert (
        verdict.statuses.scientific_qualification
        == ScientificQualificationStatus.DISQUALIFIED
    )
    assert verdict.passed is False
    assert any("whole marker RMSE" in g.reason for g in verdict.gates if not g.passed)


def test_reduced_model_planar_pendulum_profile() -> None:
    """Planar driven pendulum has distinct attainable-geometry gates without modifying full-body thresholds."""
    pkg = _make_package(
        model_id="double_pendulum_golf",
        topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
        horizon="G1",
        rmse=0.090,  # 90 mm would fail full-body G1 (25 mm), but passes planar educational baseline (150 mm)
        club_rmse=0.110,
        out_of_plane=0.025,  # <= 50 mm planarity
        tracked_markers=("Grip", "Marker_2", "Marker_3"),
    )
    profile = PlanarDrivenPendulumProfile()
    verdict = evaluate_baseline_qualification(pkg, profile)

    assert verdict.passed is True
    assert (
        verdict.statuses.scientific_qualification
        == ScientificQualificationStatus.QUALIFIED
    )
    assert profile.max_club_rmse_m == 0.150  # 150 mm frozen gate
    assert len(profile.rationale) > 20  # documented rationale


def test_reduced_model_upper_body_profile() -> None:
    """Constrained upper-body golfer qualifies under its observable subset."""
    pkg = _make_package(
        model_id="upper_body_golfer",
        topology=ModelTopology.CONSTRAINED_UPPER_BODY,
        horizon="G1",
        rmse=0.045,  # <= 55 mm
        closure_m=0.003,  # <= 5 mm weld
        tracked_markers=(
            "Clavicle",
            "ShoulderLeft",
            "ShoulderRight",
            "Marker_2",
            "Marker_3",
        ),
    )
    profile = UpperBodyGolferProfile()
    verdict = evaluate_baseline_qualification(pkg, profile)

    assert verdict.passed is True
    assert (
        verdict.statuses.scientific_qualification
        == ScientificQualificationStatus.QUALIFIED
    )


def test_missing_native_replay_blocks_scientific_qualification() -> None:
    """Even if all numeric error thresholds pass, missing native replay cannot qualify."""
    pkg = _make_package(
        model_id="full_body_g1",
        topology=ModelTopology.FULL_BODY_MULTIBODY,
        horizon="G1",
        rmse=0.015,
        club_rmse=0.030,
        has_native_replay=False,  # MISSING NATIVE REPLAY
    )
    profile = AuthoritativeFullBodyProfile()
    verdict = evaluate_baseline_qualification(pkg, profile)

    assert verdict.passed is False
    assert (
        verdict.statuses.scientific_qualification
        == ScientificQualificationStatus.UNVERIFIED
    )


def test_profile_lookup_by_topology() -> None:
    """Lookup default qualification profile by model topology."""
    p_full = get_qualification_profile(ModelTopology.FULL_BODY_MULTIBODY)
    assert isinstance(p_full, AuthoritativeFullBodyProfile)

    p_planar = get_qualification_profile(ModelTopology.PLANAR_DRIVEN_PENDULUM)
    assert isinstance(p_planar, PlanarDrivenPendulumProfile)

    p_upper = get_qualification_profile(ModelTopology.CONSTRAINED_UPPER_BODY)
    assert isinstance(p_upper, UpperBodyGolferProfile)

    p_triple = get_qualification_profile(ModelTopology.KINEMATIC_RECONSTRUCTION)
    assert isinstance(p_triple, TriplePendulumProfile)
