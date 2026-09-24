"""Tests for independent baseline qualification and model adequacy (TB-09 #10594).

TDD suite verifying:
1. Fresh package loading, complete hash chain verification, and failure on corruption.
2. Dynamic rollout reconstruction from (q0, v0) without target-state injection.
3. Rejection of target-state resets, hidden base actuation, time-varying geometry, and nonfinite values.
4. Independent recomputation of 3D/in-plane RMSE, p95, max, per-marker/phase coverage, and physical constraints.
5. Model adequacy decomposition: distinguishing missing expressiveness, optimization failure, and integration error.
6. Like-for-like observation set enforcement for cross-complexity comparisons.
7. Numerical refinement and perturbation sensitivity diagnostics.
8. Force identifiability and parameter nonuniqueness disclaimers.
9. Reviewable roster verdicts across all registered models without false promotion to G3.
10. Verifiable expert signoff receipts referencing exact package and profile hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.tour_baselines.baseline_package import (
    BaselineIdentity,
    BaselinePackage,
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
    StatusBundle,
    export_baseline_package,
)
from src.shared.python.tour_baselines.coverage import generate_coverage_matrix
from src.shared.python.tour_baselines.fit_metrics import (
    MarkerMetricSummary,
    PhaseMetricSummary,
    PhysicalFitMetrics,
)
from src.shared.python.tour_baselines.models import (
    BackendType,
    FitMode,
    ModelTopology,
)
from src.shared.python.tour_baselines.qualification import (
    ExpertSignoff,
    ForceIdentifiabilityDisclaimer,
    IndependentBaselineQualifier,
    IntegrityViolation,
    ModelAdequacyDecomposition,
    RefinementSensitivityRecord,
    RosterVerdict,
    compute_package_digest,
    evaluate_full_roster_qualification,
)
from src.shared.python.tour_baselines.qualification_profiles import (
    AuthoritativeFullBodyProfile,
    PlanarDrivenPendulumProfile,
    TriplePendulumProfile,
    UpperBodyGolferProfile,
)


def _make_test_package(
    *,
    model_id: str = "driven_double_pendulum",
    topology: ModelTopology = ModelTopology.PLANAR_DRIVEN_PENDULUM,
    capture: str = "driver",
    horizon: str = "G1",
    n_nodes: int = 50,
    dt: float = 0.01,
    rmse: float = 0.025,
    club_rmse: float = 0.035,
    has_native_replay: bool = True,
    tau_limit: float = 50.0,
    tamper_data: bool = False,
    inject_nan: bool = False,
    scientific_qualification: ScientificQualificationStatus | None = None,
) -> BaselinePackage:
    """Construct a well-formed BaselinePackage with exact hash chain."""
    time_arr = np.linspace(0.0, (n_nodes - 1) * dt, n_nodes)

    # Simple simulated 2-DOF states
    q_arr = np.zeros((n_nodes, 2), dtype=np.float64)
    q_arr[:, 0] = 0.5 * np.sin(2.0 * np.pi * time_arr)
    q_arr[:, 1] = 1.0 * (1.0 - np.cos(np.pi * time_arr))

    v_arr = np.zeros((n_nodes, 2), dtype=np.float64)
    v_arr[:, 0] = np.pi * np.cos(2.0 * np.pi * time_arr)
    v_arr[:, 1] = np.pi * np.sin(np.pi * time_arr)

    tau_arr = np.zeros((n_nodes, 2), dtype=np.float64)
    tau_arr[:, 0] = 10.0 * np.sin(2.0 * np.pi * time_arr)
    tau_arr[:, 1] = 5.0 * np.cos(np.pi * time_arr)

    if inject_nan:
        q_arr[10, 0] = np.nan

    q0_bytes = q_arr[0].tobytes()
    v0_bytes = v_arr[0].tobytes()
    controls_bytes = tau_arr.tobytes()
    geom_bytes = b"fixed_geometry_v1"
    inertia_bytes = b"fixed_inertia_v1"

    q0_hash = hashlib.sha256(q0_bytes).hexdigest()
    v0_hash = hashlib.sha256(v0_bytes).hexdigest()
    controls_hash = hashlib.sha256(controls_bytes).hexdigest()
    fixed_geometry_hash = hashlib.sha256(geom_bytes).hexdigest()
    fixed_inertia_hash = hashlib.sha256(inertia_bytes).hexdigest()

    if tamper_data:
        # Alter the data so it mismatches the controls_hash
        tau_arr[0, 0] += 999.0

    ident = BaselineIdentity(
        model_id=model_id,
        topology=topology,
        backend=BackendType.SCIPY_ODE,
        provider_pin="df8f6eeb3",
        fit_mode=FitMode.TORQUE_DRIVEN,
        capture=capture,
        capture_sha256="545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba",
        horizon=horizon,
        frame_convention="z_up_y_forward",
        plane_convention="transverse_sagittal_frontal",
        measurement_map_version="tour-measurement-map/1.0.0",
        fixed_geometry_hash=fixed_geometry_hash,
        fixed_inertia_hash=fixed_inertia_hash,
        q0_hash=q0_hash,
        v0_hash=v0_hash,
        controls_hash=controls_hash,
        solver_name="scipy_ivp",
        solver_config={"rtol": 1e-6, "atol": 1e-8},
        integrator="rk45",
        seed=42,
        wall_clock_budget_s=30.0,
        max_evaluations_budget=500,
        candidate_ancestry=(),
        runtime_hashes={"python": "3.12.0", "numpy": "1.26.4"},
        file_hashes={},
    )

    if scientific_qualification is None:
        sci_status = (
            ScientificQualificationStatus.QUALIFIED
            if has_native_replay
            else ScientificQualificationStatus.UNVERIFIED
        )
    else:
        sci_status = scientific_qualification

    statuses = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=sci_status,
        product_promotion=ProductPromotionStatus.EXPLORATORY,
        has_native_replay=has_native_replay,
    )

    per_marker = {
        "Grip": MarkerMetricSummary(
            rmse_m=rmse,
            max_m=0.04,
            p95_m=0.03,
            valid_count=n_nodes,
            total_count=n_nodes,
        ),
        "Marker_2": MarkerMetricSummary(
            rmse_m=club_rmse,
            max_m=0.05,
            p95_m=0.04,
            valid_count=n_nodes,
            total_count=n_nodes,
        ),
        "Marker_3": MarkerMetricSummary(
            rmse_m=club_rmse,
            max_m=0.05,
            p95_m=0.04,
            valid_count=n_nodes,
            total_count=n_nodes,
        ),
    }

    metrics = PhysicalFitMetrics(
        whole_marker_rmse_m=float(np.mean([rmse, club_rmse])),
        p95_marker_error_m=0.045,
        max_marker_error_m=0.050,
        per_marker=per_marker,
        per_phase={
            "address_to_backswing": PhaseMetricSummary(
                rmse_m=0.02, max_m=0.03, p95_m=0.025, valid_count=n_nodes
            ),
        },
        endpoint_error_m=0.03,
        impact_error_m=0.025,
        in_plane_rmse_m=0.020,
        out_of_plane_residual_m=0.010,
        pelvis_yaw_rmse_rad=0.0,
        optimizer_weighted_loss=1.23,
        n_valid=n_nodes,
        n_excluded=0,
        total_observations=n_nodes,
        coverage_fraction=1.0,
        landmark_set_signature="sig_grip_club",
    )

    reports = {
        "joint_limits_satisfied": True,
        "max_joint_torque_nm": float(np.max(np.abs(tau_arr))),
        "torque_limit_nm": tau_limit,
        "closure_residual_m": 0.001,
        "impact_time_s": 0.45,
        "follow_through_time_s": 0.49,
    }

    trajectories = {
        "time": time_arr,
        "q": q_arr,
        "v": v_arr,
        "tau": tau_arr,
    }

    return BaselinePackage(
        identity=ident,
        statuses=statuses,
        metrics=metrics,
        replay_command="python -m tour_baselines.replay",
        trajectories=trajectories,
        reports=reports,
    )


# ---------------------------------------------------------------------------
# 1. Integrity and Hash Chain Verification
# ---------------------------------------------------------------------------


def test_hash_chain_verification_passes_on_valid_package() -> None:
    """Valid package has intact hash chain and passes integrity check."""
    package = _make_test_package()
    qualifier = IndependentBaselineQualifier()
    report = qualifier.verify_integrity(package)
    assert report.is_intact is True
    assert len(report.violations) == 0
    assert report.verified_identity_hash == package.identity.compute_hash()


def test_corrupted_package_hash_mismatch_fails_closed() -> None:
    """Tampered arrays violate hash chain and trigger IntegrityViolation."""
    package = _make_test_package(tamper_data=True)
    qualifier = IndependentBaselineQualifier()
    report = qualifier.verify_integrity(package)
    assert report.is_intact is False
    assert any("controls_hash" in v for v in report.violations)


def test_nonfinite_trajectory_fails_closed() -> None:
    """Package with NaN or Inf in state/control trajectory fails closed."""
    package = _make_test_package(inject_nan=True)
    qualifier = IndependentBaselineQualifier()
    report = qualifier.verify_integrity(package)
    assert report.is_intact is False
    assert any("non-finite" in v.lower() for v in report.violations)


def test_stale_acceptance_profile_fails_closed() -> None:
    """Requesting an outdated qualification profile version fails closed."""
    package = _make_test_package()
    qualifier = IndependentBaselineQualifier()
    with pytest.raises(IntegrityViolation, match="stale profile version"):
        qualifier.qualify(
            package, profile_version="tour-qualification-profile/0.9.0-deprecated"
        )


# ---------------------------------------------------------------------------
# 2. Dynamic Rollout Reconstruction (No Target-State Reset)
# ---------------------------------------------------------------------------


def test_forward_rollout_reconstruction_without_target_injection() -> None:
    """Forward rollout reconstructs continuous trajectory from single (q0, v0)."""
    package = _make_test_package()
    qualifier = IndependentBaselineQualifier()
    rollout_result = qualifier.reconstruct_rollout(package)
    assert rollout_result.successful is True
    assert rollout_result.target_state_injections == 0
    assert rollout_result.max_divergence_m < 0.05


def test_target_state_injection_detected_and_rejected() -> None:
    """Target-state resets or discontinuous state jumps mid-trajectory are rejected."""
    package = _make_test_package()
    # Introduce discontinuous jump in q
    q_tampered = package.trajectories["q"].copy()
    q_tampered[25:] += 10.0  # sudden jump
    bad_trajectories = dict(package.trajectories)
    bad_trajectories["q"] = q_tampered
    bad_package = BaselinePackage(
        identity=package.identity,
        statuses=package.statuses,
        metrics=package.metrics,
        replay_command=package.replay_command,
        trajectories=bad_trajectories,
        reports=package.reports,
    )
    qualifier = IndependentBaselineQualifier()
    rollout_result = qualifier.reconstruct_rollout(bad_package)
    assert rollout_result.successful is False
    assert rollout_result.target_state_injections > 0


def test_hidden_base_actuation_rejected() -> None:
    """Planar pendulum or free-base models cannot have unauthorized root actuation."""
    package = _make_test_package(topology=ModelTopology.PLANAR_DRIVEN_PENDULUM)
    # Add hidden base actuation report
    reports = dict(package.reports)
    reports["unauthorized_base_wrench_nm"] = 15.0  # non-zero unactuated torque
    bad_package = BaselinePackage(
        identity=package.identity,
        statuses=package.statuses,
        metrics=package.metrics,
        replay_command=package.replay_command,
        trajectories=package.trajectories,
        reports=reports,
    )
    qualifier = IndependentBaselineQualifier()
    evaluation = qualifier.evaluate_constraints(bad_package)
    assert evaluation.passed is False
    assert any("unauthorized base actuation" in v.lower() for v in evaluation.failures)


def test_time_varying_geometry_rejected() -> None:
    """Varying limb/shaft lengths frame-by-frame violates fixed-geometry contract."""
    package = _make_test_package()
    reports = dict(package.reports)
    reports["geometry_variance_m"] = 0.015  # 15 mm variance in bone/shaft length
    bad_package = BaselinePackage(
        identity=package.identity,
        statuses=package.statuses,
        metrics=package.metrics,
        replay_command=package.replay_command,
        trajectories=package.trajectories,
        reports=reports,
    )
    qualifier = IndependentBaselineQualifier()
    evaluation = qualifier.evaluate_constraints(bad_package)
    assert evaluation.passed is False
    assert any("time-varying geometry" in v.lower() for v in evaluation.failures)


# ---------------------------------------------------------------------------
# 3. Independent Metric Recomputation & Constraints
# ---------------------------------------------------------------------------


def test_independent_metric_recomputation() -> None:
    """Independent calculation matches declared 3D and in-plane RMSE."""
    package = _make_test_package(rmse=0.020, club_rmse=0.030)
    qualifier = IndependentBaselineQualifier()
    recomputed = qualifier.recompute_metrics(package)
    assert recomputed.in_plane_rmse_m == pytest.approx(0.020, abs=1e-3)
    assert recomputed.club_marker_rmse_m == pytest.approx(0.030, abs=1e-3)
    assert recomputed.per_marker_coverage["Marker_2"] == 1.0


def test_window_endpoints_and_phases_checked() -> None:
    """Declared window endpoints (impact, follow-through) and phases are checked."""
    package = _make_test_package()
    qualifier = IndependentBaselineQualifier()
    endpoint_check = qualifier.verify_endpoints(package)
    assert endpoint_check.has_impact is True
    assert endpoint_check.impact_time_s == 0.45
    assert endpoint_check.has_follow_through is True


def test_model_specific_physical_constraints() -> None:
    """Enforces joint limits, torque bounds, and weld/loop closure constraints."""
    # Exceed torque limit
    package = _make_test_package(tau_limit=2.0)  # max tau in package is ~10 Nm > 2 Nm
    qualifier = IndependentBaselineQualifier()
    evaluation = qualifier.evaluate_constraints(package)
    assert evaluation.passed is False
    assert any("torque limit" in f.lower() for f in evaluation.failures)


# ---------------------------------------------------------------------------
# 4. Model Adequacy vs. Optimization Failure vs. Integration Error
# ---------------------------------------------------------------------------


def test_model_adequacy_decomposition() -> None:
    """Decomposes errors into expressiveness gap, optimization gap, and integration error."""
    # Suppose geometric limit for planar 2-DOF on 3D data is 0.040 m,
    # kinematic fit achieved was 0.045 m, and dynamic replay error was 0.048 m.
    qualifier = IndependentBaselineQualifier()
    decomp = qualifier.decompose_model_adequacy(
        projected_geometric_residual_m=0.040,
        kinematic_fit_rmse_m=0.045,
        dynamic_replay_rmse_m=0.048,
    )
    assert decomp.expressiveness_gap_m == pytest.approx(0.040)
    assert decomp.optimization_gap_m == pytest.approx(0.005)
    assert decomp.integration_error_m == pytest.approx(0.003)
    assert decomp.primary_limitation == "expressiveness_gap"


def test_like_for_like_observation_sets_enforced() -> None:
    """Rejects cross-complexity ranking when observation marker sets differ."""
    qualifier = IndependentBaselineQualifier()
    with pytest.raises(ValueError, match="Like-for-like observation set required"):
        qualifier.compare_cross_complexity(
            model_a_id="planar_double_pendulum",
            markers_a=("Grip", "Marker_2"),
            rmse_a=0.030,
            model_b_id="full_body_mujoco",
            markers_b=("WaistLeft", "HeadFront", "Marker_2"),
            rmse_b=0.025,
        )


# ---------------------------------------------------------------------------
# 5. Sensitivity & Force Identifiability Disclaimers
# ---------------------------------------------------------------------------


def test_numerical_refinement_sensitivity() -> None:
    """Simulating with tighter timestep records perturbation sensitivity."""
    package = _make_test_package()
    qualifier = IndependentBaselineQualifier()
    sensitivity = qualifier.evaluate_refinement_sensitivity(package, dt_refined=0.005)
    assert isinstance(sensitivity, RefinementSensitivityRecord)
    assert sensitivity.max_coordinate_diff_m >= 0.0
    assert sensitivity.stable_under_refinement is True


def test_force_identifiability_disclaimer_mandatory() -> None:
    """Disclaimer explicitly warns against physiological force / injury inference."""
    disclaimer = ForceIdentifiabilityDisclaimer()
    text = disclaimer.render()
    assert "nonuniqueness" in text.lower()
    assert "not independent force measurements" in text.lower()
    assert "injury" in text.lower()


# ---------------------------------------------------------------------------
# 6. Roster Cell Verdicts & G3 Boundary Protection
# ---------------------------------------------------------------------------


def test_full_roster_evaluation_covers_all_registered_models() -> None:
    """Every registered model cell in coverage matrix receives an explicit verdict."""
    matrix = generate_coverage_matrix()
    verdicts = evaluate_full_roster_qualification(matrix)
    assert len(verdicts) == len(matrix)
    assert len(verdicts) == 40

    # Ensure every cell has a typed verdict
    for verdict in verdicts.values():
        assert isinstance(verdict.verdict, RosterVerdict)
        assert len(verdict.rationale) > 0


def test_reduced_profile_never_translates_to_g3() -> None:
    """A reduced model profile pass can NEVER be marked as G3."""
    qualifier = IndependentBaselineQualifier()
    package = _make_test_package(
        topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
        horizon="G3",  # Attempting to qualify planar pendulum for G3
    )
    with pytest.raises(
        IntegrityViolation, match="Reduced models cannot qualify under G3"
    ):
        qualifier.qualify(package)


def test_expert_signoff_receipt_generation() -> None:
    """Generates auditable signoff receipt referencing exact hashes and disclaimers."""
    package = _make_test_package()
    qualifier = IndependentBaselineQualifier()
    signoff = qualifier.generate_expert_signoff(
        package=package,
        reviewer="independent_scientific_reviewer",
        verdict=RosterVerdict.QUALIFIED_REDUCED,
        notes="Educational planar baseline satisfies frozen PlanarDrivenPendulumProfile.",
    )
    assert isinstance(signoff, ExpertSignoff)
    assert signoff.package_hash == compute_package_digest(package)
    assert signoff.profile_version == "tour-qualification-profile/1.0.0"
    receipt_json = signoff.to_json()
    parsed = json.loads(receipt_json)
    assert parsed["verdict"] == "qualified_reduced"
    assert "disclaimer" in parsed


def test_expert_signoff_digest_changes_on_evidence_tampering() -> None:
    """Modifying trajectory or artifact evidence after signoff alters package digest (#10787)."""
    package = _make_test_package()
    digest_before = compute_package_digest(package)

    # Alter trajectory data
    package.trajectories["q"][0, 0] += 0.05
    digest_after = compute_package_digest(package)
    assert digest_before != digest_after


def test_missing_arrays_or_empty_hashes_rejected_by_integrity() -> None:
    """Integrity verification fails closed when arrays or hashes are empty (#10786)."""
    qualifier = IndependentBaselineQualifier()

    # Package missing q array
    pkg_missing_q = _make_test_package()
    del pkg_missing_q.trajectories["q"]
    report = qualifier.verify_integrity(pkg_missing_q)
    assert not report.is_intact
    assert any(
        "Required trajectory array 'q' is missing or empty" in v
        for v in report.violations
    )

    # Package missing q0_hash
    pkg_empty_hash = _make_test_package()
    from dataclasses import replace

    new_ident = replace(pkg_empty_hash.identity, q0_hash="")
    object.__setattr__(pkg_empty_hash, "identity", new_ident)
    report2 = qualifier.verify_integrity(pkg_empty_hash)
    assert not report2.is_intact
    assert any("missing required 'q0_hash'" in v for v in report2.violations)


def test_pendulum_assembled_package_qualifies_cleanly() -> None:
    """TB-10 bot review #10794: packages from pendulum _assemble_baseline_package are qualifiable."""
    from src.engines.physics_engines.pendulum.python.motion_matching.qualification import (
        _assemble_baseline_package,
    )
    from src.shared.python.motion_matching.club_target import (
        ClubTarget,
        SourceProvenance,
    )
    from src.shared.python.motion_matching.fit_result import CanonicalFitResult
    from src.shared.python.motion_matching.acceptance import qualify_tour_baseline

    n_samples = 26
    time_arr = np.linspace(0.0, 0.25, n_samples)
    club_quat = np.zeros((n_samples, 4))
    club_quat[:, 0] = 1.0  # unit quaternion

    target = ClubTarget(
        time=time_arr,
        butt=np.zeros((n_samples, 3)),
        clubhead=np.ones((n_samples, 3)) * 0.5,
        club_quat=club_quat,
        impact_idx=20,
        source=SourceProvenance(
            filename="test.c3d",
            format="c3d",
            subject_id="tour_avg",
            trial_id="driver_01",
            sha256="0" * 64,
        ),
    )
    fit_result = CanonicalFitResult(
        theta_optimal=np.zeros(14),
        final_cost=0.001,
        final_rmse_m=0.01,
        solver_status="success",
        iterations=10,
        n_evaluations=20,
        wall_clock_s=1.0,
        message="converged",
        history=(0.01, 0.001),
        method="trf",
        git_commit="abcdef0",
        engine_version="1.0.0",
        target_hash="0" * 64,
        timestamp_utc="2026-09-24T00:00:00Z",
    )
    trajs = (
        np.zeros((n_samples, 2)),
        np.zeros((n_samples, 2)),
        np.zeros((n_samples, 2)),
        np.zeros((n_samples, 2)),
    )
    dists = (
        np.zeros(n_samples),
        np.zeros(n_samples),
        [0.001] * n_samples,
    )
    pkg = _assemble_baseline_package(
        target=target,
        capture_kind="driver",
        result=fit_result,
        trajectories=trajs,
        dists=dists,
        lengths=(0.65, 1.05),
        maxiter=10,
    )

    # Package must contain time and tau
    assert "time" in pkg.trajectories
    assert "tau" in pkg.trajectories
    assert pkg.identity.q0_hash is not None and len(pkg.identity.q0_hash) > 0
    assert pkg.identity.v0_hash is not None and len(pkg.identity.v0_hash) > 0
    assert (
        pkg.identity.controls_hash is not None and len(pkg.identity.controls_hash) > 0
    )
    assert pkg.identity.fixed_geometry_hash != ""
    assert pkg.identity.fixed_inertia_hash != ""

    verdict = qualify_tour_baseline(pkg)
    assert verdict is not None
    assert verdict.passed is True


def test_qualify_legacy_package_migration() -> None:
    """TB-10 bot review #10794: qualify handles legacy packages missing time/tau/hashes via migration."""
    from src.shared.python.tour_baselines.baseline_package import (
        BackendType,
        BaselineIdentity,
        BaselinePackage,
        DynamicFeasibilityStatus,
        FitMode,
        KinematicAccuracyStatus,
        ModelTopology,
        PhysicalFitMetrics,
        ProductPromotionStatus,
        ScientificQualificationStatus,
        SolverConvergenceStatus,
        StatusBundle,
    )
    from dataclasses import replace
    from src.shared.python.tour_baselines.qualification import (
        IndependentBaselineQualifier,
        migrate_legacy_package,
    )

    package = _make_test_package(
        topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
        horizon="G1",
    )
    # Strip time, tau, and identity hashes to simulate a legacy package
    del package.trajectories["time"]
    del package.trajectories["tau"]
    legacy_ident = replace(
        package.identity,
        q0_hash=None,
        v0_hash=None,
        controls_hash=None,
        fixed_geometry_hash="",
        fixed_inertia_hash="",
    )
    object.__setattr__(package, "identity", legacy_ident)

    # migrate_legacy_package should populate time, tau, and hashes
    migrated = migrate_legacy_package(package)
    assert "time" in migrated.trajectories
    assert "tau" in migrated.trajectories
    assert migrated.identity.q0_hash is not None and len(migrated.identity.q0_hash) > 0
    assert migrated.identity.v0_hash is not None and len(migrated.identity.v0_hash) > 0
    assert (
        migrated.identity.controls_hash is not None
        and len(migrated.identity.controls_hash) > 0
    )
    assert migrated.identity.fixed_geometry_hash != ""
    assert migrated.identity.fixed_inertia_hash != ""

    # qualify must fail-closed on unmigrated legacy packages missing evidence/hashes (#10799)
    qualifier = IndependentBaselineQualifier()
    with pytest.raises(IntegrityViolation) as exc_info:
        qualifier.qualify(package)
    assert "Package integrity failure" in str(exc_info.value)

    # migrate_legacy_package is an explicit operation that leaves qualification unverified until regenerated
    assert (
        migrated.statuses.scientific_qualification
        == ScientificQualificationStatus.UNVERIFIED
    )


def test_pendulum_fixed_inertia_hash_changes_with_inertia_parameters() -> None:
    """TB-10 bot review #10800: fixed_inertia_hash digests actual dynamics parameters."""
    from src.engines.physics_engines.pendulum.python.motion_matching.adapters import (
        create_calibrated_double_pendulum_dynamics,
    )
    from src.engines.physics_engines.pendulum.python.motion_matching.qualification import (
        _assemble_baseline_package,
    )
    from src.shared.python.motion_matching.club_target import (
        ClubTarget,
        SourceProvenance,
    )
    from src.shared.python.motion_matching.fit_result import CanonicalFitResult

    n_samples = 10
    time_arr = np.linspace(0.0, 0.1, n_samples)
    club_quat = np.zeros((n_samples, 4))
    club_quat[:, 0] = 1.0

    target = ClubTarget(
        time=time_arr,
        butt=np.zeros((n_samples, 3)),
        clubhead=np.ones((n_samples, 3)) * 0.5,
        club_quat=club_quat,
        impact_idx=5,
        source=SourceProvenance(
            filename="test.c3d",
            format="c3d",
            subject_id="tour_avg",
            trial_id="driver_01",
            sha256="0" * 64,
        ),
    )
    fit_result = CanonicalFitResult(
        theta_optimal=np.zeros(14),
        final_cost=0.001,
        final_rmse_m=0.01,
        solver_status="success",
        iterations=10,
        n_evaluations=20,
        wall_clock_s=1.0,
        message="converged",
        history=(0.01, 0.001),
        method="trf",
        git_commit="abcdef0",
        engine_version="1.0.0",
        target_hash="0" * 64,
        timestamp_utc="2026-09-24T00:00:00Z",
    )
    trajs = (
        np.zeros((n_samples, 2)),
        np.zeros((n_samples, 2)),
        np.zeros((n_samples, 2)),
        np.zeros((n_samples, 2)),
    )
    dists = (
        np.zeros(n_samples),
        np.zeros(n_samples),
        [0.001] * n_samples,
    )

    dyn1 = create_calibrated_double_pendulum_dynamics(0.65, 1.05)
    pkg1 = _assemble_baseline_package(
        target=target,
        capture_kind="driver",
        result=fit_result,
        trajectories=trajs,
        dists=dists,
        lengths=(0.65, 1.05),
        maxiter=10,
        dynamics=dyn1,
    )

    dyn2 = create_calibrated_double_pendulum_dynamics(0.65, 1.05)
    # Modify inertia parameter: clubhead mass
    dyn2.parameters.lower_segment.clubhead_mass_kg += 0.05
    pkg2 = _assemble_baseline_package(
        target=target,
        capture_kind="driver",
        result=fit_result,
        trajectories=trajs,
        dists=dists,
        lengths=(0.65, 1.05),
        maxiter=10,
        dynamics=dyn2,
    )

    assert pkg1.identity.fixed_inertia_hash != ""
    assert pkg2.identity.fixed_inertia_hash != ""
    assert pkg1.identity.fixed_inertia_hash != pkg2.identity.fixed_inertia_hash
