"""Unit tests for Phase 6 loop-nullspace uncertainty and lower-body ensembles.

Covers:
- #12585 (NSX-60): Lower-body two-foot ground contact loop with optional GRF observations
- #12586 (NSX-61): Kinematic ensemble over admitted trajectory samples & reproject_onto_closure
- #12587 (NSX-62): Parameter ensemble over MOSAIC identifiability-gated inertial parameters
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.inertial import (
    PLANAR_PARAMETERS_PER_BODY,
    SPATIAL_PARAMETERS_PER_BODY,
    is_planar_consistent,
    is_spatial_consistent,
)
from src.shared.python.loop_nullspace.ensemble import (
    AdmittedTrajectorySample,
    EnsembleEnvelopeBand,
    KinematicEnsembleReport,
    kinematic_ensemble,
)
from src.shared.python.loop_nullspace.lower_body_loop import (
    LowerBodyLoopResult,
    actuated_lower_body_two_foot_snapshot,
    analyze_lower_body_loop,
    floating_base_two_foot_snapshot,
    include_ground_loop,
)
from src.shared.python.loop_nullspace.parameter_ensemble import (
    ParameterSample,
    ParameterSensitivityReport,
    parameter_ensemble,
)
from src.shared.python.motion_matching.club_only.nullspace_proposals import (
    reproject_onto_closure,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Slice A (#12585 — NSX-60): Lower-Body Two-Foot Unilateral Loop & GRF
# ---------------------------------------------------------------------------


def test_floating_base_two_foot_oracle_dimensions_match_golden() -> None:
    """Floating-base two-foot oracle dimension counts match golden_fixtures.json."""
    snapshot, golden_dims = floating_base_two_foot_snapshot()
    result = analyze_lower_body_loop(snapshot, grf_observed=None, mu_ground=0.6)

    assert isinstance(result, LowerBodyLoopResult)
    assert result.grf_observed_used is False
    for key, expected_val in golden_dims.items():
        assert result.dimensions[key] == expected_val, (
            f"Dimension mismatch for {key}: {result.dimensions[key]} != {expected_val}"
        )
    assert result.equality_nullity == 3
    assert result.rank_Jg == 3


def test_exact_grf_observation_reduces_equality_nullity_by_rank_jg() -> None:
    """Exact GRF observation reduces equality nullity by rank(J_g^T) and tightens sign classes."""
    # 1. Actuated lower-body + two-foot full-row-rank ground contact loop:
    #    Free GRF includes f_g in decision variables; exact GRF observation
    #    substitutes J_g^T f_g_obs into b_eff and reduces equality nullity by rank(J_g^T).
    act_snap, f_g_obs = actuated_lower_body_two_foot_snapshot()
    res_free = analyze_lower_body_loop(act_snap, grf_observed=None, torque_bound=120.0)
    res_obs = analyze_lower_body_loop(
        act_snap, grf_observed=f_g_obs, torque_bound=120.0
    )

    rank_jg = int(np.linalg.matrix_rank(act_snap.J_g.T))
    assert res_free.rank_Jg == rank_jg
    assert res_free.equality_nullity - res_obs.equality_nullity == rank_jg
    assert res_obs.nullity_reduction == rank_jg
    assert res_obs.grf_observed_used is True

    # Verify effective RHS b_eff = M a + h - J_g^T f_g_obs
    expected_b_eff = act_snap.generalized_load() - act_snap.J_g.T @ f_g_obs
    np.testing.assert_allclose(res_obs.effective_rhs, expected_b_eff, atol=1e-12)

    # Observing GRF tightens torque/hand-split envelopes and locks previously indeterminate signs
    assert np.all(res_obs.torque_lower >= res_free.torque_lower - 1e-7)
    assert np.all(res_obs.torque_upper <= res_free.torque_upper + 1e-7)
    assert len(res_obs.changed_torque_signs) >= 1

    # 2. Floating-base oracle with and without GRF via include_ground_loop
    fb_snap, _ = floating_base_two_foot_snapshot()
    fb_free = include_ground_loop(fb_snap, use_observed_grf=False)
    fb_obs = include_ground_loop(fb_snap, use_observed_grf=True)
    assert fb_free.equality_nullity == 3
    assert fb_obs.equality_nullity == 2
    assert fb_obs.ground_wrench_nullity_reduction == fb_snap.J_g.shape[0] - 1


def test_grf_uncertainty_sigma_interpolates_between_exact_and_free() -> None:
    """Bounded GRF uncertainty sigma produces envelopes containing exact GRF and inside free GRF."""
    act_snap, f_g_obs = actuated_lower_body_two_foot_snapshot()
    res_exact = analyze_lower_body_loop(
        act_snap, grf_observed=f_g_obs, grf_uncertainty_sigma=0.0, torque_bound=120.0
    )
    res_noisy = analyze_lower_body_loop(
        act_snap, grf_observed=f_g_obs, grf_uncertainty_sigma=15.0, torque_bound=120.0
    )
    res_free = analyze_lower_body_loop(act_snap, grf_observed=None, torque_bound=120.0)

    assert np.all(res_noisy.torque_lower <= res_exact.torque_lower + 1e-7)
    assert np.all(res_noisy.torque_upper >= res_exact.torque_upper - 1e-7)
    assert np.all(res_noisy.torque_lower >= res_free.torque_lower - 1e-7)
    assert np.all(res_noisy.torque_upper <= res_free.torque_upper + 1e-7)


# ---------------------------------------------------------------------------
# Slice B (#12586 — NSX-61): Kinematic Ensemble & reproject_onto_closure
# ---------------------------------------------------------------------------


def test_reproject_onto_closure_delegates_to_project_to_closure() -> None:
    """reproject_onto_closure delegates to project_to_closure on a ClosurePlant."""

    class _LinearClosurePlant:
        def __init__(self, j_c: np.ndarray, q_ref: np.ndarray) -> None:
            self._j_c = j_c
            self._q_ref = q_ref

        def closure_pose_residual(self, q: np.ndarray) -> np.ndarray:
            return self._j_c @ (q - self._q_ref)

        def closure_rate_matrix(self, q: np.ndarray) -> np.ndarray:
            _ = q
            return self._j_c

    snap, _ = floating_base_two_foot_snapshot()
    plant = _LinearClosurePlant(snap.J_c, snap.q)
    q_perturbed = snap.q + np.array([0.0, 0.0, 0.0, 0.05, -0.03, 0.02, -0.04])
    assert float(np.max(np.abs(plant.closure_pose_residual(q_perturbed)))) > 1e-3

    q_reproj = reproject_onto_closure(
        q_perturbed,
        closure_residual_m=float(
            np.max(np.abs(plant.closure_pose_residual(q_perturbed)))
        ),
        tol_m=1e-10,
        plant=plant,
    )
    assert float(np.max(np.abs(plant.closure_pose_residual(q_reproj)))) <= 1e-10


def test_kinematic_ensemble_seeded_determinism_and_union_containment() -> None:
    """Identical seed yields identical samples/bounds, and union contains every sample envelope."""
    snap0, _ = floating_base_two_foot_snapshot()
    base_trajectory = [snap0, snap0]

    rep1 = kinematic_ensemble(
        base_trajectory,
        n=6,
        seed=12345,
        perturbation_sigma=1e-3,
        closure_tol=1e-4,
    )
    rep2 = kinematic_ensemble(
        base_trajectory,
        n=6,
        seed=12345,
        perturbation_sigma=1e-3,
        closure_tol=1e-4,
    )

    assert isinstance(rep1, KinematicEnsembleReport)
    assert rep1.n_admitted == 6
    assert rep1.rejected_count == 0
    assert isinstance(rep1.envelope_union, EnsembleEnvelopeBand)
    assert isinstance(rep1.envelope_intersection, EnsembleEnvelopeBand)
    assert rep1.envelope_union.kind == "UNION"
    assert rep1.envelope_intersection.kind == "INTERSECTION"
    assert rep1.sampling_hypothesis

    # Seeded determinism
    np.testing.assert_allclose(
        rep1.envelope_union.lower, rep2.envelope_union.lower, atol=1e-13
    )
    np.testing.assert_allclose(
        rep1.envelope_union.upper, rep2.envelope_union.upper, atol=1e-13
    )
    for s1, s2 in zip(rep1.admitted_samples, rep2.admitted_samples, strict=True):
        assert isinstance(s1, AdmittedTrajectorySample)
        np.testing.assert_allclose(s1.q_trajectory, s2.q_trajectory, atol=1e-13)
        np.testing.assert_allclose(s1.v_trajectory, s2.v_trajectory, atol=1e-13)

    # Every admitted sample satisfies closure_tol and its envelope lies inside the union
    for sample in rep1.admitted_samples:
        assert sample.max_pose_residual <= rep1.closure_tol
        assert sample.max_rate_residual <= rep1.closure_tol
        assert np.all(sample.envelope_lower >= rep1.envelope_union.lower - 1e-12)
        assert np.all(sample.envelope_upper <= rep1.envelope_union.upper + 1e-12)
        if not rep1.envelope_intersection.is_empty:
            assert np.all(
                sample.envelope_lower <= rep1.envelope_intersection.lower + 1e-12
            )
            assert np.all(
                sample.envelope_upper >= rep1.envelope_intersection.upper - 1e-12
            )


def test_kinematic_ensemble_rejection_accounting() -> None:
    """Perturbations that violate closure_tol after projection are counted as rejected."""
    snap0, _ = floating_base_two_foot_snapshot()

    # Large perturbation sigma with tight closure tolerance causes curvature rejection
    rep_rejected = kinematic_ensemble(
        [snap0],
        n=8,
        seed=77,
        perturbation_sigma=0.25,
        closure_tol=1e-6,
    )
    assert rep_rejected.rejected_count > 0
    assert rep_rejected.n_admitted + rep_rejected.rejected_count == 8

    # Custom project_fn that rejects half the samples
    call_counter = {"count": 0}

    def _half_rejecting_projector(
        snap: object, q_pert: np.ndarray, v_pert: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        call_counter["count"] += 1
        _ = snap
        if call_counter["count"] % 2 == 0:
            return q_pert, v_pert, 1e-2, 1e-2
        return q_pert, v_pert, 1e-9, 1e-9

    rep_half = kinematic_ensemble(
        [snap0],
        n=6,
        seed=99,
        perturbation_sigma=1e-3,
        closure_tol=1e-6,
        project_fn=_half_rejecting_projector,
    )
    assert rep_half.n_admitted == 3
    assert rep_half.rejected_count == 3


# ---------------------------------------------------------------------------
# Slice C (#12587 — NSX-62): Parameter Ensemble Over MOSAIC Base Parameters
# ---------------------------------------------------------------------------


def test_parameter_ensemble_planar_and_spatial_lmi_and_invariance() -> None:
    """Every accepted parameter sample satisfies the pseudo-inertia LMI and keeps
    identified base combinations V_id^T theta fixed (< 1e-12).
    """
    snap0, _ = floating_base_two_foot_snapshot()

    # 1. Two-body planar inertial parameter vector (8 parameters: 2 x [m, hx, hy, Io])
    nominal_planar = np.array(
        [3.0, 0.2, -0.1, 0.45, 2.5, -0.15, 0.1, 0.35],
        dtype=np.float64,
    )
    assert bool(
        np.all(
            is_planar_consistent(nominal_planar.reshape(-1, PLANAR_PARAMETERS_PER_BODY))
        )
    )

    # Regressor Y with rank 4 < 8 (4 identified base combinations, 4 unidentifiable null directions)
    rng = np.random.default_rng(2026)
    u_mat = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    s_vals = np.array([5.0, 3.2, 1.8, 0.9], dtype=np.float64)
    regressor_planar = np.diag(s_vals) @ u_mat[:4, :]

    report_planar = parameter_ensemble(
        snap0,
        regressor_matrix=regressor_planar,
        nominal_theta=nominal_planar,
        n=10,
        seed=42,
        perturbation_scale=0.08,
    )

    assert isinstance(report_planar, ParameterSensitivityReport)
    assert report_planar.identified_rank == 4
    assert report_planar.null_dimension == 4
    assert len(report_planar.samples) == 10
    assert report_planar.max_identified_drift < 1e-12

    v_id = report_planar.identified_basis  # shape (8, 4)
    base_nominal = v_id.T @ nominal_planar

    for sample in report_planar.samples:
        assert isinstance(sample, ParameterSample)
        assert sample.lmi_satisfied is True
        # Check planar LMI directly via MOSAIC inertial.py
        bodies = sample.theta.reshape(-1, PLANAR_PARAMETERS_PER_BODY)
        assert bool(np.all(is_planar_consistent(bodies)))
        # Identified base combinations stay fixed to < 1e-12
        drift = float(np.max(np.abs(v_id.T @ sample.theta - base_nominal)))
        assert drift < 1e-12
        assert sample.identified_drift < 1e-12
        # Perturbation along ker(Y) is non-zero
        assert float(np.linalg.norm(sample.theta - nominal_planar)) > 1e-5

    # Envelope width grows along unidentified directions compared to nominal single-parameter envelope
    assert np.all(
        report_planar.envelope_union_width
        >= report_planar.nominal_envelope_width - 1e-12
    )
    assert float(np.max(report_planar.envelope_width_growth)) > 1e-4
    assert report_planar.per_quantity_sensitivity.shape == (
        report_planar.nominal_envelope_width.shape
    )

    # 2. Spatial 10-parameter body (m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz)
    nominal_spatial = np.array(
        [4.0, 0.1, -0.1, 0.2, 0.35, 0.01, -0.01, 0.40, 0.02, 0.45],
        dtype=np.float64,
    )
    assert bool(
        np.all(
            is_spatial_consistent(
                nominal_spatial.reshape(-1, SPATIAL_PARAMETERS_PER_BODY)
            )
        )
    )
    regressor_spatial = np.eye(10, dtype=np.float64)[
        :5, :
    ]  # First 5 identified, last 5 in ker(Y)

    report_spatial = parameter_ensemble(
        snap0,
        regressor_matrix=regressor_spatial,
        nominal_theta=nominal_spatial,
        n=8,
        seed=101,
        perturbation_scale=0.04,
    )
    assert report_spatial.identified_rank == 5
    assert report_spatial.null_dimension == 5
    assert report_spatial.max_identified_drift < 1e-12
    for sample in report_spatial.samples:
        assert bool(
            np.all(
                is_spatial_consistent(
                    sample.theta.reshape(-1, SPATIAL_PARAMETERS_PER_BODY)
                )
            )
        )
