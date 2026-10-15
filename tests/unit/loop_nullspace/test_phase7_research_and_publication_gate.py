"""Unit tests for NSX Phase 7 research spikes (NSX-70..73) and publication gate (NSX-74).

Covers:
- NSX-70 (#12588): Hit-and-run sampling of the feasible set on a low-rank temporal basis.
- NSX-71 (#12589): Inverse optimal control (IOC) of allocation weights with held-out swings.
- NSX-72 (#12590): True closure alternatives with re-derived kinematics per grip model.
- NSX-73 (#12591): Muscle-level view (redundancy, co-contraction, and impedance).
- NSX-74 (#12592): Publication gate for NSX results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace.publication_gate import (
    PublicationGateItem,
    PublicationGateReport,
    evaluate_nsx_publication_gate,
)
from src.shared.python.loop_nullspace.research_spikes import (
    IOCFitResult,
    MuscleRedundancyAnalysis,
    TemporalHitAndRunResult,
    TrueClosureModelResult,
    analyze_muscle_redundancy_layer,
    compare_true_closure_models,
    fit_ioc_allocation_weights,
    hit_and_run_temporal_sample,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# NSX-70 (#12588): Hit-and-Run Sampling on a Low-Rank Temporal Basis
# ---------------------------------------------------------------------------


def _build_box_polytope_per_frame(
    n_frames: int, dim_z: int, half_width: float = 2.0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Construct box constraints -half_width <= z_j(t) <= half_width per frame."""
    eye = np.eye(dim_z, dtype=float)
    a_ub_frame = np.vstack([eye, -eye])
    b_ub_frame = np.full(2 * dim_z, half_width, dtype=float)
    return [a_ub_frame.copy() for _ in range(n_frames)], [
        b_ub_frame.copy() for _ in range(n_frames)
    ]


def test_hit_and_run_temporal_sample_stays_inside_polytope_and_respects_lp_envelope() -> (
    None
):
    n_frames = 8
    dim_z = 2
    half_width = 1.5
    a_ub_list, b_ub_list = _build_box_polytope_per_frame(
        n_frames, dim_z, half_width=half_width
    )

    result = hit_and_run_temporal_sample(
        a_ub_list,
        b_ub_list,
        basis_rank=3,
        n_samples=60,
        seed=42,
        burn_in=25,
    )

    assert isinstance(result, TemporalHitAndRunResult)
    assert result.samples_z.shape == (60, n_frames, dim_z)
    assert result.basis_matrix.shape == (n_frames, 3)
    assert result.chebyshev_radius > 0.0
    assert result.max_constraint_violation <= 1e-9

    # Orthonormality of the temporal basis Phi^T Phi = I_R
    gram = result.basis_matrix.T @ result.basis_matrix
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)

    # Every generated trajectory sample must satisfy A_ub(t) z(t) <= b_ub(t)
    for sample_idx in range(result.samples_z.shape[0]):
        for t in range(n_frames):
            lhs = a_ub_list[t] @ result.samples_z[sample_idx, t]
            assert np.all(lhs <= b_ub_list[t] + 1e-9)

    # Sample min/max must lie strictly within the exact LP envelope bounds
    assert np.all(result.sample_min >= result.lp_envelope_min - 1e-9)
    assert np.all(result.sample_max <= result.lp_envelope_max + 1e-9)
    # And explore a non-trivial fraction of the interior
    assert np.all(result.sample_max - result.sample_min > 0.25)
    assert result.effective_sample_size > 1.0
    assert -1.0 <= result.lag1_autocorrelation <= 1.0


def test_hit_and_run_temporal_sample_is_deterministic_by_seed() -> None:
    a_ub_list, b_ub_list = _build_box_polytope_per_frame(6, 2, half_width=1.0)
    res_a = hit_and_run_temporal_sample(
        a_ub_list, b_ub_list, basis_rank=2, n_samples=20, seed=12345, burn_in=10
    )
    res_b = hit_and_run_temporal_sample(
        a_ub_list, b_ub_list, basis_rank=2, n_samples=20, seed=12345, burn_in=10
    )
    res_c = hit_and_run_temporal_sample(
        a_ub_list, b_ub_list, basis_rank=2, n_samples=20, seed=99999, burn_in=10
    )
    np.testing.assert_allclose(res_a.samples_z, res_b.samples_z, atol=1e-14)
    assert not np.allclose(res_a.samples_z, res_c.samples_z)


def test_hit_and_run_temporal_sample_rejects_invalid_inputs() -> None:
    a_ub_list, b_ub_list = _build_box_polytope_per_frame(4, 2, half_width=1.0)
    with pytest.raises(ValueError, match="basis_rank"):
        hit_and_run_temporal_sample(
            a_ub_list, b_ub_list, basis_rank=0, n_samples=10, seed=1
        )
    with pytest.raises(ValueError, match="basis_rank"):
        hit_and_run_temporal_sample(
            a_ub_list, b_ub_list, basis_rank=10, n_samples=10, seed=1
        )


# ---------------------------------------------------------------------------
# NSX-71 (#12589): Inverse Optimal Control With Held-Out Swings
# ---------------------------------------------------------------------------


def _generate_synthetic_ioc_swing(
    weights_true: np.ndarray, seed: int, n_frames: int = 12, dim_z: int = 3
) -> dict[str, np.ndarray]:
    """Generate synthetic trajectory minimizing sum_k w_k * 0.5 * (z - r_k)^T H_k (z - r_k)."""
    rng = np.random.default_rng(seed)
    n_terms = len(weights_true)
    hessians = np.zeros((n_frames, n_terms, dim_z, dim_z), dtype=float)
    linear_terms = np.zeros((n_frames, n_terms, dim_z), dtype=float)
    z_star = np.zeros((n_frames, dim_z), dtype=float)

    for t in range(n_frames):
        h_total = np.zeros((dim_z, dim_z), dtype=float)
        g_total = np.zeros(dim_z, dtype=float)
        for k in range(n_terms):
            raw = rng.standard_normal((dim_z, dim_z))
            h_k = raw.T @ raw + (k + 1.0) * np.eye(dim_z)
            ref_k = rng.uniform(-1.5, 1.5, size=dim_z)
            g_k = -h_k @ ref_k
            hessians[t, k] = h_k
            linear_terms[t, k] = g_k
            h_total += weights_true[k] * h_k
            g_total += weights_true[k] * g_k
        z_star[t] = -np.linalg.solve(h_total, g_total)

    return {
        "basis_hessians": hessians,
        "basis_linear": linear_terms,
        "z_observed": z_star,
    }


def test_fit_ioc_allocation_weights_recovers_ground_truth_and_enforces_non_claim() -> (
    None
):
    weights_true = np.array([0.55, 0.30, 0.15], dtype=float)
    training = [
        _generate_synthetic_ioc_swing(weights_true, seed=10 + i) for i in range(4)
    ]
    held_out = [
        _generate_synthetic_ioc_swing(weights_true, seed=100 + i) for i in range(3)
    ]

    result = fit_ioc_allocation_weights(training, held_out)

    assert isinstance(result, IOCFitResult)
    np.testing.assert_allclose(result.fitted_weights, weights_true, atol=1e-4)
    assert np.isclose(np.sum(result.fitted_weights), 1.0, atol=1e-12)
    assert np.all(result.fitted_weights >= 0.0)
    assert result.training_kkt_residual < 1e-6
    assert result.held_out_kkt_residual < 1e-6
    assert result.held_out_trajectory_rmse < 1e-5
    assert result.validates_physical_hand_loads is False
    assert (
        result.non_claim
        == "Weights fitted to IK-derived data do not validate physical hand loads"
    )


def test_fit_ioc_allocation_weights_rejects_empty_splits() -> None:
    weights_true = np.array([0.6, 0.4], dtype=float)
    swing = _generate_synthetic_ioc_swing(weights_true, seed=1)
    with pytest.raises(ValueError, match="training_trajectories"):
        fit_ioc_allocation_weights([], [swing])
    with pytest.raises(ValueError, match="held_out_trajectories"):
        fit_ioc_allocation_weights([swing], [])


# ---------------------------------------------------------------------------
# NSX-72 (#12590): True Closure Alternatives With Re-Derived Kinematics
# ---------------------------------------------------------------------------


def test_compare_true_closure_models_rederives_kinematics_and_admits_each_model() -> (
    None
):
    # Target trajectory with non-zero desired orientation / axial twist change
    t_grid = np.linspace(0.0, 0.4, 9)
    target_poses = np.column_stack(
        [
            0.55 + 0.08 * np.cos(2.0 * np.pi * t_grid),
            0.15 + 0.06 * np.sin(2.0 * np.pi * t_grid),
            0.35 * np.sin(np.pi * t_grid),  # shaft axial roll angle
            0.25 * np.cos(np.pi * t_grid),  # wrist bending angle 1
            -0.20 * np.sin(np.pi * t_grid),  # wrist bending angle 2
        ]
    )

    results = compare_true_closure_models(
        target_poses, models=("WELD6", "SHAFT_FREE5", "BALL3")
    )

    assert set(results.keys()) == {"WELD6", "SHAFT_FREE5", "BALL3"}
    for model_name, model_res in results.items():
        assert isinstance(model_res, TrueClosureModelResult)
        assert model_res.model_name == model_name
        assert model_res.analysis_kind == "TRUE_CLOSURE_ALTERNATIVE"
        assert model_res.admitted is True
        assert model_res.max_phi_residual <= 1e-6
        assert model_res.max_phidot_residual <= 1e-5
        assert model_res.max_phiddot_residual <= 1e-4
        assert model_res.q_trajectory.shape[0] == target_poses.shape[0]

    # True closure re-derives kinematics per grip model: BALL3 and SHAFT_FREE5
    # follow distinct joint trajectories compared to WELD6 when moments are released
    q_weld = results["WELD6"].q_trajectory
    q_shaft = results["SHAFT_FREE5"].q_trajectory
    q_ball = results["BALL3"].q_trajectory
    assert np.linalg.norm(q_ball - q_weld) > 1e-2
    assert np.linalg.norm(q_shaft - q_weld) > 1e-2
    assert np.linalg.norm(q_ball - q_shaft) > 1e-2


# ---------------------------------------------------------------------------
# NSX-73 (#12591): Muscle-Level View (Redundancy, Co-Contraction & Impedance)
# ---------------------------------------------------------------------------


def test_analyze_muscle_redundancy_layer_proves_co_contraction_nullspace() -> None:
    # 2-DOF joint system (e.g., wrist flexion/extension and radial/ulnar deviation)
    # actuated by 6 muscles (agonists and antagonists with opposing moment arms)
    moment_arm_matrix = np.array(
        [
            [0.035, -0.035, 0.025, -0.025, 0.015, -0.015],
            [0.020, 0.020, -0.025, -0.025, 0.030, -0.030],
        ],
        dtype=float,
    )
    f_max = np.array([600.0, 600.0, 450.0, 450.0, 350.0, 350.0], dtype=float)
    tau_target = np.array([4.5, -2.0], dtype=float)

    analysis = analyze_muscle_redundancy_layer(
        tau_target=tau_target,
        moment_arm_matrix=moment_arm_matrix,
        f_max=f_max,
    )

    assert isinstance(analysis, MuscleRedundancyAnalysis)
    assert analysis.n_joints == 2
    assert analysis.n_muscles == 6
    assert analysis.muscle_nullspace_dim == 4
    assert analysis.nullspace_basis.shape == (6, 4)

    # Nullspace basis columns must lie in ker(R)
    np.testing.assert_allclose(
        moment_arm_matrix @ analysis.nullspace_basis,
        np.zeros((2, 4)),
        atol=1e-11,
    )

    # Both min-activation and max-co-contraction solutions must produce the exact same net joint torque
    f_min = f_max * analysis.min_effort_activations
    f_max_cc = f_max * analysis.max_cocontraction_activations
    np.testing.assert_allclose(moment_arm_matrix @ f_min, tau_target, atol=1e-8)
    np.testing.assert_allclose(moment_arm_matrix @ f_max_cc, tau_target, atol=1e-8)

    # Activations stay within physiological bounds [0, 1]
    assert np.all(analysis.min_effort_activations >= -1e-9)
    assert np.all(analysis.min_effort_activations <= 1.0 + 1e-9)
    assert np.all(analysis.max_cocontraction_activations >= -1e-9)
    assert np.all(analysis.max_cocontraction_activations <= 1.0 + 1e-9)

    # Co-contraction index and joint stiffness trace are strictly larger at max co-contraction
    assert analysis.max_cocontraction_index > analysis.min_cocontraction_index + 0.1
    assert (
        np.trace(analysis.stiffness_at_max_cocontraction)
        > np.trace(analysis.stiffness_at_min_effort) * 1.5
    )
    assert analysis.net_torque_determines_stiffness is False


# ---------------------------------------------------------------------------
# NSX-74 (#12592): Publication Gate for NSX Results
# ---------------------------------------------------------------------------


def test_evaluate_nsx_publication_gate_enforces_deferred_external_validation() -> None:
    report = evaluate_nsx_publication_gate(external_validation_completed=False)

    assert isinstance(report, PublicationGateReport)
    assert report.all_hypotheses_verified is True
    assert report.no_unverified_Active_citations is True
    assert report.deferred_validation_recorded is True
    assert len(report.explicit_non_claims) >= 5
    assert all(isinstance(item, PublicationGateItem) for item in report.checklist)
    # Without physical DV-12554 completion, software/methodology publication is allowed
    # but physiological claims are strictly blocked.
    assert report.ready_for_methodology_publication is True
    assert report.ready_for_physiological_claim is False

    report_validated = evaluate_nsx_publication_gate(external_validation_completed=True)
    assert report_validated.ready_for_physiological_claim is True


def test_evaluate_nsx_publication_gate_detects_missing_dv_file(tmp_path: Path) -> None:
    missing_dv = tmp_path / "nonexistent_DV.md"
    report = evaluate_nsx_publication_gate(
        dv_path=missing_dv, external_validation_completed=False
    )
    assert report.deferred_validation_recorded is False
    assert report.ready_for_methodology_publication is False
