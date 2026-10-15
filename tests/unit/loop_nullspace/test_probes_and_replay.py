"""Unit tests for NSX-25 (Pareto fronts & certified affine probes) and NSX-31 (replay).

Covers:
- #12568: Epsilon-constraint Pareto fronts, certified affine probes, trajectory envelopes,
  and weight-sweep deprecation shim in ``src.shared.python.loop_nullspace.probes``.
- #12571: Open-loop vs feedback-assisted replay with separate feedback torque budget
  in ``src.shared.python.loop_nullspace.replay``.
"""

from __future__ import annotations

import json
from typing import Any, cast
import numpy as np
import pytest

from src.engines.feedback_comparison import ReplayPolicy
from src.shared.python.loop_nullspace.probes import (
    AffinePredicate,
    Certificate,
    FrameFeasibleSet,
    LinearObjective,
    ParetoFront,
    certify,
    pareto_front,
    trajectory_envelope,
    weight_sweep_tradeoffs_shim,
)
from src.shared.python.loop_nullspace.replay import (
    ConstrainedEOMProvider,
    FeedbackBudgetReport,
    ReplayAllocation,
    ReplayReport,
    replay,
)
from src.shared.python.motion_matching.force_nullspace import (
    ForceConstraints,
    ForceNullSpace,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures and Oracle Helpers
# ---------------------------------------------------------------------------


def _make_analytic_lp_oracle_frames() -> list[FrameFeasibleSet]:
    """Create a 2-frame oracle polytope where z_t in R^2 satisfies:
    z_{t,0} + z_{t,1} >= 4  (i.e. -z_{t,0} - z_{t,1} <= -4)
    0 <= z_{t,0} <= 6, 0 <= z_{t,1} <= 6.
    """
    a_ub = np.array([[-1.0, -1.0]], dtype=float)
    b_ub = np.array([-4.0], dtype=float)
    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([6.0, 6.0], dtype=float)
    return [
        FrameFeasibleSet(
            A_ub=a_ub,
            b_ub=b_ub,
            lb=lb,
            ub=ub,
            hypotheses=("H-CLOSURE-WELD6", "H-BOUNDS"),
        ),
        FrameFeasibleSet(
            A_ub=a_ub,
            b_ub=b_ub,
            lb=lb,
            ub=ub,
            hypotheses=("H-CLOSURE-WELD6", "H-BOUNDS"),
        ),
    ]


def _make_closed_loop_replay_oracle(
    n_steps: int = 40,
    dt: float = 0.01,
    mismatch_scale: float = 0.0,
) -> tuple[ReplayAllocation, ConstrainedEOMProvider]:
    """Build a 3-DOF constrained closed-loop oracle with 1 holonomic loop constraint.

    Constraint: J_c = [1, 1, 0], so q_0 + q_1 = const and a_0 + a_1 = 0.
    """
    mass_true = np.diag([2.0, 1.5, 1.0])
    bias_true = np.array([0.2, -0.1, 0.05], dtype=float)
    j_c = np.array([[1.0, 1.0, 0.0]], dtype=float)
    jdot_v = np.zeros(1, dtype=float)

    true_provider = ConstrainedEOMProvider(
        mass_matrix=mass_true,
        bias_vector=bias_true,
        closure_jacobian=j_c,
        closure_drift=jdot_v,
    )

    q_ref = np.zeros((n_steps, 3), dtype=float)
    v_ref = np.zeros((n_steps, 3), dtype=float)
    tau_alloc = np.zeros((n_steps, 3), dtype=float)

    q_curr = np.array([0.3, -0.3, 0.1], dtype=float)
    v_curr = np.array([0.1, -0.1, 0.2], dtype=float)

    for k in range(n_steps):
        t = k * dt
        tau_k = np.array(
            [1.2 * np.sin(2.0 * t), -0.8 * np.cos(2.0 * t), 0.5], dtype=float
        )
        q_ref[k] = q_curr
        v_ref[k] = v_curr
        tau_alloc[k] = tau_k
        acc_k = true_provider.acceleration(q_curr, v_curr, tau_k)
        q_curr = q_curr + v_curr * dt + 0.5 * acc_k * (dt**2)
        v_curr = v_curr + acc_k * dt

    alloc = ReplayAllocation(
        allocated_torques=tau_alloc,
        q_ref=q_ref,
        v_ref=v_ref,
        dt=dt,
    )

    mass_replay = mass_true * (1.0 + mismatch_scale)
    bias_replay = bias_true + mismatch_scale * np.array([0.8, -0.6, 0.4], dtype=float)
    replay_provider = ConstrainedEOMProvider(
        mass_matrix=mass_replay,
        bias_vector=bias_replay,
        closure_jacobian=j_c,
        closure_drift=jdot_v,
    )
    return alloc, replay_provider


# ---------------------------------------------------------------------------
# Slice A (#12568): AffinePredicate & Certified Probes
# ---------------------------------------------------------------------------


def test_affine_predicate_rejects_non_affine_and_invalid_inputs() -> None:
    """Non-affine predicates or non-finite weights must be rejected at construction."""
    with pytest.raises(ValueError, match="affine"):
        AffinePredicate(
            name="nonlinear_pred",
            weights_by_frame={0: np.array([1.0, 0.0])},
            rhs_threshold=0.0,
            is_affine=False,
        )

    with pytest.raises((TypeError, ValueError), match="affine|callable|array"):
        AffinePredicate(
            name="callable_pred",
            weights_by_frame=cast(Any, {0: lambda z: float(z[0] ** 2)}),
            rhs_threshold=0.0,
        )

    with pytest.raises(ValueError, match="finite"):
        AffinePredicate(
            name="nan_weights",
            weights_by_frame={0: np.array([np.nan, 1.0])},
            rhs_threshold=0.0,
        )

    with pytest.raises(ValueError, match="sense"):
        AffinePredicate(
            name="bad_sense",
            weights_by_frame={0: np.array([1.0, 1.0])},
            rhs_threshold=0.0,
            sense=cast(Any, "=="),
        )


def test_certify_holds_and_fails_with_valid_serializable_witness() -> None:
    """certify returns HOLDS when predicate holds everywhere and FAILS with a
    constraint-satisfying witness_z (serializable via to_dict) when violated.
    """
    frames = _make_analytic_lp_oracle_frames()

    # In each frame, z_0 + z_1 >= 4. So across frame 0, z_0 + z_1 >= 3.5 HOLDS.
    pred_holds = AffinePredicate(
        name="sum_load_positive",
        weights_by_frame={0: np.array([1.0, 1.0])},
        rhs_threshold=3.5,
        sense=">=",
    )
    cert_ok = certify(frames, pred_holds)
    assert isinstance(cert_ok, Certificate)
    assert cert_ok.status == "HOLDS"
    assert cert_ok.worst_value == pytest.approx(4.0, abs=1e-7)
    assert cert_ok.margin == pytest.approx(0.5, abs=1e-7)
    assert cert_ok.hypotheses == ("H-CLOSURE-WELD6", "H-BOUNDS")

    # Predicate: z_{0,0} >= 1.0 FAILS because (z_{0,0}=0, z_{0,1}=4) is feasible!
    pred_fails = AffinePredicate(
        name="lead_axial_load_at_least_one",
        weights_by_frame={0: np.array([1.0, 0.0])},
        rhs_threshold=1.0,
        sense=">=",
    )
    cert_fail = certify(frames, pred_fails, window=[0, 1])
    assert cert_fail.status == "FAILS"
    assert cert_fail.worst_value == pytest.approx(0.0, abs=1e-7)
    assert cert_fail.margin == pytest.approx(-1.0, abs=1e-7)
    assert cert_fail.witness_z is not None
    assert len(cert_fail.witness_z) == 2

    # Verify witness satisfies all frame constraints: z_0 + z_1 >= 4 and 0 <= z_i <= 6
    for frame_idx, z_vec in enumerate(cert_fail.witness_z):
        assert np.all(frames[frame_idx].A_ub @ z_vec <= frames[frame_idx].b_ub + 1e-7)
        assert np.all(z_vec >= frames[frame_idx].lb - 1e-7)
        assert np.all(z_vec <= frames[frame_idx].ub + 1e-7)
    assert cert_fail.witness_z[0][0] == pytest.approx(0.0, abs=1e-7)

    # Verify JSON serialization via to_dict()
    serialized = json.dumps(cert_fail.to_dict())
    decoded = json.loads(serialized)
    assert decoded["status"] == "FAILS"
    assert decoded["predicate_name"] == "lead_axial_load_at_least_one"
    assert len(decoded["witness_z"]) == 2


def test_certify_undecided_on_infeasible_set() -> None:
    """certify returns UNDECIDED when the trajectory feasible set is empty."""
    # Contradictory constraints: z_0 <= -1 but lb = [0, 0]
    infeasible_frame = FrameFeasibleSet(
        A_ub=np.array([[1.0, 0.0]], dtype=float),
        b_ub=np.array([-1.0], dtype=float),
        lb=np.array([0.0, 0.0], dtype=float),
        ub=np.array([2.0, 2.0], dtype=float),
    )
    pred = AffinePredicate(
        name="any_pred",
        weights_by_frame={0: np.array([1.0, 0.0])},
        rhs_threshold=0.0,
    )
    cert = certify([infeasible_frame], pred)
    assert cert.status == "UNDECIDED"
    assert cert.witness_z is None
    assert cert.reason


def test_trajectory_envelope_contained_within_static_envelope() -> None:
    """Trajectory-coupled envelope with rate_limit is tighter than static envelope."""
    # Frame 0 requires z in [0, 1]; Frame 1 allows z in [0, 10].
    # With rate_limit = 2.0, Frame 1 cannot exceed 1.0 + 2.0 = 3.0!
    f0 = FrameFeasibleSet(
        A_ub=np.zeros((0, 2), dtype=float),
        b_ub=np.zeros(0, dtype=float),
        lb=np.array([0.0, 0.0], dtype=float),
        ub=np.array([1.0, 1.0], dtype=float),
    )
    f1 = FrameFeasibleSet(
        A_ub=np.zeros((0, 2), dtype=float),
        b_ub=np.zeros(0, dtype=float),
        lb=np.array([0.0, 0.0], dtype=float),
        ub=np.array([10.0, 10.0], dtype=float),
    )
    static_env = trajectory_envelope([f0, f1], rate_limit=None)
    coupled_env = trajectory_envelope([f0, f1], rate_limit=2.0)

    assert static_env.feasible and coupled_env.feasible
    # Containment: coupled lower >= static lower, coupled upper <= static upper
    assert np.all(coupled_env.lower >= static_env.lower - 1e-7)
    assert np.all(coupled_env.upper <= static_env.upper + 1e-7)
    # Frame 1 upper bound is strictly tightened from 10.0 to 3.0
    assert static_env.upper[1, 0] == pytest.approx(10.0, abs=1e-7)
    assert coupled_env.upper[1, 0] == pytest.approx(3.0, abs=1e-7)


def test_pareto_front_monotonicity_and_analytic_lp_front() -> None:
    """pareto_front recovers the exact affine front f_1*(eps) = max(0, 4 - eps)
    including non-supported interior points that a weight sweep misses.
    """
    frames = [_make_analytic_lp_oracle_frames()[0]]
    obj1 = LinearObjective("lead_load", {0: np.array([1.0, 0.0])})
    obj2 = LinearObjective("trail_load", {0: np.array([0.0, 1.0])})
    eps_grid = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    front = pareto_front(frames, obj1, obj2, eps_grid)
    assert isinstance(front, ParetoFront)
    assert np.all(front.feasible_mask)

    expected_primary = np.maximum(0.0, 4.0 - eps_grid)
    np.testing.assert_allclose(front.primary_values, expected_primary, atol=1e-7)

    # Monotonicity: primary_values is non-increasing as eps increases
    diffs = np.diff(front.primary_values[front.feasible_mask])
    assert np.all(diffs <= 1e-9)

    # Tuple objectives signature from contract_appendix also works identically
    front_tuple_sig = pareto_front(frames, (obj1, obj2), eps_grid)
    np.testing.assert_allclose(
        front_tuple_sig.primary_values, front.primary_values, atol=1e-9
    )


def test_weight_sweep_tradeoffs_shim_warns_and_preserves_shape() -> None:
    """weight_sweep_tradeoffs_shim emits a warning naming weight sweeps and returns
    the legacy TorqueDistributionTradeoff shape.
    """
    a_mat = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], dtype=float)
    b_vec = np.array([2.0, 2.0], dtype=float)
    space = ForceNullSpace.from_balance(a_mat, b_vec)
    constraints = ForceConstraints(
        lower=np.full(4, -10.0),
        upper_bound=np.full(4, 10.0),
    )

    with pytest.warns(UserWarning, match="weight sweep"):
        tradeoff = weight_sweep_tradeoffs_shim(
            space,
            constraints,
            lead_indices=[0, 2],
            trail_indices=[1, 3],
        )

    assert len(tradeoff.alternatives) >= 5
    table = tradeoff.to_pareto_table()
    assert isinstance(table, list)
    assert "lead_effort" in table[0]


# ---------------------------------------------------------------------------
# Slice B (#12571): Open-Loop vs Feedback-Assisted Replay
# ---------------------------------------------------------------------------


def test_open_loop_replay_exact_oracle_vs_model_mismatch() -> None:
    """Open-loop replay stays within 1e-6 on exact oracle allocation and diverges
    in finite time under injected model mismatch.
    """
    alloc_exact, provider_exact = _make_closed_loop_replay_oracle(
        n_steps=50, dt=0.01, mismatch_scale=0.0
    )
    report_exact = replay(
        alloc_exact,
        provider_exact,
        mode="OPEN_LOOP",
        divergence_threshold=0.05,
        dt=0.01,
    )
    assert isinstance(report_exact, ReplayReport)
    assert report_exact.mode == "OPEN_LOOP"
    assert report_exact.feedback_budget is None
    assert float(np.max(report_exact.position_drift_norm)) < 1e-6
    assert float(np.max(report_exact.velocity_drift_norm)) < 1e-6
    assert report_exact.time_to_divergence_s is None

    # Injected mismatch causes open-loop drift exceeding threshold
    alloc_mm, provider_mm = _make_closed_loop_replay_oracle(
        n_steps=60, dt=0.01, mismatch_scale=0.25
    )
    report_mm = replay(
        alloc_mm,
        provider_mm,
        mode=ReplayPolicy.INDEPENDENT_TIME_ONLY,
        divergence_threshold=0.01,
        dt=0.01,
    )
    assert report_mm.mode == "OPEN_LOOP"
    assert float(np.max(report_mm.position_drift_norm)) > 0.01
    assert report_mm.time_to_divergence_s is not None
    assert report_mm.time_to_divergence_s > 0.0


def test_feedback_replay_budget_grows_with_mismatch_and_forbids_validation() -> None:
    """Feedback-assisted replay has ~0 feedback budget on exact allocations,
    grows monotonically with mismatch, reports feedback torque separately,
    and enforces validation_claim_allowed is False.
    """
    mismatch_levels = [0.0, 0.05, 0.15, 0.30]
    l2_ratios: list[float] = []

    for mismatch in mismatch_levels:
        alloc, provider = _make_closed_loop_replay_oracle(
            n_steps=40, dt=0.01, mismatch_scale=mismatch
        )
        rep = replay(
            alloc,
            provider,
            mode="FEEDBACK",
            kp=50.0,
            kd=10.0,
            divergence_threshold=0.5,
            dt=0.01,
        )
        assert rep.mode == "FEEDBACK"
        assert rep.validation_claim_allowed is False
        assert isinstance(rep.feedback_budget, FeedbackBudgetReport)
        assert (
            rep.feedback_budget.feedback_torque_series.shape
            == rep.feedback_budget.allocated_torque_series.shape
        )
        l2_ratios.append(rep.feedback_budget.l2_budget_ratio)

    # Exact allocation has ~0 feedback budget
    assert l2_ratios[0] < 1e-9

    # Monotonically increasing feedback budget with increasing model mismatch
    for i in range(len(l2_ratios) - 1):
        assert l2_ratios[i + 1] > l2_ratios[i]

    # Attempting to construct a feedback ReplayReport with validation_claim_allowed=True raises
    with pytest.raises(ValueError, match="validation"):
        ReplayReport(
            mode="FEEDBACK",
            q_trajectory=np.zeros((2, 3)),
            v_trajectory=np.zeros((2, 3)),
            position_drift_norm=np.zeros(2),
            velocity_drift_norm=np.zeros(2),
            time_to_divergence_s=None,
            feedback_budget=FeedbackBudgetReport(
                feedback_torque_series=np.zeros((2, 3)),
                allocated_torque_series=np.ones((2, 3)),
                l2_budget_ratio=0.0,
                l_inf_budget_ratio=0.0,
                peak_feedback_nm=0.0,
            ),
            validation_claim_allowed=True,
            label="FEEDBACK_ASSISTED_TRACKING",
        )
