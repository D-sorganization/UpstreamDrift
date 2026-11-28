"""Unit tests for shared null-space body strategies and observability (#15787)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_only.canonical_v2_nullspace import (
    CandidateGateEvaluator,
    JointObservabilityReport,
    NullspaceStrategyReport,
    SharedJacobianSpec,
    compute_joint_observability,
    generate_and_qualify_nullspace_strategies,
)

pytestmark = pytest.mark.unit
JOINTS = (
    "torso_twist",
    "torso_bend",
    "shoulder_elev",
    "elbow_flex",
    "wrist_ulnar",
    "left_ankle",
    "right_ankle",
    "club_shaft",
)


class _StubGate:
    """Stub CandidateGateEvaluator enforcing tracking/dynamics/replay rules."""

    def __init__(
        self, *, reject_strategy: str | None = None, max_norm: float | None = None
    ) -> None:
        """Configure optional strategy or norm rejection rules."""
        self.reject_strategy, self.max_norm = reject_strategy, max_norm

    def evaluate(
        self, q_cand: NDArray[np.float64], dq: NDArray[np.float64], strategy: str
    ) -> tuple[bool, tuple[str, ...]]:
        """Evaluate tracking, dynamics, and replay gates."""
        _ = q_cand
        reasons: list[str] = []
        if self.reject_strategy is not None and strategy == self.reject_strategy:
            reasons.append("replay_gate_failed")
        if self.max_norm is not None and float(np.linalg.norm(dq)) > self.max_norm:
            reasons.append("dynamics_torque_limit_exceeded")
        return (False, tuple(reasons)) if reasons else (True, ())


def _spec() -> SharedJacobianSpec:
    """Construct club, grip, and active-foot Jacobians leaving torso/arm null space."""
    jc = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    jg = np.array([[0.5, 0.0, 0.4, 0.3, 0.2, 0.0, 0.0, -0.5]])
    jf = np.array([[0, 0, 0, 0, 0, 1.0, 0, 0], [0, 0, 0, 0, 0, 0, 1.0, 0]])
    return SharedJacobianSpec(jc, jg, jf, JOINTS)


def test_every_generated_offset_lies_in_shared_kernel() -> None:
    """Every generated offset delta_q satisfies ||J_shared @ delta_q|| < tol."""
    spec, q0 = _spec(), np.zeros(len(JOINTS))
    j_shared, gate = spec.stacked(), _StubGate()
    assert isinstance(gate, CandidateGateEvaluator)
    amps = (-0.08, -0.04, 0.04, 0.08)
    rep = generate_and_qualify_nullspace_strategies(
        spec=spec, q_nominal=q0, gate_evaluator=gate, amplitudes=amps
    )
    assert isinstance(rep, NullspaceStrategyReport) and rep.candidates
    for c in rep.candidates:
        res = float(np.linalg.norm(j_shared @ c.delta_q))
        assert res < 1.0e-8
        assert c.constraint_residual_norm == pytest.approx(res, abs=1e-12)


def test_gate_enforcement_rejects_and_excludes_from_envelopes() -> None:
    """Candidates failing tracking/dynamics/replay gates are excluded from envelopes."""
    gate = _StubGate(reject_strategy="arm_wrist_led", max_norm=0.05)
    amps, q0 = (-0.10, -0.03, 0.03, 0.10), np.zeros(len(JOINTS))
    rep = generate_and_qualify_nullspace_strategies(
        spec=_spec(), q_nominal=q0, gate_evaluator=gate, amplitudes=amps
    )
    ae, te = rep.envelopes["arm_wrist_led"], rep.envelopes["torso_led"]
    assert ae.passing_count == 0 and ae.rejected_count == 4
    assert ae.min_q is None and ae.max_q is None
    assert te.passing_count == 2 and te.rejected_count == 2
    max_ok = max(float(np.linalg.norm(c.delta_q)) for c in rep.passing_candidates)
    assert max_ok <= 0.05
    assert all(not c.passed and c.rejection_reasons for c in rep.rejected_candidates)


def test_torso_vs_arm_wrist_strategies_differentiate_bounded_envelopes() -> None:
    """torso_led and arm_wrist_led yield distinct body envelopes at zero constraint drift."""
    spec = _spec()
    j_shared = spec.stacked()
    q0 = np.array([0.1, -0.1, 0.2, 0.3, -0.1, 0.0, 0.0, 0.5])
    lo, hi = q0 - 0.12, q0 + 0.12
    rep = generate_and_qualify_nullspace_strategies(
        spec=spec,
        q_nominal=q0,
        gate_evaluator=_StubGate(),
        amplitudes=(-0.08, 0.08),
        q_bounds=(lo, hi),
    )
    te, ae = rep.envelopes["torso_led"], rep.envelopes["arm_wrist_led"]
    assert te.min_delta_q is not None and te.max_delta_q is not None
    assert ae.min_delta_q is not None and ae.max_delta_q is not None
    t_t = float(np.linalg.norm(te.max_delta_q[:2] - te.min_delta_q[:2]))
    a_t = float(np.linalg.norm(ae.max_delta_q[:2] - ae.min_delta_q[:2]))
    a_a = float(np.linalg.norm(ae.max_delta_q[2:5] - ae.min_delta_q[2:5]))
    t_a = float(np.linalg.norm(te.max_delta_q[2:5] - te.min_delta_q[2:5]))
    assert t_t > a_t and a_a > t_a
    for c in rep.passing_candidates:
        assert np.all(c.q_candidate >= lo - 1e-12)
        assert np.all(c.q_candidate <= hi + 1e-12)
        assert np.allclose(j_shared @ c.q_candidate, j_shared @ q0, atol=1e-8)


def test_joint_observability_report_matches_analytical_rank_and_norms() -> None:
    """Observability scores in [0, 1] match SVD row/null norms and rank."""
    spec = _spec()
    jc, jg, jf = spec.j_club, spec.j_grip, spec.j_foot
    obs: JointObservabilityReport = compute_joint_observability(jc, jg, jf, JOINTS)
    rank = int(np.linalg.matrix_rank(spec.stacked()))
    assert obs.rank == rank and obs.null_dim == len(JOINTS) - rank
    by_name = {e.joint_name: e for e in obs.entries}
    for name in ("left_ankle", "right_ankle", "club_shaft"):
        assert by_name[name].observability_score == pytest.approx(1.0, abs=1e-12)
        assert by_name[name].null_space_norm == pytest.approx(0.0, abs=1e-12)
    assert by_name["torso_bend"].observability_score == pytest.approx(0.0, abs=1e-12)
    assert by_name["torso_bend"].null_space_norm == pytest.approx(1.0, abs=1e-12)
    for e in obs.entries:
        assert 0.0 <= e.observability_score <= 1.0
        norm_sq = e.row_space_norm**2 + e.null_space_norm**2
        assert norm_sq == pytest.approx(1.0, abs=1e-12)
    assert sum(e.observability_score for e in obs.entries) == pytest.approx(float(rank))
