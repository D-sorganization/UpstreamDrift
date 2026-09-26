"""Tests for circular control replay defect fixes (#10960 P1-6).

Validates:
(a) status is never 'passed' for the reduced path;
(b) a non-monotonic time vector gives interval_timing_ok=False;
(c) all-zero Jacobians are not a pass (rejected with ValueError);
(d) circular replay reports unassessed rather than 0.0 forward residual.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.control_replay import (
    DEFAULT_INTERVAL_TIMING_TOLERANCE_S,
    STATUS_SOFTWARE_CONTRACT_CONSISTENT,
    ControlRecoveryRequest,
    ImpactRegime,
    IndependentReplayResult,
    recover_feasible_controls,
    verify_interval_timing,
)

pytestmark = pytest.mark.unit


def _times(n: int = 21, dt: float = 0.01) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * dt


def _known_torque_trajectory(
    times: np.ndarray, *, n_actuated: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = times.size
    q = np.zeros((n, n_actuated), dtype=np.float64)
    v = np.zeros((n, n_actuated), dtype=np.float64)
    a = np.zeros((n, n_actuated), dtype=np.float64)
    tau = np.zeros((n, n_actuated), dtype=np.float64)
    for i, t in enumerate(times):
        tau[i, 0] = np.sin(2.0 * np.pi * t)
        if n_actuated > 1:
            tau[i, 1] = 0.5 * np.cos(2.0 * np.pi * t)
        a[i] = tau[i]
        if i == 0:
            q[i] = 0.0
            v[i] = 0.0
        else:
            dt = float(times[i] - times[i - 1])
            v[i] = v[i - 1] + a[i - 1] * dt
            q[i] = q[i - 1] + v[i - 1] * dt
    return q, v, a, tau


def _reduced_request(
    *,
    times: np.ndarray | None = None,
    declared_interval_s: float | None = None,
    is_circular_plant: bool | None = None,
) -> ControlRecoveryRequest:
    t = _times() if times is None else times
    q, v, a, tau = _known_torque_trajectory(t)
    return ControlRecoveryRequest(
        candidate_id="cand-reduced",
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        timestamps_s=t,
        q=q,
        v=v,
        a=a,
        tau_rnea=tau,
        n_actuated=2,
        n_contact_spheres=0,
        declared_interval_s=declared_interval_s,
        is_circular_plant=is_circular_plant,
    )


def _floating_base_request(
    *,
    j_ground: np.ndarray | None = None,
    j_grip: np.ndarray | None = None,
) -> ControlRecoveryRequest:
    t = _times(5)
    nv = 7
    n = t.size
    q = np.zeros((n, nv), dtype=np.float64)
    v = np.zeros((n, nv), dtype=np.float64)
    a = np.zeros((n, nv), dtype=np.float64)
    tau = np.ones((n, nv), dtype=np.float64)
    return ControlRecoveryRequest(
        candidate_id="cand-floating",
        trial_id="TW_wiffle",
        model_id="floating_base_humanoid",
        timestamps_s=t,
        q=q,
        v=v,
        a=a,
        tau_rnea=tau,
        n_actuated=nv,
        n_contact_spheres=2,
        j_ground=j_ground,
        j_grip=j_grip,
    )


def test_status_is_never_passed_for_reduced_path() -> None:
    """Reduced software plant replay must report software_contract_consistent, never passed."""
    req = _reduced_request()
    result = recover_feasible_controls(req)
    assert result.torque_replay_status != "passed"
    assert result.torque_replay_status == STATUS_SOFTWARE_CONTRACT_CONSISTENT
    replay = result.replay
    assert replay is not None
    assert replay.contact_feasible is None
    assert "reduced_plant_contact_unassessed" in result.limitations


def test_non_monotonic_time_vector_gives_interval_timing_false() -> None:
    """Non-monotonic timestamps or interval jitter must fail interval timing."""
    times_duplicate = np.array([0.0, 0.01, 0.01, 0.02, 0.03], dtype=np.float64)
    assert verify_interval_timing(times_duplicate) is False

    times_decreasing = np.array([0.0, 0.02, 0.01, 0.03], dtype=np.float64)
    assert verify_interval_timing(times_decreasing) is False

    times_jitter = np.array([0.0, 0.01, 0.025, 0.035], dtype=np.float64)
    assert verify_interval_timing(times_jitter, declared_interval_s=0.01) is False

    times_valid = np.arange(5, dtype=np.float64) * 0.01
    assert verify_interval_timing(times_valid, declared_interval_s=0.01) is True

    # IndependentReplayResult reflects invalid timing
    q = np.zeros((times_duplicate.size, 2), dtype=np.float64)
    replay = IndependentReplayResult(
        timestamps_s=times_duplicate,
        q_replay=q,
        forward_residual=None,
        tighter_step_residual=None,
        used_measured_state_reset=False,
        root_slack_norm=0.0,
        work_balance_error=0.0,
        torque_rate_ok=True,
        contact_feasible=None,
        closure_residual_m=None,
        impact_regime=ImpactRegime.PRE_IMPACT_ONLY,
        interval_timing_ok=verify_interval_timing(times_duplicate),
    )
    assert replay.interval_timing_ok is False

    # Jitter against declared interval rejects in recover_feasible_controls
    req_jitter = _reduced_request(times=times_jitter, declared_interval_s=0.01)
    res_jitter = recover_feasible_controls(req_jitter)
    replay_jitter = res_jitter.replay
    assert replay_jitter is not None
    assert replay_jitter.interval_timing_ok is False
    assert res_jitter.torque_replay_status == "rejected"
    assert "interval_timing_mismatch" in res_jitter.rejection_reasons


def test_all_zero_jacobians_are_not_a_pass() -> None:
    """Floating-base path with all-zero Jacobians must be rejected with ValueError."""
    req = _floating_base_request()
    with pytest.raises(ValueError, match="all-zero Jacobian"):
        recover_feasible_controls(req)

    # Explicit all-zero arrays also rejected
    n = 5
    nv = 7
    n_ground_vars = 2 * 3
    zero_jg = np.zeros((n_ground_vars, nv), dtype=np.float64)
    zero_jk = np.zeros((6, nv), dtype=np.float64)
    req_explicit = _floating_base_request(j_ground=zero_jg, j_grip=zero_jk)
    with pytest.raises(ValueError, match="all-zero Jacobian"):
        recover_feasible_controls(req_explicit)


def test_circular_replay_reports_unassessed_not_zero() -> None:
    """Circular plant replay reports forward residual unassessed, not 0.0."""
    req = _reduced_request(is_circular_plant=True)
    result = recover_feasible_controls(req)
    replay = result.replay
    assert replay is not None
    assert replay.forward_residual is None
    assert replay.forward_residual_reason == "circular_plant"
    assert "circular_plant" in result.limitations
    assert replay.closure_residual_m is None
    assert replay.tighter_step_residual is None
