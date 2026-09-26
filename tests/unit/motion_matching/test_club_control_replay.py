"""CO-06 recover feasible controls and independently replay candidates (#10610)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.control_replay import (
    CONTROL_REPLAY_SCHEMA,
    STATUS_SOFTWARE_CONTRACT_CONSISTENT,
    ControlRecoveryRequest,
    ControlRecoveryResult,
    ImpactRegime,
    IndependentReplayResult,
    detect_measured_state_resets,
    evaluate_tighter_step_sensitivity,
    independent_forward_residual,
    recover_feasible_controls,
    recover_and_replay_candidates,
    control_replay_evidence_payload,
)
from src.shared.python.motion_matching.club_only.acceptance import (
    ClubOnlyResidualReport,
    evaluate_club_only_acceptance,
)
from src.shared.python.motion_matching.club_only.ambiguity import CandidateScore
from src.shared.python.motion_matching.club_only.profiles import get_club_only_profile
from src.shared.python.tour_baselines.registry import init_default_registry

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "club_only_matching"
    / "evidence"
    / "club_control_replay.json"
)


def _times(n: int = 21, dt: float = 0.01) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * dt


def _known_torque_trajectory(
    times: np.ndarray, *, n_actuated: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic double-integrator plant with known actuator torque."""
    n = times.size
    q = np.zeros((n, n_actuated), dtype=np.float64)
    v = np.zeros((n, n_actuated), dtype=np.float64)
    a = np.zeros((n, n_actuated), dtype=np.float64)
    tau = np.zeros((n, n_actuated), dtype=np.float64)
    for i, t in enumerate(times):
        # tau = [sin(2pi t), 0.5 cos(2pi t)]; a = tau (unit inertia)
        tau[i, 0] = np.sin(2.0 * np.pi * t)
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


def _request(
    *,
    candidate_id: str = "cand-known",
    trial_id: str = "TW_wiffle",
    model_id: str = "driven_double_pendulum",
    times: np.ndarray | None = None,
    q: np.ndarray | None = None,
    v: np.ndarray | None = None,
    a: np.ndarray | None = None,
    tau_truth: np.ndarray | None = None,
    impact_time_s: float | None = None,
    impact_modeled: bool = False,
    allow_measured_resets: bool = False,
    measured_q_inject: np.ndarray | None = None,
    n_contact_spheres: int = 0,
    force_infeasible_contact: bool = False,
    root_slack_override: float | None = None,
    declared_interval_s: float | None = None,
    is_circular_plant: bool | None = None,
) -> ControlRecoveryRequest:
    times = _times() if times is None else times
    if q is None or v is None or a is None or tau_truth is None:
        q, v, a, tau_truth = _known_torque_trajectory(times)
    return ControlRecoveryRequest(
        candidate_id=candidate_id,
        trial_id=trial_id,
        model_id=model_id,
        timestamps_s=times,
        q=q,
        v=v,
        a=a,
        tau_rnea=tau_truth,  # unit-mass plant: net generalized torque equals tau
        n_actuated=tau_truth.shape[1],
        n_contact_spheres=n_contact_spheres,
        impact_time_s=impact_time_s,
        impact_modeled=impact_modeled,
        allow_measured_resets=allow_measured_resets,
        measured_q_inject=measured_q_inject,
        force_infeasible_contact=force_infeasible_contact,
        root_slack_override=root_slack_override,
        declared_interval_s=declared_interval_s,
        is_circular_plant=is_circular_plant,
        kinematic_preview_ok=True,
        qualification_blockers=(
            "native_g1_qualification_requires_desk_native_receipt",
            "software_contract_replay_is_not_native_evidence",
        ),
    )


def test_known_torque_case_recovers_minimum_effort_allocation() -> None:
    result = recover_feasible_controls(_request())
    assert result.allocation_objective == "minimum_effort"
    assert result.policy is not None
    assert result.policy.claims_unique_measured_torques is False
    np.testing.assert_allclose(
        result.policy.tau_actuated, result.policy.net_generalized_torque, atol=1e-6
    )
    assert result.torque_replay_status == STATUS_SOFTWARE_CONTRACT_CONSISTENT
    assert result.replay is not None
    assert result.replay.contact_feasible is None


def test_forward_residual_independently_recomputed() -> None:
    times = _times()
    q, v, a, tau = _known_torque_trajectory(times)
    req = _request(times=times, q=q, v=v, a=a, tau_truth=tau, is_circular_plant=False)
    result = recover_feasible_controls(req)
    assert result.policy is not None
    replay = result.replay
    assert replay is not None
    # Claimed residual must not be trusted; recompute from policy + plant.
    recomputed = independent_forward_residual(result.policy, times=times, q_reference=q)
    assert recomputed is not None
    assert replay.forward_residual is not None
    assert abs(recomputed - replay.forward_residual) < 1e-9

    # Circular replay marks residual unassessed rather than reporting 0.0
    req_circ = _request(
        times=times, q=q, v=v, a=a, tau_truth=tau, is_circular_plant=True
    )
    res_circ = recover_feasible_controls(req_circ)
    replay_circ = res_circ.replay
    assert replay_circ is not None
    assert replay_circ.forward_residual is None
    assert replay_circ.forward_residual_reason == "circular_plant"

    # Poisoned claimed residual cannot qualify — lie opposite of truth.
    lied = 0.0 if recomputed > 1e-6 else 1.0
    poisoned = IndependentReplayResult(
        timestamps_s=replay.timestamps_s,
        q_replay=replay.q_replay,
        forward_residual=lied,
        tighter_step_residual=replay.tighter_step_residual,
        used_measured_state_reset=False,
        root_slack_norm=replay.root_slack_norm,
        work_balance_error=replay.work_balance_error,
        torque_rate_ok=True,
        contact_feasible=True,
        closure_residual_m=0.0,
        impact_regime=ImpactRegime.PRE_IMPACT_ONLY,
        interval_timing_ok=True,
    )
    assert abs(poisoned.forward_residual - recomputed) > 1e-6
    # Independent recompute remains the authority over any claimed residual.
    assert (
        abs(
            independent_forward_residual(result.policy, times=times, q_reference=q)
            - recomputed
        )
        < 1e-12
    )


def test_root_slack_cannot_qualify() -> None:
    result = recover_feasible_controls(_request(root_slack_override=0.5))
    assert result.torque_replay_status == "rejected"
    assert "root_slack_cannot_qualify" in result.rejection_reasons
    assert result.claims_native_qualification is False
    assert result.native_g1_pass is False


def test_tighter_step_sensitivity_flags_unstable_replay() -> None:
    times = _times(11, dt=0.02)
    q, v, a, tau = _known_torque_trajectory(times)
    # Inject a stiff spike so coarse vs fine residual diverge.
    a = a.copy()
    a[5] *= 50.0
    tau = tau.copy()
    tau[5] *= 50.0
    req = _request(times=times, q=q, v=v, a=a, tau_truth=tau, is_circular_plant=False)
    result = recover_feasible_controls(req)
    assert result.replay is not None
    sens = evaluate_tighter_step_sensitivity(
        result.policy,
        times=times,
        q_reference=q,
        coarse_residual=result.replay.forward_residual,
    )
    assert sens.tighter_step_residual >= 0.0
    assert sens.ratio >= 0.0


def test_interval_timing_rejects_non_monotonic_clock() -> None:
    times = _times()
    times = times.copy()
    times[10] = times[9]  # duplicate stamp
    with pytest.raises(ValueError, match="strictly increasing"):
        recover_feasible_controls(_request(times=times))


def test_infeasible_reaction_contact_rejects_with_actionable_cause() -> None:
    result = recover_feasible_controls(
        _request(n_contact_spheres=2, force_infeasible_contact=True)
    )
    assert result.torque_replay_status == "rejected"
    assert any("contact" in r or "infeasible" in r for r in result.rejection_reasons)
    assert result.kinematic_preview_ok is True  # retain good kinematic preview


def test_no_reset_detection_rejects_measured_state_injection() -> None:
    times = _times()
    q, v, a, tau = _known_torque_trajectory(times)
    measured = q.copy()
    measured[10:] = measured[10:] + 0.25  # inject measured jump mid-horizon
    result = recover_feasible_controls(
        _request(
            times=times,
            q=q,
            v=v,
            a=a,
            tau_truth=tau,
            allow_measured_resets=True,
            measured_q_inject=measured,
        )
    )
    assert result.replay is not None
    assert result.replay.used_measured_state_reset is True
    assert result.torque_replay_status == "rejected"
    assert "measured_state_reset_detected" in result.rejection_reasons
    flags = detect_measured_state_resets(q_claimed=measured, q_open_loop=q, atol=1e-3)
    assert flags.any()


def test_modeled_versus_unknown_impact_event() -> None:
    unknown = recover_feasible_controls(
        _request(impact_time_s=0.05, impact_modeled=False)
    )
    assert unknown.replay is not None
    assert unknown.replay.impact_regime is ImpactRegime.UNKNOWN_IMPACT
    assert "unknown_impact_event_not_modeled" in unknown.limitations

    modeled = recover_feasible_controls(
        _request(impact_time_s=0.05, impact_modeled=True, n_contact_spheres=1)
    )
    assert modeled.replay is not None
    assert modeled.replay.impact_regime in {
        ImpactRegime.MODELED_CONTACT,
        ImpactRegime.SPLIT_REGIME,
    }


def test_rejected_dynamics_keeps_separate_kinematic_status() -> None:
    init_default_registry()
    result = recover_feasible_controls(_request(root_slack_override=1.0))
    assert result.kinematic_preview_ok is True
    assert result.torque_replay_status == "rejected"
    residual = ClubOnlyResidualReport(
        grip_position_rmse_m=0.01,
        face_position_rmse_m=0.01,
        grip_orientation_rmse_rad=None,
        face_orientation_rmse_rad=None,
        native_coverage_fraction=1.0,
        speed_error_m_s=None,
        phase_error_s=None,
        unweighted_physical={"closure_residual_m": 0.001},
    )
    scores = (
        CandidateScore(
            candidate_id=result.candidate_id,
            measured_residual_m=0.01,
            prior_score=0.8,
            body_configuration_hash="synthetic-body",
            contact_feasible=False,
            closure_residual_m=0.001,
            claims_force_measurement=False,
        ),
    )
    profile = get_club_only_profile("driven_double_pendulum")
    verdict = evaluate_club_only_acceptance(
        residual,
        scores,
        profile,
        body_markers_present=False,
        force_labels_synthetic=True,
        torque_replay_validated=False,
        kinematic_preview_ok=result.kinematic_preview_ok,
    )
    assert verdict.statuses.kinematic_preview == "passed"
    assert verdict.statuses.torque_replay == "unevaluated"


def test_batch_recover_and_replay_never_invents_native_pass() -> None:
    report = recover_and_replay_candidates(
        (
            _request(candidate_id="a"),
            _request(candidate_id="b", root_slack_override=0.2),
        )
    )
    assert report.schema == CONTROL_REPLAY_SCHEMA
    assert report.governing_issue == 10610
    assert all(not c.native_g1_pass for c in report.results)
    assert all(not c.claims_native_qualification for c in report.results)
    assert all(c.qualification_blockers for c in report.results)
    payload = control_replay_evidence_payload(report)
    assert payload["schema"] == CONTROL_REPLAY_SCHEMA
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert len(digest) == 64


def test_dbc_rejects_nonfinite_and_wrong_dims() -> None:
    times = _times()
    q, v, a, tau = _known_torque_trajectory(times)
    bad_q = q.copy()
    bad_q[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        recover_feasible_controls(
            _request(times=times, q=bad_q, v=v, a=a, tau_truth=tau)
        )
    with pytest.raises(ValueError, match="shape|dimension"):
        recover_feasible_controls(
            _request(
                times=times,
                q=q[:, :1],
                v=v,
                a=a,
                tau_truth=tau,
            )
        )


def test_evidence_fixture_matches_schema_when_present() -> None:
    if not EVIDENCE.is_file():
        pytest.skip("evidence written after GREEN implementation")
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert data["schema"] == CONTROL_REPLAY_SCHEMA
    assert data["governing_issue"] == 10610
    assert data["native_g1_pass"] is False
    assert data["claims_native_qualification"] is False
    assert data["qualification_blockers"]
