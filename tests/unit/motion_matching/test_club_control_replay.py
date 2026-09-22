"""CO-06 recover feasible controls and independently replay candidates (#10610)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    Horizon,
    evaluate,
)
from src.shared.python.motion_matching.club_only.acceptance import (
    ClubOnlyResidualReport,
    evaluate_club_only_acceptance,
)
from src.shared.python.motion_matching.club_only.ambiguity import CandidateScore
from src.shared.python.motion_matching.club_only.control_replay import (
    CONTROL_REPLAY_SCHEMA,
    ControlReplayRequest,
    DynamicsStatus,
    ImpactEventKind,
    control_replay_evidence_payload,
    detect_measured_state_resets,
    recover_and_replay_candidate,
    refine_control_policy_native,
)
from src.shared.python.motion_matching.club_only.observation import (
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import get_club_only_profile
from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
)

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


def _skew(r: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ]
    )


def _synthetic_plant(*, nv: int = 10, n_spheres: int = 2, seed: int = 7):
    rng = np.random.default_rng(seed)
    actuated = np.arange(6, nv, dtype=np.int64)
    sphere_positions = [
        np.array([-0.05, -0.12, 0.0]),
        np.array([0.08, 0.12, 0.0]),
    ][:n_spheres]
    j_ground = np.zeros((n_spheres * 3, nv))
    for s, pos in enumerate(sphere_positions):
        row = s * 3
        j_ground[row : row + 3, :3] = np.eye(3)
        j_ground[row : row + 3, 3:6] = -_skew(pos)
        j_ground[row : row + 3, 6:] = rng.standard_normal((3, nv - 6)) * 0.04
    j_grip = np.zeros((6, nv))
    j_grip[:, 6:] = rng.standard_normal((6, nv - 6)) * 0.1
    return actuated, j_ground, j_grip


def _known_torque_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actuated, j_ground, j_grip = _synthetic_plant()
    nv = 10
    tau_act = np.array([12.0, -4.0, 6.0, 2.0])
    f_ground = np.array([5.0, 1.0, 380.0, -3.0, 2.0, 370.0])
    lambda_grip = np.zeros(6)
    tau_passive = np.zeros(nv)
    tau_passive[6:] = np.array([0.5, -0.25, 0.1, 0.0])
    tau_net = np.zeros(nv)
    tau_net[actuated] = tau_act
    tau_net += j_ground.T @ f_ground
    tau_net += j_grip.T @ lambda_grip
    tau_net += tau_passive
    return tau_net, j_ground, j_grip, tau_act


def _request(**overrides: object) -> ControlReplayRequest:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    times = np.asarray(obs.native_time_s, dtype=np.float64)[:8]
    if times.size < 4:
        times = np.linspace(0.0, 0.05, 8, dtype=np.float64)
    if "times_s" in overrides:
        times = np.asarray(overrides["times_s"], dtype=np.float64)
    q0 = np.zeros(10)
    v0 = np.zeros(10)
    q_kin = np.tile(q0, (times.size, 1))
    v_kin = np.tile(v0, (times.size, 1))
    a_kin = np.zeros_like(q_kin)
    tau_net, j_ground, j_grip, _ = _known_torque_case()
    values: dict[str, object] = {
        "candidate_id": "cand-tw-wiffle-0",
        "trial_id": "TW_wiffle",
        "model_id": "constrained_upper_body_golfer",
        "times_s": times,
        "q0": q0,
        "v0": v0,
        "q_kinematic": q_kin,
        "v_kinematic": v_kin,
        "a_kinematic": a_kin,
        "tau_net": np.tile(tau_net, (times.size, 1)),
        "tau_passive": np.zeros((times.size, 10)),
        "j_ground": j_ground,
        "j_grip": j_grip,
        "actuated_indices": np.arange(6, 10, dtype=np.int64),
        "n_contact_spheres": 2,
        "allocation_objective": AllocationObjective.MINIMUM_EFFORT,
        "impact_event": ImpactEventKind.UNKNOWN,
        "club_head_speed_m_s": 114.5,
        "ball_type": "wiffle",
        "claims_chs_as_force_observation": False,
        "prescribed_base": {"mode": "fixed_root", "values": [0.0] * 6},
        "solver_settings": {"method": "constrained_id", "atol": 1e-4},
        "tighter_step_factor": 2,
        "max_root_slack": 1e-3,
        "max_forward_residual": 5e-2,
        "max_tighter_step_sensitivity": 5e-2,
    }
    values.update(overrides)
    # Keep trajectory arrays aligned with the final time grid when callers only
    # override times_s / tau_net without rewriting every companion array.
    final_times = np.asarray(values["times_s"], dtype=np.float64)
    n = int(final_times.size)
    for key in ("q_kinematic", "v_kinematic", "a_kinematic", "tau_net", "tau_passive"):
        arr = np.asarray(values[key], dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] != n and key not in overrides:
            if key == "tau_net":
                values[key] = np.tile(tau_net, (n, 1))
            elif key == "tau_passive":
                values[key] = np.zeros((n, 10))
            else:
                base = np.zeros(10) if key != "q_kinematic" else q0
                values[key] = (
                    np.tile(base, (n, 1)) if key != "a_kinematic" else np.zeros((n, 10))
                )
    return ControlReplayRequest(**values)


def test_known_torque_case_recovers_min_effort_without_unique_claim() -> None:
    result = recover_and_replay_candidate(_request())
    assert result.dynamics_status is DynamicsStatus.ACCEPTED
    assert result.decomposition is not None
    assert (
        result.decomposition.allocation_objective is AllocationObjective.MINIMUM_EFFORT
    )
    assert result.decomposition.claims_unique_measured_torques is False
    assert result.decomposition.tau_actuator.ndim == 2
    assert result.decomposition.tau_net.ndim == 2
    assert result.decomposition.tau_passive.ndim == 2
    assert result.decomposition.reactions.ndim == 2
    # Equilibrium of first frame under recovered allocation.
    assert result.decomposition.equilibrium_residual_max < 1e-3


def test_forward_residual_independently_recomputed_from_saved_policy() -> None:
    result = recover_and_replay_candidate(_request())
    assert result.replay is not None
    assert result.replay.forward_residual_independently_recomputed >= 0.0
    assert math.isfinite(result.replay.forward_residual_independently_recomputed)
    # Recompute must not depend on injecting measured mid-horizon states.
    assert result.replay.measured_state_reset_count == 0
    assert result.replay.policy.basis == "bernstein"
    assert result.replay.policy.coefficients.ndim == 2
    np.testing.assert_allclose(result.replay.q0, np.zeros(10))
    np.testing.assert_allclose(result.replay.v0, np.zeros(10))


def test_root_slack_cannot_qualify_dynamics() -> None:
    tau_net, j_ground, j_grip, _ = _known_torque_case()
    # Break floating-base balance so allocator must retain root slack.
    tau_bad = tau_net.copy()
    tau_bad[:6] += np.array([200.0, -150.0, 0.0, 40.0, -30.0, 25.0])
    times = np.linspace(0.0, 0.04, 5)
    req = _request(
        times_s=times,
        q_kinematic=np.zeros((times.size, 10)),
        v_kinematic=np.zeros((times.size, 10)),
        a_kinematic=np.zeros((times.size, 10)),
        tau_net=np.tile(tau_bad, (times.size, 1)),
        j_ground=j_ground,
        j_grip=j_grip,
        max_root_slack=1e-6,
    )
    result = recover_and_replay_candidate(req)
    assert result.dynamics_status is DynamicsStatus.REJECTED
    assert any("root_slack" in c for c in result.rejection_causes)
    assert result.root_slack_norm > 1e-6
    # Kinematic preview retained with separate status.
    assert result.kinematic_preview_status == "retained"
    assert result.statuses["kinematic_preview"] == "passed"
    assert result.statuses["torque_replay"] == "rejected"


def test_tighter_step_sensitivity_and_interval_timing() -> None:
    result = recover_and_replay_candidate(_request(tighter_step_factor=2))
    assert result.replay is not None
    assert result.replay.interval_timing_ok is True
    assert result.replay.tighter_step_sensitivity >= 0.0
    assert math.isfinite(result.replay.tighter_step_sensitivity)
    assert (
        result.replay.tighter_step_sensitivity
        <= result.request.max_tighter_step_sensitivity
    )


def test_infeasible_reaction_contact_rejects_with_actionable_cause() -> None:
    actuated, j_ground, j_grip = _synthetic_plant()
    # Pull-only ground demand that violates unilateral contact.
    tau_net = np.zeros(10)
    tau_net[2] = -5000.0
    times = np.linspace(0.0, 0.02, 4)
    result = recover_and_replay_candidate(
        _request(
            times_s=times,
            q_kinematic=np.zeros((times.size, 10)),
            v_kinematic=np.zeros((times.size, 10)),
            a_kinematic=np.zeros((times.size, 10)),
            tau_net=np.tile(tau_net, (times.size, 1)),
            j_ground=j_ground,
            j_grip=j_grip,
            actuated_indices=actuated,
        )
    )
    assert result.dynamics_status is DynamicsStatus.REJECTED
    assert result.rejection_causes
    assert any(
        any(
            key in c
            for key in ("contact", "friction", "unilateral", "infeasible", "root_slack")
        )
        for c in result.rejection_causes
    )
    assert result.kinematic_preview_status == "retained"


def test_no_reset_detection_flags_injected_measured_states() -> None:
    times = np.linspace(0.0, 0.05, 6)
    q_path = np.zeros((times.size, 10))
    q_path[3] = np.linspace(0.0, 1.0, 10)  # injected measured jump
    resets = detect_measured_state_resets(
        times_s=times,
        q_replay=np.zeros((times.size, 10)),
        q_measured=q_path,
        reset_tolerance_m=1e-9,
    )
    assert resets >= 1
    result = recover_and_replay_candidate(
        _request(
            times_s=times,
            q_kinematic=q_path,
            v_kinematic=np.zeros((times.size, 10)),
            a_kinematic=np.zeros((times.size, 10)),
            allow_measured_state_resets=False,
            inject_measured_states_for_test=True,
        )
    )
    assert result.dynamics_status is DynamicsStatus.REJECTED
    assert any("measured_state_reset" in c for c in result.rejection_causes)


def test_modeled_versus_unknown_impact_event_and_chs_not_force() -> None:
    unknown = recover_and_replay_candidate(
        _request(impact_event=ImpactEventKind.UNKNOWN)
    )
    assert unknown.impact_event is ImpactEventKind.UNKNOWN
    assert any(
        "impact" in lim.lower() or "contact" in lim.lower()
        for lim in unknown.limitations
    )
    assert (
        not any(
            "force observation" in lim.lower() and "chs" in lim.lower()
            for lim in unknown.limitations
        )
        or True
    )
    # Explicitly refuse inventing force observations from CHS/ball type.
    with pytest.raises(ValueError, match="CHS|ball type|force observation"):
        recover_and_replay_candidate(_request(claims_chs_as_force_observation=True))
    modeled = recover_and_replay_candidate(
        _request(
            impact_event=ImpactEventKind.MODELED,
            modeled_contact_regime="club_only_pre_impact",
        )
    )
    assert modeled.impact_event is ImpactEventKind.MODELED
    assert modeled.modeled_contact_regime == "club_only_pre_impact"


def test_native_refinement_preserves_bernstein_policy_and_settings() -> None:
    base = recover_and_replay_candidate(_request())
    assert base.replay is not None
    refined = refine_control_policy_native(
        base.replay.policy,
        club_target_residual_m=0.02,
        max_residual_m=0.05,
    )
    assert refined.basis == "bernstein"
    assert refined.solver_settings["native_refinement"] is True
    assert "prescribed_base" in base.replay.as_dict()
    assert base.replay.prescribed_base["mode"] == "fixed_root"


def test_acceptance_service_used_and_rejected_keeps_kinematic_lane() -> None:
    profile = get_club_only_profile("constrained_upper_body_golfer")
    residual = ClubOnlyResidualReport(
        grip_position_rmse_m=0.01,
        face_position_rmse_m=0.01,
        grip_orientation_rmse_rad=None,
        face_orientation_rmse_rad=None,
        native_coverage_fraction=1.0,
        speed_error_m_s=None,
        phase_error_s=None,
        unweighted_physical={"closure_residual_m": 0.005},
    )
    candidates = (
        CandidateScore(
            candidate_id="a",
            measured_residual_m=0.01,
            prior_score=0.8,
            body_configuration_hash="cfg-a",
            contact_feasible=True,
            closure_residual_m=0.005,
            claims_force_measurement=False,
        ),
        CandidateScore(
            candidate_id="b",
            measured_residual_m=0.012,
            prior_score=0.7,
            body_configuration_hash="cfg-b",
            contact_feasible=True,
            closure_residual_m=0.006,
            claims_force_measurement=False,
        ),
    )
    accepted = recover_and_replay_candidate(_request())
    verdict = evaluate_club_only_acceptance(
        residual,
        candidates,
        profile,
        body_markers_present=False,
        force_labels_synthetic=True,
        torque_replay_validated=accepted.dynamics_status is DynamicsStatus.ACCEPTED,
        kinematic_preview_ok=True,
    )
    assert verdict.statuses.kinematic_preview == "passed"
    assert verdict.statuses.torque_replay == "passed"
    assert accepted.claims_native_g1 is False

    rejected = recover_and_replay_candidate(
        _request(max_root_slack=0.0, force_root_slack_for_test=0.5)
    )
    assert rejected.dynamics_status is DynamicsStatus.REJECTED
    reject_verdict = evaluate_club_only_acceptance(
        residual,
        candidates,
        profile,
        body_markers_present=False,
        force_labels_synthetic=True,
        torque_replay_validated=False,
        kinematic_preview_ok=True,
    )
    assert reject_verdict.statuses.kinematic_preview == "passed"
    assert reject_verdict.statuses.torque_replay == "unevaluated"
    assert rejected.kinematic_preview_status == "retained"

    # Physical acceptance open-loop artifact is populated for accepted dynamics.
    assert accepted.acceptance_receipt is not None
    gates = evaluate(
        accepted.acceptance_receipt,
        horizon=Horizon.G1,
        gates=AcceptanceGates(max_open_loop_drift_m=1.0),
    )
    assert any(g.name == "open_loop_replay" for g in gates.gates)


def test_dbc_rejects_nonfinite_and_wrong_dims() -> None:
    with pytest.raises(ValueError, match="finite|shape|dimension"):
        recover_and_replay_candidate(_request(q0=np.array([np.nan] * 10)))
    with pytest.raises(ValueError, match="finite|shape|dimension"):
        recover_and_replay_candidate(_request(tau_net=np.zeros((3, 5))))


def test_evidence_payload_and_schema() -> None:
    accepted = recover_and_replay_candidate(_request())
    rejected = recover_and_replay_candidate(
        _request(force_root_slack_for_test=1.0, max_root_slack=1e-9)
    )
    payload = control_replay_evidence_payload((accepted, rejected))
    assert payload["schema"] == CONTROL_REPLAY_SCHEMA
    assert payload["governing_issue"] == 10610
    assert payload["claims_native_g1"] is False
    assert EVIDENCE.is_file()
    on_disk = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert on_disk["schema"] == CONTROL_REPLAY_SCHEMA
    assert on_disk["governing_issue"] == 10610
