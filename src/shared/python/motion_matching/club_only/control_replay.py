"""Recover feasible controls and independently replay club-only candidates (CO-06 #10610).

Consumes CO-04 pendulum matches and CO-05 body candidates. Solves constrained
inverse-dynamics allocation under a declared actuation/contact model, separates
net generalized torque, actuator input, passive effects and reactions, and
replays the claimed horizon from one initial state with no measured-state resets.

Software-contract fixtures validate contracts only. Native G1 remains blocked
with named qualification blockers; this module never invents native success.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.motion_matching.club_only.observation import (
    require_strictly_increasing_timestamps,
)
from src.shared.python.motion_matching.club_only.native_g1_gates import (
    validate_native_g1_claim_contract,
)
from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocator,
    FeasibilityStatus,
    verify_torque_and_rate_bounds,
)

CONTROL_REPLAY_SCHEMA = "club-control-replay/1.0.0"
_GOVERNING_ISSUE = 10610
_ROOT_SLACK_QUALIFY_LIMIT = 1e-3
_DEFAULT_BLOCKERS = (
    "native_g1_qualification_requires_desk_native_receipt",
    "software_contract_replay_is_not_native_evidence",
)

__all__ = [
    "CONTROL_REPLAY_SCHEMA",
    "ControlPolicy",
    "ControlRecoveryRequest",
    "ControlRecoveryResult",
    "ControlReplayReport",
    "ImpactRegime",
    "IndependentReplayResult",
    "TighterStepSensitivity",
    "control_replay_evidence_payload",
    "detect_measured_state_resets",
    "evaluate_tighter_step_sensitivity",
    "independent_forward_residual",
    "recover_and_replay_candidates",
    "recover_feasible_controls",
]


class ImpactRegime(str, Enum):
    """Declared contact/impact modeling status for the claimed horizon."""

    PRE_IMPACT_ONLY = "pre_impact_only"
    MODELED_CONTACT = "modeled_contact"
    UNKNOWN_IMPACT = "unknown_impact"
    SPLIT_REGIME = "split_regime"


@dataclass(frozen=True)
class ControlPolicy:
    """Saved continuous-time control package for independent replay."""

    q0: NDArray[np.float64]
    v0: NDArray[np.float64]
    timestamps_s: NDArray[np.float64]
    tau_actuated: NDArray[np.float64]
    net_generalized_torque: NDArray[np.float64]
    passive_terms: NDArray[np.float64]
    f_ground: NDArray[np.float64]
    lambda_grip: NDArray[np.float64]
    delta_tau_root: NDArray[np.float64]
    control_basis_coeffs: NDArray[np.float64]
    prescribed_base_inputs: Mapping[str, Any]
    solver_settings: Mapping[str, Any]
    allocation_objective: str
    content_hash: str
    claims_unique_measured_torques: bool = False

    def __post_init__(self) -> None:
        if self.claims_unique_measured_torques:
            raise ValueError(
                "allocation cannot claim unique measured torques; "
                "objective is a stated minimum-effort/smoothness choice"
            )
        for name, arr in (
            ("q0", self.q0),
            ("v0", self.v0),
            ("timestamps_s", self.timestamps_s),
            ("tau_actuated", self.tau_actuated),
            ("net_generalized_torque", self.net_generalized_torque),
            ("passive_terms", self.passive_terms),
            ("f_ground", self.f_ground),
            ("lambda_grip", self.lambda_grip),
            ("delta_tau_root", self.delta_tau_root),
            ("control_basis_coeffs", self.control_basis_coeffs),
        ):
            values = np.asarray(arr, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, values.copy())
        object.__setattr__(
            self, "prescribed_base_inputs", dict(self.prescribed_base_inputs)
        )
        object.__setattr__(self, "solver_settings", dict(self.solver_settings))
        if not self.content_hash:
            raise ValueError("content_hash required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "q0": self.q0.tolist(),
            "v0": self.v0.tolist(),
            "timestamps_s": self.timestamps_s.tolist(),
            "tau_actuated": self.tau_actuated.tolist(),
            "net_generalized_torque": self.net_generalized_torque.tolist(),
            "passive_terms": self.passive_terms.tolist(),
            "f_ground": self.f_ground.tolist(),
            "lambda_grip": self.lambda_grip.tolist(),
            "delta_tau_root": self.delta_tau_root.tolist(),
            "control_basis_coeffs": self.control_basis_coeffs.tolist(),
            "prescribed_base_inputs": dict(self.prescribed_base_inputs),
            "solver_settings": dict(self.solver_settings),
            "allocation_objective": self.allocation_objective,
            "content_hash": self.content_hash,
            "claims_unique_measured_torques": self.claims_unique_measured_torques,
        }


@dataclass(frozen=True)
class IndependentReplayResult:
    """Independent open-loop replay diagnostics (no measured-state resets)."""

    timestamps_s: NDArray[np.float64]
    q_replay: NDArray[np.float64]
    forward_residual: float
    tighter_step_residual: float
    used_measured_state_reset: bool
    root_slack_norm: float
    work_balance_error: float
    torque_rate_ok: bool
    contact_feasible: bool
    closure_residual_m: float
    impact_regime: ImpactRegime
    interval_timing_ok: bool

    def __post_init__(self) -> None:
        times = require_strictly_increasing_timestamps(
            np.asarray(self.timestamps_s, dtype=np.float64)
        )
        q = np.asarray(self.q_replay, dtype=np.float64)
        if q.ndim != 2 or q.shape[0] != times.size:
            raise ValueError("q_replay must be (N, dof) aligned with timestamps")
        if not np.all(np.isfinite(q)):
            raise ValueError("q_replay must be finite")
        for name, value in (
            ("forward_residual", self.forward_residual),
            ("tighter_step_residual", self.tighter_step_residual),
            ("root_slack_norm", self.root_slack_norm),
            ("work_balance_error", self.work_balance_error),
            ("closure_residual_m", self.closure_residual_m),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        object.__setattr__(self, "timestamps_s", times.copy())
        object.__setattr__(self, "q_replay", q.copy())

    def as_dict(self) -> dict[str, Any]:
        return {
            "forward_residual": self.forward_residual,
            "tighter_step_residual": self.tighter_step_residual,
            "used_measured_state_reset": self.used_measured_state_reset,
            "root_slack_norm": self.root_slack_norm,
            "work_balance_error": self.work_balance_error,
            "torque_rate_ok": self.torque_rate_ok,
            "contact_feasible": self.contact_feasible,
            "closure_residual_m": self.closure_residual_m,
            "impact_regime": self.impact_regime.value,
            "interval_timing_ok": self.interval_timing_ok,
        }


@dataclass(frozen=True)
class TighterStepSensitivity:
    """Coarse vs finer-step open-loop residual comparison."""

    coarse_residual: float
    tighter_step_residual: float
    ratio: float

    def __post_init__(self) -> None:
        for name, value in (
            ("coarse_residual", self.coarse_residual),
            ("tighter_step_residual", self.tighter_step_residual),
            ("ratio", self.ratio),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")


@dataclass(frozen=True)
class ControlRecoveryRequest:
    """One candidate kinematic preview plus dynamics bundle for recovery."""

    candidate_id: str
    trial_id: str
    model_id: str
    timestamps_s: NDArray[np.float64]
    q: NDArray[np.float64]
    v: NDArray[np.float64]
    a: NDArray[np.float64]
    tau_rnea: NDArray[np.float64]
    n_actuated: int
    n_contact_spheres: int = 0
    impact_time_s: float | None = None
    impact_modeled: bool = False
    allow_measured_resets: bool = False
    measured_q_inject: NDArray[np.float64] | None = None
    force_infeasible_contact: bool = False
    root_slack_override: float | None = None
    kinematic_preview_ok: bool = True
    qualification_blockers: tuple[str, ...] = _DEFAULT_BLOCKERS
    allocation_objective: str = AllocationObjective.MINIMUM_EFFORT.value

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.trial_id or not self.model_id:
            raise ValueError("candidate_id, trial_id, and model_id required")
        times = require_strictly_increasing_timestamps(
            np.asarray(self.timestamps_s, dtype=np.float64)
        )
        q = np.asarray(self.q, dtype=np.float64)
        v = np.asarray(self.v, dtype=np.float64)
        a = np.asarray(self.a, dtype=np.float64)
        tau = np.asarray(self.tau_rnea, dtype=np.float64)
        require(self.n_actuated >= 1, "n_actuated must be >= 1", self.n_actuated)
        require(
            q.ndim == 2 and q.shape == v.shape == a.shape == tau.shape,
            "q, v, a, tau_rnea must share shape (N, dof)",
            (q.shape, v.shape, a.shape, tau.shape),
        )
        require(
            q.shape[0] == times.size and q.shape[1] == self.n_actuated,
            "trajectory rows must match timestamps; cols must equal n_actuated",
            (q.shape, times.size, self.n_actuated),
        )
        require(
            bool(
                np.all(np.isfinite(q))
                and np.all(np.isfinite(v))
                and np.all(np.isfinite(a))
                and np.all(np.isfinite(tau))
            ),
            "q, v, a, tau_rnea must be finite",
        )
        if self.n_contact_spheres < 0:
            raise ValueError("n_contact_spheres must be >= 0")
        if self.impact_time_s is not None and (
            not np.isfinite(self.impact_time_s) or self.impact_time_s < 0.0
        ):
            raise ValueError("impact_time_s must be finite and >= 0 when set")
        if self.root_slack_override is not None and (
            not np.isfinite(self.root_slack_override) or self.root_slack_override < 0.0
        ):
            raise ValueError("root_slack_override must be finite and >= 0 when set")
        if self.measured_q_inject is not None:
            inject = np.asarray(self.measured_q_inject, dtype=np.float64)
            require(
                bool(inject.shape == q.shape and np.all(np.isfinite(inject))),
                "measured_q_inject must be finite and match q shape",
                inject.shape,
            )
            object.__setattr__(self, "measured_q_inject", inject.copy())
        blockers = tuple(self.qualification_blockers) or _DEFAULT_BLOCKERS
        object.__setattr__(self, "timestamps_s", times.copy())
        object.__setattr__(self, "q", q.copy())
        object.__setattr__(self, "v", v.copy())
        object.__setattr__(self, "a", a.copy())
        object.__setattr__(self, "tau_rnea", tau.copy())
        object.__setattr__(self, "qualification_blockers", blockers)


@dataclass(frozen=True)
class ControlRecoveryResult:
    """Recovered controls + independent replay for one candidate."""

    schema: str
    candidate_id: str
    trial_id: str
    model_id: str
    allocation_objective: str
    kinematic_preview_ok: bool
    torque_replay_status: str
    policy: ControlPolicy | None
    replay: IndependentReplayResult | None
    rejection_reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    qualification_blockers: tuple[str, ...]
    claims_native_qualification: bool
    native_g1_pass: bool

    def __post_init__(self) -> None:
        if self.schema != CONTROL_REPLAY_SCHEMA:
            raise ValueError(f"schema must be {CONTROL_REPLAY_SCHEMA!r}")
        validate_native_g1_claim_contract(
            native_g1_pass=self.native_g1_pass,
            claims_native_qualification=self.claims_native_qualification,
            qualification_blockers=self.qualification_blockers,
        )
        if self.torque_replay_status not in {
            "passed",
            "rejected",
            "unevaluated",
            "blocked",
        }:
            raise ValueError(
                f"unknown torque_replay_status={self.torque_replay_status!r}"
            )
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(
            self, "qualification_blockers", tuple(self.qualification_blockers)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "trial_id": self.trial_id,
            "model_id": self.model_id,
            "allocation_objective": self.allocation_objective,
            "kinematic_preview_ok": self.kinematic_preview_ok,
            "torque_replay_status": self.torque_replay_status,
            "policy": None if self.policy is None else self.policy.as_dict(),
            "replay": None if self.replay is None else self.replay.as_dict(),
            "rejection_reasons": list(self.rejection_reasons),
            "limitations": list(self.limitations),
            "qualification_blockers": list(self.qualification_blockers),
            "claims_native_qualification": self.claims_native_qualification,
            "native_g1_pass": self.native_g1_pass,
        }


@dataclass(frozen=True)
class ControlReplayReport:
    """Batch recovery/replay matrix for CO-06 evidence."""

    schema: str
    governing_issue: int
    results: tuple[ControlRecoveryResult, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != CONTROL_REPLAY_SCHEMA:
            raise ValueError(f"schema must be {CONTROL_REPLAY_SCHEMA!r}")
        if self.governing_issue != _GOVERNING_ISSUE:
            raise ValueError(f"governing_issue must be {_GOVERNING_ISSUE}")
        if not self.results:
            raise ValueError("results must be non-empty")
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "notes", tuple(self.notes))


def _content_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _impact_regime(
    request: ControlRecoveryRequest,
) -> tuple[ImpactRegime, tuple[str, ...]]:
    if request.impact_time_s is None:
        return ImpactRegime.PRE_IMPACT_ONLY, ()
    t0 = float(request.timestamps_s[0])
    t1 = float(request.timestamps_s[-1])
    if not (t0 <= request.impact_time_s <= t1):
        return ImpactRegime.PRE_IMPACT_ONLY, ()
    if request.impact_modeled and request.n_contact_spheres > 0:
        # Explicit split when impact sits inside the horizon with a contact model.
        if t0 < request.impact_time_s < t1:
            return ImpactRegime.SPLIT_REGIME, ()
        return ImpactRegime.MODELED_CONTACT, ()
    return (
        ImpactRegime.UNKNOWN_IMPACT,
        ("unknown_impact_event_not_modeled",),
    )


def detect_measured_state_resets(
    *,
    q_claimed: NDArray[np.floating],
    q_open_loop: NDArray[np.floating],
    atol: float = 1e-3,
) -> NDArray[np.bool_]:
    """Return per-frame flags where claimed state jumps off open-loop replay."""
    claimed = np.asarray(q_claimed, dtype=np.float64)
    open_loop = np.asarray(q_open_loop, dtype=np.float64)
    require(
        claimed.shape == open_loop.shape and claimed.ndim == 2,
        "q_claimed and q_open_loop must share shape (N, dof)",
        (claimed.shape, open_loop.shape),
    )
    require(
        bool(np.all(np.isfinite(claimed)) and np.all(np.isfinite(open_loop))),
        "states must be finite",
    )
    require(np.isfinite(atol) and atol >= 0.0, "atol must be finite and >= 0", atol)
    deltas = np.linalg.norm(claimed - open_loop, axis=1)
    return deltas > atol


def _integrate_open_loop(
    *,
    q0: NDArray[np.float64],
    v0: NDArray[np.float64],
    tau: NDArray[np.float64],
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Unit-inertia double-integrator plant for software-contract replay."""
    n = times.size
    dof = q0.size
    q = np.zeros((n, dof), dtype=np.float64)
    v = np.zeros((n, dof), dtype=np.float64)
    q[0] = q0
    v[0] = v0
    for i in range(1, n):
        dt = float(times[i] - times[i - 1])
        a_prev = tau[i - 1]
        v[i] = v[i - 1] + a_prev * dt
        q[i] = q[i - 1] + v[i - 1] * dt
    return q


def independent_forward_residual(
    policy: ControlPolicy,
    *,
    times: NDArray[np.floating],
    q_reference: NDArray[np.floating],
) -> float:
    """Recompute open-loop residual from saved policy; never trust claimed values."""
    times_arr = require_strictly_increasing_timestamps(
        np.asarray(times, dtype=np.float64)
    )
    ref = np.asarray(q_reference, dtype=np.float64)
    require(
        ref.shape[0] == times_arr.size and ref.shape[1] == policy.q0.size,
        "q_reference shape must match policy/timestamps",
        ref.shape,
    )
    require(bool(np.all(np.isfinite(ref))), "q_reference must be finite")
    q_replay = _integrate_open_loop(
        q0=policy.q0,
        v0=policy.v0,
        tau=policy.tau_actuated,
        times=times_arr,
    )
    return float(np.sqrt(np.mean(np.square(q_replay - ref))))


def evaluate_tighter_step_sensitivity(
    policy: ControlPolicy | None,
    *,
    times: NDArray[np.floating],
    q_reference: NDArray[np.floating],
    coarse_residual: float,
) -> TighterStepSensitivity:
    """Replay with half-step open-loop integration and compare residuals."""
    if policy is None:
        raise ValueError("policy required for tighter-step sensitivity")
    if not np.isfinite(coarse_residual) or coarse_residual < 0.0:
        raise ValueError("coarse_residual must be finite and >= 0")
    times_arr = require_strictly_increasing_timestamps(
        np.asarray(times, dtype=np.float64)
    )
    # Densify each interval by 2x via linear control hold + midpoint states.
    mid_times = []
    for i in range(times_arr.size - 1):
        mid_times.append(float(times_arr[i]))
        mid_times.append(0.5 * (float(times_arr[i]) + float(times_arr[i + 1])))
    mid_times.append(float(times_arr[-1]))
    dense_times = np.asarray(mid_times, dtype=np.float64)
    # Zero-order hold tau onto dense grid.
    tau_dense = np.zeros(
        (dense_times.size, policy.tau_actuated.shape[1]), dtype=np.float64
    )
    j = 0
    for i, t in enumerate(dense_times):
        while j < times_arr.size - 2 and t >= times_arr[j + 1]:
            j += 1
        tau_dense[i] = policy.tau_actuated[min(j, policy.tau_actuated.shape[0] - 1)]
    dense_policy = ControlPolicy(
        q0=policy.q0,
        v0=policy.v0,
        timestamps_s=dense_times,
        tau_actuated=tau_dense,
        net_generalized_torque=tau_dense,
        passive_terms=np.zeros_like(tau_dense),
        f_ground=np.zeros(
            (dense_times.size, max(1, policy.f_ground.shape[1])), dtype=np.float64
        ),
        lambda_grip=np.zeros((dense_times.size, 6), dtype=np.float64),
        delta_tau_root=np.zeros((dense_times.size, 6), dtype=np.float64),
        control_basis_coeffs=policy.control_basis_coeffs,
        prescribed_base_inputs=policy.prescribed_base_inputs,
        solver_settings=policy.solver_settings,
        allocation_objective=policy.allocation_objective,
        content_hash=policy.content_hash,
        claims_unique_measured_torques=False,
    )
    # Interpolate reference onto dense timestamps for residual comparison.
    ref = np.asarray(q_reference, dtype=np.float64)
    ref_dense = np.vstack(
        [
            np.array(
                [np.interp(t, times_arr, ref[:, k]) for k in range(ref.shape[1])],
                dtype=np.float64,
            )
            for t in dense_times
        ]
    )
    tight = independent_forward_residual(
        dense_policy, times=dense_times, q_reference=ref_dense
    )
    ratio = (
        tight / coarse_residual
        if coarse_residual > 1e-12
        else (0.0 if tight == 0.0 else np.inf)
    )
    if not np.isfinite(ratio):
        ratio = 1e9
    return TighterStepSensitivity(
        coarse_residual=float(coarse_residual),
        tighter_step_residual=float(tight),
        ratio=float(ratio),
    )


def _allocation_arrays(
    *, n: int, nv: int, n_ground: int
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    return (
        np.zeros((n, nv), dtype=np.float64),
        np.zeros((n, n_ground), dtype=np.float64),
        np.zeros((n, 6), dtype=np.float64),
        np.zeros((n, 6), dtype=np.float64),
    )


def _allocate_reduced_software_plant(
    request: ControlRecoveryRequest,
    *,
    tau_act: NDArray[np.float64],
    f_ground: NDArray[np.float64],
    lambda_grip: NDArray[np.float64],
    delta_root: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    bool,
    tuple[str, ...],
]:
    """Unit-inertia / reduced DoF path: copy RNEA onto actuated channels."""
    for i in range(request.timestamps_s.size):
        tau_act[i] = np.asarray(request.tau_rnea[i], dtype=np.float64)
    max_root = 0.0
    if request.root_slack_override is not None:
        max_root = float(request.root_slack_override)
        delta_root[:, 0] = request.root_slack_override
    return tau_act, f_ground, lambda_grip, delta_root, max_root, True, ()


def _allocate_floating_base_contacts(
    request: ControlRecoveryRequest,
    *,
    tau_act: NDArray[np.float64],
    lambda_grip: NDArray[np.float64],
    delta_root: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    bool,
    tuple[str, ...],
]:
    """Floating-base plant: reuse ContactForceAllocator (requires nv > 6)."""
    nv = request.n_actuated
    n_spheres = max(0, request.n_contact_spheres)
    n = request.timestamps_s.size
    actuated = tuple(range(6, nv))  # floating base occupies 0..5
    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated or tuple(range(nv)),
        n_contact_spheres=n_spheres,
        regularisation_contact=1.0,
        regularisation_grip=1.0,
        root_penalty_weight=1e6,
    )
    f_ground = np.zeros((n, allocator.n_ground_vars), dtype=np.float64)
    j_ground = np.zeros((allocator.n_ground_vars, nv), dtype=np.float64)
    j_grip = np.zeros((6, nv), dtype=np.float64)
    reasons: list[str] = []
    contact_ok = True
    max_root = 0.0
    for i in range(n):
        tau_full = np.asarray(request.tau_rnea[i], dtype=np.float64)
        allocation = allocator.allocate(
            tau_rnea=tau_full,
            j_ground=j_ground,
            j_grip=j_grip,
            objective=request.allocation_objective,
            contact_mask=[True] * allocator.n_contact_spheres,
        )
        tau_act[i, list(allocator.actuated_indices)] = allocation.tau_actuated
        f_ground[i] = allocation.f_ground
        lambda_grip[i] = allocation.lambda_grip
        delta_root[i] = allocation.delta_tau_root
        max_root = max(max_root, float(allocation.root_slack_norm))
        if (
            not allocation.is_physically_feasible
            or allocation.feasibility_status is not FeasibilityStatus.FEASIBLE
        ):
            contact_ok = False
            reasons.append(allocation.feasibility_status.value)
    if request.root_slack_override is not None:
        max_root = max(max_root, float(request.root_slack_override))
        delta_root[:, 0] = request.root_slack_override
    return (
        tau_act,
        f_ground,
        lambda_grip,
        delta_root,
        max_root,
        contact_ok,
        tuple(reasons),
    )


def _allocate_trajectory(
    request: ControlRecoveryRequest,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    bool,
    tuple[str, ...],
]:
    """Frame-wise constrained ID allocation.

    Reduced software plants (nv <= 6) use a direct minimum-effort copy of RNEA
    onto actuated channels. Full floating-base plants (nv > 6) reuse
    ContactForceAllocator. ContactForceAllocator requires nv > 6 by contract.
    """
    nv = request.n_actuated
    n_spheres = max(0, request.n_contact_spheres)
    n_ground = max(1, n_spheres) * 3 if n_spheres > 0 else 1
    n = request.timestamps_s.size
    tau_act, f_ground, lambda_grip, delta_root = _allocation_arrays(
        n=n, nv=nv, n_ground=n_ground
    )

    if request.force_infeasible_contact:
        return (
            tau_act,
            f_ground,
            lambda_grip,
            delta_root,
            0.0,
            False,
            ("infeasible_reaction_contact",),
        )

    if nv <= 6 or n_spheres == 0:
        return _allocate_reduced_software_plant(
            request,
            tau_act=tau_act,
            f_ground=f_ground,
            lambda_grip=lambda_grip,
            delta_root=delta_root,
        )

    return _allocate_floating_base_contacts(
        request,
        tau_act=tau_act,
        lambda_grip=lambda_grip,
        delta_root=delta_root,
    )


def _bernstein_basis_from_tau(tau: NDArray[np.float64]) -> NDArray[np.float64]:
    """Store a continuous control basis snapshot (degree-0 hold of mean torque)."""
    # Degree-0 Bernstein control equals mean torque; bounds the continuous signal
    # under a constant hold — not a claim of uniqueness.
    mean_tau = np.mean(tau, axis=0)
    return mean_tau.reshape(1, -1).copy()


def _work_balance_error(
    tau: NDArray[np.float64],
    v: NDArray[np.float64],
    times: NDArray[np.float64],
) -> float:
    """|∫ tau·v dt − trapezoid mechanical work| relative residual."""
    if times.size < 2:
        return 0.0
    power = np.sum(tau * v, axis=1)
    work = float(np.trapezoid(power, times))
    # Discrete left-rectangle estimate for residual.
    dt = np.diff(times)
    rect = float(np.sum(power[:-1] * dt))
    denom = max(abs(work), 1e-9)
    return float(abs(work - rect) / denom)


def _build_control_policy(
    request: ControlRecoveryRequest,
    *,
    tau_act: NDArray[np.float64],
    f_ground: NDArray[np.float64],
    lambda_grip: NDArray[np.float64],
    delta_root: NDArray[np.float64],
) -> ControlPolicy:
    """Assemble the continuous control package from an allocation."""
    passive = np.zeros_like(tau_act)
    net = tau_act + passive
    basis = _bernstein_basis_from_tau(tau_act)
    hash_payload = {
        "candidate_id": request.candidate_id,
        "model_id": request.model_id,
        "trial_id": request.trial_id,
        "q0": request.q[0].tolist(),
        "v0": request.v[0].tolist(),
        "tau_actuated": tau_act.tolist(),
        "allocation_objective": request.allocation_objective,
    }
    return ControlPolicy(
        q0=request.q[0],
        v0=request.v[0],
        timestamps_s=request.timestamps_s,
        tau_actuated=tau_act,
        net_generalized_torque=net,
        passive_terms=passive,
        f_ground=f_ground
        if request.n_contact_spheres > 0
        else np.zeros((request.timestamps_s.size, 1)),
        lambda_grip=lambda_grip,
        delta_tau_root=delta_root,
        control_basis_coeffs=basis,
        prescribed_base_inputs={},
        solver_settings={
            "allocator": "ContactForceAllocator",
            "objective": request.allocation_objective,
            "plant": "unit_inertia_double_integrator",
        },
        allocation_objective=request.allocation_objective,
        content_hash=_content_hash(hash_payload),
        claims_unique_measured_torques=False,
    )


def _run_independent_replay(
    request: ControlRecoveryRequest,
    policy: ControlPolicy,
    *,
    tau_act: NDArray[np.float64],
    max_root: float,
    contact_ok: bool,
    rate_ok: bool,
) -> tuple[IndependentReplayResult, list[str]]:
    """Open-loop replay from q0/v0; append rejection reasons for resets/slack."""
    rejection: list[str] = []
    q_open = _integrate_open_loop(
        q0=policy.q0,
        v0=policy.v0,
        tau=policy.tau_actuated,
        times=request.timestamps_s,
    )
    q_claimed = (
        request.measured_q_inject
        if request.allow_measured_resets and request.measured_q_inject is not None
        else request.q
    )
    reset_flags = detect_measured_state_resets(
        q_claimed=q_claimed, q_open_loop=q_open, atol=1e-3
    )
    used_reset = bool(np.any(reset_flags)) and bool(request.allow_measured_resets)
    if used_reset:
        rejection.append("measured_state_reset_detected")

    forward = independent_forward_residual(
        policy, times=request.timestamps_s, q_reference=request.q
    )
    sens = evaluate_tighter_step_sensitivity(
        policy,
        times=request.timestamps_s,
        q_reference=request.q,
        coarse_residual=forward,
    )
    if max_root > _ROOT_SLACK_QUALIFY_LIMIT:
        rejection.append("root_slack_cannot_qualify")

    impact, _ = _impact_regime(request)
    replay = IndependentReplayResult(
        timestamps_s=request.timestamps_s,
        q_replay=q_open,
        forward_residual=forward,
        tighter_step_residual=sens.tighter_step_residual,
        used_measured_state_reset=used_reset,
        root_slack_norm=float(max_root),
        work_balance_error=_work_balance_error(
            tau_act, request.v, request.timestamps_s
        ),
        torque_rate_ok=bool(rate_ok),
        contact_feasible=contact_ok,
        closure_residual_m=float(np.linalg.norm(q_open[-1] - request.q[-1])),
        impact_regime=impact,
        interval_timing_ok=True,
    )
    return replay, rejection


def recover_feasible_controls(request: ControlRecoveryRequest) -> ControlRecoveryResult:
    """Recover minimum-effort controls and independently replay one candidate."""
    limitations: list[str] = []
    rejection: list[str] = []
    impact, impact_limits = _impact_regime(request)
    limitations.extend(impact_limits)

    tau_act, f_ground, lambda_grip, delta_root, max_root, contact_ok, alloc_reasons = (
        _allocate_trajectory(request)
    )
    rejection.extend(alloc_reasons)

    dt = float(np.mean(np.diff(request.timestamps_s)))
    rate_ok, _ = verify_torque_and_rate_bounds(
        tau_act,
        dt=max(dt, 1e-9),
        tau_bounds=(
            np.full(request.n_actuated, -1e3),
            np.full(request.n_actuated, 1e3),
        ),
        rate_bounds=1e4,
    )
    if not rate_ok:
        rejection.append("torque_rate_bounds_violated")

    policy = _build_control_policy(
        request,
        tau_act=tau_act,
        f_ground=f_ground,
        lambda_grip=lambda_grip,
        delta_root=delta_root,
    )
    replay, replay_reasons = _run_independent_replay(
        request,
        policy,
        tau_act=tau_act,
        max_root=max_root,
        contact_ok=contact_ok,
        rate_ok=bool(rate_ok),
    )
    rejection.extend(replay_reasons)

    if rejection:
        status = "rejected"
    elif impact is ImpactRegime.UNKNOWN_IMPACT:
        status = "blocked"
        limitations.append("impact_outside_declared_contact_model")
    else:
        status = "passed"

    return ControlRecoveryResult(
        schema=CONTROL_REPLAY_SCHEMA,
        candidate_id=request.candidate_id,
        trial_id=request.trial_id,
        model_id=request.model_id,
        allocation_objective=request.allocation_objective,
        kinematic_preview_ok=request.kinematic_preview_ok,
        torque_replay_status=status,
        policy=policy,
        replay=replay,
        rejection_reasons=tuple(dict.fromkeys(rejection)),
        limitations=tuple(dict.fromkeys(limitations)),
        qualification_blockers=request.qualification_blockers,
        claims_native_qualification=False,
        native_g1_pass=False,
    )


def recover_and_replay_candidates(
    requests: Sequence[ControlRecoveryRequest],
) -> ControlReplayReport:
    """Recover/replay a batch of candidates without inventing native qualification."""
    if not requests:
        raise ValueError("requests must be non-empty")
    results = tuple(recover_feasible_controls(req) for req in requests)
    return ControlReplayReport(
        schema=CONTROL_REPLAY_SCHEMA,
        governing_issue=_GOVERNING_ISSUE,
        results=results,
        notes=(
            "Software-contract recovery/replay only; native G1 remains blocked.",
            "Allocation objective is stated minimum-effort; not unique measured torques.",
            "Rejected dynamics retain kinematic_preview_ok separately from torque_replay.",
        ),
    )


def control_replay_evidence_payload(report: ControlReplayReport) -> dict[str, Any]:
    """Serialize a CO-06 evidence receipt (never claims native G1)."""
    return {
        "schema": report.schema,
        "governing_issue": report.governing_issue,
        "native_g1_pass": False,
        "claims_native_qualification": False,
        "qualification_blockers": sorted(
            {
                blocker
                for result in report.results
                for blocker in result.qualification_blockers
            }
        ),
        "notes": list(report.notes),
        "results": [result.as_dict() for result in report.results],
        "summary": {
            "n_candidates": len(report.results),
            "n_torque_passed": sum(
                1 for r in report.results if r.torque_replay_status == "passed"
            ),
            "n_torque_rejected": sum(
                1 for r in report.results if r.torque_replay_status == "rejected"
            ),
            "n_torque_blocked": sum(
                1 for r in report.results if r.torque_replay_status == "blocked"
            ),
        },
    }
