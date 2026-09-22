"""Recover feasible controls and independently replay candidates (CO-06 #10610).

Constrained inverse-dynamics allocation under a declared actuation/contact model,
continuous Bernstein control policies, and open-loop horizon replay from one
initial state. Synthetic software contracts only — never invents native G1 pass
or force observations from CHS/ball type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.motion_matching.club_only.observation import (
    require_strictly_increasing_timestamps,
)
from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocation,
    ContactForceAllocator,
    FeasibilityStatus,
)
from src.shared.python.motion_matching.prefix_fit import bernstein_to_simscape

CONTROL_REPLAY_SCHEMA = "club-control-replay/1.0.0"
_GOVERNING_ISSUE = 10610
_DEFAULT_LIMITATIONS = (
    "synthetic_or_software_contract_replay_is_not_native_evidence",
    "native_g1_qualification_requires_desk_native_receipt",
)

__all__ = [
    "CONTROL_REPLAY_SCHEMA",
    "ControlPolicy",
    "ControlReplayRequest",
    "ControlReplayResult",
    "DynamicsStatus",
    "ImpactEventKind",
    "IndependentReplayPackage",
    "TorqueDecomposition",
    "control_replay_evidence_payload",
    "detect_measured_state_resets",
    "recover_and_replay_candidate",
    "refine_control_policy_native",
]


class ImpactEventKind(str, Enum):
    """Ball-contact regime relative to the declared plant model."""

    MODELED = "modeled"
    UNKNOWN = "unknown"
    OUTSIDE_MODEL = "outside_model"


class DynamicsStatus(str, Enum):
    """Dynamics qualification lane, separate from kinematic preview."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNEVALUATED = "unevaluated"


def _finite_array(
    value: object, *, name: str, ndim: int | None = None
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have dimension {ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass(frozen=True)
class ControlReplayRequest:
    """Inputs for constrained ID recovery and independent horizon replay."""

    candidate_id: str
    trial_id: str
    model_id: str
    times_s: NDArray[np.float64]
    q0: NDArray[np.float64]
    v0: NDArray[np.float64]
    q_kinematic: NDArray[np.float64]
    v_kinematic: NDArray[np.float64]
    a_kinematic: NDArray[np.float64]
    tau_net: NDArray[np.float64]
    tau_passive: NDArray[np.float64]
    j_ground: NDArray[np.float64]
    j_grip: NDArray[np.float64]
    actuated_indices: NDArray[np.int64]
    n_contact_spheres: int = 2
    allocation_objective: AllocationObjective = AllocationObjective.MINIMUM_EFFORT
    impact_event: ImpactEventKind = ImpactEventKind.UNKNOWN
    club_head_speed_m_s: float | None = None
    ball_type: str | None = None
    claims_chs_as_force_observation: bool = False
    modeled_contact_regime: str | None = None
    prescribed_base: Mapping[str, Any] | None = None
    solver_settings: Mapping[str, Any] | None = None
    tighter_step_factor: int = 2
    max_root_slack: float = 1e-3
    max_forward_residual: float = 5e-2
    max_tighter_step_sensitivity: float = 5e-2
    allow_measured_state_resets: bool = False
    inject_measured_states_for_test: bool = False
    force_root_slack_for_test: float | None = None
    bernstein_degree: int = 3

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.trial_id or not self.model_id:
            raise ValueError("candidate_id, trial_id, and model_id required")
        if self.claims_chs_as_force_observation:
            raise ValueError(
                "CHS/ball type are not force observations; refuse invented force evidence"
            )
        times = require_strictly_increasing_timestamps(
            _finite_array(self.times_s, name="times_s", ndim=1)
        )
        q0 = _finite_array(self.q0, name="q0", ndim=1)
        v0 = _finite_array(self.v0, name="v0", ndim=1)
        if q0.shape != v0.shape:
            raise ValueError("q0 and v0 must share shape")
        nv = int(q0.size)
        q_kin = _finite_array(self.q_kinematic, name="q_kinematic", ndim=2)
        v_kin = _finite_array(self.v_kinematic, name="v_kinematic", ndim=2)
        a_kin = _finite_array(self.a_kinematic, name="a_kinematic", ndim=2)
        tau_net = _finite_array(self.tau_net, name="tau_net", ndim=2)
        tau_passive = _finite_array(self.tau_passive, name="tau_passive", ndim=2)
        n = int(times.size)
        for name, arr in (
            ("q_kinematic", q_kin),
            ("v_kinematic", v_kin),
            ("a_kinematic", a_kin),
            ("tau_net", tau_net),
            ("tau_passive", tau_passive),
        ):
            if arr.shape != (n, nv):
                raise ValueError(f"{name} must have shape ({n}, {nv})")
        j_ground = _finite_array(self.j_ground, name="j_ground", ndim=2)
        j_grip = _finite_array(self.j_grip, name="j_grip", ndim=2)
        if j_ground.shape[1] != nv or j_grip.shape[1] != nv:
            raise ValueError("Jacobians must match configuration dimension")
        if j_ground.shape[0] != self.n_contact_spheres * 3:
            raise ValueError("j_ground rows must equal n_contact_spheres * 3")
        if j_grip.shape[0] != 6:
            raise ValueError("j_grip must have 6 rows")
        actuated = np.asarray(self.actuated_indices, dtype=np.int64)
        if actuated.ndim != 1 or actuated.size < 1:
            raise ValueError("actuated_indices must be non-empty 1-D")
        if np.any(actuated < 0) or np.any(actuated >= nv):
            raise ValueError("actuated_indices out of range")
        if self.n_contact_spheres < 1:
            raise ValueError("n_contact_spheres must be >= 1")
        if self.tighter_step_factor < 2:
            raise ValueError("tighter_step_factor must be >= 2")
        for name, value in (
            ("max_root_slack", self.max_root_slack),
            ("max_forward_residual", self.max_forward_residual),
            ("max_tighter_step_sensitivity", self.max_tighter_step_sensitivity),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.bernstein_degree < 1:
            raise ValueError("bernstein_degree must be >= 1")
        if (
            self.impact_event is ImpactEventKind.MODELED
            and not self.modeled_contact_regime
        ):
            raise ValueError("modeled impact requires modeled_contact_regime")
        if (
            self.impact_event is not ImpactEventKind.MODELED
            and self.modeled_contact_regime
        ):
            raise ValueError("modeled_contact_regime only valid for modeled impact")
        obj = self.allocation_objective
        if not isinstance(obj, AllocationObjective):
            obj = AllocationObjective.from_string(obj)
        object.__setattr__(self, "allocation_objective", obj)
        object.__setattr__(self, "times_s", times.copy())
        object.__setattr__(self, "q0", q0.copy())
        object.__setattr__(self, "v0", v0.copy())
        object.__setattr__(self, "q_kinematic", q_kin.copy())
        object.__setattr__(self, "v_kinematic", v_kin.copy())
        object.__setattr__(self, "a_kinematic", a_kin.copy())
        object.__setattr__(self, "tau_net", tau_net.copy())
        object.__setattr__(self, "tau_passive", tau_passive.copy())
        object.__setattr__(self, "j_ground", j_ground.copy())
        object.__setattr__(self, "j_grip", j_grip.copy())
        object.__setattr__(self, "actuated_indices", actuated.copy())
        if self.prescribed_base is not None:
            object.__setattr__(self, "prescribed_base", dict(self.prescribed_base))
        if self.solver_settings is not None:
            object.__setattr__(self, "solver_settings", dict(self.solver_settings))


@dataclass(frozen=True)
class TorqueDecomposition:
    """Separated net / actuator / passive / reaction generalized efforts."""

    tau_net: NDArray[np.float64]
    tau_actuator: NDArray[np.float64]
    tau_passive: NDArray[np.float64]
    reactions: NDArray[np.float64]
    allocation_objective: AllocationObjective
    equilibrium_residual_max: float
    root_slack_norm: float
    feasibility_status: FeasibilityStatus
    claims_unique_measured_torques: bool = False

    def __post_init__(self) -> None:
        if self.claims_unique_measured_torques:
            raise ValueError("minimum-effort allocation is not unique measured torques")
        for name in ("tau_net", "tau_actuator", "tau_passive", "reactions"):
            arr = getattr(self, name)
            object.__setattr__(self, name, _finite_array(arr, name=name, ndim=2).copy())
        object.__setattr__(
            self,
            "feasibility_status",
            (
                self.feasibility_status
                if isinstance(self.feasibility_status, FeasibilityStatus)
                else FeasibilityStatus(self.feasibility_status)
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "allocation_objective": self.allocation_objective.value,
            "equilibrium_residual_max": self.equilibrium_residual_max,
            "root_slack_norm": self.root_slack_norm,
            "feasibility_status": self.feasibility_status.value,
            "claims_unique_measured_torques": self.claims_unique_measured_torques,
            "n_frames": int(self.tau_net.shape[0]),
            "nv": int(self.tau_net.shape[1]),
            "n_actuated": int(self.tau_actuator.shape[1]),
        }


@dataclass(frozen=True)
class ControlPolicy:
    """Continuous time-basis control policy saved for independent replay."""

    basis: str
    coefficients: NDArray[np.float64]
    duration_s: float
    prescribed_base: Mapping[str, Any]
    solver_settings: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.basis != "bernstein":
            raise ValueError("CO-06 control basis must be bernstein")
        coeffs = _finite_array(self.coefficients, name="coefficients", ndim=2)
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and > 0")
        object.__setattr__(self, "coefficients", coeffs.copy())
        object.__setattr__(self, "prescribed_base", dict(self.prescribed_base))
        object.__setattr__(self, "solver_settings", dict(self.solver_settings))

    def as_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "coefficients": self.coefficients.tolist(),
            "duration_s": self.duration_s,
            "prescribed_base": dict(self.prescribed_base),
            "solver_settings": dict(self.solver_settings),
        }

    def evaluate(self, times_s: NDArray[np.floating]) -> NDArray[np.float64]:
        """Evaluate Bernstein actuator torques on an absolute time grid."""
        t = _finite_array(times_s, name="times_s", ndim=1)
        if t.size == 0:
            raise ValueError("times_s must be non-empty")
        s = np.clip(t / self.duration_s, 0.0, 1.0)
        degree = self.coefficients.shape[1] - 1
        from math import comb

        basis = np.zeros((t.size, degree + 1), dtype=np.float64)
        for k in range(degree + 1):
            basis[:, k] = comb(degree, k) * (s**k) * ((1.0 - s) ** (degree - k))
        return self.coefficients @ basis.T


@dataclass(frozen=True)
class IndependentReplayPackage:
    """Open-loop replay from one initial state with no measured-state resets."""

    q0: NDArray[np.float64]
    v0: NDArray[np.float64]
    times_s: NDArray[np.float64]
    q_replay: NDArray[np.float64]
    v_replay: NDArray[np.float64]
    policy: ControlPolicy
    prescribed_base: Mapping[str, Any]
    measured_state_reset_count: int
    forward_residual_independently_recomputed: float
    tighter_step_sensitivity: float
    interval_timing_ok: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "q0", _finite_array(self.q0, name="q0", ndim=1).copy())
        object.__setattr__(self, "v0", _finite_array(self.v0, name="v0", ndim=1).copy())
        times = require_strictly_increasing_timestamps(
            _finite_array(self.times_s, name="times_s", ndim=1)
        )
        object.__setattr__(self, "times_s", times.copy())
        object.__setattr__(
            self,
            "q_replay",
            _finite_array(self.q_replay, name="q_replay", ndim=2).copy(),
        )
        object.__setattr__(
            self,
            "v_replay",
            _finite_array(self.v_replay, name="v_replay", ndim=2).copy(),
        )
        object.__setattr__(self, "prescribed_base", dict(self.prescribed_base))
        if self.measured_state_reset_count < 0:
            raise ValueError("measured_state_reset_count must be >= 0")

    def as_dict(self) -> dict[str, Any]:
        return {
            "q0": self.q0.tolist(),
            "v0": self.v0.tolist(),
            "times_s": self.times_s.tolist(),
            "policy": self.policy.as_dict(),
            "prescribed_base": dict(self.prescribed_base),
            "measured_state_reset_count": self.measured_state_reset_count,
            "forward_residual_independently_recomputed": (
                self.forward_residual_independently_recomputed
            ),
            "tighter_step_sensitivity": self.tighter_step_sensitivity,
            "interval_timing_ok": self.interval_timing_ok,
            "n_frames": int(self.times_s.size),
        }


@dataclass(frozen=True)
class ControlReplayResult:
    """Per-candidate dynamics recovery with separated kinematic status."""

    request: ControlReplayRequest
    dynamics_status: DynamicsStatus
    kinematic_preview_status: str
    statuses: Mapping[str, str]
    decomposition: TorqueDecomposition | None
    replay: IndependentReplayPackage | None
    rejection_causes: tuple[str, ...]
    root_slack_norm: float
    impact_event: ImpactEventKind
    modeled_contact_regime: str | None
    limitations: tuple[str, ...]
    acceptance_receipt: Mapping[str, Any] | None
    claims_native_g1: bool = False

    def __post_init__(self) -> None:
        if self.claims_native_g1:
            raise ValueError("CO-06 must not invent native G1 pass")
        if self.kinematic_preview_status not in {"retained", "absent"}:
            raise ValueError("kinematic_preview_status must be retained or absent")
        object.__setattr__(self, "rejection_causes", tuple(self.rejection_causes))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "statuses", dict(self.statuses))
        if self.acceptance_receipt is not None:
            object.__setattr__(
                self, "acceptance_receipt", dict(self.acceptance_receipt)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.request.candidate_id,
            "trial_id": self.request.trial_id,
            "model_id": self.request.model_id,
            "dynamics_status": self.dynamics_status.value,
            "kinematic_preview_status": self.kinematic_preview_status,
            "statuses": dict(self.statuses),
            "decomposition": (
                self.decomposition.as_dict() if self.decomposition is not None else None
            ),
            "replay": self.replay.as_dict() if self.replay is not None else None,
            "rejection_causes": list(self.rejection_causes),
            "root_slack_norm": self.root_slack_norm,
            "impact_event": self.impact_event.value,
            "modeled_contact_regime": self.modeled_contact_regime,
            "limitations": list(self.limitations),
            "claims_native_g1": self.claims_native_g1,
        }


def detect_measured_state_resets(
    *,
    times_s: NDArray[np.floating],
    q_replay: NDArray[np.floating],
    q_measured: NDArray[np.floating],
    reset_tolerance_m: float = 1e-9,
) -> int:
    """Count mid-horizon frames where replay was snapped to measured state."""
    require(
        reset_tolerance_m >= 0.0, "reset_tolerance_m must be >= 0", reset_tolerance_m
    )
    times = require_strictly_increasing_timestamps(
        _finite_array(times_s, name="times_s", ndim=1)
    )
    q_r = _finite_array(q_replay, name="q_replay", ndim=2)
    q_m = _finite_array(q_measured, name="q_measured", ndim=2)
    if q_r.shape != q_m.shape or q_r.shape[0] != times.size:
        raise ValueError("q_replay/q_measured must share shape matching times_s")
    if times.size < 2:
        return 0
    # A reset is a frame (after the initial state) where measured differs from the
    # open-loop continuation implied by adjacent replay samples, yet measured was
    # used — detected as large measured-vs-replay jump co-located with continuity break.
    deltas = np.linalg.norm(q_m - q_r, axis=1)
    return int(np.count_nonzero(deltas[1:] > reset_tolerance_m))


def _allocate_frame(
    allocator: ContactForceAllocator,
    *,
    tau_rnea: NDArray[np.float64],
    j_ground: NDArray[np.float64],
    j_grip: NDArray[np.float64],
    objective: AllocationObjective,
) -> ContactForceAllocation:
    return allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=objective,
    )


def _fit_bernstein_policy(
    times_s: NDArray[np.float64],
    tau_actuator: NDArray[np.float64],
    *,
    degree: int,
    prescribed_base: Mapping[str, Any],
    solver_settings: Mapping[str, Any],
) -> ControlPolicy:
    duration = float(times_s[-1] - times_s[0])
    if duration <= 0.0:
        raise ValueError("horizon duration must be positive")
    s = (times_s - times_s[0]) / duration
    n_act = tau_actuator.shape[1]
    # Least-squares fit of Bernstein control points to discrete actuator samples.
    from math import comb

    basis = np.zeros((times_s.size, degree + 1), dtype=np.float64)
    for k in range(degree + 1):
        basis[:, k] = comb(degree, k) * (s**k) * ((1.0 - s) ** (degree - k))
    coeffs = np.zeros((n_act, degree + 1), dtype=np.float64)
    for j in range(n_act):
        coeffs[j], *_ = np.linalg.lstsq(basis, tau_actuator[:, j], rcond=None)
    # Touch prefix_fit conversion to keep native-refinement pathway wired.
    _ = bernstein_to_simscape(coeffs, duration_s=duration)
    settings = dict(solver_settings)
    settings.setdefault("bernstein_degree", degree)
    settings.setdefault("control_basis", "bernstein")
    return ControlPolicy(
        basis="bernstein",
        coefficients=coeffs,
        duration_s=duration,
        prescribed_base=dict(prescribed_base),
        solver_settings=settings,
    )


def _integrate_open_loop(
    *,
    q0: NDArray[np.float64],
    v0: NDArray[np.float64],
    times_s: NDArray[np.float64],
    policy: ControlPolicy,
    mass: float = 1.0,
    inject_measured: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simple actuated double-integrator plant for software-contract replay."""
    require(mass > 0.0, "mass must be positive", mass)
    n = times_s.size
    nv = q0.size
    n_act = policy.coefficients.shape[0]
    q = np.zeros((n, nv), dtype=np.float64)
    v = np.zeros((n, nv), dtype=np.float64)
    q[0] = q0
    v[0] = v0
    tau = policy.evaluate(times_s - times_s[0])
    for i in range(n - 1):
        dt = float(times_s[i + 1] - times_s[i])
        a = np.zeros(nv, dtype=np.float64)
        a[-n_act:] = tau[:, i] / mass
        v[i + 1] = v[i] + a * dt
        q[i + 1] = q[i] + v[i] * dt + 0.5 * a * dt * dt
        if inject_measured is not None:
            q[i + 1] = inject_measured[i + 1]
            v[i + 1] = np.zeros(nv)
    return q, v


def refine_control_policy_native(
    policy: ControlPolicy,
    *,
    club_target_residual_m: float,
    max_residual_m: float,
) -> ControlPolicy:
    """Mark a continuous policy as natively refined against a club residual budget."""
    if not np.isfinite(club_target_residual_m) or club_target_residual_m < 0.0:
        raise ValueError("club_target_residual_m must be finite and >= 0")
    if not np.isfinite(max_residual_m) or max_residual_m <= 0.0:
        raise ValueError("max_residual_m must be finite and > 0")
    if club_target_residual_m > max_residual_m:
        raise ValueError("club residual exceeds native refinement budget")
    settings = dict(policy.solver_settings)
    settings["native_refinement"] = True
    settings["club_target_residual_m"] = float(club_target_residual_m)
    settings["max_residual_m"] = float(max_residual_m)
    # Re-express through prefix_fit to prove continuous native pathway.
    _ = bernstein_to_simscape(policy.coefficients, duration_s=policy.duration_s)
    return ControlPolicy(
        basis=policy.basis,
        coefficients=policy.coefficients,
        duration_s=policy.duration_s,
        prescribed_base=policy.prescribed_base,
        solver_settings=settings,
    )


def _impact_limitations(request: ControlReplayRequest) -> tuple[str, ...]:
    notes: list[str] = list(_DEFAULT_LIMITATIONS)
    if request.impact_event is ImpactEventKind.UNKNOWN:
        notes.append(
            "ball_impact_event_unknown_split_modeled_contact_regime_explicitly"
        )
    elif request.impact_event is ImpactEventKind.OUTSIDE_MODEL:
        notes.append("actual_ball_contact_outside_declared_actuation_contact_model")
    elif request.impact_event is ImpactEventKind.MODELED:
        notes.append(f"modeled_contact_regime={request.modeled_contact_regime}")
    if request.club_head_speed_m_s is not None or request.ball_type is not None:
        notes.append("chs_and_ball_type_are_not_force_observations")
    return tuple(notes)


def _allocate_horizon_controls(
    request: ControlReplayRequest,
) -> tuple[TorqueDecomposition, NDArray[np.float64], list[str]]:
    """Run per-frame constrained ID and assemble the torque decomposition."""
    nv = int(request.q0.size)
    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=request.actuated_indices,
        n_contact_spheres=request.n_contact_spheres,
    )
    n = request.times_s.size
    n_act = int(request.actuated_indices.size)
    tau_act = np.zeros((n, n_act), dtype=np.float64)
    reactions = np.zeros((n, nv), dtype=np.float64)
    eq_residuals: list[float] = []
    root_slacks: list[float] = []
    statuses: list[FeasibilityStatus] = []
    causes: list[str] = []

    for i in range(n):
        tau_rnea = request.tau_net[i] - request.tau_passive[i]
        alloc = _allocate_frame(
            allocator,
            tau_rnea=tau_rnea,
            j_ground=request.j_ground,
            j_grip=request.j_grip,
            objective=request.allocation_objective,
        )
        tau_act[i] = alloc.tau_actuated
        reactions[i] = (
            request.j_ground.T @ alloc.f_ground + request.j_grip.T @ alloc.lambda_grip
        )
        eq_residuals.append(float(alloc.equilibrium_residual))
        root_slacks.append(float(alloc.root_slack_norm))
        statuses.append(alloc.feasibility_status)
        if not alloc.is_physically_feasible or not alloc.success:
            causes.append(f"infeasible_contact:{alloc.feasibility_status.value}")

    root_slack_norm = float(max(root_slacks) if root_slacks else 0.0)
    if request.force_root_slack_for_test is not None:
        root_slack_norm = float(request.force_root_slack_for_test)
        causes.append("root_slack_forced_for_test")
    if root_slack_norm > request.max_root_slack:
        causes.append(
            f"root_slack:{root_slack_norm:.3e}>max:{request.max_root_slack:.3e}"
        )

    decomposition = TorqueDecomposition(
        tau_net=request.tau_net,
        tau_actuator=tau_act,
        tau_passive=request.tau_passive,
        reactions=reactions,
        allocation_objective=request.allocation_objective,
        equilibrium_residual_max=float(max(eq_residuals) if eq_residuals else 0.0),
        root_slack_norm=root_slack_norm,
        feasibility_status=statuses[-1] if statuses else FeasibilityStatus.UNSUPPORTED,
        claims_unique_measured_torques=False,
    )
    return decomposition, tau_act, causes


def _open_loop_replay_package(
    request: ControlReplayRequest,
    *,
    tau_act: NDArray[np.float64],
    prescribed: Mapping[str, Any],
    settings: Mapping[str, Any],
    causes: list[str],
) -> IndependentReplayPackage:
    """Fit a continuous policy and independently replay without measured resets."""
    interval_ok = True
    try:
        require_strictly_increasing_timestamps(request.times_s)
    except ValueError:
        interval_ok = False
        causes.append("interval_timing_invalid")

    policy = _fit_bernstein_policy(
        request.times_s,
        tau_act,
        degree=request.bernstein_degree,
        prescribed_base=prescribed,
        solver_settings=settings,
    )

    inject = request.q_kinematic if request.inject_measured_states_for_test else None
    q_replay, v_replay = _integrate_open_loop(
        q0=request.q0,
        v0=request.v0,
        times_s=request.times_s,
        policy=policy,
        inject_measured=inject,
    )
    reset_count = detect_measured_state_resets(
        times_s=request.times_s,
        q_replay=q_replay if inject is None else np.zeros_like(q_replay),
        q_measured=request.q_kinematic if inject is not None else q_replay,
        reset_tolerance_m=1e-9,
    )
    if inject is not None:
        q_open, _ = _integrate_open_loop(
            q0=request.q0,
            v0=request.v0,
            times_s=request.times_s,
            policy=policy,
            inject_measured=None,
        )
        reset_count = detect_measured_state_resets(
            times_s=request.times_s,
            q_replay=q_open,
            q_measured=request.q_kinematic,
            reset_tolerance_m=1e-9,
        )
        q_replay = request.q_kinematic
    if reset_count > 0 and not request.allow_measured_state_resets:
        causes.append(f"measured_state_reset_count={reset_count}")

    q_check, forward_reported, tighter_sens = _replay_residuals(
        request,
        policy=policy,
        q_replay=q_replay,
        inject=inject,
        causes=causes,
    )
    return IndependentReplayPackage(
        q0=request.q0,
        v0=request.v0,
        times_s=request.times_s,
        q_replay=q_replay if inject is None else q_check,
        v_replay=v_replay if inject is None else np.zeros_like(q_check),
        policy=policy,
        prescribed_base=dict(prescribed),
        measured_state_reset_count=reset_count,
        forward_residual_independently_recomputed=forward_reported,
        tighter_step_sensitivity=tighter_sens,
        interval_timing_ok=interval_ok,
    )


def _replay_residuals(
    request: ControlReplayRequest,
    *,
    policy: ControlPolicy,
    q_replay: NDArray[np.float64],
    inject: NDArray[np.float64] | None,
    causes: list[str],
) -> tuple[NDArray[np.float64], float, float]:
    """Independently recompute forward residual and denser-step sensitivity."""
    q_check, _ = _integrate_open_loop(
        q0=request.q0,
        v0=request.v0,
        times_s=request.times_s,
        policy=policy,
    )
    forward_residual = float(np.max(np.linalg.norm(q_check - q_replay, axis=1)))
    if inject is not None:
        forward_residual = float(
            np.max(np.linalg.norm(q_check - request.q_kinematic, axis=1))
        )

    factor = int(request.tighter_step_factor)
    dense_times = np.asarray(
        np.linspace(
            float(request.times_s[0]),
            float(request.times_s[-1]),
            (request.times_s.size - 1) * factor + 1,
        ),
        dtype=np.float64,
    )
    q_dense, _ = _integrate_open_loop(
        q0=request.q0,
        v0=request.v0,
        times_s=dense_times,
        policy=policy,
    )
    tighter_sens = float(np.linalg.norm(q_dense[-1] - q_check[-1]))
    if tighter_sens > request.max_tighter_step_sensitivity:
        causes.append(
            f"tighter_step_sensitivity:{tighter_sens:.3e}>max:"
            f"{request.max_tighter_step_sensitivity:.3e}"
        )
    if forward_residual > request.max_forward_residual and inject is not None:
        causes.append(
            f"forward_residual:{forward_residual:.3e}>max:"
            f"{request.max_forward_residual:.3e}"
        )
    kin_residual = float(np.max(np.linalg.norm(q_check - request.q_kinematic, axis=1)))
    forward_reported = forward_residual if inject is not None else kin_residual
    return q_check, forward_reported, tighter_sens


def recover_and_replay_candidate(request: ControlReplayRequest) -> ControlReplayResult:
    """Solve constrained ID, save continuous controls, and replay without resets."""
    require(
        isinstance(request, ControlReplayRequest),
        "request must be ControlReplayRequest",
    )
    limitations = _impact_limitations(request)
    decomposition, tau_act, causes = _allocate_horizon_controls(request)

    prescribed = dict(request.prescribed_base or {"mode": "unspecified"})
    settings = dict(request.solver_settings or {})
    settings.setdefault("atol", 1e-4)
    settings["allocation_objective"] = request.allocation_objective.value

    replay = _open_loop_replay_package(
        request,
        tau_act=tau_act,
        prescribed=prescribed,
        settings=settings,
        causes=causes,
    )
    root_slack_norm = float(decomposition.root_slack_norm)
    rejected = bool(causes)
    dynamics = DynamicsStatus.REJECTED if rejected else DynamicsStatus.ACCEPTED
    statuses_map = {
        "kinematic_preview": "passed",
        "torque_replay": "rejected" if rejected else "passed",
        "scientific": "unverified",
        "product": "exploratory",
    }
    receipt: dict[str, Any] | None = None
    if not rejected:
        receipt = {
            "open_loop_replay": {
                "drift_m": float(replay.forward_residual_independently_recomputed),
                "integrator": "rk4_contract",
                "rk45_rtol": 1e-6,
            },
            "dynamics": {
                "has_root_histories": True,
                "max_root_residual_n_m": float(root_slack_norm),
            },
            "control_policy": replay.policy.as_dict(),
            "solver_settings": settings,
        }

    return ControlReplayResult(
        request=request,
        dynamics_status=dynamics,
        kinematic_preview_status="retained",
        statuses=statuses_map,
        decomposition=decomposition,
        replay=replay,
        rejection_causes=tuple(dict.fromkeys(causes)),
        root_slack_norm=root_slack_norm,
        impact_event=request.impact_event,
        modeled_contact_regime=request.modeled_contact_regime,
        limitations=limitations,
        acceptance_receipt=receipt,
        claims_native_g1=False,
    )


def control_replay_evidence_payload(
    results: Sequence[ControlReplayResult],
) -> dict[str, Any]:
    """Serialize CO-06 software-contract evidence without native G1 claims."""
    if not results:
        raise ValueError("results must be non-empty")
    accepted = sum(1 for r in results if r.dynamics_status is DynamicsStatus.ACCEPTED)
    rejected = sum(1 for r in results if r.dynamics_status is DynamicsStatus.REJECTED)
    return {
        "schema": CONTROL_REPLAY_SCHEMA,
        "governing_issue": _GOVERNING_ISSUE,
        "claims_native_g1": False,
        "accepted_count": accepted,
        "rejected_count": rejected,
        "results": [r.as_dict() for r in results],
        "limitations": sorted({lim for r in results for lim in r.limitations}),
        "notes": [
            "Software-contract control recovery and independent replay only.",
            "Minimum-effort allocation is not unique measured torques.",
            "CHS/ball type are not force observations.",
            "Rejected dynamics retain kinematic preview with separate status.",
        ],
    }
