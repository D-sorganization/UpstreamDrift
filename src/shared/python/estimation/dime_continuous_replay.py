"""DIME Offline Smoothing and Independent Continuous Replay (#11421, #11430).

Provides:
1. ContinuousReplayOptions specifying declared controller, contact policy, model hash,
   and frozen numerical acceptance thresholds.
2. ReplayReceipt tracking reset count, assistance channels, root wrench integrity,
   and cryptographic provenance.
3. IndependentReplayMetrics separating optimization cost from independent replay metrics.
4. execute_continuous_replay performing single-shot forward integration from a saved
   initial state with saved controls, detecting undeclared root forces, per-frame resets,
   and unmodeled assistance channels.
5. smooth_backward_trajectory performing offline backward smoothing using marginalized
   arrival information while strictly forbidding reverse-time contact integration.
6. Public adapters connecting estimation results to Shadow Tracker and Simscape replay harnesses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import hashlib
from typing import Any
import numpy as np

from src.shared.python.contracts import PreconditionError, require
from src.shared.python.estimation.dime_contracts import (
    ContactPolicy,
    DimeCompleteState,
    DimeFullStepRequest,
    DimeDynamicsProvider,
)
from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    DimeProvenanceRecord,
    NumericAcceptanceThresholds,
    compute_alignment_metric,
)
from src.shared.python.estimation.moving_horizon import ArrivalFactor
from src.shared.python.motion_matching.simscape_replay_harness import (
    ContinuousReplayTrajectory,
)
from src.shared.python.motion_matching.provenance import git_commit_full
from src.shared.python.shadow_tracker.contracts import RolloutRequest

DIME_REPLAY_SCHEMA_VERSION = "dime-continuous-replay-receipt/1.0"


def _utc_now() -> datetime:
    """Current wall-clock time in UTC."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class ReplayProvenanceSources:
    """Where replay provenance comes from when no record is supplied (#11551).

    ``git_commit`` returns the full commit SHA or ``"unknown"``; ``clock``
    returns an aware datetime. Both are injectable so tests stay hermetic.
    """

    git_commit: Callable[[], str] = git_commit_full
    clock: Callable[[], datetime] = _utc_now

    def __post_init__(self) -> None:
        require(
            callable(self.git_commit) and callable(self.clock),
            "provenance sources must be callable",
        )


@dataclass(frozen=True)
class ContinuousReplayOptions:
    """Declared configuration for continuous forward replay."""

    declared_controller: str = "open_loop_feedforward"
    declared_contact_policy: ContactPolicy | str = ContactPolicy.NATIVE_ELIMINATED
    expected_model_hash: str | None = None
    allow_per_frame_resets: bool = False
    allow_hidden_feedback: bool = False
    allow_undeclared_root_forces: bool = False
    floating_base_root_dofs: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    tolerances: NumericAcceptanceThresholds = field(
        default_factory=NumericAcceptanceThresholds
    )
    integrator_name: str = "rk4"
    intermediate_resets: Sequence[tuple[int, DimeCompleteState]] | None = None
    provenance_sources: ReplayProvenanceSources = field(
        default_factory=ReplayProvenanceSources
    )


@dataclass(frozen=True)
class ReplayReceipt:
    """Structured receipt recording every reset and assistance channel during replay."""

    receipt_id: str
    schema_version: str
    model_hash: str
    provider_id: str
    integrator_name: str
    coverage_start_s: float
    coverage_end_s: float
    reset_count: int
    assistance_channels: tuple[str, ...]
    declared_controller: str
    declared_contact_policy: str
    has_undeclared_root_forces: bool
    is_physically_accepted: bool
    provenance: DimeProvenanceRecord

    def to_dict(self) -> dict[str, Any]:
        prov_dict = self.provenance.to_dict()
        return {
            "receipt_id": self.receipt_id,
            "schema_version": self.schema_version,
            "model_hash": self.model_hash,
            "provider_id": self.provider_id,
            "integrator_name": self.integrator_name,
            "coverage_start_s": self.coverage_start_s,
            "coverage_end_s": self.coverage_end_s,
            "reset_count": self.reset_count,
            "assistance_channels": list(self.assistance_channels),
            "declared_controller": self.declared_controller,
            "declared_contact_policy": self.declared_contact_policy,
            "has_undeclared_root_forces": self.has_undeclared_root_forces,
            "is_physically_accepted": self.is_physically_accepted,
            "provenance": prov_dict,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReplayReceipt:
        prov = DimeProvenanceRecord.from_dict(data["provenance"])
        return cls(
            receipt_id=str(data["receipt_id"]),
            schema_version=str(data["schema_version"]),
            model_hash=str(data["model_hash"]),
            provider_id=str(data["provider_id"]),
            integrator_name=str(data["integrator_name"]),
            coverage_start_s=float(data["coverage_start_s"]),
            coverage_end_s=float(data["coverage_end_s"]),
            reset_count=int(data["reset_count"]),
            assistance_channels=tuple(
                str(x) for x in data.get("assistance_channels", ())
            ),
            declared_controller=str(data["declared_controller"]),
            declared_contact_policy=str(data["declared_contact_policy"]),
            has_undeclared_root_forces=bool(data["has_undeclared_root_forces"]),
            is_physically_accepted=bool(data["is_physically_accepted"]),
            provenance=prov,
        )


@dataclass(frozen=True)
class IndependentReplayMetrics:
    """Independently recomputed replay metrics separated from optimization cost.

    ``reproducibility_error`` and ``alignment_metric`` are ``None`` when no
    reference trajectory was supplied: they were not measured (issue #11551).
    ``max_angular_drift_rad``, ``cancellation_ratio`` and
    ``grf_vertical_equilibrium_rms`` are ``None`` because the replay does not
    compute them yet: a literal ``0.0`` would read as a perfect result (#11545).
    """

    max_position_drift_m: float
    rms_position_drift_m: float
    max_velocity_drift: float
    max_angular_drift_rad: float | None
    alignment_metric: float | None
    cancellation_ratio: float | None
    reproducibility_error: float | None
    grf_vertical_equilibrium_rms: float | None = None
    satisfies_frozen_tolerances: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_position_drift_m": self.max_position_drift_m,
            "rms_position_drift_m": self.rms_position_drift_m,
            "max_velocity_drift": self.max_velocity_drift,
            "max_angular_drift_rad": self.max_angular_drift_rad,
            "alignment_metric": self.alignment_metric,
            "cancellation_ratio": self.cancellation_ratio,
            "reproducibility_error": self.reproducibility_error,
            "grf_vertical_equilibrium_rms": self.grf_vertical_equilibrium_rms,
            "satisfies_frozen_tolerances": self.satisfies_frozen_tolerances,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IndependentReplayMetrics:
        grf = data.get("grf_vertical_equilibrium_rms")
        ang = data.get("max_angular_drift_rad")
        canc = data.get("cancellation_ratio")
        align = data.get("alignment_metric")
        repro = data.get("reproducibility_error")
        return cls(
            max_position_drift_m=float(data["max_position_drift_m"]),
            rms_position_drift_m=float(data["rms_position_drift_m"]),
            max_velocity_drift=float(data["max_velocity_drift"]),
            max_angular_drift_rad=float(ang) if ang is not None else None,
            alignment_metric=float(align) if align is not None else None,
            cancellation_ratio=float(canc) if canc is not None else None,
            reproducibility_error=float(repro) if repro is not None else None,
            grf_vertical_equilibrium_rms=float(grf) if grf is not None else None,
            satisfies_frozen_tolerances=bool(data["satisfies_frozen_tolerances"]),
        )


@dataclass(frozen=True)
class ContinuousReplayResult:
    """Standardized result containing trajectory, independent metrics, and replay receipt."""

    trajectory: tuple[DimeCompleteState, ...]
    controls: np.ndarray
    receipt: ReplayReceipt
    metrics: IndependentReplayMetrics
    is_physically_accepted: bool
    provenance: DimeProvenanceRecord
    unqualified_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        rec_dict = self.receipt.to_dict()
        met_dict = self.metrics.to_dict()
        prov_dict = self.provenance.to_dict()
        return {
            "trajectory": [s.to_dict() for s in self.trajectory],
            "controls": self.controls.tolist(),
            "receipt": rec_dict,
            "metrics": met_dict,
            "is_physically_accepted": self.is_physically_accepted,
            "provenance": prov_dict,
            "unqualified_reasons": list(self.unqualified_reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContinuousReplayResult:
        traj = tuple(DimeCompleteState.from_dict(s) for s in data["trajectory"])
        ctrl = np.array(data["controls"], dtype=np.float64)
        rec = ReplayReceipt.from_dict(data["receipt"])
        met = IndependentReplayMetrics.from_dict(data["metrics"])
        prov = DimeProvenanceRecord.from_dict(data["provenance"])
        return cls(
            trajectory=traj,
            controls=ctrl,
            receipt=rec,
            metrics=met,
            is_physically_accepted=bool(data["is_physically_accepted"]),
            provenance=prov,
            unqualified_reasons=tuple(
                str(x) for x in data.get("unqualified_reasons", ())
            ),
        )


@dataclass(frozen=True)
class SmoothedTrajectoryResult:
    """Output of offline backward trajectory smoothing."""

    smoothed_states: tuple[DimeCompleteState, ...]
    arrival_factors: tuple[ArrivalFactor, ...]
    max_step_jump: float
    continuity_metric: float
    is_continuous: bool


def _validate_replay_inputs(
    provider: DimeDynamicsProvider,
    initial_state: DimeCompleteState,
    controls: np.ndarray,
    options: ContinuousReplayOptions,
    assistance_forces: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """Validate all replay preconditions and return sanitized controls and assistance list."""
    cap = provider.capability
    require(cap.status != "unavailable", "Provider is unavailable")

    intermediate_resets = options.intermediate_resets
    if intermediate_resets is not None and len(intermediate_resets) > 0:
        raise PreconditionError(
            "Per-frame state resets are strictly forbidden in continuous replay; "
            "replay must integrate continuously from single initial state."
        )

    expected_hash = options.expected_model_hash
    p_hash = provider.model_hash
    init_hash = initial_state.model_hash
    if expected_hash is not None:
        require(
            expected_hash == p_hash,
            f"Model hash mismatch: expected {expected_hash}, got {p_hash}",
        )
        require(
            expected_hash == init_hash,
            f"Model hash mismatch: expected {expected_hash}, got {init_hash}",
        )
    else:
        require(
            init_hash == p_hash,
            f"Model hash mismatch: initial_state ({init_hash}) != provider ({p_hash})",
        )

    assistance_channels: list[str] = []
    if assistance_forces is not None:
        af = np.asarray(assistance_forces, dtype=np.float64)
        if np.any(np.abs(af) > 1e-9):
            if not options.allow_hidden_feedback:
                raise PreconditionError(
                    "Hidden target-force feedback detected in continuous replay"
                )
            assistance_channels.append("hidden_target_force_feedback")

    ctrls = np.asarray(controls, dtype=np.float64)
    require(ctrls.ndim == 2, "controls must be a 2D array")
    require(ctrls.shape[0] >= 1, "controls must have >= 1 step")
    n_ch = len(cap.control_channels)
    require(
        ctrls.shape[1] == n_ch,
        f"Missing controls or dimension mismatch: expected {n_ch} channels, got {ctrls.shape[1]}",
    )

    fb_dofs = options.floating_base_root_dofs
    for dof in fb_dofs:
        if dof < ctrls.shape[1]:
            dof_vals = ctrls[:, dof]
            if np.any(np.abs(dof_vals) > 1e-9):
                if not options.allow_undeclared_root_forces:
                    raise PreconditionError(
                        f"Undeclared root wrench detected on unactuated root DoF {dof}"
                    )

    return ctrls, assistance_channels


def _rollout_continuous_replay(
    provider: DimeDynamicsProvider,
    initial_state: DimeCompleteState,
    controls: np.ndarray,
    dt: float,
) -> list[DimeCompleteState]:
    """Execute uninterrupted forward integration loop from initial state."""
    provider.set_state(initial_state)
    n_steps = controls.shape[0]
    p_hash = provider.model_hash
    states = [initial_state]
    curr = initial_state

    for k in range(n_steps):
        req = DimeFullStepRequest(
            state=curr,
            controls=controls[k],
            dt=dt,
            model_hash=p_hash,
        )
        res = provider.step(req)
        curr = res.next_state
        states.append(curr)

    return states


_GATE_METRIC_NAMES: tuple[str, ...] = (
    "max_angular_drift_rad",
    "cancellation_ratio",
    "grf_vertical_equilibrium_rms",
)


def _unmeasured_gate_metrics(metrics: IndependentReplayMetrics) -> tuple[str, ...]:
    """Names of acceptance metrics that were not measured (value is None)."""
    return tuple(n for n in _GATE_METRIC_NAMES if getattr(metrics, n) is None)


def _compute_replay_metrics(
    traj_states: list[DimeCompleteState],
    reference_trajectory: Sequence[DimeCompleteState] | None,
    options: ContinuousReplayOptions,
) -> tuple[IndependentReplayMetrics, bool]:
    """Compute independent trajectory comparison metrics and evaluate acceptance."""
    tol = options.tolerances
    init_state = traj_states[0]

    if reference_trajectory is not None:
        require(
            len(reference_trajectory) == len(traj_states),
            "reference_trajectory length must match replay trajectory",
        )
        errs = [
            float(np.linalg.norm(s.q - r.q))
            for s, r in zip(traj_states, reference_trajectory, strict=True)
        ]
        max_pos_drift = float(np.max(errs))
        rms_pos_drift = float(np.sqrt(np.mean(np.array(errs) ** 2)))
        reproducibility_err = max_pos_drift
        q_replay = np.array([s.q for s in traj_states])
        q_ref = np.array([s.q for s in reference_trajectory])
        align = compute_alignment_metric(q_replay, q_ref, policy="guarded_zero")
    else:
        init_q = init_state.q
        drift_init = [float(np.linalg.norm(s.q - init_q)) for s in traj_states]
        max_pos_drift = float(np.max(drift_init))
        rms_pos_drift = float(np.sqrt(np.mean(np.array(drift_init) ** 2)))
        # No reference: reproducibility and alignment were NOT measured.
        reproducibility_err = None
        align = None

    v_norms = [float(np.linalg.norm(s.v)) for s in traj_states]
    max_v = float(np.max(v_norms)) if v_norms else 0.0

    metrics = IndependentReplayMetrics(
        max_position_drift_m=max_pos_drift,
        rms_position_drift_m=rms_pos_drift,
        max_velocity_drift=max_v,
        max_angular_drift_rad=None,  # not computed: not measured
        alignment_metric=align,
        cancellation_ratio=None,  # not computed: not measured
        reproducibility_error=reproducibility_err,
        grf_vertical_equilibrium_rms=None,  # not computed: not measured
        satisfies_frozen_tolerances=False,
    )
    # Fail closed: without a reference there is nothing to qualify against, and a
    # frozen acceptance metric that was not measured cannot pass (#11545).
    satisfies_tol = (
        reproducibility_err is not None
        and not _unmeasured_gate_metrics(metrics)
        and bool(
            max_pos_drift <= tol.max_drift_m
            or reproducibility_err <= tol.reproducibility_atol
        )
    )
    return replace(metrics, satisfies_frozen_tolerances=satisfies_tol), satisfies_tol


def execute_continuous_replay(
    provider: DimeDynamicsProvider,
    initial_state: DimeCompleteState,
    controls: np.ndarray,
    dt: float,
    options: ContinuousReplayOptions | None = None,
    reference_trajectory: Sequence[DimeCompleteState] | None = None,
    assistance_forces: np.ndarray | None = None,
    provenance: DimeProvenanceRecord | None = None,
) -> ContinuousReplayResult:
    """Execute continuous forward replay from saved initial state and controls.

    Postconditions (issue #11551): without ``reference_trajectory`` the result is
    not physically accepted, ``unqualified_reasons`` says why and reproducibility
    is reported as not measured (``None``). When ``provenance`` is not supplied
    the git commit and timestamp come from ``options.provenance_sources`` (default
    :class:`ReplayProvenanceSources`: real git, ``"unknown"`` when git is
    unavailable, and the UTC wall clock); it is injectable so tests stay hermetic. ``reset_count`` counts the initial-state placement plus any
    declared intermediate resets (the latter are rejected during validation).
    """
    opts = options or ContinuousReplayOptions()
    ctrls, assistance_channels = _validate_replay_inputs(
        provider, initial_state, controls, opts, assistance_forces
    )

    src = opts.provenance_sources
    traj_states = _rollout_continuous_replay(provider, initial_state, ctrls, dt)
    metrics, satisfies_tol = _compute_replay_metrics(
        traj_states, reference_trajectory, opts
    )
    unqualified_reasons: tuple[str, ...] = ()
    if reference_trajectory is None:
        unqualified_reasons = (
            "No reference trajectory supplied: reproducibility not measured",
        )
    unmeasured = _unmeasured_gate_metrics(metrics)
    if unmeasured:
        unqualified_reasons += (
            "Acceptance metrics not measured: " + ", ".join(unmeasured),
        )

    cap = provider.capability
    p_hash = provider.model_hash
    prov = provenance or DimeProvenanceRecord(
        engine=cap.provider_id,
        engine_version=cap.version,
        model_hash=p_hash,
        param_hash="default-param-hash",
        git_commit=src.git_commit(),
        created_at=src.clock().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes="Independent continuous replay execution",
    )

    p_created = prov.created_at
    rid = f"receipt-replay-{hashlib.sha256(p_created.encode()).hexdigest()[:12]}"
    receipt = ReplayReceipt(
        receipt_id=rid,
        schema_version=DIME_REPLAY_SCHEMA_VERSION,
        model_hash=p_hash,
        provider_id=cap.provider_id,
        integrator_name=opts.integrator_name,
        coverage_start_s=float(traj_states[0].t),
        coverage_end_s=float(traj_states[-1].t),
        reset_count=1 + len(opts.intermediate_resets or ()),
        assistance_channels=tuple(assistance_channels),
        declared_controller=opts.declared_controller,
        declared_contact_policy=str(opts.declared_contact_policy),
        has_undeclared_root_forces=False,
        is_physically_accepted=satisfies_tol,
        provenance=prov,
    )

    return ContinuousReplayResult(
        trajectory=tuple(traj_states),
        controls=ctrls,
        receipt=receipt,
        metrics=metrics,
        is_physically_accepted=satisfies_tol,
        provenance=prov,
        unqualified_reasons=unqualified_reasons,
    )


def smooth_backward_trajectory(
    states: Sequence[DimeCompleteState],
    arrival_factors: Sequence[ArrivalFactor] | None = None,
    dt: float = 0.01,
    allow_reverse_contact: bool = False,
) -> SmoothedTrajectoryResult:
    """Perform offline backward trajectory smoothing."""
    require(not allow_reverse_contact, "Reverse-time contact integration is forbidden")
    require(dt > 0.0 and np.isfinite(dt), "dt must be strictly positive and finite")
    require(len(states) >= 1, "states must contain at least 1 sample")

    n = len(states)
    if n == 1:
        return SmoothedTrajectoryResult(
            smoothed_states=tuple(states),
            arrival_factors=tuple(arrival_factors or ()),
            max_step_jump=0.0,
            continuity_metric=0.0,
            is_continuous=True,
        )

    dim_q = states[0].q.size
    raw_q = np.array([s.q for s in states], dtype=np.float64)

    # Weights from arrival factors if supplied
    weights = np.ones(n, dtype=np.float64)
    if arrival_factors is not None and len(arrival_factors) == n:
        for i, af in enumerate(arrival_factors):
            si = af.sqrt_information
            w = float(np.trace(si.T @ si))
            weights[i] = float(np.clip(w, 0.1, 50.0))

    # Quadratic smoothing: min_q sum w_i (q_i - y_i)^2 + lambda sum (q_{i+1} - q_i)^2
    lam = 5.0 / (dt**2)
    smoothed_q = np.zeros_like(raw_q)

    # Tridiagonal system matrix for each coordinate
    a_mat = np.diag(weights)
    for i in range(n - 1):
        a_mat[i, i] += lam
        a_mat[i + 1, i + 1] += lam
        a_mat[i, i + 1] -= lam
        a_mat[i + 1, i] -= lam

    for d in range(dim_q):
        rhs = weights * raw_q[:, d]
        smoothed_q[:, d] = np.linalg.solve(a_mat, rhs)

    # Compute smoothed velocities via central differences
    smoothed_v = np.zeros((n, dim_q), dtype=np.float64)
    if n > 1:
        smoothed_v[0] = (smoothed_q[1] - smoothed_q[0]) / dt
        smoothed_v[-1] = (smoothed_q[-1] - smoothed_q[-2]) / dt
        for i in range(1, n - 1):
            smoothed_v[i] = (smoothed_q[i + 1] - smoothed_q[i - 1]) / (2.0 * dt)

    out_states: list[DimeCompleteState] = []
    for i, s in enumerate(states):
        out_states.append(
            DimeCompleteState(
                t=s.t,
                q=smoothed_q[i],
                v=smoothed_v[i],
                internal_state=dict(s.internal_state),
                model_hash=s.model_hash,
                units=dict(s.units),
            )
        )

    jumps = np.linalg.norm(np.diff(smoothed_q, axis=0), axis=1)
    max_jump = float(np.max(jumps)) if len(jumps) > 0 else 0.0
    mean_jump = float(np.mean(jumps)) if len(jumps) > 0 else 0.0
    is_cont = bool(max_jump < 0.25)

    return SmoothedTrajectoryResult(
        smoothed_states=tuple(out_states),
        arrival_factors=tuple(arrival_factors or ()),
        max_step_jump=max_jump,
        continuity_metric=mean_jump,
        is_continuous=is_cont,
    )


def to_shadow_tracker_rollout_request(
    initial_state: DimeCompleteState,
    controls: np.ndarray,
    times: Sequence[float],
) -> RolloutRequest:
    """Convert replay parameters to Shadow Tracker RolloutRequest."""
    ctrls = np.asarray(controls, dtype=np.float64)
    return RolloutRequest(
        initial_state=tuple(float(x) for x in initial_state.q),
        controls=tuple(tuple(float(x) for x in col) for col in ctrls.T),
        time_points_s=tuple(float(t) for t in times),
    )


def to_simscape_continuous_trajectory(
    replay_result: ContinuousReplayResult,
    coordinate_names: Sequence[str] | None = None,
    marker_labels: Sequence[str] | None = None,
) -> ContinuousReplayTrajectory:
    """Convert replay result to Simscape ContinuousReplayTrajectory."""
    traj = replay_result.trajectory
    n = len(traj)
    times = np.array([s.t for s in traj], dtype=np.float64)
    q = np.array([s.q for s in traj], dtype=np.float64)
    v = np.array([s.v for s in traj], dtype=np.float64)
    labels = tuple(marker_labels or ("marker_0",))
    bodies = tuple("body_0" for _ in labels)
    coords = tuple(coordinate_names or tuple(f"q_{i}" for i in range(q.shape[1])))

    return ContinuousReplayTrajectory(
        time_s=times,
        q=q,
        v=v,
        markers_m=np.zeros((n, len(labels), 3), dtype=np.float64),
        target_m=np.zeros((n, len(labels), 3), dtype=np.float64),
        valid=np.ones((n, len(labels)), dtype=bool),
        marker_labels=labels,
        marker_bodies=bodies,
        coordinate_names=coords,
        tau=replay_result.controls,
    )
