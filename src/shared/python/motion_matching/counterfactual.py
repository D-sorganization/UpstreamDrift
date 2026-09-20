"""Counterfactual semantics and acceleration decomposition (MV-06, #10482).

Provides:
- Exact instantaneous acceleration decomposition (gravity, drift, control).
- Zero-Torque Counterfactual (ZTCF) and Zero-Velocity Counterfactual (ZVCF).
- Counterfactual rollout fork generation with strict baseline immutability.
- Divergence measurement and acceptance verification against biomechanical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Sequence
import uuid

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition

if TYPE_CHECKING:
    from src.shared.python.motion_matching.candidate_session import CandidateSession

logger = logging.getLogger(__name__)

__all__ = [
    "AccelerationDecomposition",
    "CounterfactualFork",
    "CounterfactualStrategy",
    "create_counterfactual_rollout",
]


def _make_readonly(arr: np.ndarray | None) -> None:
    """Set numpy array flags to read-only while satisfying Law of Demeter."""
    if arr is not None:
        flags = arr.flags
        flags.writeable = False


class CounterfactualStrategy(str, Enum):
    """Supported counterfactual intervention strategies."""

    ZERO_TRAIL_ARM_TORQUE = "zero_trail_arm_torque"
    CLAMPED_ACTUATOR_TORQUE = "clamped_actuator_torque"
    NULLSPACE_EXPLORATION = "nullspace_exploration"
    CUSTOM = "custom"


@dataclass(frozen=True)
class AccelerationDecomposition:
    """Instantaneous decomposition of generalized accelerations.

    Invariants:
    - a_grav: generalized acceleration induced purely by gravity.
    - a_drift: velocity-product drift (Coriolis, centrifugal, passive damping).
    - a_ctrl: generalized acceleration induced by active control torques.
    - ztcf = a_grav + a_drift (Zero-Torque Counterfactual).
    - zvcf = a_grav + a_ctrl (Zero-Velocity Counterfactual).
    """

    a_grav: NDArray[np.float64]
    a_drift: NDArray[np.float64]
    a_ctrl: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not (
            np.all(np.isfinite(self.a_grav))
            and np.all(np.isfinite(self.a_drift))
            and np.all(np.isfinite(self.a_ctrl))
        ):
            raise ValueError("All acceleration decomposition components must be finite")
        if (
            self.a_grav.shape != self.a_drift.shape
            or self.a_grav.shape != self.a_ctrl.shape
        ):
            raise ValueError(
                f"Component shape mismatch: grav={self.a_grav.shape}, "
                f"drift={self.a_drift.shape}, ctrl={self.a_ctrl.shape}"
            )
        _make_readonly(self.a_grav)
        _make_readonly(self.a_drift)
        _make_readonly(self.a_ctrl)

    @property
    def ztcf(self) -> NDArray[np.float64]:
        """Zero-Torque Counterfactual acceleration (a_grav + a_drift)."""
        res = self.a_grav + self.a_drift
        res.flags.writeable = False
        return res

    @property
    def zvcf(self) -> NDArray[np.float64]:
        """Zero-Velocity Counterfactual acceleration (a_grav + a_ctrl)."""
        res = self.a_grav + self.a_ctrl
        res.flags.writeable = False
        return res

    @property
    def total_accel(self) -> NDArray[np.float64]:
        """Total instantaneous acceleration (a_grav + a_drift + a_ctrl)."""
        res = self.a_grav + self.a_drift + self.a_ctrl
        res.flags.writeable = False
        return res

    def verify_decomposition(
        self, a_total: NDArray[np.float64], rtol: float = 1e-4, atol: float = 1e-4
    ) -> bool:
        """Verify whether decomposition matches total acceleration within tolerance."""
        return bool(np.allclose(self.total_accel, a_total, rtol=rtol, atol=atol))


@dataclass(frozen=True)
class CounterfactualFork:
    """Immutable record of an intervened trajectory rollout diverging from a baseline."""

    fork_id: str
    baseline_candidate_sha256: str
    strategy: CounterfactualStrategy
    fork_time_s: float
    fork_frame_idx: int
    initial_state: tuple[NDArray[np.float64], NDArray[np.float64]]
    time_s: NDArray[np.float64]
    q: NDArray[np.float64]
    v: NDArray[np.float64] | None = None
    a: NDArray[np.float64] | None = None
    altered_tau: NDArray[np.float64] | None = None
    divergence_rms: float = 0.0
    constraint_status: dict[str, Any] = field(default_factory=dict)
    is_accepted: bool = True
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _make_readonly(self.initial_state[0])
        _make_readonly(self.initial_state[1])
        _make_readonly(self.time_s)
        _make_readonly(self.q)
        _make_readonly(self.v)
        _make_readonly(self.a)
        _make_readonly(self.altered_tau)

    @property
    def frame_count(self) -> int:
        return int(len(self.time_s))

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0


def _compute_baseline_hash(session: CandidateSession) -> str:
    """Compute combined SHA-256 digest of session coordinate arrays to guarantee immutability."""
    hasher = hashlib.sha256()
    hasher.update(session.q.tobytes())
    hasher.update(session.time_s.tobytes())
    if session.v is not None:
        hasher.update(session.v.tobytes())
    if session.tau is not None:
        hasher.update(session.tau.tobytes())
    return hasher.hexdigest()


def _resolve_trail_arm_indices(
    coord_names: tuple[str, ...], explicit_indices: Sequence[int] | None
) -> tuple[int, ...]:
    """Identify coordinate columns for trail arm joints."""
    if explicit_indices is not None:
        return tuple(explicit_indices)

    trail_keywords = (
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "r_shoulder",
        "r_elbow",
        "r_wrist",
        "trail_arm",
        "trail_shoulder",
        "trail_elbow",
    )
    matches: list[int] = []
    for idx, name in enumerate(coord_names):
        lower_name = name.lower()
        if any(kw in lower_name for kw in trail_keywords):
            matches.append(idx)
    return tuple(matches)


def _alter_torques(
    base_tau: NDArray[np.float64],
    strategy: CounterfactualStrategy,
    coord_names: tuple[str, ...],
    control_override: NDArray[np.float64] | None,
    clamp_limits: tuple[float, float] | None,
    trail_arm_indices: Sequence[int] | None,
) -> NDArray[np.float64]:
    """Compute altered torques according to chosen counterfactual intervention strategy."""
    altered = base_tau.copy()
    if strategy == CounterfactualStrategy.ZERO_TRAIL_ARM_TORQUE:
        arm_indices = _resolve_trail_arm_indices(coord_names, trail_arm_indices)
        for idx in arm_indices:
            if idx < altered.shape[1]:
                altered[:, idx] = 0.0
    elif strategy == CounterfactualStrategy.CLAMPED_ACTUATOR_TORQUE:
        c_min, c_max = clamp_limits if clamp_limits is not None else (-50.0, 50.0)
        altered = np.clip(altered, c_min, c_max)
    elif strategy == CounterfactualStrategy.NULLSPACE_EXPLORATION:
        t = np.linspace(0.0, np.pi, len(altered))[:, np.newaxis]
        altered = altered + 5.0 * np.sin(t)
    elif strategy == CounterfactualStrategy.CUSTOM:
        if control_override is None:
            raise ValueError("control_override must be provided for CUSTOM strategy")
        if control_override.shape != altered.shape:
            raise ValueError(
                f"control_override shape {control_override.shape} does not match {altered.shape}"
            )
        altered = control_override.copy()
    return altered


def _integrate_dynamics(
    time_s: NDArray[np.float64],
    q0: NDArray[np.float64],
    v0: NDArray[np.float64],
    base_q: NDArray[np.float64],
    base_v: NDArray[np.float64],
    base_tau: NDArray[np.float64],
    altered_tau: NDArray[np.float64],
    base_a: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Numerically integrate counterfactual divergence from baseline trajectory."""
    n_frames = len(time_s)
    n_dofs = len(q0)
    q_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)
    v_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)
    a_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)

    q_cf[0] = q0
    v_cf[0] = v0

    # Effective diagonal inertia approximation for acceleration perturbation
    m_eff = 1.0

    for i in range(n_frames):
        dt = float(time_s[i + 1] - time_s[i]) if i + 1 < n_frames else 0.001
        if dt <= 0.0:
            dt = 0.001

        # Baseline acceleration at frame i
        if base_a is not None and i < len(base_a):
            a_nom = base_a[i]
        elif i + 1 < len(base_v):
            a_nom = (base_v[i + 1] - base_v[i]) / dt
        else:
            a_nom = np.zeros(n_dofs, dtype=np.float64)

        delta_tau = altered_tau[i] - base_tau[i]
        delta_a = np.zeros(n_dofs, dtype=np.float64)
        n_map = min(len(delta_tau), n_dofs)
        delta_a[:n_map] = delta_tau[:n_map] / m_eff
        a_cf[i] = a_nom + delta_a

        if i + 1 < n_frames:
            v_cf[i + 1] = v_cf[i] + a_cf[i] * dt
            q_cf[i + 1] = q_cf[i] + v_cf[i + 1] * dt

    return q_cf, v_cf, a_cf


def _evaluate_constraints(
    q: NDArray[np.float64],
    v: NDArray[np.float64],
    specification: dict[str, Any],
    coord_names: tuple[str, ...],
) -> tuple[dict[str, Any], bool, tuple[str, ...]]:
    """Validate counterfactual rollout trajectory against system constraints."""
    reasons: list[str] = []
    status: dict[str, Any] = {"finite": True, "limits_satisfied": True}

    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v))):
        status["finite"] = False
        reasons.append("Non-finite state encountered during counterfactual rollout")

    # Joint limits evaluation if present in specification
    limits = specification.get("joint_limits")
    if isinstance(limits, dict):
        limit_violations = 0
        for idx, name in enumerate(coord_names):
            if name in limits and idx < q.shape[1]:
                min_val, max_val = limits[name]
                col = q[:, idx]
                if np.any(col < min_val) or np.any(col > max_val):
                    limit_violations += 1
        if limit_violations > 0:
            status["limits_satisfied"] = False
            reasons.append(
                f"Joint limits violated on {limit_violations} generalized coordinates"
            )

    is_accepted = len(reasons) == 0
    return status, is_accepted, tuple(reasons)


@precondition(
    lambda session, fork_frame_idx, **_: (
        session.supports_counterfactuals
        and 0 <= fork_frame_idx < session.frame_count - 1
    ),
    "Candidate session must support counterfactuals and have a valid fork frame index",
)
def create_counterfactual_rollout(
    session: CandidateSession,
    fork_frame_idx: int,
    strategy: CounterfactualStrategy = CounterfactualStrategy.ZERO_TRAIL_ARM_TORQUE,
    *,
    control_override: NDArray[np.float64] | None = None,
    duration_frames: int | None = None,
    clamp_limits: tuple[float, float] | None = None,
    trail_arm_indices: Sequence[int] | None = None,
) -> CounterfactualFork:
    """Generate an immutable counterfactual fork guaranteeing zero mutation of baseline data."""
    # Strict immutability guarantee: hash baseline state before rollout
    baseline_digest_before = _compute_baseline_hash(session)

    end_idx = (
        session.frame_count
        if duration_frames is None
        else min(fork_frame_idx + duration_frames, session.frame_count)
    )
    if end_idx <= fork_frame_idx + 1:
        raise ValueError(
            f"Rollout window too short: [{fork_frame_idx}, {end_idx}) frames"
        )

    fork_time_s = float(session.time_s[fork_frame_idx])
    time_window = session.time_s[fork_frame_idx:end_idx].copy()
    q0 = session.q[fork_frame_idx].copy()
    v0 = (
        session.v[fork_frame_idx].copy() if session.v is not None else np.zeros_like(q0)
    )

    base_q = session.q[fork_frame_idx:end_idx]
    base_v = (
        session.v[fork_frame_idx:end_idx]
        if session.v is not None
        else np.zeros_like(base_q)
    )
    base_a = session.a[fork_frame_idx:end_idx] if session.a is not None else None
    base_tau = (
        session.tau[fork_frame_idx:end_idx].copy()
        if session.tau is not None
        else np.zeros_like(base_q)
    )

    altered_tau = _alter_torques(
        base_tau=base_tau,
        strategy=strategy,
        coord_names=session.coordinate_names,
        control_override=control_override,
        clamp_limits=clamp_limits,
        trail_arm_indices=trail_arm_indices,
    )

    q_cf, v_cf, a_cf = _integrate_dynamics(
        time_s=time_window,
        q0=q0,
        v0=v0,
        base_q=base_q,
        base_v=base_v,
        base_tau=base_tau,
        altered_tau=altered_tau,
        base_a=base_a,
    )

    divergence_rms = float(np.sqrt(np.mean((q_cf - base_q) ** 2)))
    status, is_accepted, reasons = _evaluate_constraints(
        q=q_cf,
        v=v_cf,
        specification=session.specification,
        coord_names=session.coordinate_names,
    )

    # Verify zero-mutation invariant on baseline session
    baseline_digest_after = _compute_baseline_hash(session)
    if baseline_digest_before != baseline_digest_after:
        raise RuntimeError(
            "Baseline candidate session was mutated during counterfactual rollout calculation!"
        )

    fork_id = f"cf_{session.candidate_sha256[:8]}_{strategy.value}_{fork_frame_idx}"

    return CounterfactualFork(
        fork_id=fork_id,
        baseline_candidate_sha256=session.candidate_sha256,
        strategy=strategy,
        fork_time_s=fork_time_s,
        fork_frame_idx=fork_frame_idx,
        initial_state=(q0, v0),
        time_s=time_window,
        q=q_cf,
        v=v_cf,
        a=a_cf,
        altered_tau=altered_tau,
        divergence_rms=divergence_rms,
        constraint_status=status,
        is_accepted=is_accepted,
        rejection_reasons=reasons,
    )
