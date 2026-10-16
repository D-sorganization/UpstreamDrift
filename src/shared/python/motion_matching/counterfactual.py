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
from src.shared.python.core.contracts import require
from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    constrained_acceleration_at,
    is_constrained_provider,
)
from src.shared.python.simulation_backends.ztcf_zvcf import (
    drift_and_control_split,
    zvcf_acceleration,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from src.shared.python.loop_nullspace.decomposition import Decomposition
    from src.shared.python.motion_matching.candidate_session import CandidateSession
    from src.shared.python.simulation_backends.protocol import EquationsOfMotionProvider

logger = logging.getLogger(__name__)

__all__ = [
    "AccelerationDecomposition",
    "CounterfactualDynamics",
    "CounterfactualFork",
    "CounterfactualStrategy",
    "NullspaceExploration",
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

    ZVCF has one definition in the repository: velocity *and* control set to
    zero, ``-M(q)^-1 h(q, 0)``, implemented by
    :func:`simulation_backends.ztcf_zvcf.zvcf_acceleration` (issue #11553).
    :meth:`from_dynamics` builds the components from that module.

    Invariants:
    - a_grav: configuration-only acceleration at zero velocity and zero control
      (gravity plus configuration-dependent passive loads); this *is* the ZVCF.
    - a_drift: velocity-dependent drift (Coriolis, centrifugal, passive damping).
    - a_ctrl: generalized acceleration induced by active control torques.
    - ztcf = a_grav + a_drift (Zero-Torque Counterfactual).
    - zvcf = a_grav (Zero-Velocity Counterfactual: v = 0 and u = 0).
    - zero_velocity_control_preserved = a_grav + a_ctrl (not a ZVCF).
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

    @classmethod
    def from_dynamics(
        cls,
        provider: EquationsOfMotionProvider,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        tau: NDArray[np.float64],
    ) -> AccelerationDecomposition:
        """Decompose the acceleration at ``(q, v)`` under ``tau`` canonically.

        Delegates to :mod:`simulation_backends.ztcf_zvcf` so the ZTCF and ZVCF
        reported here are the same numbers as that module's for the same state.

        Args:
            provider: Backend exposing ``mass_matrix`` and ``bias_forces``.
            q: Joint positions ``(n,)``.
            v: Joint velocities ``(n,)``.
            tau: Applied generalized control ``(n,)``.

        Returns:
            Decomposition whose ``zvcf`` equals ``zvcf_acceleration(provider, q)``
            and whose ``ztcf`` equals ``ztcf_acceleration(provider, q, v)``.

        Raises:
            ValueError: If the state or control vectors are empty, non-finite or
                of inconsistent length.
        """
        ztcf, a_ctrl = drift_and_control_split(provider, q, v, tau)
        a_grav = zvcf_acceleration(provider, q)
        require(
            a_grav.shape == ztcf.shape,
            "zvcf and ztcf accelerations must share a shape",
            value=(a_grav.shape, ztcf.shape),
        )
        return cls(a_grav=a_grav, a_drift=ztcf - a_grav, a_ctrl=a_ctrl)

    @property
    def ztcf(self) -> NDArray[np.float64]:
        """Zero-Torque Counterfactual acceleration (a_grav + a_drift)."""
        res = self.a_grav + self.a_drift
        res.flags.writeable = False
        return res

    @property
    def zvcf(self) -> NDArray[np.float64]:
        """Zero-Velocity Counterfactual acceleration (v = 0 and u = 0): a_grav."""
        res = self.a_grav.copy()
        res.flags.writeable = False
        return res

    @property
    def zero_velocity_control_preserved(self) -> NDArray[np.float64]:
        """Zero-velocity acceleration that keeps the control (a_grav + a_ctrl).

        Matches ``zero_velocity_control_preserved_acceleration``; never a ZVCF.
        """
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


@dataclass(frozen=True)
class NullspaceExploration:
    """Null-space sampling for ``NULLSPACE_EXPLORATION`` (NSX-15 coordinates).

    ``decompositions`` holds one NSX-15 :class:`Decomposition` per rollout
    frame (the fork frame first); ``actuated_indices`` maps its actuated
    torques onto the session's generalised coordinates.  ``z`` are the null
    coordinates: ``None`` samples ``N(0, scale^2)`` per frame from ``seed``;
    a ``(nullity,)`` vector is reused every frame; a ``(T, nullity)`` array is
    used per frame.  Every member of the family reproduces the measured
    acceleration *at the frame state it was sampled for*, so this explores
    allocations, not motions; the load is held across each integration step,
    so later frames drift by O(dt).
    """

    decompositions: Sequence[Decomposition]
    actuated_indices: tuple[int, ...]
    z: ArrayLike | None = None
    seed: int = 0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.decompositions) == 0:
            raise ValueError("decompositions must not be empty")
        if not (np.isfinite(self.scale) and self.scale >= 0.0):
            raise ValueError(f"scale must be finite and >= 0, got {self.scale}")
        object.__setattr__(self, "actuated_indices", tuple(self.actuated_indices))

    def torques(self, base_tau: NDArray[np.float64]) -> NDArray[np.float64]:
        """``base_tau`` with the actuated columns replaced by ``tau(x_p + N z)``."""
        n_frames, n_dofs = base_tau.shape
        if len(self.decompositions) != n_frames:
            raise ValueError(
                f"decompositions must have one entry per rollout frame "
                f"({n_frames}), got {len(self.decompositions)}"
            )
        if any(not 0 <= i < n_dofs for i in self.actuated_indices):
            raise ValueError(f"actuated_indices outside [0, {n_dofs})")
        rng = np.random.default_rng(self.seed)
        out = base_tau.copy()
        for k, dec in enumerate(self.decompositions):
            if dec.n_actuated != len(self.actuated_indices):
                raise ValueError(
                    f"frame {k}: decomposition has {dec.n_actuated} actuated "
                    f"torques, actuated_indices has {len(self.actuated_indices)}"
                )
            out[k, list(self.actuated_indices)] = dec.torque(self._z_at(k, dec, rng))
        return out

    def _z_at(
        self, frame: int, dec: Decomposition, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        if self.z is None:
            return np.asarray(rng.normal(0.0, self.scale, size=dec.nullity))
        z = np.asarray(self.z, dtype=np.float64)
        return z[frame] if z.ndim == 2 else z


@dataclass(frozen=True)
class CounterfactualDynamics:
    """The dynamics a rollout integrates (no placeholder inertia exists).

    ``provider`` is an ``EquationsOfMotionProvider``; a
    ``ConstrainedEquationsOfMotionProvider`` is integrated on the closed loop
    with the reaction re-solved at every step.  ``nullspace`` is required by
    ``NULLSPACE_EXPLORATION`` only.
    """

    provider: Any
    nullspace: NullspaceExploration | None = None


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
    nullspace: NullspaceExploration | None = None,
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
        if nullspace is None:
            raise ValueError(
                "NULLSPACE_EXPLORATION requires CounterfactualDynamics.nullspace "
                "(per-frame NSX-15 decompositions)"
            )
        altered = nullspace.torques(altered)
    elif strategy == CounterfactualStrategy.CUSTOM:
        if control_override is None:
            raise ValueError("control_override must be provided for CUSTOM strategy")
        if control_override.shape != altered.shape:
            raise ValueError(
                f"control_override shape {control_override.shape} does not match {altered.shape}"
            )
        altered = control_override.copy()
    return altered


def _forward_acceleration(
    provider: Any,
    q: NDArray[np.float64],
    v: NDArray[np.float64],
    tau: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Acceleration under generalised load ``tau``: closed-loop if constrained."""
    if is_constrained_provider(provider):
        return constrained_acceleration_at(provider, q, v, tau)[0]
    mass = np.asarray(provider.mass_matrix(q), dtype=np.float64)
    bias = np.asarray(provider.bias_forces(q, v), dtype=np.float64)
    return np.asarray(np.linalg.solve(mass, tau - bias))


def _rk4_step(
    provider: Any,
    q: NDArray[np.float64],
    v: NDArray[np.float64],
    tau: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One RK4 step of ``q' = v, v' = a(q, v, tau)`` with ``tau`` held."""
    k1v = _forward_acceleration(provider, q, v, tau)
    q2, v2 = q + 0.5 * dt * v, v + 0.5 * dt * k1v
    k2v = _forward_acceleration(provider, q2, v2, tau)
    q3, v3 = q + 0.5 * dt * v2, v + 0.5 * dt * k2v
    k3v = _forward_acceleration(provider, q3, v3, tau)
    q4, v4 = q + dt * v3, v + dt * k3v
    k4v = _forward_acceleration(provider, q4, v4, tau)
    q_next = q + dt / 6.0 * (v + 2.0 * v2 + 2.0 * v3 + v4)
    v_next = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return q_next, v_next


def _integrate_dynamics(
    time_s: NDArray[np.float64],
    q0: NDArray[np.float64],
    v0: NDArray[np.float64],
    altered_tau: NDArray[np.float64],
    provider: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Forward-simulate ``provider`` from ``(q0, v0)`` under ``altered_tau``.

    Each frame interval is one RK4 step with the frame's generalised load held
    constant; ``a_cf[i]`` is the acceleration at frame ``i``'s state.  A
    constrained provider is integrated on the closed loop (reaction re-solved
    at every stage), an unconstrained one by ``M^-1 (tau - h)``.

    Postcondition: ``q_cf[0] == q0``, ``v_cf[0] == v0``; arrays share the
    ``(n_frames, n_dofs)`` shape.  Raises ``ValueError`` on non-increasing time
    or a state that does not match the load width.
    """
    n_frames, n_dofs = len(time_s), len(q0)
    if altered_tau.shape != (n_frames, n_dofs) or v0.shape != q0.shape:
        raise ValueError(
            f"altered_tau must be ({n_frames}, {n_dofs}) and v0 must match q0; "
            f"got {altered_tau.shape} and {v0.shape}"
        )
    steps = np.diff(time_s)
    if np.any(steps <= 0.0):
        raise ValueError("time_s must be strictly increasing for integration")
    q_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)
    v_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)
    a_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)
    q_cf[0], v_cf[0] = q0, v0
    for i in range(n_frames):
        a_cf[i] = _forward_acceleration(provider, q_cf[i], v_cf[i], altered_tau[i])
        if i + 1 < n_frames:
            q_cf[i + 1], v_cf[i + 1] = _rk4_step(
                provider, q_cf[i], v_cf[i], altered_tau[i], float(steps[i])
            )
    return q_cf, v_cf, a_cf


def _closure_velocity_residual(
    provider: Any, q: NDArray[np.float64], v: NDArray[np.float64]
) -> float | None:
    """``max |J_c v|`` over the rollout, or ``None`` for an unconstrained provider."""
    if not is_constrained_provider(provider):
        return None
    if len(q) != len(v):
        raise ValueError("q and v must have the same length")
    return float(
        max(
            np.max(np.abs(provider.closure_jacobian(q[i]) @ v[i]), initial=0.0)
            for i in range(len(q))
        )
    )


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


def _rollout_window(
    session: CandidateSession, fork_frame_idx: int, duration_frames: int | None
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Copies of ``(time, q0, v0, base_q, base_tau)`` for the rollout window."""
    end_idx = (
        session.frame_count
        if duration_frames is None
        else min(fork_frame_idx + duration_frames, session.frame_count)
    )
    if end_idx <= fork_frame_idx + 1:
        raise ValueError(
            f"Rollout window too short: [{fork_frame_idx}, {end_idx}) frames"
        )
    window = slice(fork_frame_idx, end_idx)
    q0 = session.q[fork_frame_idx].copy()
    v0 = (
        session.v[fork_frame_idx].copy() if session.v is not None else np.zeros_like(q0)
    )
    base_q = session.q[window]
    base_tau = (
        session.tau[window].copy() if session.tau is not None else np.zeros_like(base_q)
    )
    return session.time_s[window].copy(), q0, v0, base_q, base_tau


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
    dynamics: CounterfactualDynamics | None = None,
) -> CounterfactualFork:
    """Generate an immutable counterfactual fork guaranteeing zero mutation of baseline data.

    The fork is a forward simulation of ``dynamics.provider`` from the measured
    fork state under the altered generalised load (NSX-30): on a
    ``ConstrainedEquationsOfMotionProvider`` the closed loop is respected and
    the reaction is re-solved at every step.  Without ``dynamics`` the provider is
    built from the session's own model specification; if that is missing the
    call raises ``ValueError`` naming the input (no placeholder inertia).
    ``constraint_status`` reports ``closure_velocity_residual_max`` for a
    constrained provider.
    """
    if dynamics is None:
        from src.shared.python.motion_matching.counterfactual_provider import (
            resolve_session_dynamics,
        )

        dynamics = resolve_session_dynamics(session)
    # Strict immutability guarantee: hash baseline state before rollout
    baseline_digest_before = _compute_baseline_hash(session)

    time_window, q0, v0, base_q, base_tau = _rollout_window(
        session, fork_frame_idx, duration_frames
    )
    fork_time_s = float(time_window[0])

    altered_tau = _alter_torques(
        base_tau=base_tau,
        strategy=strategy,
        coord_names=session.coordinate_names,
        control_override=control_override,
        clamp_limits=clamp_limits,
        trail_arm_indices=trail_arm_indices,
        nullspace=dynamics.nullspace,
    )

    q_cf, v_cf, a_cf = _integrate_dynamics(
        time_s=time_window,
        q0=q0,
        v0=v0,
        altered_tau=altered_tau,
        provider=dynamics.provider,
    )

    divergence_rms = float(np.sqrt(np.mean((q_cf - base_q) ** 2)))
    status, is_accepted, reasons = _evaluate_constraints(
        q=q_cf,
        v=v_cf,
        specification=session.specification,
        coord_names=session.coordinate_names,
    )
    closure_residual = _closure_velocity_residual(dynamics.provider, q_cf, v_cf)
    if closure_residual is not None:
        status["closure_velocity_residual_max"] = closure_residual

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
