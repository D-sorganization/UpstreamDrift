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
from src.shared.python.simulation_backends.ztcf_zvcf import (
    drift_and_control_split,
    zvcf_acceleration,
)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.candidate_session import CandidateSession
    from src.shared.python.simulation_backends.protocol import EquationsOfMotionProvider

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


def _sample_nullspace_torques(
    base_tau: NDArray[np.float64],
    snapshot: Any | None = None,
    nullspace_z: NDArray[np.float64] | Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Sample dynamically equivalent or null-space perturbed torques x_p + N z (NSX-15, NSX-30)."""
    from src.shared.python.loop_nullspace.decomposition import decompose
    from src.shared.python.motion_matching.force_nullspace import ForceNullSpace

    n_frames, n_tau = base_tau.shape
    if snapshot is not None:
        decomp = decompose(snapshot)
        tau_p = decomp.x_particular[:n_tau]
        n_tau_basis = decomp.null_basis[:n_tau, :]
        z_vec = (
            np.ones(decomp.nullity, dtype=np.float64)
            if nullspace_z is None
            else np.asarray(nullspace_z, dtype=np.float64)
        )
        sampled = tau_p + n_tau_basis @ z_vec
        return np.tile(sampled, (n_frames, 1))

    j_c_t = np.zeros((n_tau, 1), dtype=np.float64)
    j_c_t[0, 0] = 1.0
    if n_tau > 1:
        j_c_t[-1, 0] = -1.0
    a_mat = np.hstack([np.eye(n_tau, dtype=np.float64), j_c_t])
    out = np.zeros_like(base_tau)
    for k in range(n_frames):
        space = ForceNullSpace.from_balance(a_mat, base_tau[k])
        z_k = (
            np.ones(space.basis.shape[1], dtype=np.float64)
            if nullspace_z is None
            else np.asarray(nullspace_z, dtype=np.float64)
        )
        x_k = space.particular + space.basis @ z_k
        out[k] = x_k[:n_tau]
    return out


def _alter_torques(
    base_tau: NDArray[np.float64],
    strategy: CounterfactualStrategy,
    coord_names: tuple[str, ...],
    **options: Any,
) -> NDArray[np.float64]:
    """Compute altered torques according to chosen counterfactual intervention strategy."""
    control_override = options.get("control_override")
    clamp_limits = options.get("clamp_limits")
    trail_arm_indices = options.get("trail_arm_indices")
    snapshot = options.get("snapshot")
    nullspace_z = options.get("nullspace_z")

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
        altered = _sample_nullspace_torques(
            base_tau, snapshot=snapshot, nullspace_z=nullspace_z
        )
    elif strategy == CounterfactualStrategy.CUSTOM:
        if control_override is None:
            raise ValueError("control_override must be provided for CUSTOM strategy")
        if control_override.shape != altered.shape:
            raise ValueError(
                f"control_override shape {control_override.shape} does not match {altered.shape}"
            )
        altered = control_override.copy()
    return altered


def _compute_step_acceleration(
    q_i: NDArray[np.float64],
    v_i: NDArray[np.float64],
    tau_i: NDArray[np.float64],
    delta_tau_i: NDArray[np.float64],
    a_nom_i: NDArray[np.float64],
    **dyn_ctx: Any,
) -> NDArray[np.float64]:
    """Evaluate step acceleration using constrained KKT projector or provider EOM."""
    from src.shared.python.loop_nullspace.constrained_dynamics import (
        ConstrainedEquationsOfMotionProvider,
        constrained_acceleration,
    )

    provider = dyn_ctx.get("provider")
    snapshot = dyn_ctx.get("snapshot")
    q_ref = dyn_ctx.get("q_ref", q_i)
    alpha = float(dyn_ctx.get("baumgarte_alpha", 10.0))
    beta = float(dyn_ctx.get("baumgarte_beta", 10.0))

    if provider is not None:
        m_mat = np.asarray(provider.mass_matrix(q_i), dtype=np.float64)
        h_vec = np.asarray(provider.bias_forces(q_i, v_i), dtype=np.float64)
        if isinstance(provider, ConstrainedEquationsOfMotionProvider):
            j_c = np.asarray(provider.closure_jacobian(q_i), dtype=np.float64)
            jdot_v = np.asarray(provider.closure_drift(q_i, v_i), dtype=np.float64)
            q_err = j_c @ (q_i - q_ref)
            v_err = j_c @ v_i
            q_ddot, _ = constrained_acceleration(
                m_mat,
                h_vec,
                tau_i,
                j_c,
                jdot_v,
                q_err=q_err,
                v_err=v_err,
                baumgarte_alpha=alpha,
                baumgarte_beta=beta,
            )
            return np.asarray(q_ddot, dtype=np.float64)
        return np.asarray(np.linalg.solve(m_mat, tau_i - h_vec), dtype=np.float64)

    if snapshot is not None:
        q_err = snapshot.J_c @ (q_i - q_ref)
        v_err = snapshot.J_c @ v_i
        q_ddot, _ = constrained_acceleration(
            snapshot.M,
            snapshot.h,
            tau_i,
            snapshot.J_c,
            snapshot.Jdot_v,
            q_err=q_err,
            v_err=v_err,
            baumgarte_alpha=alpha,
            baumgarte_beta=beta,
        )
        return np.asarray(q_ddot, dtype=np.float64)

    n_dofs = len(q_i)
    mass_mat = dyn_ctx.get("mass_matrix")
    m_mat = (
        np.asarray(mass_mat, dtype=np.float64)
        if mass_mat is not None
        else np.eye(n_dofs, dtype=np.float64)
    )
    delta_rhs = np.zeros(n_dofs, dtype=np.float64)
    n_map = min(len(delta_tau_i), n_dofs)
    delta_rhs[:n_map] = delta_tau_i[:n_map]
    return a_nom_i + np.linalg.solve(m_mat, delta_rhs)


def _integrate_dynamics(
    time_s: NDArray[np.float64],
    q0: NDArray[np.float64],
    v0: NDArray[np.float64],
    base_v: NDArray[np.float64],
    base_tau: NDArray[np.float64],
    altered_tau: NDArray[np.float64],
    **dyn_ctx: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Numerically integrate counterfactual trajectory using provider or KKT dynamics."""
    n_frames = len(time_s)
    n_dofs = len(q0)
    q_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)
    v_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)
    a_cf = np.zeros((n_frames, n_dofs), dtype=np.float64)

    q_cf[0] = q0
    v_cf[0] = v0
    base_a = dyn_ctx.get("base_a")

    for i in range(n_frames):
        dt = float(time_s[i + 1] - time_s[i]) if i + 1 < n_frames else 0.001
        if dt <= 0.0:
            dt = 0.001

        if base_a is not None and i < len(base_a):
            a_nom = base_a[i]
        elif i + 1 < len(base_v):
            a_nom = (base_v[i + 1] - base_v[i]) / dt
        else:
            a_nom = np.zeros(n_dofs, dtype=np.float64)

        delta_tau = altered_tau[i] - base_tau[i]
        a_cf[i] = _compute_step_acceleration(
            q_cf[i],
            v_cf[i],
            altered_tau[i],
            delta_tau,
            a_nom,
            q_ref=q0,
            **dyn_ctx,
        )

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
    **options: Any,
) -> CounterfactualFork:
    """Generate an immutable counterfactual fork guaranteeing zero mutation of baseline data."""
    # Strict immutability guarantee: hash baseline state before rollout
    baseline_digest_before = _compute_baseline_hash(session)

    duration_frames = options.get("duration_frames")
    end_idx = (
        session.frame_count
        if duration_frames is None
        else min(fork_frame_idx + int(duration_frames), session.frame_count)
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
        **options,
    )

    q_cf, v_cf, a_cf = _integrate_dynamics(
        time_s=time_window,
        q0=q0,
        v0=v0,
        base_v=base_v,
        base_tau=base_tau,
        altered_tau=altered_tau,
        base_a=base_a,
        mass_matrix=session.specification.get("mass_matrix"),
        **options,
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
