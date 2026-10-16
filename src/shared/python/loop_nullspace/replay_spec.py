"""Value types and the feedback budget of the replay of allocated torques (NSX-31).

Three things are kept apart (``AGENTS.md``, "Modeling Reference Documentation"):
inverse kinematics, feedback-assisted tracking and independently replayed
open-loop dynamics.  This module holds the declared inputs of a replay (settings,
setup, reference trajectory, feedback law) and the accounting of the effort a
feedback law spends.  :func:`feedback_budget` is a pure function of series, so
the budget can be checked by hand; the feedback torque is never added to the
allocated torque in any of these quantities.

Pure numpy and scipy; no engine imports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicHermiteSpline

from src.shared.python.loop_nullspace.evidence import Array, readonly_float

__all__ = [
    "INTERPOLATIONS",
    "AccelerationFeedbackLaw",
    "BaumgarteGains",
    "ClosureDrift",
    "FeedbackBudget",
    "FeedbackLaw",
    "ReferenceTrajectory",
    "ReplayMode",
    "ReplaySettings",
    "ReplaySetup",
    "acceleration_feedback_law",
    "check_gain",
    "feedback_budget",
]

INTERPOLATIONS = ("zoh", "linear", "cubic")
_INF = float("inf")


class ReplayMode(str, Enum):
    """What kind of evidence a replay is (the mandatory label of every report)."""

    OPEN_LOOP = "open_loop"
    FEEDBACK = "feedback_assisted"


@dataclass(frozen=True)
class BaumgarteGains:
    """Declared Baumgarte gains ``alpha`` (velocity) and ``beta`` (position) [1/s].

    Stabilisation is not physics: it adds ``2 alpha J_c v + beta^2 Phi(q)`` to
    the closure drift and is reported apart from the dynamics.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        for name in ("alpha", "beta"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Baumgarte {name} must be finite and >= 0: {value}")


@dataclass(frozen=True)
class ReplaySettings:
    """Numerical choices of a replay (all declared in the report).

    Attributes:
        substeps: RK4 steps per allocation interval (``>= 1``).
        interpolation: How the allocated torque is evaluated between frames:
            ``zoh`` (held, first order), ``linear`` (second order) or ``cubic``
            (not-a-knot spline, fourth order for smooth torques; needs four
            frames and can overshoot a torque that saturates a bound).
        divergence_tolerance: Largest absolute coordinate error (rad or m)
            before the replay counts as diverged from the reference.
        baumgarte: Declared stabilisation, or ``None`` for pure dynamics.
    """

    substeps: int = 1
    interpolation: str = "linear"
    divergence_tolerance: float = 0.05
    baumgarte: BaumgarteGains | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.substeps, (int, np.integer)) or self.substeps < 1:
            raise ValueError(f"substeps must be an integer >= 1, got {self.substeps}")
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {INTERPOLATIONS}, got "
                f"{self.interpolation!r}"
            )
        if not np.isfinite(self.divergence_tolerance) or self.divergence_tolerance <= 0:
            raise ValueError("divergence_tolerance must be finite and positive")
        if self.baumgarte is not None and not isinstance(
            self.baumgarte, BaumgarteGains
        ):
            raise TypeError("baumgarte must be a BaumgarteGains or None")


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Sampled ``(q, v)`` the feedback tracks and the divergence is measured on.

    Positions are interpolated by the cubic Hermite spline through the samples
    and their velocities, so an exact reference is reproduced to fourth order.

    Raises:
        ValueError: Fewer than two strictly increasing times, mismatched or
            non-finite arrays.
    """

    times_s: Array
    q: Array
    v: Array
    _spline: CubicHermiteSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        t = readonly_float(self.times_s, "reference times_s")
        q = readonly_float(self.q, "reference q")
        v = readonly_float(self.v, "reference v")
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
            raise ValueError("reference times_s must be strictly increasing, >= 2")
        if q.ndim != 2 or q.shape != v.shape or q.shape[0] != t.size:
            raise ValueError(
                f"reference q and v must both be ({t.size}, n); got {q.shape} and "
                f"{v.shape}"
            )
        object.__setattr__(self, "times_s", t)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "_spline", CubicHermiteSpline(t, q, v, axis=0))

    @property
    def n(self) -> int:
        """Number of coordinates."""
        return int(self.q.shape[1])

    def covers(self, t_start: float, t_end: float) -> bool:
        """Whether ``[t_start, t_end]`` lies inside the sampled window."""
        return bool(
            self.times_s[0] <= t_start + 1e-12 and t_end <= self.times_s[-1] + 1e-12
        )

    def state_at(self, t: float) -> tuple[Array, Array]:
        """``(q_ref(t), v_ref(t))``; ``t`` is clipped to the sampled window."""
        tc = float(np.clip(t, self.times_s[0], self.times_s[-1]))
        return (
            np.asarray(self._spline(tc), dtype=np.float64),
            np.asarray(self._spline(tc, 1), dtype=np.float64),
        )

    def states_at(self, times: Array) -> tuple[Array, Array]:
        """``(q_ref, v_ref)`` at every time of ``times`` (clipped), ``(T, n)``."""
        tc = np.clip(times, self.times_s[0], self.times_s[-1])
        return (
            np.asarray(self._spline(tc), dtype=np.float64),
            np.asarray(self._spline(tc, 1), dtype=np.float64),
        )


@dataclass(frozen=True)
class FeedbackLaw:
    """Declared joint-space PD law ``tau_fb = kp (q_ref - q) + kd (v_ref - v)``.

    Acts on the actuated coordinates only, in the allocated torques' units
    [N m per rad, N m s per rad].  ``kp`` and ``kd`` are per-joint vectors
    ``(n_a,)`` or symmetric positive semi-definite matrices ``(n_a, n_a)``
    (see :func:`acceleration_feedback_law`).  ``torque_limit`` (``N m`` per
    joint) is a declared saturation; the fraction of saturated samples is
    reported.

    Raises:
        ValueError: Gains negative, non-finite or not PSD, shape mismatch, or a
            non-positive limit.
    """

    kp: Array
    kd: Array
    torque_limit: Array | None = None
    name: str = "joint-space PD"

    def __post_init__(self) -> None:
        kp = readonly_float(self.kp, "feedback kp")
        kd = readonly_float(self.kd, "feedback kd")
        if kp.ndim not in (1, 2) or kp.shape != kd.shape or kp.size == 0:
            raise ValueError(f"kp and kd must share a (n,) or (n, n) shape: {kp.shape}")
        for gain, label in ((kp, "kp"), (kd, "kd")):
            check_gain(gain, label)
        n = kp.shape[0]
        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "kd", kd)
        if self.torque_limit is not None:
            lim = readonly_float(self.torque_limit, "feedback torque_limit")
            if lim.shape != (n,) or np.any(lim <= 0.0):
                raise ValueError("torque_limit must be positive, one per joint")
            object.__setattr__(self, "torque_limit", lim)

    @property
    def n_actuated(self) -> int:
        """Number of actuated joints the law acts on."""
        return int(self.kp.shape[0])

    def torque(self, q_error: Array, v_error: Array, q: Array | None = None) -> Array:
        """Saturated feedback torque for the actuated errors ``ref - state``.

        ``q`` (the full configuration) is accepted for the state-dependent laws
        and ignored here.
        """
        if self.kp.ndim == 1:
            raw = self.kp * q_error + self.kd * v_error
        else:
            raw = self.kp @ q_error + self.kd @ v_error
        if self.torque_limit is None:
            return np.asarray(raw)
        return np.asarray(np.clip(raw, -self.torque_limit, self.torque_limit))


@dataclass(frozen=True)
class AccelerationFeedbackLaw:
    """State-dependent acceleration-space PD (computed-torque style feedback).

    ``tau_fb = Lambda(q) (omega^2 e + 2 zeta omega e_dot)`` with ``Lambda(q) =
    pinv(response(q))`` and ``response(q) = S P_c(q) S^T`` supplied by the caller
    (the actuated acceleration per unit actuated torque on the closed loop at
    the *current* configuration).  The actuated error then obeys ``e_ddot + 2
    zeta omega e_dot + omega^2 e = 0`` wherever the actuated response is
    invertible, whatever the inertia of the joint - which a fixed per-joint
    gain cannot do on a coupled chain with joints three orders of magnitude
    apart in inertia.  The feedback torque is still reported apart from the
    allocated one.

    Raises:
        ValueError: ``omega``/``zeta``/``rcond`` out of range, or limit invalid.
    """

    response: Callable[[Array], Array]
    n_actuated: int
    omega_rad_s: float
    zeta: float = 1.0
    rcond: float = 1e-6
    torque_limit: Array | None = None
    name: str = "acceleration PD (state-dependent)"

    def __post_init__(self) -> None:
        if not callable(self.response):
            raise TypeError("response must be callable q -> (n_a, n_a)")
        if self.n_actuated < 1:
            raise ValueError("n_actuated must be >= 1")
        if not (np.isfinite(self.omega_rad_s) and self.omega_rad_s > 0.0):
            raise ValueError(f"omega_rad_s must be positive, got {self.omega_rad_s}")
        if not (np.isfinite(self.zeta) and self.zeta >= 0.0):
            raise ValueError(f"zeta must be non-negative, got {self.zeta}")
        if not (np.isfinite(self.rcond) and 0.0 < self.rcond < 1.0):
            raise ValueError(f"rcond must lie in (0, 1), got {self.rcond}")
        if self.torque_limit is not None:
            lim = readonly_float(self.torque_limit, "feedback torque_limit")
            if lim.shape != (self.n_actuated,) or np.any(lim <= 0.0):
                raise ValueError("torque_limit must be positive, one per joint")
            object.__setattr__(self, "torque_limit", lim)

    def torque(self, q_error: Array, v_error: Array, q: Array | None = None) -> Array:
        """Saturated ``Lambda(q) (omega^2 e + 2 zeta omega e_dot)``; needs ``q``."""
        if q is None:
            raise ValueError("a state-dependent feedback law needs the configuration q")
        resp = np.asarray(self.response(q), dtype=np.float64)
        lam = np.linalg.pinv(0.5 * (resp + resp.T), rcond=self.rcond, hermitian=True)
        raw = lam @ (
            self.omega_rad_s**2 * q_error + 2.0 * self.zeta * self.omega_rad_s * v_error
        )
        if self.torque_limit is None:
            return np.asarray(raw)
        return np.asarray(np.clip(raw, -self.torque_limit, self.torque_limit))


def check_gain(gain: Array, label: str) -> None:
    """Require a gain vector ``>= 0`` or a symmetric PSD gain matrix (``ValueError``)."""
    if gain.ndim == 1:
        if np.any(gain < 0.0):
            raise ValueError(f"feedback gain entries ({label}) must be >= 0")
        return
    if gain.shape[0] != gain.shape[1] or np.max(np.abs(gain - gain.T)) > 1e-9 * max(
        1.0, float(np.max(np.abs(gain)))
    ):
        raise ValueError(f"feedback gain matrix ({label}) must be symmetric")
    floor = -1e-9 * max(1.0, float(np.max(np.abs(gain))))
    if np.linalg.eigvalsh(0.5 * (gain + gain.T))[0] < floor:
        raise ValueError(f"feedback gain matrix ({label}) must be PSD")


def acceleration_feedback_law(
    response: ArrayLike,
    omega_rad_s: float,
    zeta: float = 1.0,
    *,
    rcond: float = 1e-6,
    torque_limit: ArrayLike | None = None,
) -> FeedbackLaw:
    """Feedback law that shapes the actuated acceleration like a damped spring.

    With ``response = S P_c S^T`` (the actuated acceleration produced by unit
    actuated torque on the closed loop, frozen at one configuration) the law
    ``tau_fb = Lambda (omega^2 e + 2 zeta omega e_dot)``, ``Lambda =
    pinv(response)``, makes the actuated error obey ``e_ddot + 2 zeta omega
    e_dot + omega^2 e = 0`` to the extent that ``response`` stays valid.  A
    per-joint gain scaled by the diagonal mass would be unstable on a coupled
    chain; this scaling is why the law is declared in acceleration terms.

    Args:
        response: Symmetric PSD ``(n_a, n_a)`` actuated response matrix.
        omega_rad_s: Natural frequency [rad/s], positive.
        zeta: Damping ratio, non-negative.
        rcond: Singular values below ``rcond * max`` are dropped by the
            pseudo-inverse (directions the loop makes uncontrollable).
        torque_limit: Optional declared per-joint saturation.
    """
    r = np.asarray(response, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or not np.isfinite(r).all():
        raise ValueError(f"response must be a finite square matrix, got {r.shape}")
    if not (np.isfinite(omega_rad_s) and omega_rad_s > 0.0):
        raise ValueError(f"omega_rad_s must be positive, got {omega_rad_s}")
    if not (np.isfinite(zeta) and zeta >= 0.0):
        raise ValueError(f"zeta must be non-negative, got {zeta}")
    lam = np.linalg.pinv(0.5 * (r + r.T), rcond=rcond, hermitian=True)
    lam = 0.5 * (lam + lam.T)
    return FeedbackLaw(
        kp=omega_rad_s**2 * lam,
        kd=2.0 * zeta * omega_rad_s * lam,
        torque_limit=None if torque_limit is None else np.asarray(torque_limit),
        name=f"acceleration PD (omega={omega_rad_s:g} rad/s, zeta={zeta:g})",
    )


@dataclass(frozen=True)
class ReplaySetup:
    """What a replay needs besides the allocation and the provider.

    Attributes:
        actuation: ``S`` ``(n_a, n)``, a coordinate-selection matrix: row ``i``
            is the unit vector of the coordinate that torque ``i`` drives.
        q0, v0: Initial state (the only state the replay is given).
        reference: Needed by the feedback mode, optional otherwise (it enables
            the error curve and the time to divergence).
        external_load: Known generalised load ``(t, q) -> (n,)`` added to the
            allocated one, for example the allocated ground loads ``J_g^T f_g``;
            it is neither feedback nor part of the budget.
    """

    actuation: Array
    q0: Array
    v0: Array
    reference: ReferenceTrajectory | None = None
    external_load: Callable[[float, Array], Array] | None = None

    def __post_init__(self) -> None:
        s = readonly_float(self.actuation, "actuation")
        if s.ndim != 2 or s.shape[0] == 0:
            raise ValueError(f"actuation must be (n_a, n), got {s.shape}")
        n = s.shape[1]
        cols = np.argmax(s, axis=1)
        selection = np.zeros_like(s)
        selection[np.arange(s.shape[0]), cols] = 1.0
        if not np.array_equal(s, selection) or len(set(cols.tolist())) != s.shape[0]:
            raise ValueError(
                "actuation must be a coordinate-selection matrix (one unit entry "
                "per row, distinct columns)"
            )
        for name in ("q0", "v0"):
            vec = readonly_float(getattr(self, name), name)
            if vec.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {vec.shape}")
            object.__setattr__(self, name, vec)
        object.__setattr__(self, "actuation", s)
        if self.reference is not None and self.reference.n != n:
            raise ValueError(
                f"reference has {self.reference.n} coordinates, expected {n}"
            )
        if self.external_load is not None and not callable(self.external_load):
            raise TypeError("external_load must be callable (t, q) -> (n,) or None")

    @property
    def n(self) -> int:
        """Number of coordinates."""
        return int(self.actuation.shape[1])

    @property
    def n_actuated(self) -> int:
        """Number of actuated coordinates."""
        return int(self.actuation.shape[0])


@dataclass(frozen=True)
class ClosureDrift:
    """Closure residuals along the replay (no stabilisation counted as physics).

    Attributes:
        velocity: ``max |J_c v|`` per node (m/s and rad/s mixed rows).
        position: ``max |Phi(q)|`` per node, or ``None`` when the provider has
            no position-level closure error (never reported as zero).
    """

    velocity: Array
    position: Array | None

    @property
    def velocity_max(self) -> float:
        """Largest velocity-level residual."""
        return float(np.max(self.velocity))

    @property
    def position_max(self) -> float | None:
        """Largest position-level residual, ``None`` if unavailable."""
        return None if self.position is None else float(np.max(self.position))


@dataclass(frozen=True)
class FeedbackBudget:
    """The effort a feedback law spent, relative to the allocated torques.

    Attributes:
        peak_norm_ratio: ``max |tau_fb|_2 / max |tau_a|_2``.
        energy_ratio: ``int |tau_fb|^2 dt / int |tau_a|^2 dt`` (``inf`` if the
            allocated torque is identically zero while feedback is not).
        rms_ratio: ``sqrt(energy_ratio)``.
        abs_work_feedback_j, abs_work_allocated_j: ``int sum_i |tau_i w_i| dt``.
        abs_work_ratio: their quotient.
        per_joint_rms_feedback_nm, per_joint_rms_allocated_nm: RMS per joint.
        saturated_fraction: Share of (node, joint) samples at the declared limit.
    """

    peak_norm_ratio: float
    energy_ratio: float
    rms_ratio: float
    abs_work_feedback_j: float
    abs_work_allocated_j: float
    abs_work_ratio: float
    per_joint_rms_feedback_nm: Array
    per_joint_rms_allocated_nm: Array
    saturated_fraction: float


def _ratio(num: float, den: float) -> float:
    if den > 0.0:
        return float(num / den)
    return 0.0 if num == 0.0 else _INF


def _series(value: ArrayLike, shape: tuple[int, int], name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {arr.shape}")
    return arr


def feedback_budget(
    times_s: ArrayLike,
    allocated: ArrayLike,
    feedback: ArrayLike,
    velocity: ArrayLike,
    torque_limit: ArrayLike | None = None,
) -> FeedbackBudget:
    """Account the feedback effort against the allocated torques.

    Args:
        times_s: Strictly increasing node times ``(T,)``, ``T >= 2``.
        allocated: Allocated torques ``(T, n_a)``.
        feedback: Feedback torques ``(T, n_a)`` (never summed into ``allocated``).
        velocity: Actuated-coordinate velocities ``(T, n_a)`` for the work terms.
        torque_limit: Declared per-joint saturation, if any.

    Postconditions: integrals are trapezoids over ``times_s``; a zero feedback
    series has an exactly zero budget; no value is NaN.
    """
    t = np.asarray(times_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times_s must be strictly increasing with >= 2 nodes")
    n_a_cols = np.shape(allocated)[1] if np.ndim(allocated) == 2 else -1
    shape = (t.size, n_a_cols)
    tau_a = _series(allocated, shape, "allocated")
    tau_f = _series(feedback, shape, "feedback")
    omega = _series(velocity, shape, "velocity")
    span = float(t[-1] - t[0])
    n_a = np.linalg.norm(tau_a, axis=1)
    n_f = np.linalg.norm(tau_f, axis=1)
    energy = _ratio(float(np.trapezoid(n_f**2, t)), float(np.trapezoid(n_a**2, t)))
    work_f = float(np.trapezoid(np.sum(np.abs(tau_f * omega), axis=1), t))
    work_a = float(np.trapezoid(np.sum(np.abs(tau_a * omega), axis=1), t))
    saturated = 0.0
    if torque_limit is not None:
        lim = np.asarray(torque_limit, dtype=np.float64)
        saturated = float(np.mean(np.abs(tau_f) >= lim * (1.0 - 1e-12)))
    return FeedbackBudget(
        peak_norm_ratio=_ratio(float(n_f.max()), float(n_a.max())),
        energy_ratio=energy,
        rms_ratio=float(np.sqrt(energy)) if np.isfinite(energy) else _INF,
        abs_work_feedback_j=work_f,
        abs_work_allocated_j=work_a,
        abs_work_ratio=_ratio(work_f, work_a),
        per_joint_rms_feedback_nm=np.sqrt(np.trapezoid(tau_f**2, t, axis=0) / span),
        per_joint_rms_allocated_nm=np.sqrt(np.trapezoid(tau_a**2, t, axis=0) / span),
        saturated_fraction=saturated,
    )
