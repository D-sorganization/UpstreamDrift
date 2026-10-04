"""ZTCF-anchored, control-bounded one-step prediction (DIME-04, epic #11421).

Within a smooth, contact-free mode the dynamics are input-affine::

    a = f(q, v) + B(q) tau,      f = M^-1 (-bias)   (ZTCF drift)
                                 B = M^-1 S^T        (control influence)

The zero-torque counterfactual drift ``f`` is evaluated without knowing the
applied torques, so it anchors the prediction of the next state. Bounded
torques (``ControlBand``) bound every admissible deviation from that anchor.
Together they give the range of viable next states; anything outside it cannot
be explained by reasonable inputs and is evidence of noise, occlusion or model
error.

Scope and limits:

* Pointwise drift reuses :func:`simulation_backends.ztcf_zvcf.ztcf_acceleration`
  (DRY). Providers expose only ``mass_matrix`` and ``bias_forces``.
* Contact switching is not modelled here. ``contact_active=True`` is refused
  rather than silently extrapolating free-flight drift through stance.
* Integration uses constant acceleration over one step in local coordinates.
  Quaternion (manifold) configurations are refused.
* Zero control mean is an *initialisation* with declared covariance, never an
  assertion that the joints are inactive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import check_finite, ensure, require
from src.shared.python.simulation_backends.ztcf_zvcf import ztcf_acceleration

if TYPE_CHECKING:
    from src.shared.python.simulation_backends.protocol import DynamicsProvider

__all__ = [
    "ControlBand",
    "DriftLinearization",
    "UncertainPrediction",
    "drift_dominance_index",
    "integrate_step",
    "linearize_drift",
    "predict_step",
    "reachable_acceleration_interval",
    "uncertain_control_prediction",
]

_PSD_TOLERANCE = 1e-12


def _finite_vector(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    require(arr.size > 0, f"{name} must be non-empty", value=arr.shape)
    require(check_finite(arr), f"{name} must contain only finite values", value=arr)
    return arr


def _require_psd(name: str, matrix: np.ndarray, size: int) -> np.ndarray:
    mat = np.asarray(matrix, dtype=float)
    require(mat.shape == (size, size), f"{name} must be ({size}, {size})", mat.shape)
    require(check_finite(mat), f"{name} must be finite")
    require(np.allclose(mat, mat.T, atol=1e-12), f"{name} must be symmetric")
    scale = max(1.0, float(np.max(np.abs(mat))))
    require(
        float(np.min(np.linalg.eigvalsh(mat))) >= -_PSD_TOLERANCE * scale,
        f"{name} must be positive semi-definite",
        value=np.linalg.eigvalsh(mat),
    )
    return mat


@dataclass(frozen=True)
class ControlBand:
    """Admissible joint-torque box with an optional torque-rate limit.

    ``lower``/``upper`` are the global actuator limits [N m]. ``rate_limit``
    [N m/s] enforces torque-profile continuity: from the previous torque the
    next step may move at most ``rate_limit * dt`` per channel.
    """

    lower: np.ndarray
    upper: np.ndarray
    rate_limit: np.ndarray | None = None

    def __post_init__(self) -> None:
        lower = _finite_vector("lower", self.lower)
        upper = _finite_vector("upper", self.upper)
        require(lower.shape == upper.shape, "lower and upper must share shape")
        require(bool(np.all(lower <= upper)), "lower must not exceed upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if self.rate_limit is not None:
            rate = _finite_vector("rate_limit", self.rate_limit)
            require(rate.shape == lower.shape, "rate_limit must match lower shape")
            require(bool(np.all(rate >= 0.0)), "rate_limit must be non-negative")
            object.__setattr__(self, "rate_limit", rate)

    @property
    def size(self) -> int:
        """Number of actuated channels."""
        return int(self.lower.size)

    def local(self, previous: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the tight band reachable from ``previous`` within ``dt``.

        The previous torque is clipped into the global box first, so an
        out-of-range warm start narrows toward the box instead of producing an
        empty band. Postcondition: ``lower <= lo <= hi <= upper``.
        """
        prev = _finite_vector("previous", previous)
        require(prev.shape == self.lower.shape, "previous must match band size")
        require(np.isfinite(dt) and dt > 0.0, "dt must be positive and finite", dt)
        if self.rate_limit is None:
            return self.lower.copy(), self.upper.copy()
        anchor = np.clip(prev, self.lower, self.upper)
        step = self.rate_limit * dt
        lo = np.maximum(self.lower, anchor - step)
        hi = np.minimum(self.upper, anchor + step)
        ensure(bool(np.all(lo <= hi)), "local band must be non-empty")
        return lo, hi


@dataclass(frozen=True)
class DriftLinearization:
    """Input-affine split ``a = drift + B tau`` at one state."""

    drift_acceleration: np.ndarray
    control_influence: np.ndarray
    mass_condition_number: float

    def acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Return total acceleration under torque ``tau``."""
        u = _finite_vector("tau", tau)
        require(
            u.size == self.control_influence.shape[1],
            "tau must match the actuated dimension",
            value=u.shape,
        )
        return self.drift_acceleration + self.control_influence @ u


@dataclass(frozen=True)
class UncertainPrediction:
    """Gaussian next-state prediction ``x+ ~ N(mean, covariance)``; x=(q, v)."""

    mean: np.ndarray
    covariance: np.ndarray


def linearize_drift(
    provider: DynamicsProvider,
    q: np.ndarray,
    v: np.ndarray,
    *,
    selection: np.ndarray | None = None,
    contact_active: bool = False,
) -> DriftLinearization:
    """Evaluate the ZTCF drift and control influence at ``(q, v)``.

    Args:
        provider: Dynamics provider (``mass_matrix``/``bias_forces`` only).
        q: Configuration ``(n,)`` [rad].
        v: Velocity ``(n,)`` [rad/s].
        selection: Actuation map ``S^T`` of shape ``(n, m)``; identity when
            ``None`` (fully actuated).
        contact_active: Must be ``False``; contact modes need a constrained
            provider (#10286) and are refused here.

    Returns:
        :class:`DriftLinearization` with drift ``(n,)`` and ``B`` ``(n, m)``.

    Raises:
        ValueError: On contact, non-finite inputs or inconsistent shapes.
    """
    require(
        not contact_active,
        "contact mode is active: free-flight drift is invalid in stance; use a "
        "constrained #10286 provider (prediction disabled)",
    )
    q_arr = _finite_vector("q", q)
    v_arr = _finite_vector("v", v)
    drift = ztcf_acceleration(provider, q_arr, v_arr)
    n = drift.size
    sel = np.eye(n) if selection is None else np.asarray(selection, dtype=float)
    require(
        sel.ndim == 2 and sel.shape[0] == n and sel.shape[1] >= 1,
        f"selection must be ({n}, m); got {sel.shape}",
        value=sel.shape,
    )
    mass = np.asarray(provider.mass_matrix(q_arr), dtype=float)
    influence = np.linalg.solve(mass, sel)
    ensure(check_finite(influence), "control influence must be finite")
    return DriftLinearization(
        drift_acceleration=drift,
        control_influence=influence,
        mass_condition_number=float(np.linalg.cond(mass)),
    )


def reachable_acceleration_interval(
    lin: DriftLinearization, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-coordinate acceleration interval over a torque box.

    For ``tau`` in ``[lower, upper]`` the image ``drift + B tau`` is a
    parallelotope; its axis-aligned hull is ``centre +/- |B| half_width``.
    """
    lo = _finite_vector("lower", lower)
    hi = _finite_vector("upper", upper)
    require(bool(np.all(lo <= hi)), "lower must not exceed upper")
    centre = lin.acceleration(0.5 * (lo + hi))
    half = np.abs(lin.control_influence) @ (0.5 * (hi - lo))
    return centre - half, centre + half


def predict_step(
    lin: DriftLinearization,
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-acceleration step ``q+ = q + dt v + dt^2 a/2``, ``v+ = v + dt a``.

    Raises:
        ValueError: If ``q`` is not in local coordinates (``len(q) != len(v)``,
            e.g. a quaternion manifold) or ``dt`` is not positive.
    """
    q_arr = _finite_vector("q", q)
    v_arr = _finite_vector("v", v)
    require(
        q_arr.shape == v_arr.shape,
        "q must use local coordinates (len(q) == len(v)); manifold "
        "configurations need a retraction and are not supported here",
        value=(q_arr.shape, v_arr.shape),
    )
    require(np.isfinite(dt) and dt > 0.0, "dt must be positive and finite", dt)
    acc = lin.acceleration(tau)
    return q_arr + dt * v_arr + 0.5 * dt * dt * acc, v_arr + dt * acc


def integrate_step(
    provider: DynamicsProvider,
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    dt: float,
    *,
    selection: np.ndarray | None = None,
    contact_active: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One classic RK4 step of ``a = drift + B tau`` with zero-order-hold torque.

    Re-evaluates the drift at every stage, so this is the full nonlinear
    forward dynamics, not the one-step constant-acceleration approximation of
    :func:`predict_step`. Same local-coordinate and contact limits apply.
    """
    q_arr = _finite_vector("q", q)
    v_arr = _finite_vector("v", v)
    require(q_arr.shape == v_arr.shape, "q must use local coordinates (manifold)")
    require(np.isfinite(dt) and dt > 0.0, "dt must be positive and finite", dt)

    def accel(qs: np.ndarray, vs: np.ndarray) -> np.ndarray:
        lin = linearize_drift(
            provider, qs, vs, selection=selection, contact_active=contact_active
        )
        return lin.acceleration(tau)

    k1q, k1v = v_arr, accel(q_arr, v_arr)
    k2q = v_arr + 0.5 * dt * k1v
    k2v = accel(q_arr + 0.5 * dt * k1q, k2q)
    k3q = v_arr + 0.5 * dt * k2v
    k3v = accel(q_arr + 0.5 * dt * k2q, k3q)
    k4q = v_arr + dt * k3v
    k4v = accel(q_arr + dt * k3q, k4q)
    q_next = q_arr + dt / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    v_next = v_arr + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    ensure(check_finite(q_next) and check_finite(v_next), "RK4 step must be finite")
    return q_next, v_next


def uncertain_control_prediction(
    lin: DriftLinearization,
    q: np.ndarray,
    v: np.ndarray,
    control_mean: np.ndarray,
    control_covariance: np.ndarray,
    dt: float,
    *,
    state_covariance: np.ndarray | None = None,
) -> UncertainPrediction:
    """Drift-centred prediction with an explicitly uncertain control.

    Marginalises ``tau ~ N(mu, Sigma_u)`` through the step map::

        x+ = A x + c + G tau,   A = [[I, dt I], [0, I]],
        G = [[dt^2/2 B], [dt B]],
        Cov(x+) = A P A^T + G Sigma_u G^T.

    State/control cross-covariance is assumed zero and the Jacobian of the
    drift with respect to the state is neglected over one step (first-order
    local approximation; disclose it where reported).
    """
    q_arr = _finite_vector("q", q)
    n = q_arr.size
    m = lin.control_influence.shape[1]
    sigma_u = _require_psd("control_covariance", control_covariance, m)
    mean_q, mean_v = predict_step(lin, q_arr, v, control_mean, dt)
    gain = np.vstack(
        [0.5 * dt * dt * lin.control_influence, dt * lin.control_influence]
    )
    covariance = gain @ sigma_u @ gain.T
    if state_covariance is not None:
        p = _require_psd("state_covariance", state_covariance, 2 * n)
        a = np.block([[np.eye(n), dt * np.eye(n)], [np.zeros((n, n)), np.eye(n)]])
        covariance = covariance + a @ p @ a.T
    covariance = 0.5 * (covariance + covariance.T)
    return UncertainPrediction(mean=np.r_[mean_q, mean_v], covariance=covariance)


def drift_dominance_index(
    lin: DriftLinearization, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Share of the admissible acceleration budget owed to drift, in ``[0, 1]``.

    ``|drift| / (|drift| + | |B| half_width |)``. It compares drift with the
    control *authority* in the band, not with the realised total
    acceleration, so cancelling control (total ~ 0) cannot blow it up.
    Returns ``1.0`` when the band has zero width, ``0.0`` when drift is zero.
    """
    lo = _finite_vector("lower", lower)
    hi = _finite_vector("upper", upper)
    require(bool(np.all(lo <= hi)), "lower must not exceed upper")
    drift = float(np.linalg.norm(lin.drift_acceleration))
    authority = float(np.linalg.norm(np.abs(lin.control_influence) @ (0.5 * (hi - lo))))
    if drift + authority == 0.0:
        return 0.0
    index = drift / (drift + authority)
    ensure(0.0 <= index <= 1.0, "dominance index must lie in [0, 1]", index)
    return index
