"""Model-independent static-optimisation numerics (SPN-10b, #12761).

Shared by the leg static optimisation (``musculoskeletal_static_opt``, issue
#11617) and the trunk static optimisation (``spine.static_optimization``).
Nothing here knows about a model: a frame is described by

* ``active`` ``(nm,)``: muscle force per unit activation [N];
* ``passive`` ``(nm,)``: muscle force at zero activation [N] (passive
  elastic plus damping terms; may be negative for a damping force);
* ``moment`` ``(nm, nc)``: moment arms [m] (dimensionless for a translational
  coordinate), ``moment[m, c] * F_m`` being the generalised force of muscle
  ``m`` on coordinate ``c``;
* ``tau`` ``(nc,)``: required generalised forces [N*m or N].

The muscle generalised force is affine in the activation,
``M(a) = moment.T @ (passive + a * active) = offset + G a``.

Two solvers:

``solve_frame`` (leg, unchanged)
    OpenSim-style penalty: ``min sum a^2 + w sum r^2`` with unit reserves,
    solved as one bounded linear least-squares problem (BVLS).

``solve_activations`` (trunk; exponent ``p`` in {2, 3})
    Lexicographic (exact) reserves.  Stage 1 finds the smallest weighted
    reserve ``min_a ||(tau - M(a)) / R||_2`` over ``0 <= a <= 1`` (BVLS); its
    remainder ``r*`` is unique even when ``a`` is not.  Stage 2 resolves the
    redundancy: ``min sum a^p`` s.t. ``G a = G a_1`` (the stage-1 muscle
    moment), ``0 <= a <= 1``, by SLSQP with the analytic gradient ``p a^(p-1)``
    and the constant constraint Jacobian.  For ``p = 2`` stage 2 is a convex
    QP (SLSQP is exact on a QP); for ``p = 3`` it is a smooth convex NLP on
    ``a >= 0``.  This is the ``w -> infinity`` limit of the penalty
    ``sum a^p + w sum (r / R)^2``: reserves carry exactly the remainder the
    muscles cannot produce and are zero (to solver precision) when the target
    is reachable.

    From ``CONIC_MIN_MUSCLES`` muscles stage 2 uses convex QPs (Clarabel; p = 3
    by iteratively reweighted least norm) because SLSQP does not converge on a
    full-body frame (446 muscles); SLSQP remains below the threshold and when
    Clarabel is not installed.

A muscle with no force capacity at the frame (an all-zero gain column) is held
at zero activation and excluded from stage 2 (see ``_redistribute``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear, minimize

from src.shared.python.contracts import ensure, require

__all__ = [
    "CONIC_MIN_MUSCLES",
    "SUPPORTED_EXPONENTS",
    "ActivationSolution",
    "FrameSolution",
    "activation_cost",
    "participation_ratio",
    "solve_activations",
    "solve_frame",
]

SUPPORTED_EXPONENTS: tuple[int, ...] = (2, 3)
_RANK_TOL = 1e-10
_SLSQP_FTOL = 1e-14
_SLSQP_MAXITER = 500
_BOUND_TOL = 1e-12
CONIC_MIN_MUSCLES = 100
_CONIC_TOL = 1e-8
_WEIGHT_FLOOR = 1e-6
_IRLS_MAXITER = 200
_IRLS_TOL = 1e-10
_CONIC_EQ_TOL = 1e-7
_CONIC_OK = ("Solved", "AlmostSolved")


@dataclass(frozen=True)
class FrameSolution:
    """Result of one frame: activations, reserves and the optimiser status."""

    activation: np.ndarray
    reserve: np.ndarray
    cost: float
    success: bool


def solve_frame(
    active: np.ndarray,
    passive: np.ndarray,
    moment: np.ndarray,
    tau: np.ndarray,
    *,
    reserve_weight: float = 1.0,
) -> FrameSolution:
    """Resolve muscle redundancy for one frame.

    Args:
        active: ``(nm,)`` force per unit activation (N), >= 0.
        passive: ``(nm,)`` force at zero activation (N), >= 0.
        moment: ``(nm, nc)`` moment arms (m); ``moment[m, c] * F_m`` is the
            generalised force of muscle ``m`` on coordinate ``c``.
        tau: ``(nc,)`` required generalised forces (N m).
        reserve_weight: relative weight of the squared reserve in the cost.

    Returns:
        The minimiser.  Postconditions: ``0 <= activation <= 1`` and
        ``reserve = tau - moment.T @ (passive + activation * active)``.

    Raises:
        ValueError: on shape mismatch, non-finite input or a non-positive weight.
    """
    nm = active.shape[0]
    require(passive.shape == (nm,), "passive must be (nm,)")
    require(moment.ndim == 2 and moment.shape[0] == nm, "moment must be (nm, nc)")
    require(tau.shape == (moment.shape[1],), "tau must be (nc,)")
    require(reserve_weight > 0.0, "reserve_weight must be positive")
    for name, array in (
        ("active", active),
        ("passive", passive),
        ("moment", moment),
        ("tau", tau),
    ):
        require(bool(np.isfinite(array).all()), f"{name} must be finite")
    gain = moment.T * active[None, :]  # (nc, nm)
    offset = moment.T @ passive
    root = np.sqrt(reserve_weight)
    design = np.vstack([np.eye(nm), root * gain])
    target = np.concatenate([np.zeros(nm), root * (tau - offset)])
    result = lsq_linear(design, target, bounds=(0.0, 1.0), method="bvls")
    a = np.clip(result.x, 0.0, 1.0)
    reserve = tau - offset - gain @ a
    cost = float(a @ a + reserve_weight * (reserve @ reserve))
    return FrameSolution(a, reserve, cost, bool(result.success))


# --------------------------------------------------------- exponent-p solver


def _check_exponent(exponent: object) -> int:
    if not isinstance(exponent, int) or isinstance(exponent, bool):
        raise TypeError(f"exponent must be an int, got {type(exponent).__name__}")
    if exponent not in SUPPORTED_EXPONENTS:
        raise ValueError(f"exponent must be one of {SUPPORTED_EXPONENTS}: {exponent}")
    return exponent


def activation_cost(activation: np.ndarray, exponent: int) -> tuple[float, np.ndarray]:
    """``(sum a^p, p a^(p-1))``: the activation cost and its analytic gradient.

    Preconditions: ``exponent`` in :data:`SUPPORTED_EXPONENTS` (``TypeError``
    for a non-int, ``ValueError`` otherwise).  For ``p = 3`` the cost is convex
    only on ``a >= 0``, which the bounds guarantee.
    """
    p = _check_exponent(exponent)
    a = np.asarray(activation, dtype=float)
    return float(np.sum(a**p)), p * a ** (p - 1)


def participation_ratio(values: np.ndarray) -> float:
    """``(sum |x|)^2 / sum x^2`` in ``[1, n]``: the effective number of muscles.

    Equal to ``n`` when all entries are equal and to 1 when one entry carries
    everything.  Raises ``ValueError`` for an all-zero vector.
    """
    x = np.abs(np.asarray(values, dtype=float))
    squares = float(x @ x)
    if not squares > 0.0:
        raise ValueError("participation ratio needs a non-zero vector")
    return float(np.sum(x) ** 2 / squares)


@dataclass(frozen=True)
class ActivationSolution:
    """One frame of :func:`solve_activations`.

    ``activation`` ``(nm,)`` in [0, 1]; ``muscle_moment`` ``(nc,)`` the muscle
    generalised force ``offset + G a``; ``reserve`` ``(nc,)`` equal to
    ``tau - muscle_moment``; ``cost`` is ``sum a^p``; ``success`` combines both
    stages; ``message`` is the stage-2 optimiser message; ``solver`` names the
    stage-2 solver (``slsqp``, ``clarabel_qp`` or ``clarabel_irls``).
    """

    activation: np.ndarray
    muscle_moment: np.ndarray
    reserve: np.ndarray
    cost: float
    success: bool
    message: str
    solver: str = "slsqp"


def _as_float_array(name: str, value: object, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(value).__name__}")
    out = np.asarray(value, dtype=float)
    if out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _validate(
    active: object,
    passive: object,
    moment: object,
    tau: object,
    reserve_scale: object,
) -> tuple[np.ndarray, ...]:
    act = _as_float_array("active", active, 1)
    pas = _as_float_array("passive", passive, 1)
    arm = _as_float_array("moment", moment, 2)
    req = _as_float_array("tau", tau, 1)
    nm, nc = arm.shape
    if act.shape != (nm,) or pas.shape != (nm,):
        raise ValueError(f"active and passive must have shape ({nm},)")
    if req.shape != (nc,):
        raise ValueError(f"tau must have shape ({nc},), got {req.shape}")
    if np.any(act < 0.0):
        raise ValueError("active force per unit activation must be >= 0")
    scale = np.ones(nc) if reserve_scale is None else reserve_scale
    scale = _as_float_array("reserve_scale", scale, 1)
    if scale.shape != (nc,) or np.any(scale <= 0.0):
        raise ValueError(f"reserve_scale must be ({nc},) and strictly positive")
    return act, pas, arm, req, scale


def _min_reserve_activation(
    gain: np.ndarray, rhs: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Stage 1: ``argmin ||(rhs - G a) / R||`` on the unit box (BVLS)."""
    result = lsq_linear(
        gain / scale[:, None],
        rhs / scale,
        bounds=(0.0, 1.0),
        method="bvls",
        tol=1e-14,
    )
    return np.clip(result.x, 0.0, 1.0), bool(result.success)


def _range_basis(gain: np.ndarray) -> np.ndarray:
    """Orthonormal basis ``(nc, k)`` of the column space of ``gain``."""
    u, s, _vt = np.linalg.svd(gain, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return u[:, :0]
    return u[:, s > _RANK_TOL * s[0]]


def _polish(a: np.ndarray, gain: np.ndarray, target: np.ndarray) -> np.ndarray:
    """One minimum-norm Newton step on the free activations toward ``G a = t``."""
    free = (a > _BOUND_TOL) & (a < 1.0 - _BOUND_TOL)
    if not np.any(free):
        return a
    step = np.linalg.lstsq(gain[:, free], target - gain @ a, rcond=None)[0]
    trial = a.copy()
    trial[free] += step
    if np.all((trial >= 0.0) & (trial <= 1.0)):
        return trial
    return a


def _redistribute(
    a_start: np.ndarray, gain: np.ndarray, exponent: int
) -> tuple[np.ndarray, bool, str]:
    """Stage 2: ``min sum a^p`` s.t. ``G a = G a_start`` on the unit box.

    A muscle whose gain column is identically zero (no force at any
    activation: slack, or shorter than its minimum fibre length) is inert: it
    cannot change ``G a`` and its cost is minimal at 0, so it is held at 0 and
    left out of the SLSQP problem.  Keeping the zero column made SLSQP stop
    with "Inequality constraints incompatible" (#12827).
    """
    live = np.any(gain != 0.0, axis=0)
    if not live.any():
        return np.zeros_like(a_start), True, "no muscle has force capacity"
    if live.all():
        return _redistribute_live(a_start, gain, exponent)
    a = np.zeros_like(a_start)
    a[live], ok, message = _redistribute_live(a_start[live], gain[:, live], exponent)
    return a, ok, message


def _redistribute_live(
    a_start: np.ndarray, gain: np.ndarray, exponent: int
) -> tuple[np.ndarray, bool, str]:
    target = gain @ a_start
    basis = _range_basis(gain)
    reduced = basis.T @ gain  # (k, nm), full row rank
    rhs = basis.T @ target
    norm = max(float(np.max(np.abs(reduced), initial=0.0)), 1.0)
    constraint: dict[str, Any] = {
        "type": "eq",
        "fun": lambda a: (reduced @ a - rhs) / norm,
        "jac": lambda _a: reduced / norm,
    }
    constraints: list[Any] = [constraint] if reduced.shape[0] else []
    result = minimize(
        activation_cost,
        a_start,
        args=(exponent,),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * a_start.size,
        constraints=constraints,
        options={"ftol": _SLSQP_FTOL, "maxiter": _SLSQP_MAXITER},
    )
    a = _polish(np.clip(result.x, 0.0, 1.0), gain, target)
    return a, bool(result.success), str(result.message)


def _weighted_qp(
    reduced: np.ndarray, rhs: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, bool, str]:
    """``min sum w_i a_i^2`` s.t. ``reduced a = rhs``, ``0 <= a <= 1`` (Clarabel)."""
    import clarabel  # lazy: optional extra
    import scipy.sparse as sp

    n = weight.size
    eye = sp.identity(n, format="csc")
    eq = sp.csc_matrix(reduced)
    a_mat = sp.vstack([eq, eye, -eye], format="csc")
    b = np.concatenate([rhs, np.ones(n), np.zeros(n)])
    cones: list[Any] = []
    if eq.shape[0]:
        cones.append(clarabel.ZeroConeT(eq.shape[0]))
    else:
        a_mat, b = a_mat[eq.shape[0] :], b[eq.shape[0] :]
    cones.append(clarabel.NonnegativeConeT(2 * n))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = _CONIC_TOL
    result = clarabel.DefaultSolver(
        sp.diags(2.0 * weight, format="csc"), np.zeros(n), a_mat, b, cones, settings
    ).solve()
    x = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    return x, str(result.status) in _CONIC_OK, str(result.status)


def _redistribute_conic(
    a_start: np.ndarray, gain: np.ndarray, exponent: int
) -> tuple[np.ndarray, bool, str]:
    """Stage 2 by convex QPs (Clarabel): exact for p=2, reweighted for p=3.

    Same problem as :func:`_redistribute`, in time polynomial in the muscle
    count (SLSQP does not finish a full-body frame).  ``p = 3`` is solved by
    iteratively reweighted least norm: each step minimises ``sum w_i a_i^2``
    with ``w_i = max(a_i, _WEIGHT_FLOOR)`` from the previous iterate, whose
    fixed point satisfies the KKT conditions of ``min sum a^3`` (stationarity
    ``w_i a_i = a_i^2``).  Iterates are averaged with the previous one (step
    1/2) for stability and stop when the cost change is below ``_IRLS_TOL``.
    Raises ``ImportError`` without Clarabel.
    """
    n = a_start.size
    target = gain @ a_start
    basis = _range_basis(gain)
    norm = max(float(np.max(np.abs(basis.T @ gain), initial=0.0)), 1.0)
    reduced = (basis.T @ gain) / norm
    rhs = (basis.T @ target) / norm
    a, ok, status = _weighted_qp(reduced, rhs, np.ones(n))
    iterations = 1
    if exponent == 3 and ok:
        cost = float(np.sum(a**3))
        for _step in range(2, _IRLS_MAXITER + 1):
            iterations += 1
            trial, ok, status = _weighted_qp(reduced, rhs, np.maximum(a, _WEIGHT_FLOOR))
            if not ok:
                break
            a = 0.5 * (a + trial)  # feasible: a convex combination of feasible points
            new_cost = float(np.sum(a**3))
            done = abs(cost - new_cost) <= _IRLS_TOL * max(1.0, abs(new_cost))
            cost = new_cost
            if done:
                break
        else:
            ok, status = False, "IRLS iteration limit"
    a = _polish(np.clip(a, 0.0, 1.0), gain, target)
    held = float(np.max(np.abs(reduced @ a - rhs), initial=0.0)) <= _CONIC_EQ_TOL
    return a, ok and held, f"clarabel {status}, {iterations} QP solves"


def _stage_two(
    a_start: np.ndarray, gain: np.ndarray, exponent: int
) -> tuple[np.ndarray, bool, str]:
    """Conic at full-body scale (SLSQP does not converge there), else SLSQP."""
    if a_start.size >= CONIC_MIN_MUSCLES:
        try:
            return _redistribute_conic(a_start, gain, exponent)
        except ImportError:
            pass  # Clarabel not installed: fall back to SLSQP
    return _redistribute(a_start, gain, exponent)


def solve_activations(
    active: np.ndarray,
    passive: np.ndarray,
    moment: np.ndarray,
    tau: np.ndarray,
    *,
    exponent: int = 2,
    reserve_scale: np.ndarray | None = None,
) -> ActivationSolution:
    """Minimum ``sum a^p`` activations with lexicographic (exact) reserves.

    Args:
        active, passive, moment, tau: the frame (see the module docstring).
        exponent: ``p`` in :data:`SUPPORTED_EXPONENTS`.
        reserve_scale: ``(nc,)`` declared reserve limits ``R_j`` (> 0) used as
            the stage-1 weights ``(r_j / R_j)^2``; ``None`` means all ones.

    Returns:
        :class:`ActivationSolution`.  Postconditions: ``0 <= a <= 1``,
        ``reserve = tau - muscle_moment`` exactly, and ``reserve`` is the
        smallest achievable weighted remainder (zero to solver precision when
        ``tau`` is reachable).

    Raises:
        TypeError: a non-ndarray input or a non-int exponent.
        ValueError: shape mismatch, non-finite values, negative ``active``,
            a non-positive reserve scale or an unsupported exponent.
    """
    p = _check_exponent(exponent)
    act, pas, arm, req, scale = _validate(active, passive, moment, tau, reserve_scale)
    gain = arm.T * act[None, :]
    offset = arm.T @ pas
    a1, ok1 = _min_reserve_activation(gain, req - offset, scale)
    a, ok2, message = _stage_two(a1, gain, p)
    solver = (
        ("clarabel_irls" if p == 3 else "clarabel_qp")
        if message.startswith("clarabel")
        else "slsqp"
    )
    muscle = offset + gain @ a
    reserve = req - muscle
    ensure(bool(np.all((a >= 0.0) & (a <= 1.0))), "activations must lie in [0, 1]")
    return ActivationSolution(
        activation=a,
        muscle_moment=muscle,
        reserve=reserve,
        cost=activation_cost(a, p)[0],
        success=ok1 and ok2,
        message=message,
        solver=solver,
    )
