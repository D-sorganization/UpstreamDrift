"""Cross-backend correctness gate for the golf double-pendulum model (M5).

This module is the *correctness harness* that proves the interchangeable
simulation backends (ODE reference, MuJoCo CPU, MuJoCo Warp GPU) agree on the
physics they compute: mass matrices, bias forces, integrated trajectories, and
energy behaviour. It depends only on the frozen Protocols
(:class:`~simulation_backends.protocol.DynamicsProvider`,
:class:`~simulation_backends.protocol.SimulationBackend`) so any conforming
backend can be cross-checked without this module importing a concrete class.

Why every cross-check is tolerance-based (never ``==``)
-------------------------------------------------------
Bit-for-bit equality across backends is **physically impossible** and must
never be asserted:

* **Fused multiply-add (FMA).** A CPU and a GPU evaluate ``a * b + c`` with
  different rounding — the GPU fuses the multiply and add into one rounding
  step while the CPU rounds twice. The last-bit results differ even for the
  identical algebraic expression.
* **Non-associative parallel reductions.** Summing a vector on thousands of GPU
  lanes regroups the additions (``(a + b) + c`` vs ``a + (b + c)``). Floating
  point addition is not associative, so the reduction order — which is
  non-deterministic across launches — changes the low-order bits.
* **float32 defaults.** GPU physics kernels (and MuJoCo Warp in particular)
  default to single precision, whereas the analytical reference accumulates in
  float64. The representable precision gap alone forbids exact comparison.

Consequently **every** comparison here uses :func:`numpy.allclose` with an
explicit, physically justified ``rtol``/``atol`` and *never* the ``==``
operator. The default tolerances widen as error accumulates down the pipeline:
algebraic quantities evaluated at a single state (mass matrix) are tight
(``rtol=1e-7``); a quantity that sums several contributions (bias forces) is a
decade looser (``rtol=1e-6``); an *integrated* trajectory, where per-step error
compounds over the horizon, is looser still (``rtol=1e-4``). Energy drift over a
long passive rollout is compared with a relative tolerance (``rel_tol=1e-2``)
because a symplectic-ish integrator conserves energy only approximately.

Each public check returns a :class:`ValidationReport` rather than asserting, so
callers (tests, CI gates, notebooks) can inspect the realised ``max_abs_error``
and the tolerances that were applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from .protocol import SimState

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .protocol import DynamicsProvider, SimulationBackend

logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of a single tolerance-based cross-check.

    Attributes:
        name: Human-readable identifier for the check (e.g. ``"mass_matrix"``).
        max_abs_error: Largest absolute elementwise discrepancy observed across
            all samples [check-specific units].
        rtol: Relative tolerance that was applied by :func:`numpy.allclose`.
        atol: Absolute tolerance that was applied by :func:`numpy.allclose`.
        passed: Whether every sample satisfied the tolerance.
        detail: Free-form diagnostic string (sample count, worst-case index,
            tolerance rationale).
    """

    name: str
    max_abs_error: float
    rtol: float
    atol: float
    passed: bool
    detail: str


def _require_positive(value: float, label: str) -> float:
    """Validate that a scalar step/horizon precondition is strictly positive.

    Args:
        value: Candidate value.
        label: Name used in the error message.

    Returns:
        ``float(value)`` once validated.

    Raises:
        ValueError: If ``value`` is not strictly positive.
    """
    numeric = float(value)
    if not numeric > 0.0:
        raise ValueError(f"{label} must be > 0, got {value!r}")
    return numeric


def _as_pair(arr: np.ndarray, label: str) -> np.ndarray:
    """Coerce input to a 1-D ``float`` array and require length 2.

    The planar driven double pendulum has exactly two generalised coordinates
    (``[theta1, theta2]``) / velocities (``[omega1, omega2]``), so every state
    vector handed to a check must be length-2.

    Args:
        arr: Candidate array-like.
        label: Name used in error messages.

    Returns:
        A contiguous ``float`` array of shape ``(2,)``.

    Raises:
        ValueError: If the flattened array does not have exactly two entries.
    """
    out = np.asarray(arr, dtype=float).reshape(-1)
    if out.size != 2:
        raise ValueError(f"{label} must have exactly 2 entries, got shape {out.shape}")
    return out


def _compare(
    name: str,
    values_a: np.ndarray,
    values_b: np.ndarray,
    rtol: float,
    atol: float,
    detail: str,
) -> ValidationReport:
    """Build a :class:`ValidationReport` from two stacked result arrays.

    Centralises the tolerance comparison so every check applies *exactly* the
    same ``numpy.allclose`` semantics (DRY) and never falls back to ``==``.

    Args:
        name: Check identifier stored on the report.
        values_a: Stacked results from backend ``a``.
        values_b: Stacked results from backend ``b`` (same shape).
        rtol: Relative tolerance forwarded to :func:`numpy.allclose`.
        atol: Absolute tolerance forwarded to :func:`numpy.allclose`.
        detail: Diagnostic string stored on the report.

    Returns:
        The populated :class:`ValidationReport`.

    Postcondition:
        ``report.max_abs_error >= 0`` and equals ``max |a - b|``.
    """
    arr_a = np.asarray(values_a, dtype=float)
    arr_b = np.asarray(values_b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"{name}: result shapes disagree, {arr_a.shape} vs {arr_b.shape}"
        )
    abs_error = np.abs(arr_a - arr_b)
    max_abs_error = float(abs_error.max()) if abs_error.size else 0.0
    passed = bool(np.allclose(arr_a, arr_b, rtol=rtol, atol=atol))
    logger.debug(
        "cross-check %s: max_abs_error=%.3e rtol=%.1e atol=%.1e passed=%s",
        name,
        max_abs_error,
        rtol,
        atol,
        passed,
    )
    return ValidationReport(
        name=name,
        max_abs_error=max_abs_error,
        rtol=rtol,
        atol=atol,
        passed=passed,
        detail=detail,
    )


def cross_validate_mass_matrix(
    a: DynamicsProvider,
    b: DynamicsProvider,
    q_samples: Sequence[np.ndarray] | Iterable[np.ndarray],
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> ValidationReport:
    """Cross-validate the joint-space inertia matrix ``M(q)`` of two backends.

    The mass matrix is a closed-form algebraic function of the configuration, so
    two correct derivations should agree to near machine precision; the tight
    default ``rtol=1e-7`` still uses :func:`numpy.allclose` (never ``==``)
    because FMA/precision differences perturb the last bits (see module
    docstring).

    Args:
        a: First dynamics provider (e.g. the ODE reference backend).
        b: Second dynamics provider (e.g. the MuJoCo CPU backend).
        q_samples: Non-empty iterable of configuration vectors, each shape
            ``(2,)``.
        rtol: Relative tolerance for the elementwise comparison.
        atol: Absolute tolerance for the elementwise comparison.

    Returns:
        A :class:`ValidationReport` over the stacked ``(N, 2, 2)`` matrices.

    Raises:
        ValueError: If ``q_samples`` is empty, a sample is not length-2, or a
            provider returns a non ``(2, 2)`` matrix.
    """
    samples = [_as_pair(q, "q sample") for q in q_samples]
    if not samples:
        raise ValueError("q_samples must be non-empty")

    mats_a = np.stack([_mass_matrix_2x2(a, q) for q in samples])
    mats_b = np.stack([_mass_matrix_2x2(b, q) for q in samples])
    detail = (
        f"{len(samples)} configuration sample(s); M(q) is algebraic so the "
        f"tolerance is tight (rtol={rtol:.0e}), but FMA/float-precision skew "
        "forbids bit-equality."
    )
    return _compare("mass_matrix", mats_a, mats_b, rtol, atol, detail)


def cross_validate_bias(
    a: DynamicsProvider,
    b: DynamicsProvider,
    states: Iterable[tuple[np.ndarray, np.ndarray]],
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> ValidationReport:
    """Cross-validate the bias forces ``C(q,v)v + g(q) (+ damping)``.

    Bias forces sum Coriolis, gravity and damping contributions, so rounding
    accumulates over a few terms; the default ``rtol=1e-6`` is therefore one
    decade looser than the mass-matrix check. The comparison is still
    tolerance-based via :func:`numpy.allclose`.

    Args:
        a: First dynamics provider.
        b: Second dynamics provider.
        states: Non-empty iterable of ``(q, v)`` pairs, each component shape
            ``(2,)``.
        rtol: Relative tolerance for the elementwise comparison.
        atol: Absolute tolerance for the elementwise comparison.

    Returns:
        A :class:`ValidationReport` over the stacked ``(N, 2)`` bias vectors.

    Raises:
        ValueError: If ``states`` is empty or any ``q``/``v`` is not length-2.
    """
    pairs = [(_as_pair(q, "q"), _as_pair(v, "v")) for q, v in states]
    if not pairs:
        raise ValueError("states must be non-empty")

    bias_a = np.stack([_bias_forces_vec(a, q, v) for q, v in pairs])
    bias_b = np.stack([_bias_forces_vec(b, q, v) for q, v in pairs])
    detail = (
        f"{len(pairs)} (q, v) sample(s); bias sums Coriolis+gravity+damping so "
        f"the tolerance (rtol={rtol:.0e}) is a decade looser than M(q)."
    )
    return _compare("bias_forces", bias_a, bias_b, rtol, atol, detail)


def cross_validate_trajectory(
    a: SimulationBackend,
    b: SimulationBackend,
    controls: np.ndarray | None,
    horizon: int,
    dt: float,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ValidationReport:
    """Cross-validate integrated ``(q, v)`` trajectories of two backends.

    Both backends are reset to the *same* canonical initial state
    (``SimState(q=[0.1, -0.1], v=[0, 0])``) and rolled out for ``horizon`` steps
    of size ``dt``. Per-step truncation/rounding error compounds along the
    horizon, so the default trajectory tolerance (``rtol=1e-4``, ``atol=1e-5``)
    is necessarily looser than the single-state algebraic checks. Agreement is
    decided by :func:`numpy.allclose` on the concatenated position and velocity
    histories — never exact equality.

    Args:
        a: First backend to integrate.
        b: Second backend to integrate.
        controls: Shared control history ``(horizon, 2)`` applied to *both*
            backends, or ``None`` for a passive (zero-torque) rollout.
        horizon: Number of integration steps (``> 0``).
        dt: Integration step size [s] (``> 0``).
        rtol: Relative tolerance for the elementwise comparison.
        atol: Absolute tolerance for the elementwise comparison.

    Returns:
        A :class:`ValidationReport` over the stacked position and velocity
        histories (each ``(horizon + 1, 2)``).

    Raises:
        ValueError: If ``horizon`` or ``dt`` is not strictly positive, or if
            ``controls`` is provided with the wrong shape, or if the two traces
            disagree on sample count.
    """
    steps = int(horizon)
    if steps <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon!r}")
    step = _require_positive(dt, "dt")
    ctrl = _validate_controls(controls, steps)

    initial = SimState(q=np.array([0.1, -0.1]), v=np.zeros(2))
    a.reset(initial.copy())
    b.reset(initial.copy())
    trace_a = a.rollout(ctrl, steps, step)
    trace_b = b.rollout(ctrl, steps, step)

    if trace_a.num_steps != trace_b.num_steps:
        raise ValueError(
            "trajectory length mismatch: "
            f"{trace_a.num_steps} vs {trace_b.num_steps} (expected {steps + 1})"
        )

    stacked_a = np.concatenate([trace_a.q, trace_a.v], axis=1)
    stacked_b = np.concatenate([trace_b.q, trace_b.v], axis=1)
    detail = (
        f"horizon={steps}, dt={step:g}, "
        f"{'passive' if ctrl is None else 'driven'}; per-step error compounds "
        f"over the rollout so the trajectory tolerance (rtol={rtol:.0e}, "
        f"atol={atol:.0e}) is looser than the algebraic checks."
    )
    return _compare("trajectory", stacked_a, stacked_b, rtol, atol, detail)


def kinetic_energy(provider: DynamicsProvider, q: np.ndarray, v: np.ndarray) -> float:
    """Return the kinetic energy ``0.5 * v^T M(q) v`` [J].

    Args:
        provider: A dynamics provider exposing :meth:`mass_matrix`.
        q: Configuration vector, shape ``(2,)``.
        v: Velocity vector, shape ``(2,)``.

    Returns:
        The (non-negative for a physical positive-definite ``M``) kinetic
        energy as a Python ``float``.

    Raises:
        ValueError: If ``q`` or ``v`` is not length-2, or ``M(q)`` is not
            ``(2, 2)``.
    """
    q_arr = _as_pair(q, "q")
    v_arr = _as_pair(v, "v")
    mass = _mass_matrix_2x2(provider, q_arr)
    return float(0.5 * v_arr @ mass @ v_arr)


def check_energy_conservation(
    backend: SimulationBackend,
    provider: DynamicsProvider,
    initial: SimState,
    horizon: int,
    dt: float,
    rel_tol: float = 1e-2,
) -> ValidationReport:
    """Check that kinetic energy stays ~constant along a *conservative* rollout.

    For a conservative model — gravity disabled and zero joint damping — the
    only energy store along a *passive* (zero-torque) rollout is kinetic energy,
    so total energy is constant and the kinetic energy must not drift. Because a
    discrete integrator conserves energy only approximately, the maximum
    relative deviation from the initial kinetic energy is compared against
    ``rel_tol`` (default 1%) with :func:`numpy.allclose`-style logic — never
    exact equality.

    Args:
        backend: The backend to roll out (must be configured conservative).
        provider: Dynamics provider supplying ``M(q)`` for the energy integral
            (typically the *same* backend if it also implements
            :class:`~simulation_backends.protocol.DynamicsProvider`).
        initial: Initial :class:`SimState` (must carry non-zero velocity for a
            meaningful relative tolerance).
        horizon: Number of passive steps to integrate (``> 0``).
        dt: Integration step size [s] (``> 0``).
        rel_tol: Allowed relative kinetic-energy drift.

    Returns:
        A :class:`ValidationReport` whose ``max_abs_error`` is the worst
        relative energy deviation and ``passed`` is whether it stayed within
        ``rel_tol``.

    Raises:
        TypeError: If ``initial`` is not a :class:`SimState`.
        ValueError: If ``horizon``/``dt`` are not positive, ``rel_tol`` is not
            positive, or the initial kinetic energy is zero (no scale for a
            relative tolerance).
    """
    if not isinstance(initial, SimState):
        raise TypeError(f"initial must be a SimState, got {type(initial).__name__}")
    steps = int(horizon)
    if steps <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon!r}")
    step = _require_positive(dt, "dt")
    tol = _require_positive(rel_tol, "rel_tol")

    e0 = kinetic_energy(provider, initial.q, initial.v)
    if not abs(e0) > 0.0:
        raise ValueError(
            "initial kinetic energy is zero; a relative-tolerance energy check "
            "needs a non-zero initial velocity"
        )

    backend.reset(initial.copy())
    trace = backend.rollout(None, steps, step)
    energies = np.array(
        [
            kinetic_energy(provider, qk, vk)
            for qk, vk in zip(trace.q, trace.v, strict=True)
        ]
    )

    rel_dev = np.abs(energies - e0) / abs(e0)
    max_rel = float(rel_dev.max()) if rel_dev.size else 0.0
    passed = bool(np.allclose(energies, e0, rtol=tol, atol=0.0))
    detail = (
        f"conservative passive rollout: horizon={steps}, dt={step:g}, "
        f"E0={e0:.6g} J; max relative KE drift {max_rel:.3e} vs rel_tol={tol:.0e}."
    )
    logger.debug(
        "energy conservation: E0=%.6g max_rel=%.3e rel_tol=%.1e passed=%s",
        e0,
        max_rel,
        tol,
        passed,
    )
    return ValidationReport(
        name="energy_conservation",
        max_abs_error=max_rel,
        rtol=tol,
        atol=0.0,
        passed=passed,
        detail=detail,
    )


def _validate_controls(controls: np.ndarray | None, horizon: int) -> np.ndarray | None:
    """Validate an optional control history against the rollout contract.

    Args:
        controls: ``None`` (passive) or a ``(horizon, 2)`` torque history.
        horizon: Expected number of control rows.

    Returns:
        ``None`` if passive, otherwise a validated ``float`` array
        ``(horizon, 2)``.

    Raises:
        ValueError: If the control array does not have shape ``(horizon, 2)``.
    """
    if controls is None:
        return None
    arr = np.asarray(controls, dtype=float)
    if arr.shape != (horizon, 2):
        raise ValueError(f"controls must have shape ({horizon}, 2), got {arr.shape}")
    return arr


def _mass_matrix_2x2(provider: DynamicsProvider, q: np.ndarray) -> np.ndarray:
    """Return ``provider.mass_matrix(q)`` validated as a ``(2, 2)`` array.

    Args:
        provider: The dynamics provider to query.
        q: A length-2 configuration vector.

    Returns:
        The mass matrix as a ``float`` array of shape ``(2, 2)``.

    Raises:
        ValueError: If the returned matrix is not ``(2, 2)``.
    """
    mass = np.asarray(provider.mass_matrix(q), dtype=float)
    if mass.shape != (2, 2):
        raise ValueError(f"mass_matrix must be (2, 2), got {mass.shape}")
    return mass


def _bias_forces_vec(
    provider: DynamicsProvider, q: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Return ``provider.bias_forces(q, v)`` validated as a ``(2,)`` array.

    Args:
        provider: The dynamics provider to query.
        q: A length-2 configuration vector.
        v: A length-2 velocity vector.

    Returns:
        The bias-force vector as a ``float`` array of shape ``(2,)``.

    Raises:
        ValueError: If the returned vector is not length-2.
    """
    bias = np.asarray(provider.bias_forces(q, v), dtype=float).reshape(-1)
    if bias.size != 2:
        raise ValueError(f"bias_forces must have 2 entries, got {bias.shape}")
    return bias
