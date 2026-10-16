"""Covariance propagation and adjoint sensitivity kernels (NSX-90, #12719).

Implements ``docs/research/loop_nullspace/loop_nullspace.tex``
``sec:var-cov`` for the discrete tangent-basis steps

    dx_{t+1} = A_t dx_t + B_t w_t,          A_t = I + dt F_t,  B_t = dt G_t,

with ``(F_t, G_t)`` from :func:`tangent_linearization.tangent_state_space`
(explicit Euler, a declared discretization).  Shapes are general in the tangent
dimension ``k`` (``nx = 2 k``); the module never inspects ``k``, so the
connection between tangent frames for ``k > 1`` (#12817) lives upstream of it.

* Forward: ``P_{t+1} = A_t P_t A_t^T + B_t Sigma_t B_t^T`` and
  ``Cov(y) = C P_N C^T``.
* Backward (adjoint): ``S_x[N] = C``, ``S_x[t] = S_x[t+1] A_t`` and
  ``S_in[t] = S_x[t+1] B_t``, so ``Cov(y) = S_x[0] P_0 S_x[0]^T
  + sum_t S_in[t] Sigma_t S_in[t]^T``.  ``S_in[t][i, j]`` is "which input
  channel ``j``, at which step ``t``" moves outcome ``i``; ``S_x[0]`` is the
  initial-state kernel ``S_{y,x0}``.
* Persistent parameter states (per-trial constants, e.g. MOSAIC parameter
  uncertainty, rule R14) are appended to the state with ``A_aug = [[A, E],
  [0, I]]``; their covariance enters ``P_0`` and their kernel is the trailing
  block of ``S_x[0]``.
* The variance budget splits the total by input class and step.  It requires
  the classes to be mutually uncorrelated (block-diagonal covariances), so the
  parts add up exactly.

Non-claim: linear predictions do not cover saturation, slip or missed impacts
(see :mod:`nonlinearity_gate`).  Pure numpy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.core.contracts import ensure
from src.shared.python.estimation.mosaic.local_policy import LinearizedDynamics
from src.shared.python.loop_nullspace.event_section import event_projection
from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.perturbation.disturbance import DisturbanceSpec, Spectrum

__all__ = [
    "AdjointKernels",
    "InputClass",
    "VarianceBudget",
    "VarianceProblem",
    "adjoint_covariance",
    "adjoint_kernels",
    "augment_persistent",
    "augmented_initial_covariance",
    "euler_dynamics",
    "event_outcome_rows",
    "pad_outcome_rows",
    "propagate_covariance",
    "stacked_covariance",
    "stacked_sensitivity",
    "step_covariances",
    "variance_budget",
]

_SYM_RTOL = 1e-9
_CLASS_RTOL = 1e-12


def _fail(message: str) -> None:
    raise ValueError(message)


def _matrix(value: ArrayLike, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        _fail(f"{name} must be 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        _fail(f"{name} must be finite")
    return arr


def _covariance(value: ArrayLike, size: int, name: str) -> Array:
    arr = _matrix(value, name)
    if arr.shape != (size, size):
        _fail(f"{name} must have shape ({size}, {size}), got {arr.shape}")
    scale = max(1.0, float(np.max(np.abs(arr))))
    if np.max(np.abs(arr - arr.T)) > _SYM_RTOL * scale:
        _fail(f"{name} must be symmetric")
    return arr


def _check_chain(
    dyn: LinearizedDynamics, p0: ArrayLike, sigma: ArrayLike
) -> tuple[Array, Array]:
    if not (np.all(np.isfinite(dyn.a)) and np.all(np.isfinite(dyn.b))):
        _fail("dynamics (A, B) must be finite")
    n_steps, n_x, n_w = dyn.b.shape
    p0_arr = _covariance(p0, n_x, "p0")
    sig = np.asarray(sigma, dtype=np.float64)
    if sig.shape != (n_steps, n_w, n_w):
        _fail(f"sigma must have shape {(n_steps, n_w, n_w)}, got {sig.shape}")
    for t in range(n_steps):
        _covariance(sig[t], n_w, f"sigma[{t}]")
    return p0_arr, sig


def euler_dynamics(
    state_spaces: Sequence[tuple[ArrayLike, ArrayLike]], dt: float
) -> LinearizedDynamics:
    """Explicit-Euler ``A_t = I + dt F_t``, ``B_t = dt G_t`` from ``(F, G)`` pairs.

    Preconditions (``ValueError``): ``dt`` positive and finite, at least one
    pair, ``F`` square ``(nx, nx)`` and ``G`` ``(nx, nw)`` with the same shapes
    at every step.  Postcondition: ``A`` is ``(T, nx, nx)``, ``B`` is
    ``(T, nx, nw)``.
    """
    if not (math.isfinite(dt) and dt > 0.0):
        _fail(f"dt must be positive and finite, got {dt}")
    if len(state_spaces) == 0:
        _fail("at least one (F, G) pair is required")
    f0, g0 = _matrix(state_spaces[0][0], "F"), _matrix(state_spaces[0][1], "G")
    if f0.shape[0] != f0.shape[1] or g0.shape[0] != f0.shape[0]:
        _fail(f"F must be square and G conformal, got {f0.shape} and {g0.shape}")
    a_all, b_all = [], []
    for f_raw, g_raw in state_spaces:
        f_mat, g_mat = _matrix(f_raw, "F"), _matrix(g_raw, "G")
        if f_mat.shape != f0.shape or g_mat.shape != g0.shape:
            _fail("every (F, G) pair must have the shapes of the first")
        a_all.append(np.eye(f_mat.shape[0]) + dt * f_mat)
        b_all.append(dt * g_mat)
    return LinearizedDynamics(np.array(a_all), np.array(b_all))


def step_covariances(
    spec: DisturbanceSpec,
    dt: float,
    n_steps: int,
    signals: ArrayLike | None = None,
) -> Array:
    """Per-step covariances ``Sigma_t`` of a white disturbance, ``(T, nw, nw)``.

    Uses :meth:`DisturbanceSpec.discrete_covariance` (``S_w / dt``, timestep
    consistent, V6) and, when the spec declares the M1 law, the nominal signal
    ``u_t`` of shape ``(T, nw)`` per step.  Persistent and coloured
    disturbances are not per-step white noise and are rejected: a persistent
    one enters through :func:`augment_persistent`.
    """
    if n_steps < 1:
        _fail(f"n_steps must be >= 1, got {n_steps}")
    if spec.spectrum is not Spectrum.WHITE:
        _fail(
            "step_covariances supports white disturbances only; persistent "
            "ones enter via augment_persistent"
        )
    if spec.has_signal_law != (signals is not None):
        _fail("signals are required iff the spec declares a signal-dependent law")
    if signals is None:
        return np.repeat(spec.discrete_covariance(dt)[None], n_steps, axis=0)
    sig = np.asarray(signals, dtype=np.float64)
    if sig.shape != (n_steps, spec.n_channels):
        _fail(f"signals must have shape {(n_steps, spec.n_channels)}, got {sig.shape}")
    return np.stack([spec.discrete_covariance(dt, sig[t]) for t in range(n_steps)])


def propagate_covariance(
    dyn: LinearizedDynamics, p0: ArrayLike, sigma: ArrayLike
) -> Array:
    """Forward recursion ``P_{t+1} = A P A^T + B Sigma B^T``; shape ``(T+1, nx, nx)``.

    Preconditions (``ValueError``): finite dynamics, ``p0`` symmetric
    ``(nx, nx)``, ``sigma`` ``(T, nw, nw)`` symmetric per step.
    Postcondition: ``traj[0] == p0`` and every ``P_t`` is symmetric.
    """
    p0_arr, sig = _check_chain(dyn, p0, sigma)
    traj = np.empty((dyn.n_steps + 1, *p0_arr.shape))
    traj[0] = p0_arr
    for t in range(dyn.n_steps):
        a_t, b_t = dyn.a[t], dyn.b[t]
        nxt = a_t @ traj[t] @ a_t.T + b_t @ sig[t] @ b_t.T
        traj[t + 1] = 0.5 * (nxt + nxt.T)
    ensure(bool(np.all(np.isfinite(traj))), "propagated covariance must be finite")
    return traj


@dataclass(frozen=True)
class AdjointKernels:
    """Backward sensitivity kernels of the outcome rows ``C``.

    ``state[t]`` is ``dy/dx_t`` (shape ``(ny, nx)``) and ``inputs[t]`` is
    ``dy/dw_t`` (shape ``(ny, nw)``).
    """

    state: Array
    inputs: Array

    @property
    def s_y_x0(self) -> Array:
        """``S_{y,x0}``: sensitivity of the outcomes to the initial state."""
        return self.state[0]

    def s_y_tau(self, step: int) -> Array:
        """``S_{y,tau}(t)``: sensitivity to the input at step ``t`` ("when")."""
        if not 0 <= step < self.inputs.shape[0]:
            _fail(f"step must be in [0, {self.inputs.shape[0]}), got {step}")
        return self.inputs[step]

    @property
    def n_steps(self) -> int:
        return int(self.inputs.shape[0])


def adjoint_kernels(dyn: LinearizedDynamics, outcome_rows: ArrayLike) -> AdjointKernels:
    """One backward sweep: ``S_x[N] = C``, ``S_x[t] = S_x[t+1] A_t``, ``S_in[t] = S_x[t+1] B_t``.

    Preconditions (``ValueError``): ``outcome_rows`` finite ``(ny, nx)``.
    Postcondition: ``S_x[0]`` equals ``C Phi(N, 0)``; kernels are finite.
    """
    c_mat = _matrix(outcome_rows, "outcome_rows")
    n_steps, n_x, n_w = dyn.b.shape
    if c_mat.shape[1] != n_x:
        _fail(f"outcome_rows must have {n_x} columns (state size), got {c_mat.shape}")
    if not (np.all(np.isfinite(dyn.a)) and np.all(np.isfinite(dyn.b))):
        _fail("dynamics (A, B) must be finite")
    state = np.empty((n_steps + 1, c_mat.shape[0], n_x))
    inputs = np.empty((n_steps, c_mat.shape[0], n_w))
    state[n_steps] = c_mat
    for t in range(n_steps - 1, -1, -1):
        inputs[t] = state[t + 1] @ dyn.b[t]
        state[t] = state[t + 1] @ dyn.a[t]
    ensure(bool(np.all(np.isfinite(state))), "adjoint kernels must be finite")
    return AdjointKernels(state=state, inputs=inputs)


def _check_kernel_inputs(
    kernels: AdjointKernels, p0: ArrayLike, sigma: ArrayLike
) -> tuple[Array, Array]:
    """Validated ``(p0, sigma)`` conformal with the kernels' state and input sizes."""
    n_x = kernels.state.shape[2]
    n_steps, _, n_w = kernels.inputs.shape
    p0_arr = _covariance(p0, n_x, "p0")
    sig = np.asarray(sigma, dtype=np.float64)
    if sig.shape != (n_steps, n_w, n_w):
        _fail(f"sigma must have shape {(n_steps, n_w, n_w)}, got {sig.shape}")
    return p0_arr, sig


def adjoint_covariance(
    kernels: AdjointKernels, p0: ArrayLike, sigma: ArrayLike
) -> Array:
    """``Cov(y) = S_x[0] P_0 S_x[0]^T + sum_t S_in[t] Sigma_t S_in[t]^T``."""
    p0_arr, sig = _check_kernel_inputs(kernels, p0, sigma)
    n_x = p0_arr.shape[0]
    n_steps = sig.shape[0]
    out = kernels.s_y_x0 @ p0_arr @ kernels.s_y_x0.T
    for t in range(n_steps):
        out = out + kernels.inputs[t] @ sig[t] @ kernels.inputs[t].T
    return np.asarray(0.5 * (out + out.T))


# ------------------------------------------------------------ persistent state


def augment_persistent(
    dyn: LinearizedDynamics, coupling: ArrayLike
) -> LinearizedDynamics:
    """Append a per-trial constant parameter state ``theta`` to the chain.

    ``x_{t+1} = A_t x_t + E_t theta + B_t w_t`` with ``theta_{t+1} = theta_t``
    becomes ``A_aug = [[A, E], [0, I]]`` and ``B_aug = [[B], [0]]``.
    ``coupling`` is ``E`` of shape ``(T, nx, np)``.  The *same* draw acts at
    every step, so its variance contribution is ``S_theta Sigma_theta S_theta^T``
    with ``S_theta = sum_t C Phi(N, t+1) E_t`` rather than a sum of independent
    per-step terms.
    """
    e_mat = np.asarray(coupling, dtype=np.float64)
    n_steps, n_x, _ = dyn.b.shape
    if e_mat.ndim != 3 or e_mat.shape[:2] != (n_steps, n_x):
        _fail(f"coupling must have shape ({n_steps}, {n_x}, np), got {e_mat.shape}")
    if not np.all(np.isfinite(e_mat)):
        _fail("coupling must be finite")
    n_p = e_mat.shape[2]
    a_aug = np.zeros((n_steps, n_x + n_p, n_x + n_p))
    b_aug = np.zeros((n_steps, n_x + n_p, dyn.b.shape[2]))
    a_aug[:, :n_x, :n_x] = dyn.a
    a_aug[:, :n_x, n_x:] = e_mat
    a_aug[:, n_x:, n_x:] = np.eye(n_p)
    b_aug[:, :n_x, :] = dyn.b
    return LinearizedDynamics(a_aug, b_aug)


def augmented_initial_covariance(
    p0: ArrayLike, parameter_covariance: ArrayLike
) -> Array:
    """Block-diagonal ``P_0`` of ``(dx_0, theta)`` (uncorrelated by assumption)."""
    p_x = _matrix(p0, "p0")
    p_t = _matrix(parameter_covariance, "parameter_covariance")
    _covariance(p_x, p_x.shape[0], "p0")
    _covariance(p_t, p_t.shape[0], "parameter_covariance")
    n_x, n_p = p_x.shape[0], p_t.shape[0]
    out = np.zeros((n_x + n_p, n_x + n_p))
    out[:n_x, :n_x] = p_x
    out[n_x:, n_x:] = p_t
    return out


def pad_outcome_rows(outcome_rows: ArrayLike, n_extra: int) -> Array:
    """Append ``n_extra`` zero columns (the outcome is not read from ``theta``)."""
    c_mat = _matrix(outcome_rows, "outcome_rows")
    if n_extra < 0:
        _fail(f"n_extra must be >= 0, got {n_extra}")
    return np.hstack([c_mat, np.zeros((c_mat.shape[0], n_extra))])


def event_outcome_rows(
    outcome_jacobian: ArrayLike,
    grad_g: ArrayLike,
    flow: ArrayLike,
    g_t: float = 0.0,
) -> Array:
    """Outcome rows at the physical event: ``C_ev = C_y Pi`` (V2, NSX-83).

    ``Pi`` (:func:`event_section.event_projection`) removes the event-time
    shift, so a perturbation along the flow changes no outcome; evaluating at a
    fixed time instead would omit it.  Postcondition: ``C_ev f = 0``.
    """
    c_y = _matrix(outcome_jacobian, "outcome_jacobian")
    pi_mat = event_projection(grad_g, flow, g_t)
    if c_y.shape[1] != pi_mat.shape[0]:
        _fail(f"outcome_jacobian needs {pi_mat.shape[0]} columns, got {c_y.shape}")
    return np.asarray(c_y @ pi_mat)


# --------------------------------------------------------------------- budget


@dataclass(frozen=True)
class InputClass:
    """A named group of input indices (noise channels or initial-state entries)."""

    name: str
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            _fail("InputClass requires a non-empty name")
        if len(self.indices) == 0 or len(set(self.indices)) != len(self.indices):
            _fail(f"InputClass {self.name!r} needs unique, non-empty indices")
        if min(self.indices) < 0:
            _fail(f"InputClass {self.name!r} has a negative index")


@dataclass(frozen=True)
class VarianceBudget:
    """Outcome-variance parts by input class (diagonal of ``Cov(y)``).

    ``contributions[name]`` has shape ``(ny,)``; ``by_time[name]`` has shape
    ``(T, ny)`` for noise classes.  Postcondition: parts sum to ``total``.
    """

    contributions: dict[str, Array]
    by_time: dict[str, Array]
    total: Array

    def fractions(self) -> Array:
        """Shares of the total, shape ``(n_classes, ny)`` in class order."""
        with np.errstate(divide="ignore", invalid="ignore"):
            rows = [self.contributions[k] / self.total for k in self.contributions]
        return np.array(rows)


def _check_partition(
    classes: Sequence[InputClass], size: int, cov: Array, label: str
) -> None:
    seen: list[int] = []
    for cls in classes:
        if max(cls.indices) >= size:
            _fail(f"{label} class {cls.name!r} index out of range for size {size}")
        seen.extend(cls.indices)
    if sorted(seen) != list(range(size)):
        _fail(f"every {label} index must belong to exactly one class")
    scale = max(1.0, float(np.max(np.abs(cov))))
    for i, first in enumerate(classes):
        for second in classes[i + 1 :]:
            block = cov[np.ix_(first.indices, second.indices)]
            if np.max(np.abs(block)) > _CLASS_RTOL * scale:
                _fail(
                    f"cross-class covariance between {first.name!r} and "
                    f"{second.name!r}; merge the classes to budget them"
                )


def _quad_diag(kernel: Array, cov: Array) -> Array:
    return np.asarray(np.einsum("ij,jk,ik->i", kernel, cov, kernel))


def variance_budget(
    kernels: AdjointKernels,
    p0: ArrayLike,
    sigma: ArrayLike,
    noise_classes: Sequence[InputClass],
    initial_classes: Sequence[InputClass],
) -> VarianceBudget:
    """Variance by input class: initial-state classes plus per-step noise classes.

    Preconditions (``ValueError``): the classes partition the state entries and
    the noise channels, and are mutually uncorrelated in ``p0`` and every
    ``Sigma_t`` (otherwise the parts would not sum to the total).
    Postcondition: ``sum(contributions) == diag(Cov(y))`` to ``1e-9`` relative.
    """
    p0_arr, sig = _check_kernel_inputs(kernels, p0, sigma)
    n_x, n_w = p0_arr.shape[0], sig.shape[1]
    n_steps = sig.shape[0]
    _check_partition(initial_classes, n_x, p0_arr, "initial-state")
    for t in range(n_steps):
        _check_partition(noise_classes, n_w, sig[t], "noise channel")
    parts: dict[str, Array] = {}
    by_time: dict[str, Array] = {}
    for cls in initial_classes:
        idx = list(cls.indices)
        parts[cls.name] = _quad_diag(kernels.s_y_x0[:, idx], p0_arr[np.ix_(idx, idx)])
    for cls in noise_classes:
        idx = list(cls.indices)
        series = np.array(
            [
                _quad_diag(kernels.inputs[t][:, idx], sig[t][np.ix_(idx, idx)])
                for t in range(n_steps)
            ]
        ).reshape(n_steps, -1)
        by_time[cls.name] = series
        parts[cls.name] = series.sum(axis=0)
    total = np.diag(adjoint_covariance(kernels, p0_arr, sig)).copy()
    got = np.sum(list(parts.values()), axis=0)
    ensure(
        bool(np.allclose(got, total, rtol=1e-9, atol=1e-15)),
        "variance budget parts must sum to the total variance",
    )
    return VarianceBudget(contributions=parts, by_time=by_time, total=total)


# ----------------------------------------------------- stacked input vector


def stacked_sensitivity(kernels: AdjointKernels) -> Array:
    """``[S_x0, S_in_0, ..., S_in_{T-1}]``: ``dy/d(x0, w_0, ..., w_{T-1})``."""
    return np.hstack([kernels.s_y_x0, *kernels.inputs])


def stacked_covariance(p0: ArrayLike, sigma: ArrayLike) -> Array:
    """Block-diagonal covariance of ``(x0, w_0, ..., w_{T-1})``."""
    p0_arr = _matrix(p0, "p0")
    sig = np.asarray(sigma, dtype=np.float64)
    blocks = [p0_arr, *sig]
    size = sum(b.shape[0] for b in blocks)
    out = np.zeros((size, size))
    pos = 0
    for block in blocks:
        n = block.shape[0]
        out[pos : pos + n, pos : pos + n] = block
        pos += n
    return out


@dataclass(frozen=True)
class VarianceProblem:
    """One variance question: dynamics, input covariances, outcome and classes."""

    dynamics: LinearizedDynamics
    initial_covariance: Array
    step_covariance: Array
    outcome_rows: Array
    noise_classes: tuple[InputClass, ...]
    initial_classes: tuple[InputClass, ...]

    def __post_init__(self) -> None:
        _check_chain(self.dynamics, self.initial_covariance, self.step_covariance)
        a_mat = self.dynamics.a
        n_x = a_mat.shape[1]
        c_mat = _matrix(self.outcome_rows, "outcome_rows")
        if c_mat.shape[1] != n_x:
            _fail(f"outcome_rows must have {n_x} columns, got {c_mat.shape}")

    def kernels(self) -> AdjointKernels:
        return adjoint_kernels(self.dynamics, self.outcome_rows)

    def forward_outcome_covariance(self) -> Array:
        """``C P_N C^T`` from the forward recursion."""
        traj = propagate_covariance(
            self.dynamics, self.initial_covariance, self.step_covariance
        )
        c_mat = np.asarray(self.outcome_rows, dtype=np.float64)
        return np.asarray(c_mat @ traj[-1] @ c_mat.T)

    def budget(self) -> VarianceBudget:
        return variance_budget(
            self.kernels(),
            self.initial_covariance,
            self.step_covariance,
            self.noise_classes,
            self.initial_classes,
        )
