"""Linear variance prediction with impedance and delayed feedback (NSX-93, #12722).

The tangent chain of NSX-84/NSX-90 (``dx = F dx + G_w w + G_u u`` in the frame
coordinates ``x = (xi, eta)``) gets two declared controllers, both state
feedback on the deviation ``dx = state - ref``:

* **Impedance** (instant, continuous): ``u = -K_imp(t) dx`` with ``K_imp = K Cq +
  D Cv``, where ``(Cq, Cv)`` map the tangent state to the actuated errors
  ``e = S dq = (S P) xi`` and ``de = S dv = (S P) eta + (S O) xi``.  It is
  folded into the *continuous* closed-loop matrix ``F - G_u K_imp`` before
  discretisation, so the stiffness is exact for any step.
* **Delayed feedback**: ``u_t = -K_fb,t dx_{t-m}`` with ``m = delay / dt``
  integer steps, appended as ``m`` delayed copies of the state (``z_t = (x_t,
  ..., x_{t-m})``).  The pre-history is the initial deviation held constant,
  matching the replay (:mod:`feedback_layer`), so ``P_0^aug = T P_0 T^T`` with
  ``T = [I; ...; I]``.

Discretisation is declared: ``zoh`` (exact matrix exponential of ``[[F_cl, G_w,
G_u], [0, 0, 0]] dt``; exact for noise and input held over a step, which is how
the NSX-91 replay holds them) or ``euler`` (``A = I + dt F_cl``, NSX-90's
default).  The augmented chain is an ordinary
:class:`~estimation.mosaic.local_policy.LinearizedDynamics`, so
:func:`variance_records.assess_variance` and the adjoint kernels of NSX-90 apply
unchanged.

Delay margin.  :func:`frozen_delay_margin` is the first delay (in steps) at
which the frozen discrete loop ``x_{k+1} = A x_k - B K x_{k-m}`` has spectral
radius above ``1 + 1e-9`` (the tolerance is the eigenvalue accuracy of a ``<=
400``-square companion matrix, ``~ n eps ||M||``).  For a delay system stability
can switch more than once as the delay grows; the *first* loss is reported.  For
a time-varying chain the per-step frozen margins are a diagnostic, not a proof:
over a finite horizon the real question is the finite-horizon gain of the
transition product, which the variance itself reports.

Non-claim: all of this is conditional on the declared impedance, gains, delay
and noise.  Pure numpy and scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm

from src.shared.python.estimation.mosaic.local_policy import LinearizedDynamics
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.impedance_spec import JointImpedance
from src.shared.python.loop_nullspace.variance_propagation import (
    InputClass,
    VarianceProblem,
    pad_outcome_rows,
)
from src.shared.python.loop_nullspace.variance_records import (
    GateSpec,
    VarianceAssessment,
    VarianceDeclaration,
    assess_variance,
)

__all__ = [
    "DelayMargin",
    "LayerLinearization",
    "LayerVariance",
    "impedance_state_gain",
    "frozen_delay_margin",
    "inflate_step_covariances",
    "layer_variance",
    "variance_ratio_record",
    "with_layer_conditioning",
]

Discretisation = Literal["zoh", "euler"]
_UNSTABLE_RHO = 1.0 + 1e-9


def _check_dims(f: Array, g_w: Array, g_u: Array) -> None:
    if f.ndim != 3 or f.shape[1] != f.shape[2]:
        raise ValueError(f"f must be (T, nx, nx), got {f.shape}")
    for name, g in (("g_w", g_w), ("g_u", g_u)):
        if g.ndim != 3 or g.shape[:2] != f.shape[:2]:
            raise ValueError(f"{name} must be (T, nx, n), got {g.shape}")
    if not all(np.all(np.isfinite(x)) for x in (f, g_w, g_u)):
        raise ValueError("f, g_w and g_u must be finite")


@dataclass(frozen=True)
class LayerLinearization:
    """The tangent chain with the declared controller gains (see module docstring).

    Attributes:
        f, g_w, g_u: ``(T, nx, nx)``, ``(T, nx, n_w)``, ``(T, nx, n_u)`` continuous
            matrices, each frozen over its step.
        dt: Step [s].
        impedance_gain: ``K_imp`` ``(T, n_u, nx)`` or ``None``.
        feedback_gain: ``K_fb`` ``(T, n_u, nx)`` or ``None``.
        delay_steps: Feedback delay in whole steps (``>= 0``).
        method: ``"zoh"`` or ``"euler"``.

    Raises:
        ValueError: Inconsistent shapes, non-finite entries, ``dt <= 0``, a
            negative delay or an unknown method.
    """

    f: Array
    g_w: Array
    g_u: Array
    dt: float
    impedance_gain: Array | None = None
    feedback_gain: Array | None = None
    delay_steps: int = 0
    method: Discretisation = "zoh"

    def __post_init__(self) -> None:
        f, g_w, g_u = (
            np.asarray(x, dtype=np.float64) for x in (self.f, self.g_w, self.g_u)
        )
        _check_dims(f, g_w, g_u)
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError(f"dt must be positive and finite, got {self.dt}")
        if self.delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {self.delay_steps}")
        if self.method not in ("zoh", "euler"):
            raise ValueError(f"method must be 'zoh' or 'euler', got {self.method!r}")
        object.__setattr__(self, "f", f)
        object.__setattr__(self, "g_w", g_w)
        object.__setattr__(self, "g_u", g_u)
        shape = (f.shape[0], g_u.shape[2], f.shape[1])
        for name in ("impedance_gain", "feedback_gain"):
            gain = getattr(self, name)
            if gain is None:
                continue
            arr = np.asarray(gain, dtype=np.float64)
            if arr.shape != shape or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, arr)

    @property
    def n_steps(self) -> int:
        return int(self.f.shape[0])

    @property
    def n_x(self) -> int:
        return int(self.f.shape[1])

    @property
    def augmented_blocks(self) -> int:
        """Copies of the state in the closed-loop chain (``1 + m`` with feedback)."""
        return 1 + (self.delay_steps if self.feedback_gain is not None else 0)

    def discrete(self) -> tuple[Array, Array, Array]:
        """``(A, B_w, B_u)`` per step with the impedance folded into ``A``."""
        t_n, n_x = self.n_steps, self.n_x
        n_w, n_u = self.g_w.shape[2], self.g_u.shape[2]
        a = np.empty((t_n, n_x, n_x))
        b_w = np.empty((t_n, n_x, n_w))
        b_u = np.empty((t_n, n_x, n_u))
        for t in range(t_n):
            f_cl = self.f[t]
            if self.impedance_gain is not None:
                f_cl = f_cl - self.g_u[t] @ self.impedance_gain[t]
            if self.method == "euler":
                a[t] = np.eye(n_x) + self.dt * f_cl
                b_w[t], b_u[t] = self.dt * self.g_w[t], self.dt * self.g_u[t]
                continue
            block = np.zeros((n_x + n_w + n_u,) * 2)
            block[:n_x, :n_x] = f_cl
            block[:n_x, n_x : n_x + n_w] = self.g_w[t]
            block[:n_x, n_x + n_w :] = self.g_u[t]
            exp = expm(self.dt * block)
            a[t], b_w[t], b_u[t] = (
                exp[:n_x, :n_x],
                exp[:n_x, n_x : n_x + n_w],
                exp[:n_x, n_x + n_w :],
            )
        return a, b_w, b_u

    def plant_dynamics(self) -> LinearizedDynamics:
        """``(A, B_u)`` with the impedance included: the plant a feedback is designed on."""
        a, _, b_u = self.discrete()
        return LinearizedDynamics(a, b_u)

    def closed_loop(self) -> LinearizedDynamics:
        """Augmented ``(A_aug, B_w_aug)`` on ``z = (x_t, ..., x_{t-m})``."""
        a, b_w, b_u = self.discrete()
        blocks, n_x = self.augmented_blocks, self.n_x
        m = blocks - 1
        a_aug = np.zeros((self.n_steps, blocks * n_x, blocks * n_x))
        b_aug = np.zeros((self.n_steps, blocks * n_x, b_w.shape[2]))
        a_aug[:, :n_x, :n_x] = a
        b_aug[:, :n_x, :] = b_w
        if self.feedback_gain is not None:
            a_aug[:, :n_x, m * n_x : (m + 1) * n_x] -= b_u @ self.feedback_gain
        for j in range(1, blocks):
            a_aug[:, j * n_x : (j + 1) * n_x, (j - 1) * n_x : j * n_x] = np.eye(n_x)
        return LinearizedDynamics(a_aug, b_aug)

    def augment_covariance(self, p0: ArrayLike) -> Array:
        """``T P_0 T^T``: the delayed copies start equal to the initial deviation."""
        p_mat = np.asarray(p0, dtype=np.float64)
        stack = np.tile(np.eye(self.n_x), (self.augmented_blocks, 1))
        return np.asarray(stack @ p_mat @ stack.T)

    def augment_outcomes(self, outcome_rows: ArrayLike) -> Array:
        """Outcome rows on the augmented state (the outcome reads ``x_N`` only)."""
        return pad_outcome_rows(outcome_rows, (self.augmented_blocks - 1) * self.n_x)

    def delay_margins(self, max_steps: int = 100) -> list[DelayMargin]:
        """Frozen margin at every step (diagnostic; see the module docstring)."""
        if self.feedback_gain is None:
            raise ValueError("a delay margin needs a feedback gain")
        a, _, b_u = self.discrete()
        return [
            frozen_delay_margin(a[t], b_u[t], self.feedback_gain[t], self.dt, max_steps)
            for t in range(self.n_steps)
        ]


def impedance_state_gain(
    impedance: JointImpedance,
    times: ArrayLike,
    selection: ArrayLike,
    basis: ArrayLike,
    offset: ArrayLike,
) -> Array:
    """``K_imp = K(t) Cq + D(t) Cv`` ``(T, n_a, 2k)`` on ``dx = (xi, eta)``.

    ``Cq = [S P, 0]`` and ``Cv = [S O, S P]`` turn the tangent deviation into the
    actuated position and velocity errors (``dq = P xi``, ``dv = P eta + O xi``).
    The torque on the deviation is ``-(K e + D de)``, hence the sign of ``u =
    -K_imp dx`` used by :class:`LayerLinearization`.  ``basis`` and ``offset`` are
    ``(n_v, k)`` or ``(T, n_v, k)``.

    Raises:
        ValueError: Shapes that do not conform.
    """
    ts = np.asarray(times, dtype=np.float64).reshape(-1)
    sel = np.asarray(selection, dtype=np.float64)
    p_arr = np.broadcast_to(
        np.asarray(basis, dtype=np.float64), (ts.size, *np.shape(basis)[-2:])
    )
    o_arr = np.broadcast_to(np.asarray(offset, dtype=np.float64), p_arr.shape)
    if sel.ndim != 2 or sel.shape != (impedance.n_actuated, p_arr.shape[1]):
        raise ValueError(
            f"selection must be ({impedance.n_actuated}, {p_arr.shape[1]}), got {sel.shape}"
        )
    k_tan = p_arr.shape[2]
    stiff, damp = impedance.gain_series(ts)
    sp, so = sel @ p_arr, sel @ o_arr
    out = np.zeros((ts.size, impedance.n_actuated, 2 * k_tan))
    out[:, :, :k_tan] = stiff @ sp + damp @ so
    out[:, :, k_tan:] = damp @ sp
    return out


def inflate_step_covariances(sigma: ArrayLike, scale: ArrayLike) -> Array:
    """``diag(s_t) Sigma_t diag(s_t)``: the declared noise-inflation penalty.

    ``scale`` is the per-channel standard-deviation multiplier ``(T, n_w)``
    (:meth:`JointImpedance.noise_scale_series`); a scale of one returns
    ``sigma`` exactly.
    """
    sig = np.asarray(sigma, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    if sig.ndim != 3 or s.shape != sig.shape[:2]:
        raise ValueError(f"scale must be {sig.shape[:2]} for sigma {sig.shape}")
    if not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("scale must be finite and non-negative")
    return np.asarray(s[:, :, None] * sig * s[:, None, :])


# ------------------------------------------------------------ delay margin


@dataclass(frozen=True)
class DelayMargin:
    """First destabilising delay of a frozen loop (see the module docstring).

    ``margin_steps`` is ``0`` if the delay-free loop is already unstable and
    ``None`` if no tested delay (``<= max_steps``) destabilises it.
    ``spectral_radius[m]`` is the radius at ``m`` steps up to the first unstable
    delay.
    """

    stable_at_zero: bool
    margin_steps: int | None
    margin_s: float | None
    spectral_radius: Array
    max_steps: int


def _companion(a: Array, b_k: Array, m: int) -> Array:
    n = a.shape[0]
    mat = np.zeros(((m + 1) * n, (m + 1) * n))
    mat[:n, :n] = a
    mat[:n, m * n :] -= b_k
    for j in range(1, m + 1):
        mat[j * n : (j + 1) * n, (j - 1) * n : j * n] = np.eye(n)
    return mat


def frozen_delay_margin(
    a: ArrayLike, b_u: ArrayLike, gain: ArrayLike, dt: float, max_steps: int = 100
) -> DelayMargin:
    """Margin of ``x_{k+1} = A x_k - B_u K x_{k-m}`` over ``m = 0, 1, ...``.

    Preconditions (``ValueError``): ``A`` square, ``B_u`` ``(nx, nu)``, ``K``
    ``(nu, nx)``, finite, ``dt > 0``, ``max_steps >= 1``.  Postcondition: the
    returned ``margin_steps`` is the smallest ``m`` whose companion matrix has
    spectral radius above ``1 + 1e-9`` (``margin_s = m dt``).
    """
    a_m, b_m, k_m = (np.asarray(x, dtype=np.float64) for x in (a, b_u, gain))
    n = a_m.shape[0]
    if a_m.shape != (n, n) or b_m.shape[0] != n or k_m.shape != (b_m.shape[1], n):
        raise ValueError("A (nx,nx), B_u (nx,nu) and K (nu,nx) must conform")
    if not all(np.all(np.isfinite(x)) for x in (a_m, b_m, k_m)):
        raise ValueError("A, B_u and K must be finite")
    if not (math.isfinite(dt) and dt > 0.0) or max_steps < 1:
        raise ValueError("dt must be positive and max_steps >= 1")
    b_k = b_m @ k_m
    radii: list[float] = []
    first: int | None = None
    for m in range(max_steps + 1):
        rho = float(np.max(np.abs(np.linalg.eigvals(_companion(a_m, b_k, m)))))
        radii.append(rho)
        if rho > _UNSTABLE_RHO:
            first = m
            break
    return DelayMargin(
        stable_at_zero=radii[0] <= _UNSTABLE_RHO,
        margin_steps=first,
        margin_s=None if first is None else first * dt,
        spectral_radius=np.array(radii),
        max_steps=max_steps,
    )


# ---------------------------------------------------------------- variance


def with_layer_conditioning(
    declaration: VarianceDeclaration, hypotheses: tuple[str, ...]
) -> VarianceDeclaration:
    """The declaration with the layer's hypotheses added to the conditioning axis.

    Impedance and delay are declared controller properties, not mechanisms by
    which the allocation changes the variance (V1), so they enter the orthogonal
    ``conditioning`` axis (NSX-81) and leave the mechanism set unchanged.
    """
    merged = tuple(sorted({*declaration.noise_hypotheses, *hypotheses}))
    return replace(declaration, noise_hypotheses=merged)


@dataclass(frozen=True)
class LayerVariance:
    """Variance assessment of the closed loop plus the margins of its feedback."""

    assessment: VarianceAssessment
    delay_margins: tuple[DelayMargin, ...]
    delay_steps: int
    n_state_copies: int


def layer_variance(
    linearization: LayerLinearization,
    p0: ArrayLike,
    sigma: ArrayLike,
    outcome_rows: ArrayLike,
    declaration: VarianceDeclaration,
    *,
    layer_hypotheses: tuple[str, ...],
    outcome_units: str,
    gate: GateSpec | None = None,
    max_margin_steps: int = 20,
) -> LayerVariance:
    """NSX-90 variance of the closed loop, typed with the layer's conditioning.

    ``sigma`` is the (already inflated, if declared) per-step noise covariance.
    Postconditions: the variance record is ``INVARIANT`` with the layer and noise
    hypotheses on the ``conditioning`` axis unless a V1 mechanism is declared (then
    ``FAMILY_DEPENDENT``); the kernels carry the same conditioning, because they
    depend on the declared gains; ``linear_validity`` is ``NOT_ASSESSED`` without
    a gate.  Delay margins are listed only when a feedback gain is present and
    searched up to ``max_margin_steps`` whole steps (cost grows as
    ``steps * (n_x * max_margin_steps)^3``; a loop with no margin inside the
    search reports ``margin_steps=None``, i.e. "none found up to the bound").
    """
    decl = with_layer_conditioning(declaration, layer_hypotheses)
    n_aug = linearization.augmented_blocks * linearization.n_x
    n_w = linearization.g_w.shape[2]
    problem = VarianceProblem(
        dynamics=linearization.closed_loop(),
        initial_covariance=linearization.augment_covariance(p0),
        step_covariance=np.asarray(sigma, dtype=np.float64),
        outcome_rows=linearization.augment_outcomes(outcome_rows),
        noise_classes=(InputClass("torque_noise", tuple(range(n_w))),),
        initial_classes=(InputClass("initial_state", tuple(range(n_aug))),),
    )
    raw = assess_variance(problem, decl, outcome_units=outcome_units, gate=gate)
    assessment = replace(
        raw,
        kernel_x0=replace(raw.kernel_x0, conditioning=decl.noise_hypotheses),
        kernel_input=replace(raw.kernel_input, conditioning=decl.noise_hypotheses),
    )
    margins = (
        tuple(linearization.delay_margins(max_margin_steps))
        if linearization.feedback_gain is not None
        else ()
    )
    return LayerVariance(
        assessment, margins, linearization.delay_steps, linearization.augmented_blocks
    )


def variance_ratio_record(
    baseline: VarianceAssessment,
    layered: VarianceAssessment | None,
    *,
    impedance_declared: bool,
) -> QuantityRecord:
    """Variance ratio ``layered / baseline`` per outcome, or ``UNAVAILABLE``.

    Raises ``ValueError`` for a zero baseline variance.  Without a declared
    impedance there is nothing to compare: a torque-only
    "co-contraction" effect cannot be read off net torques (R6, V5), so the
    record is ``UNAVAILABLE`` with that reason and ``NaN`` values, never a
    number inferred from the torque history.
    """
    base = baseline.variance
    if not impedance_declared or layered is None:
        return QuantityRecord(
            values=np.full(base.values.shape, np.nan),
            units="1",
            allocation_dependence=AllocationDependence.UNAVAILABLE,
            model_provenance=base.model_provenance,
            input_provenance=base.input_provenance,
            reason=(
                "co-contraction is not identifiable from net joint torque; it "
                "needs a declared H-DECLARED-JOINT-IMPEDANCE (rule R6, V5)"
            ),
        )
    if not np.all(base.values > 0.0):
        raise ValueError("a variance ratio needs a positive baseline variance")
    return replace(
        layered.variance, values=layered.variance.values / base.values, units="1"
    )
