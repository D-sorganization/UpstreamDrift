"""Face-angle variance model as a function of the allocation (NSX-92, #12721).

The outcome is the scalar ``y = c^T dx_N`` (a face-angle-like quantity) of the
linear tangent chain

``dx_{t+1} = A_t(z) dx_t + B_t w_t``,   ``A_t(z) = A0_t + sum_m z_m A1_{t,m}``,

with initial covariance ``P0`` and a *white* torque disturbance on the
actuated channels, ``w_t ~ (S + diag(k^2 tau_t(z)^2)) / dt`` (M1, the signal
law of :class:`~src.shared.python.perturbation.disturbance.DisturbanceSpec`;
V6 timestep consistency).  Frame ``t`` holds the torque ``tau_t`` of step
``t`` and ``z`` stacks all frames' null coordinates.  ``A1`` is the geometric
preload (M2, ``H-GEOMETRIC-PRELOAD``): the closure multiplier ``lambda(z)``
moves ``K_geom`` and so ``A_t``.  Backward kernels (NSX-90)::

    S_x[N] = c,   S_x[t] = S_x[t+1] A_t(z),   S_in[t] = S_x[t+1] B_t

give the exact decomposition (``sec:var-cov``, ``sec:var-mechanisms``)

    Var(y; z) = S_x[0] P0 S_x[0]^T                     (initial state, M2 only)
              + sum_t S_in[t] (S/dt) S_in[t]^T          (additive, M2 only)
              + sum_{t,j} k_j^2 S_in[t,j]^2 tau_tj^2 / dt  (signal dependent, M1)

Every term is the squared norm of a residual: ``Var = |r(z)|^2``.  With
``A1 = 0`` (M2 off) the kernels are constant, the first two terms are constants
and the last is a diagonal convex quadratic in ``tau`` (hence in ``z``).  With
``A1 != 0`` the residuals are polynomial in ``z`` (multilinear across steps), so
``Var`` is in general NOT convex.  :meth:`NoiseAwareModel.residuals` returns
``r`` and its exact Jacobian (product rule through the backward recursion), the
Gauss-Newton ingredients, so a first-order stationarity certificate uses the
exact gradient ``2 J^T r``.

Non-claims: linear prediction at the nominal motion only (see the NSX-90
nonlinearity gate); amplitudes are declared hypotheses, not measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.core.contracts import ensure
from src.shared.python.loop_nullspace.evidence import Array, readonly_float
from src.shared.python.perturbation.disturbance import DisturbanceSpec, Spectrum

__all__ = [
    "NoiseAwareModel",
    "fit_affine_dynamics",
    "psd_factor",
    "wrap_probability_bound",
]

AFFINE_CHECK_RTOL = 1e-9


def psd_factor(cov: ArrayLike) -> Array:
    """``L`` with ``cov = L L^T`` for a symmetric PSD matrix (eigen square root).

    Pre: ``cov`` square, finite, symmetric within ``1e-12`` relative and PSD up
    to rounding.  Post: ``L L^T = cov`` to rounding.
    """
    mat = np.asarray(cov, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"cov must be square, got shape {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("cov must be finite")
    scale = max(1.0, float(np.abs(mat).max()))
    if np.abs(mat - mat.T).max() > 1e-12 * scale:
        raise ValueError("cov must be symmetric")
    w, v = np.linalg.eigh(0.5 * (mat + mat.T))
    if w.min() < -1e-12 * scale:
        raise ValueError(f"cov must be positive semi-definite (min eig {w.min():.3e})")
    return np.asarray(v * np.sqrt(np.clip(w, 0.0, None)))


def wrap_probability_bound(variance: float, period: float = 2.0 * math.pi) -> float:
    """Gaussian bound ``P(|y| > period/2) = erfc(period / (2 sqrt(2 var)))``.

    V3: the linear variance equals the wrapped (angle) variance only when this
    probability is negligible; it is reported next to every variance.
    Pre: ``variance >= 0`` finite, ``period > 0``.  Post: in ``[0, 1]``.
    """
    if not (math.isfinite(variance) and variance >= 0.0):
        raise ValueError(f"variance must be finite and non-negative, got {variance}")
    if not (math.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be positive and finite, got {period}")
    if variance == 0.0:
        return 0.0
    return float(math.erfc(period / (2.0 * math.sqrt(2.0 * variance))))


@dataclass(frozen=True, eq=False)
class NoiseAwareModel:
    """Linear tangent chain with allocation-dependent kernels (see module doc).

    ``a0`` ``(T, n, n)``, ``b`` ``(T, n, n_ch)``, ``outcome_row`` ``(n,)``,
    ``p0`` ``(n, n)``.  ``a_slope`` ``(T, n, n, n_z)`` is ``dA_t/dz_m`` (M2) or
    ``None`` (M2 off); ``n_z`` is the stacked null dimension.  ``spec`` is a
    white :class:`DisturbanceSpec` on the ``n_ch`` torque channels and ``dt``
    the step.  Raises ``ValueError`` on any shape or value violation.
    """

    a0: Array
    b: Array
    outcome_row: Array
    p0: Array
    spec: DisturbanceSpec
    dt: float
    n_z: int
    a_slope: Array | None = None
    _l0: Array = field(init=False, repr=False)
    _ls: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        a0 = readonly_float(self.a0, "a0")
        b = readonly_float(self.b, "b")
        c = readonly_float(self.outcome_row, "outcome_row")
        p0 = readonly_float(self.p0, "p0")
        if a0.ndim != 3 or a0.shape[1] != a0.shape[2]:
            raise ValueError(f"a0 must be (T, n, n), got {a0.shape}")
        n_steps, n = a0.shape[0], a0.shape[1]
        if b.ndim != 3 or b.shape[:2] != (n_steps, n):
            raise ValueError(f"b must be (T, n, n_ch), got {b.shape}")
        if c.shape != (n,) or p0.shape != (n, n):
            raise ValueError("outcome_row must be (n,) and p0 (n, n)")
        if not isinstance(self.spec, DisturbanceSpec):
            raise TypeError("spec must be a DisturbanceSpec")
        if self.spec.spectrum is not Spectrum.WHITE or self.spec.persistent:
            raise ValueError("spec must be a white, non-persistent disturbance")
        if self.spec.n_channels != b.shape[2]:
            raise ValueError("spec channels must equal the input channels of b")
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError(f"dt must be positive and finite, got {self.dt}")
        if isinstance(self.n_z, bool) or int(self.n_z) < 1:
            raise ValueError(f"n_z must be a positive integer, got {self.n_z!r}")
        slope = None
        if self.a_slope is not None:
            slope = readonly_float(self.a_slope, "a_slope")
            if slope.shape != (n_steps, n, n, int(self.n_z)):
                raise ValueError(
                    f"a_slope must be {(n_steps, n, n, int(self.n_z))}, "
                    f"got {slope.shape}"
                )
        object.__setattr__(self, "a0", a0)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "outcome_row", c)
        object.__setattr__(self, "p0", p0)
        object.__setattr__(self, "a_slope", slope)
        object.__setattr__(self, "n_z", int(self.n_z))
        object.__setattr__(self, "_l0", psd_factor(p0))
        object.__setattr__(
            self,
            "_ls",
            psd_factor(np.asarray(self.spec.covariance)) / math.sqrt(self.dt),
        )

    # --------------------------------------------------------------- shape
    @property
    def n_steps(self) -> int:
        return int(self.a0.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.b.shape[2])

    @property
    def gain(self) -> Array:
        """Signal gain ``k_j`` (zeros when no signal law is declared)."""
        if not self.spec.has_signal_law:
            return np.zeros(self.n_channels)
        return np.asarray(self.spec.signal_gain, dtype=float)

    @property
    def preload_active(self) -> bool:
        """M2 declared: the kernels move with the allocation."""
        return self.a_slope is not None and bool(np.any(self.a_slope != 0.0))

    # ------------------------------------------------------------- kernels
    def a_at(self, z: ArrayLike) -> Array:
        """``A_t(z)`` of every step, ``(T, n, n)``."""
        zz = self._z(z)
        if self.a_slope is None:
            return np.array(self.a0)
        return np.asarray(self.a0 + np.einsum("tijm,m->tij", self.a_slope, zz))

    def kernels(self, z: ArrayLike) -> tuple[Array, Array]:
        """``(S_x[0], S_in)`` with shapes ``(n,)`` and ``(T, n_ch)`` at ``z``."""
        a_all = self.a_at(z)
        s_x = np.array(self.outcome_row)
        s_in = np.empty((self.n_steps, self.n_channels))
        for t in range(self.n_steps - 1, -1, -1):
            s_in[t] = s_x @ self.b[t]
            s_x = s_x @ a_all[t]
        return s_x, s_in

    def kernel_jacobians(self, z: ArrayLike) -> tuple[Array, Array, Array, Array]:
        """``(S_x0, S_in, dS_x0/dz, dS_in/dz)``; Jacobians put ``z`` last.

        Exact product rule through the backward recursion
        ``dS_x[t]/dz_m = dS_x[t+1]/dz_m A_t + S_x[t+1] A1_{t,m}``.
        """
        a_all = self.a_at(z)
        n_x = self.a0.shape[1]
        s_x = np.array(self.outcome_row)
        d_x = np.zeros((self.n_z, n_x))
        s_in = np.empty((self.n_steps, self.n_channels))
        d_in = np.empty((self.n_steps, self.n_channels, self.n_z))
        for t in range(self.n_steps - 1, -1, -1):
            s_in[t] = s_x @ self.b[t]
            d_in[t] = (d_x @ self.b[t]).T
            d_x = d_x @ a_all[t]
            if self.a_slope is not None:
                d_x = d_x + np.einsum("i,ijm->mj", s_x, self.a_slope[t])
            s_x = s_x @ a_all[t]
        return s_x, s_in, d_x.T, d_in

    # ------------------------------------------------------------ variance
    def variance_parts(self, z: ArrayLike, tau: ArrayLike) -> dict[str, float]:
        """``initial``, ``additive`` and ``signal_dependent`` variances at ``z``.

        ``tau`` is the ``(T, n_ch)`` torque history of the same ``z`` (the
        caller owns the affine map).  Post: all parts finite and non-negative.
        """
        torque = self._tau(tau)
        s_x0, s_in = self.kernels(z)
        r0 = self._l0.T @ s_x0
        r_add = s_in @ self._ls
        r_sig = self.gain * s_in * torque / math.sqrt(self.dt)
        parts = {
            "initial": float(r0 @ r0),
            "additive": float(np.sum(r_add**2)),
            "signal_dependent": float(np.sum(r_sig**2)),
        }
        ensure(all(math.isfinite(v) and v >= 0.0 for v in parts.values()), "parts")
        return parts

    def variance(self, z: ArrayLike, tau: ArrayLike) -> float:
        """Total linear outcome variance ``Var(y; z)``."""
        return float(sum(self.variance_parts(z, tau).values()))

    def residuals(
        self, z: ArrayLike, tau: ArrayLike, dtau_dz: ArrayLike
    ) -> tuple[Array, Array]:
        """``(r, J)`` with ``Var = |r|^2``; ``J = dr/dz`` is exact.

        ``dtau_dz`` is the ``(T * n_ch, n_z)`` torque Jacobian (frame-major).
        Post: ``grad Var = 2 J^T r``.
        """
        torque = self._tau(tau)
        dtau = np.asarray(dtau_dz, dtype=float)
        if dtau.shape != (self.n_steps * self.n_channels, self.n_z):
            raise ValueError(
                f"dtau_dz must be {(self.n_steps * self.n_channels, self.n_z)}"
            )
        s_x0, s_in, d_x0, d_in = self.kernel_jacobians(z)
        root_dt = math.sqrt(self.dt)
        dtau3 = dtau.reshape(self.n_steps, self.n_channels, self.n_z)
        r_init = self._l0.T @ s_x0
        j_init = self._l0.T @ d_x0
        r_add = (s_in @ self._ls).reshape(-1)
        j_add = np.einsum("tjm,jk->tkm", d_in, self._ls).reshape(-1, self.n_z)
        k = self.gain[None, :]
        r_sig = (k * s_in * torque / root_dt).reshape(-1)
        j_sig = (
            k[..., None]
            * (d_in * torque[..., None] + s_in[..., None] * dtau3)
            / root_dt
        ).reshape(-1, self.n_z)
        return (
            np.concatenate([r_init, r_add, r_sig]),
            np.vstack([j_init, j_add, j_sig]),
        )

    def signal_coefficients(self, z: ArrayLike) -> Array:
        """``k_j^2 S_in[t, j]^2 / dt`` at frozen kernels, ``(T, n_ch)`` (M1)."""
        _, s_in = self.kernels(z)
        return np.asarray(self.gain**2 * s_in**2 / self.dt)

    def invariant_variance(self, z: ArrayLike) -> float:
        """Initial + additive variance at the kernels of ``z`` (constant iff M2 off)."""
        s_x0, s_in = self.kernels(z)
        r0 = self._l0.T @ s_x0
        return float(r0 @ r0 + np.sum((s_in @ self._ls) ** 2))

    # -------------------------------------------------------------- checks
    def _z(self, z: ArrayLike) -> Array:
        zz = np.asarray(z, dtype=float)
        if zz.shape != (self.n_z,) or not np.all(np.isfinite(zz)):
            raise ValueError(
                f"z must be finite with shape ({self.n_z},), got {zz.shape}"
            )
        return zz

    def _tau(self, tau: ArrayLike) -> Array:
        arr = np.asarray(tau, dtype=float)
        if arr.shape != (self.n_steps, self.n_channels) or not np.all(np.isfinite(arr)):
            raise ValueError(
                f"tau must be finite with shape {(self.n_steps, self.n_channels)}"
            )
        return arr


def fit_affine_dynamics(
    dynamics_at: object, n_z: int, *, step: float = 1.0, check_points: int = 2
) -> tuple[Array, Array]:
    """Fit ``A_t(z) = A0 + sum_m z_m A1_m`` from a callable and verify affinity.

    ``dynamics_at(z)`` returns the ``(T, n, n)`` array ``A(z)`` for the stacked
    null coordinates.  ``A0 = A(0)`` and ``A1_m = (A(step e_m) - A(0)) / step``
    are exact when ``A`` is affine; the fit is then **checked** at
    ``check_points`` deterministic off-axis points and a ``ValueError`` is raised
    when the relative mismatch exceeds ``1e-9`` (a non-affine provider must not
    be passed off as the M2 model).  Returns ``(a0, a_slope)`` with ``a_slope``
    shaped ``(T, n, n, n_z)``.
    """
    if not callable(dynamics_at):
        raise TypeError("dynamics_at must be callable")
    if n_z < 1 or step <= 0.0 or not math.isfinite(step):
        raise ValueError("n_z must be >= 1 and step positive and finite")
    a0 = np.asarray(dynamics_at(np.zeros(n_z)), dtype=float)
    slope = np.empty((*a0.shape, n_z))
    for m in range(n_z):
        e = np.zeros(n_z)
        e[m] = step
        slope[..., m] = (np.asarray(dynamics_at(e), dtype=float) - a0) / step
    rng = np.random.default_rng(0)
    scale = max(1.0, float(np.abs(a0).max()))
    for _ in range(check_points):
        z = rng.uniform(-step, step, size=n_z)
        pred = a0 + np.einsum("tijm,m->tij", slope, z)
        err = float(np.abs(np.asarray(dynamics_at(z), dtype=float) - pred).max())
        if err > AFFINE_CHECK_RTOL * scale:
            raise ValueError(
                f"dynamics are not affine in z (mismatch {err:.3e}); the M2 model "
                "needs an affine A_t(z)"
            )
    return a0, slope
