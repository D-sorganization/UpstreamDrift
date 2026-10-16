"""Constrained tangent linearization of the closed-loop dynamics (NSX-84, #12713).

Implements ``loop_nullspace.tex`` Section "Variability Sensitivity", subsection
``sec:var-lin``: the linearization of ``M a + h = tau + J_c^T lambda`` with the
closure ``J_c a + Jdot_v = 0`` on the *tangent bundle* of the closure,
``J_c dq = 0`` and ``J_c dv + (dJ_c[dq]) v = 0``, with the reaction ``lambda``
re-solved for every perturbation and the feed-forward load held fixed.

Coordinates.  Perturbations are expressed in an orthonormal basis ``P`` of
``ker J_c`` (``dq = P xi``, ``dv = P eta + O xi``), so ``J_c dq = 0`` holds by
construction; the basis-independent object is the projector ``Pi = P P^T``.  The
full-state Jacobian of the index-3 DAE is never formed.

Two routes obtain the same :class:`ClosedLoopDerivatives`:

* :class:`FiniteDifferenceDerivatives` -- central differences of the single KKT
  implementation (:func:`constrained_acceleration`) along retractions.  This is
  the engine-free reference of the derivative capability seam
  (:mod:`derivative_capability`) and the fall-back declared as ``PARTIAL``.
* :class:`AssembledKKTDerivatives` -- the differentiated KKT system
  ``eq:var-kkt``: ``[[M, -J^T], [J, 0]] [da; dlam] = [ -d_q(M a + h) dq - d_v h dv
  + K_geom dq ; ... ]`` with the geometric preload ``K_geom = sum_r lambda_r
  grad^2 Phi_r`` (mechanism M2, :func:`geometric_preload`).  Omitting
  ``K_geom`` is the error the specification forbids; the NSX-80 goldens fail if
  it is dropped.

Both refuse (:class:`ClosureRankChangeError`) when the closure rank or the
contact mode changes within the differencing step, and both record
``derivative_method`` in :meth:`ClosedLoopDerivatives.provenance_tag`.

Non-claim: the linearization is local; the nonlinearity gate is NSX-90.
Pure numpy; no engine imports.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.configuration_manifold import (
    ConfigurationManifold,
    EuclideanManifold,
    FreeFlyerManifold,
    _vec,
)
from src.shared.python.loop_nullspace.constrained_dynamics import (
    constrained_acceleration,
    solve_kkt_block,
)
from src.shared.python.loop_nullspace.derivative_capability import (
    ClosedLoopDerivatives,
    DerivativeMethod,
)
from src.shared.python.loop_nullspace.evidence import Array

logger = logging.getLogger(__name__)

__all__ = [
    "AssembledKKTDerivatives",
    "ClosureRankChangeError",
    "ConfigurationManifold",
    "EuclideanManifold",
    "FiniteDifferenceDerivatives",
    "FreeFlyerManifold",
    "StepConvergence",
    "TangentStateSpace",
    "closure_rank",
    "fd_step_convergence",
    "geometric_preload",
    "tangent_basis",
    "tangent_projector",
    "tangent_state_space",
]

_RANK_RTOL = 1e-9
_ORTHO_TOL = 1e-8


class ClosureRankChangeError(ValueError):
    """The closure rank or contact mode changes inside the differencing step."""


def closure_rank(jac: ArrayLike) -> int:
    """Numerical rank of ``J_c`` (SVD, tolerance ``1e-9 * max(sigma_max, 1)``)."""
    arr = np.asarray(jac, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"J_c must be 2-D, got shape {arr.shape}")
    if arr.size == 0:
        return 0
    sing = np.linalg.svd(arr, compute_uv=False)
    return int((sing > _RANK_RTOL * max(float(sing[0]), 1.0)).sum())


def tangent_basis(jac: ArrayLike) -> Array:
    """Orthonormal basis of ``ker J_c``, shape ``(nv, nv - rank)`` (SVD)."""
    arr = np.asarray(jac, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"J_c must be 2-D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        return np.eye(arr.shape[1])
    rank = closure_rank(arr)
    return np.asarray(np.linalg.svd(arr)[2][rank:].T)


def tangent_projector(jac: ArrayLike) -> Array:
    """Basis-independent orthogonal projector ``Pi`` onto ``ker J_c``.

    Postconditions: ``Pi`` symmetric idempotent and ``J_c Pi J_c^T = 0``.
    """
    basis = tangent_basis(jac)
    return np.asarray(basis @ basis.T)


def _central(fn: Callable[[float], Array], step: float) -> Array:
    return np.asarray((fn(step) - fn(-step)) / (2.0 * step))


def geometric_preload(
    model: Any, manifold: ConfigurationManifold, q: ArrayLike, lam: ArrayLike
) -> Array:
    """``K_geom = d_q (J_c(q)^T lambda)`` at fixed ``lambda`` (mechanism M2).

    For a holonomic closure ``J_c = d Phi / dq`` this is ``sum_r lambda_r
    grad^2 Phi_r``.  Column ``i`` is the central difference of ``J_c^T lambda``
    along the retraction of the ``i``-th tangent unit vector.  Preconditions
    (``ValueError``): ``q`` of length ``nq``, ``lam`` of length ``m``.
    Postcondition: shape ``(nv, nv)``.
    """
    q_arr = _vec(q, manifold.nq, "q")
    nv = manifold.nv
    lam_arr = np.asarray(lam, dtype=np.float64).reshape(-1)
    if lam_arr.shape[0] != model.closure_jacobian(q_arr).shape[0]:
        raise ValueError("lam length must equal the number of closure rows")
    step = 1e-6
    cols = []
    for i in range(nv):
        unit = np.zeros(nv)
        unit[i] = step
        plus = model.closure_jacobian(manifold.retract(q_arr, unit)).T @ lam_arr
        minus = model.closure_jacobian(manifold.retract(q_arr, -unit)).T @ lam_arr
        cols.append((plus - minus) / (2.0 * step))
    return np.array(cols).T


@dataclass(frozen=True)
class _Nominal:
    """Nominal evaluation shared by both derivative routes."""

    q: Array
    v: Array
    tau: Array
    jac: Array
    basis: Array
    offset: Array
    acceleration: Array
    reaction: Array


class _DerivativeBase(ABC):
    """Common preconditions, guards and tau columns of both routes."""

    scheme = ""

    def __init__(
        self,
        model: Any,
        manifold: ConfigurationManifold,
        *,
        step: float = 1e-5,
        basis_field: Callable[[Array], Array] | None = None,
        expected_rank: int | None = None,
        velocity_tolerance: float = 1e-6,
    ) -> None:
        if not (np.isfinite(step) and step > 0.0):
            raise ValueError(f"step must be positive and finite, got {step}")
        if basis_field is not None and not callable(basis_field):
            raise TypeError("basis_field must be callable or None")
        if velocity_tolerance <= 0.0:
            raise ValueError("velocity_tolerance must be positive")
        self._model = model
        self._manifold = manifold
        self._step = float(step)
        self._basis_field = basis_field
        self._expected_rank = expected_rank
        self._vel_tol = float(velocity_tolerance)

    # -- model evaluation -------------------------------------------------
    def _solve(self, q: Array, v: Array, tau: Array) -> tuple[Array, Array]:
        model = self._model
        return constrained_acceleration(
            model.mass_matrix(q),
            model.bias_forces(q, v),
            tau,
            model.closure_jacobian(q),
            model.closure_drift(q, v),
        )

    def _step_to(self, q: Array, direction: Array, eps: float) -> Array:
        return np.asarray(self._manifold.retract(q, eps * direction))

    def _mode(self, q: Array, v: Array) -> object:
        probe = getattr(self._model, "contact_mode", None)
        return probe(q, v) if callable(probe) else None

    # -- guards and nominal -----------------------------------------------
    def _guard(self, nom: _Nominal) -> None:
        rank = closure_rank(nom.jac)
        if self._expected_rank is not None and rank != self._expected_rank:
            raise ClosureRankChangeError(
                f"closure rank is {rank} but expected rank {self._expected_rank}"
            )
        mode0 = self._mode(nom.q, nom.v)
        for j in range(nom.basis.shape[1]):
            for sign in (1.0, -1.0):
                probe = self._step_to(nom.q, nom.basis[:, j], sign * self._step)
                got = closure_rank(self._model.closure_jacobian(probe))
                if got != rank:
                    raise ClosureRankChangeError(
                        f"closure rank changes from {rank} to {got} within the "
                        f"differencing step {self._step:g}; refusing to differentiate"
                    )
                if self._mode(probe, nom.v) != mode0:
                    raise ClosureRankChangeError(
                        f"contact mode changes within the differencing step "
                        f"{self._step:g}; refusing to differentiate"
                    )

    def _basis(self, q: Array, jac: Array) -> Array:
        if self._basis_field is None:
            return tangent_basis(jac)
        basis = np.asarray(self._basis_field(q), dtype=np.float64)
        nv = jac.shape[1]
        if basis.ndim != 2 or basis.shape[0] != nv:
            raise ValueError(f"basis_field(q) must have shape ({nv}, k)")
        if np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1])) > _ORTHO_TOL:
            raise ValueError("basis_field(q) must be orthonormal")
        if basis.shape[1] != nv - closure_rank(jac) or (
            jac.size and np.linalg.norm(jac @ basis) > _ORTHO_TOL
        ):
            raise ValueError("basis_field(q) must span ker J_c(q)")
        return basis

    def _velocity_offset(self, q: Array, v: Array, basis: Array, jac: Array) -> Array:
        """``O`` with ``J O xi = -(dJ[P xi]) v``; minimum norm, shape ``(nv, k)``."""
        nv, k = basis.shape
        if jac.shape[0] == 0 or k == 0:
            return np.zeros((nv, k))
        pinv = np.linalg.pinv(jac, rcond=1e-12)
        cols = []
        for j in range(k):
            direction = basis[:, j]

            def jac_v(e: float, direction: Array = direction) -> Array:
                probe = self._step_to(q, direction, e)
                return np.asarray(self._model.closure_jacobian(probe) @ v)

            cols.append(-pinv @ _central(jac_v, self._step))
        return np.array(cols).T

    def _nominal(self, q: ArrayLike, v: ArrayLike, tau: ArrayLike) -> _Nominal:
        nq, nv = self._manifold.nq, self._manifold.nv
        q_arr, v_arr = _vec(q, nq, "q"), _vec(v, nv, "v")
        tau_arr = _vec(tau, nv, "tau")
        jac = np.asarray(self._model.closure_jacobian(q_arr), dtype=np.float64)
        if jac.ndim != 2 or jac.shape[1] != nv:
            raise ValueError(f"J_c must have shape (m, {nv}), got {jac.shape}")
        if jac.size and np.linalg.norm(jac @ v_arr) > self._vel_tol * (
            1.0 + np.linalg.norm(v_arr)
        ):
            raise ValueError(
                "v is not tangent to the closure (J_c v != 0); the tangent-bundle "
                "linearization needs an admissible velocity"
            )
        basis = self._basis(q_arr, jac)
        acc, lam = self._solve(q_arr, v_arr, tau_arr)
        offset = self._velocity_offset(q_arr, v_arr, basis, jac)
        nom = _Nominal(q_arr, v_arr, tau_arr, jac, basis, offset, acc, lam)
        self._guard(nom)
        return nom

    # -- result assembly --------------------------------------------------
    def _tau_columns(self, nom: _Nominal) -> tuple[Array, Array]:
        nv = nom.v.shape[0]
        d_acc, d_lam = [], []
        for j in range(nv):
            load = nom.tau.copy()
            load[j] += 1.0
            acc, lam = self._solve(nom.q, nom.v, load)
            d_acc.append(acc - nom.acceleration)
            d_lam.append(lam - nom.reaction)
        return np.array(d_acc).T, np.array(d_lam).T.reshape(len(nom.reaction), nv)

    @abstractmethod
    def _column(
        self, nom: _Nominal, ctx: Any, dq: Array, dv: Array
    ) -> tuple[Array, Array]:
        """``(delta a, delta lambda)`` for the tangent perturbation ``(dq, dv)``."""

    def _column_context(self, nom: _Nominal) -> Any:
        """Per-nominal data shared by every column (``None`` by default)."""
        return None

    def _columns(self, nom: _Nominal) -> tuple[Array, Array, Array, Array]:
        ctx = self._column_context(nom)
        k = nom.basis.shape[1]
        zero = np.zeros(nom.v.shape[0])
        xi = [
            self._column(nom, ctx, nom.basis[:, j], nom.offset[:, j]) for j in range(k)
        ]
        eta = [self._column(nom, ctx, zero, nom.basis[:, j]) for j in range(k)]
        m = len(nom.reaction)
        return (
            np.array([c[0] for c in xi]).T.reshape(-1, k),
            np.array([c[0] for c in eta]).T.reshape(-1, k),
            np.array([c[1] for c in xi]).T.reshape(m, k),
            np.array([c[1] for c in eta]).T.reshape(m, k),
        )

    def constrained_acceleration_derivatives(
        self, q: ArrayLike, v: ArrayLike, tau: ArrayLike
    ) -> ClosedLoopDerivatives:
        """Tangent-bundle derivatives of ``(a, lambda)`` at ``(q, v, tau)``.

        Preconditions (``ValueError``): finite ``q`` (``nq``), ``v`` and ``tau``
        (``nv``), ``J_c v = 0``.  Raises :class:`ClosureRankChangeError` at a
        rank or contact-mode change.  Postconditions: ``J_c @ basis = 0`` and
        the reaction is re-solved, never frozen.
        """
        nom = self._nominal(q, v, tau)
        d_axi, d_aeta, d_lxi, d_leta = self._columns(nom)
        d_atau, d_ltau = self._tau_columns(nom)
        return ClosedLoopDerivatives(
            acceleration=nom.acceleration,
            reaction=nom.reaction,
            basis=nom.basis,
            projector=nom.basis @ nom.basis.T,
            d_acc_d_xi=d_axi,
            d_acc_d_eta=d_aeta,
            d_acc_d_tau=d_atau,
            d_lambda_d_xi=d_lxi,
            d_lambda_d_eta=d_leta,
            d_lambda_d_tau=d_ltau,
            method=DerivativeMethod.FINITE_DIFFERENCE,
            scheme=self.scheme,
            step=self._step,
        )


class FiniteDifferenceDerivatives(_DerivativeBase):
    """Engine-free central-difference reference of the derivative seam.

    Differences :func:`constrained_acceleration` along retractions of the
    closure tangent basis, so ``nq != nv`` and quaternion joints are handled and
    ``J_c dq = 0`` is preserved.  Declared ``PARTIAL`` (finite difference).
    """

    scheme = "central_retraction"

    def _column(
        self, nom: _Nominal, ctx: Any, dq: Array, dv: Array
    ) -> tuple[Array, Array]:
        plus = self._solve(
            self._step_to(nom.q, dq, self._step), nom.v + self._step * dv, nom.tau
        )
        minus = self._solve(
            self._step_to(nom.q, dq, -self._step), nom.v - self._step * dv, nom.tau
        )
        return (
            (plus[0] - minus[0]) / (2.0 * self._step),
            (plus[1] - minus[1]) / (2.0 * self._step),
        )


class AssembledKKTDerivatives(_DerivativeBase):
    """The differentiated KKT system ``eq:var-kkt`` with the M2 preload term.

    Only the model functions ``M, h, J_c, Jdot_v`` are differenced; the
    algebra, the re-solved ``delta lambda`` and the geometric preload
    :func:`geometric_preload` are explicit.  This is the form analytic and
    autodiff engines implement natively.
    """

    scheme = "assembled_kkt"

    def _ddir(
        self, fn: Callable[[Array, Array], Array], nom: _Nominal, dq: Array, dv: Array
    ) -> Array:
        return _central(
            lambda e: fn(self._step_to(nom.q, dq, e), nom.v + e * dv), self._step
        )

    def _perturbation(
        self, nom: _Nominal, k_geom: Array, dq: Array, dv: Array
    ) -> tuple[Array, Array]:
        model, acc = self._model, nom.acceleration

        def dyn(q: Array, v: Array) -> Array:
            return model.mass_matrix(q) @ acc + model.bias_forces(q, v)

        def clo(q: Array, v: Array) -> Array:
            return model.closure_jacobian(q) @ acc + model.closure_drift(q, v)

        rhs1 = -self._ddir(dyn, nom, dq, dv) + k_geom @ dq
        rhs2 = -self._ddir(clo, nom, dq, dv)
        mass = model.mass_matrix(nom.q)
        d_acc, d_lam = solve_kkt_block(mass, nom.jac, rhs1, -rhs2)
        return d_acc, d_lam

    def _column_context(self, nom: _Nominal) -> Array:
        return geometric_preload(self._model, self._manifold, nom.q, nom.reaction)

    def _column(
        self, nom: _Nominal, ctx: Any, dq: Array, dv: Array
    ) -> tuple[Array, Array]:
        return self._perturbation(nom, ctx, dq, dv)


@dataclass(frozen=True)
class StepConvergence:
    """Finite-difference step-convergence report.

    ``errors[i]`` is the max-norm change of the derivative matrices between
    ``steps[i]`` and ``steps[i + 1]``; ``observed_order`` the fitted rate.
    """

    steps: tuple[float, ...]
    errors: tuple[float, ...]
    observed_order: float
    converged: bool


def _stacked(res: ClosedLoopDerivatives) -> Array:
    return np.concatenate(
        [
            res.d_acc_d_xi.ravel(),
            res.d_acc_d_eta.ravel(),
            res.d_lambda_d_xi.ravel(),
            res.d_lambda_d_eta.ravel(),
        ]
    )


def fd_step_convergence(
    model: Any,
    manifold: ConfigurationManifold,
    q: ArrayLike,
    v: ArrayLike,
    tau: ArrayLike,
    *,
    steps: tuple[float, ...] = (1e-3, 5e-4, 2.5e-4),
) -> StepConvergence:
    """Check that the central differences converge at second order as ``h`` halves.

    Preconditions: at least three strictly decreasing positive steps
    (``ValueError``).  ``converged`` is true when the successive changes shrink
    at order ``>= 1.5`` or are already below ``1e-9`` (round-off floor).
    """
    if len(steps) < 3 or any(s <= 0 for s in steps):
        raise ValueError("steps must hold at least three positive values")
    if any(b >= a for a, b in zip(steps, steps[1:], strict=False)):
        raise ValueError("steps must be strictly decreasing")
    stacked = [
        _stacked(
            FiniteDifferenceDerivatives(
                model, manifold, step=s
            ).constrained_acceleration_derivatives(q, v, tau)
        )
        for s in steps
    ]
    errors = tuple(
        float(np.max(np.abs(a - b))) for a, b in zip(stacked, stacked[1:], strict=False)
    )
    floor = 1e-9
    if errors[0] <= floor:
        return StepConvergence(tuple(steps), errors, float("inf"), True)
    ratio_h = steps[0] / steps[1]
    order = float(np.log(errors[0] / max(errors[1], 1e-300)) / np.log(ratio_h))
    ok = all(b < a for a, b in zip(errors, errors[1:], strict=False)) and order >= 1.5
    return StepConvergence(tuple(steps), errors, order, bool(ok or errors[-1] <= floor))


@dataclass(frozen=True)
class TangentStateSpace:
    """``d x_t/dt = F x_t + G du`` in the frame coordinates ``x_t = (xi, eta)``.

    ``F`` is ``(2, 2)``, ``G`` is ``(2, nu)`` for a one-dimensional tangent
    space (the case ``sec:var-lin`` specifies).
    """

    F: Array
    G: Array
    basis: Array


def tangent_state_space(
    derivs: ClosedLoopDerivatives,
    *,
    q: ArrayLike,
    manifold: ConfigurationManifold,
    basis_field: Callable[[Array], Array],
    selection: ArrayLike | None = None,
    step: float = 1e-5,
) -> TangentStateSpace:
    """``(F_t, G_t)`` in the frame ``P(q)``: ``F = [[0, 1], [P^T dA/dxi + c, P^T
    dA/deta]]`` and ``G = [[0], [P^T dA/dtau S^T]]``.

    ``c = (d_q P^T [P e]) a`` is the moving-frame (connection) term; for the
    pendulum it produces the ``omega^2`` in ``F10 = omega^2 + lambda/m``.
    ``selection`` is the actuator map ``S`` (``nu, nv``); default identity.

    Preconditions (``ValueError``): tangent dimension ``k = 1`` -- the kinematic
    connection of ``k > 1`` frames is not specified by NSX-80 and is not
    invented here; ``basis_field`` returns an ``(nv, 1)`` unit tangent.
    """
    k = derivs.tangent_dimension
    if k != 1:
        raise ValueError(
            f"tangent dimension must be 1 for a scalar frame state space, got {k}"
        )
    q_arr = _vec(q, manifold.nq, "q")
    basis = np.asarray(basis_field(q_arr), dtype=np.float64).reshape(-1, 1)
    plus = np.asarray(basis_field(manifold.retract(q_arr, step * basis[:, 0])))
    minus = np.asarray(basis_field(manifold.retract(q_arr, -step * basis[:, 0])))
    d_basis = (plus - minus).reshape(-1, 1) / (2.0 * step)
    conn = float((d_basis.T @ derivs.acceleration)[0])
    sel = np.eye(derivs.nv) if selection is None else np.asarray(selection, float)
    if sel.ndim != 2 or sel.shape[1] != derivs.nv:
        raise ValueError(f"selection must have shape (nu, {derivs.nv})")
    f_mat = np.array(
        [
            [0.0, 1.0],
            [
                float((basis.T @ derivs.d_acc_d_xi)[0, 0]) + conn,
                float((basis.T @ derivs.d_acc_d_eta)[0, 0]),
            ],
        ]
    )
    g_mat = np.zeros((2, sel.shape[0]))
    g_mat[1, :] = (basis.T @ derivs.d_acc_d_tau @ sel.T)[0]
    return TangentStateSpace(F=f_mat, G=g_mat, basis=basis)
