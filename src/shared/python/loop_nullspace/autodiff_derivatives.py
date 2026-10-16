"""Exact-differential KKT route of the derivative seam (NSX-86, #12715).

Engines that can differentiate their own model (Drake AutoDiff, later analytic
engines) supply the *differentials* of the four canonical arrays
``M(q), h(q, v), J_c(q), Jdot_v(q, v)``; this module differentiates the single
KKT solve (``constrained_dynamics``) with them, exactly as
:class:`AssembledKKTDerivatives` does with finite differences::

    [[M, -J^T], [J, 0]] [da; dlam] = [ -d(M a + h)|_a + K_geom dq ; d(J a + Jdot_v)|_a ]

with ``K_geom = d_q (J_c^T lambda) = sum_r lambda_r dJ_r/dq`` the mechanism-M2
geometric preload.  The result is expressed in the same closure tangent basis
and canonical closure convention (epic rule R8) as the finite-difference
reference, so the two are interchangeable.

There is no silent fall-back: a model that cannot supply differentials, or a
manifold other than the Euclidean one (quaternion joints are not covered by the
coordinate differentials), is refused with
:class:`DerivativeCapabilityUnavailable`.  Pure numpy; no engine imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.configuration_manifold import (
    ConfigurationManifold,
    EuclideanManifold,
)
from src.shared.python.loop_nullspace.derivative_capability import (
    ClosedLoopDerivatives,
    DerivativeCapabilityUnavailable,
    DerivativeMethod,
    _check,
)
from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.tangent_linearization import (
    AssembledKKTDerivatives,
    KKTContext,
    _Nominal,
)

__all__ = [
    "AutoDiffKKTDerivatives",
    "DifferentialsUnavailable",
    "ModelDifferentials",
]


class DifferentialsUnavailable(RuntimeError):
    """A model component cannot be differentiated exactly (engine-raised)."""


@dataclass(frozen=True)
class ModelDifferentials:
    """Partial derivatives of ``M, h, J_c, Jdot_v`` at one ``(q, v)``.

    ``n`` coordinates, ``m`` closure rows; ``k`` indexes the differentiated
    coordinate.  ``d_mass_dq[i, j, k] = dM_ij/dq_k``, ``d_bias_dq[i, k] =
    dh_i/dq_k``, ``d_bias_dv[i, k] = dh_i/dv_k``, ``d_jac_dq[r, j, k] =
    dJ_rj/dq_k``, ``d_drift_dq[r, k]`` and ``d_drift_dv[r, k]``.
    """

    d_mass_dq: Array
    d_bias_dq: Array
    d_bias_dv: Array
    d_jac_dq: Array
    d_drift_dq: Array
    d_drift_dv: Array

    def __post_init__(self) -> None:
        n = int(np.shape(self.d_bias_dq)[0])
        m = int(np.shape(self.d_drift_dq)[0])
        spec = {
            "d_mass_dq": (n, n, n),
            "d_bias_dq": (n, n),
            "d_bias_dv": (n, n),
            "d_jac_dq": (m, n, n),
            "d_drift_dq": (m, n),
            "d_drift_dv": (m, n),
        }
        for name, shape in spec.items():
            object.__setattr__(self, name, _check(getattr(self, name), shape, name))

    def map_closure_rows(self, matrix: ArrayLike) -> ModelDifferentials:
        """Rows of ``J_c`` and ``Jdot_v`` mapped by a constant ``(m', m)`` matrix.

        Valid only when the row map does not depend on ``q`` (``dC/dq = 0``);
        the caller owns that precondition.
        """
        c = np.asarray(matrix, dtype=np.float64)
        return ModelDifferentials(
            self.d_mass_dq,
            self.d_bias_dq,
            self.d_bias_dv,
            np.einsum("sr,rjk->sjk", c, self.d_jac_dq),
            c @ self.d_drift_dq,
            c @ self.d_drift_dv,
        )

    def mass_along(self, dq: Array) -> Array:
        """``dM[dq]``, shape ``(n, n)``."""
        return np.asarray(np.einsum("ijk,k->ij", self.d_mass_dq, dq))

    def jacobian_along(self, dq: Array) -> Array:
        """``dJ_c[dq]``, shape ``(m, n)``."""
        return np.asarray(np.einsum("rjk,k->rj", self.d_jac_dq, dq))

    def preload(self, reaction: Array) -> Array:
        """``K_geom = sum_r lambda_r dJ_r/dq``, shape ``(n, n)``."""
        return np.asarray(np.einsum("r,rik->ik", reaction, self.d_jac_dq))


class AutoDiffKKTDerivatives(AssembledKKTDerivatives):
    """Differentiated KKT system fed by exact model differentials.

    ``model`` provides ``mass_matrix, bias_forces, closure_jacobian,
    closure_drift`` (as for the finite-difference routes) and
    ``differentials(q, v) -> ModelDifferentials``.  ``step`` only sizes the
    rank / contact-mode guard probes; it never enters a derivative.
    """

    scheme = "assembled_kkt_exact"
    method = DerivativeMethod.AUTODIFF
    reports_step = False

    def __init__(
        self, model: Any, manifold: ConfigurationManifold, **kwargs: Any
    ) -> None:
        if not callable(getattr(model, "differentials", None)):
            raise DerivativeCapabilityUnavailable(
                f"{type(model).__name__} provides no exact differentials; refusing "
                "to fall back to finite differences"
            )
        if not isinstance(manifold, EuclideanManifold):
            raise DerivativeCapabilityUnavailable(
                f"exact coordinate differentials need a Euclidean manifold, got "
                f"{type(manifold).__name__}"
            )
        super().__init__(model, manifold, **kwargs)
        self._memo: tuple[Array, Array, ModelDifferentials] | None = None

    def _differentials(self, q: Array, v: Array) -> ModelDifferentials:
        memo = self._memo
        if (
            memo is not None
            and np.array_equal(memo[0], q)
            and np.array_equal(memo[1], v)
        ):
            return memo[2]
        try:
            diffs = self._model.differentials(q, v)
        except DifferentialsUnavailable as exc:
            raise DerivativeCapabilityUnavailable(
                f"exact differentials unavailable: {exc}"
            ) from exc
        if not isinstance(diffs, ModelDifferentials):
            raise TypeError("model.differentials must return ModelDifferentials")
        self._memo = (q.copy(), v.copy(), diffs)
        return diffs

    def _velocity_offset(self, q: Array, v: Array, basis: Array, jac: Array) -> Array:
        """``O`` with ``J O xi = -(dJ[P xi]) v`` from the exact ``dJ``."""
        nv, k = basis.shape
        if jac.shape[0] == 0 or k == 0:
            return np.zeros((nv, k))
        diffs = self._differentials(q, v)
        pinv = np.linalg.pinv(jac, rcond=1e-12)
        cols = [-pinv @ (diffs.jacobian_along(basis[:, j]) @ v) for j in range(k)]
        return np.array(cols).T

    def _column_context(self, nom: _Nominal) -> KKTContext:
        diffs = self._differentials(nom.q, nom.v)
        return KKTContext(diffs.preload(nom.reaction), diffs)

    def _source_derivatives(
        self, nom: _Nominal, ctx: KKTContext, dq: Array, dv: Array
    ) -> tuple[Array, Array]:
        d: ModelDifferentials = ctx.differentials
        acc = nom.acceleration
        d_dyn = d.mass_along(dq) @ acc + d.d_bias_dq @ dq + d.d_bias_dv @ dv
        d_clo = d.jacobian_along(dq) @ acc + d.d_drift_dq @ dq + d.d_drift_dv @ dv
        return d_dyn, d_clo

    def constrained_acceleration_derivatives(
        self, q: ArrayLike, v: ArrayLike, tau: ArrayLike
    ) -> ClosedLoopDerivatives:
        """Exact tangent-bundle derivatives of ``(a, lambda)`` at ``(q, v, tau)``.

        Same preconditions, guards and postconditions as
        :meth:`FiniteDifferenceDerivatives.constrained_acceleration_derivatives`;
        additionally raises :class:`DerivativeCapabilityUnavailable` when the
        model cannot supply exact differentials.  ``method`` is ``AUTODIFF`` and
        ``step`` is ``None`` (nothing was differenced).
        """
        return super().constrained_acceleration_derivatives(q, v, tau)
