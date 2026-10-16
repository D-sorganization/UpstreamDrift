"""Pinocchio analytic closed-loop derivatives (NSX-85, #12714).

The FULL reference engine of ``Capability.CLOSED_LOOP_DERIVATIVES``.  It plugs
into the NSX-84 seam (:mod:`loop_nullspace.tangent_linearization`) and returns
the same :class:`ClosedLoopDerivatives`, but every derivative comes from
Pinocchio's own analytic kinematics and dynamics instead of differencing:

* ``(da/dq, da/dv, da/dtau)`` and ``(dlam/dq, dlam/dv, dlam/dtau)`` are the
  total derivatives of ``constraintDynamics`` with the weld reaction re-solved
  (``computeConstraintDynamicsDerivatives``), so the geometric preload
  ``K_geom = d_q (J_c^T lam)`` (mechanism M2) is included by Pinocchio;
* the tangent-bundle velocity offset ``O`` (``J_c O xi = -(dJ_c[P xi]) v``)
  uses the analytic joint-velocity derivatives
  (:func:`weld_jacobian_directional_drift`).

Closure convention (rule R8).  The canonical ``J_c = C J_native`` with
``C = closure_conversion_matrix(...)``.  For Pinocchio the native rows are the
constraint LOCAL frame, linear then angular, relative sign ``+1``, and the
canonical grip frame *is* that frame (``closure_wrench_frame``), so ``C = I``
exactly: acceleration and reaction derivatives need no conversion.  The
identity is verified at every evaluation; if a future model broke it the
``dC/dq`` term of ``lam_c = C^{-T} lam_n`` would be required and the evaluation
is refused instead of silently dropping it.

The tangent basis, guards (:class:`ClosureRankChangeError`), admissibility
check ``J_c v = 0`` and result assembly are the NSX-84 ones, imported unchanged.
"""

from __future__ import annotations

import dataclasses
import logging
from importlib import import_module
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.loop_nullspace.closure_convention import (
    closure_conversion_matrix,
)
from src.shared.python.loop_nullspace.configuration_manifold import EuclideanManifold
from src.shared.python.loop_nullspace.derivative_capability import (
    ClosedLoopDerivatives,
    DerivativeMethod,
)
from src.shared.python.loop_nullspace.provider import AdapterConstrainedProvider
from src.shared.python.loop_nullspace.tangent_linearization import (
    _DerivativeBase,
    _Nominal,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NativeConstraintDerivatives",
    "PinocchioAnalyticDerivatives",
    "native_constraint_derivatives",
    "weld_jacobian_directional_drift",
]

Array = NDArray[np.float64]

_IDENTITY_TOL = 1e-12
_TANGENT_TOL = 1e-7
_N_NATIVE_DERIVATIVES = 6


class NativeConstraintDerivatives(NamedTuple):
    """Pinocchio constrained dynamics and its analytic derivatives (native order)."""

    acceleration: Array
    reaction: Array
    d_acc_d_q: Array
    d_acc_d_v: Array
    d_acc_d_tau: Array
    d_lambda_d_q: Array
    d_lambda_d_v: Array
    d_lambda_d_tau: Array


def native_constraint_derivatives(
    plant: Any, q: ArrayLike, v: ArrayLike, tau: ArrayLike
) -> NativeConstraintDerivatives:
    """``a, lambda`` and their partials from ``computeConstraintDynamicsDerivatives``.

    Pinocchio's sign is ``M a + h = tau + J^T lambda`` with ``lambda`` the weld
    wrench in the constraint LOCAL frame (identical to the canonical
    convention, see the module docstring).  Partials are with the other two
    arguments fixed and the reaction re-solved.  Raises ``ValueError`` on a
    missing or nonfinite derivative block and ``FloatingPointError`` on a
    nonfinite acceleration.  Postcondition: read-only finite arrays.
    """
    pin = import_module("pinocchio")
    accel = plant.constrained_acceleration(q, v, tau)
    reaction = np.array(plant.data.lambda_c, dtype=float, copy=True).reshape(-1)
    raw = pin.computeConstraintDynamicsDerivatives(
        plant.model, plant.data, plant.constraints, plant.constraint_data
    )
    if len(raw) < _N_NATIVE_DERIVATIVES:
        raise ValueError("Pinocchio returned fewer than six constraint derivatives")
    nv = plant.model.nv
    blocks = [np.array(b, dtype=float, copy=True) for b in raw[:_N_NATIVE_DERIVATIVES]]
    for i, block in enumerate(blocks):
        expected = (nv, nv) if i < 3 else (reaction.size, nv)
        if block.shape != expected or not np.isfinite(block).all():
            raise ValueError(f"Invalid native constraint derivative block {i}")
    for arr in (accel, reaction, *blocks):
        arr.setflags(write=False)
    return NativeConstraintDerivatives(accel, reaction, *blocks)


def _frame_twist_partials(
    plant: Any, joint_id: int, placement: Any, directions: Array
) -> Array:
    """``d V`` of the joint-fixed frame ``placement`` in its own axes, ``(6, k)``.

    ``V`` is the frame twist (linear at its origin, then angular) expressed in
    the frame's own axes; partials are along the tangent columns ``directions``
    at fixed ``v``.  The joint LOCAL twist partials are Pinocchio's analytic
    ``getJointVelocityDerivatives``; the constant offset gives
    ``dv = R^T (dv_j + d omega_j x p)`` and ``d omega = R^T d omega_j``.
    """
    k = directions.shape[1]
    if joint_id == 0:
        return np.zeros((6, k))
    pin = import_module("pinocchio")
    local = pin.ReferenceFrame.LOCAL
    d_dq = np.asarray(
        pin.getJointVelocityDerivatives(plant.model, plant.data, joint_id, local)[0]
    )
    partial = d_dq @ directions
    offset = np.asarray(placement.translation)
    rotation = np.asarray(placement.rotation)
    linear = partial[:3] + np.cross(partial[3:].T, offset).T
    return np.vstack([rotation.T @ linear, rotation.T @ partial[3:]])


def weld_jacobian_directional_drift(
    plant: Any, q: ArrayLike, v: ArrayLike, directions: ArrayLike
) -> Array:
    """``dJ_n[w] v`` for each tangent column ``w``, native LOCAL rows, ``(6, k)``.

    ``J_n v`` is the weld's relative twist ``V_1 - X_12 V_2`` (hand frame minus
    club frame, in the constraint LOCAL frame, linear then angular).  Its
    derivative along ``w`` at fixed ``v`` is ``dV_1 - dV_2``: the terms from
    ``dX_12 V_2`` multiply the relative pose rate ``J_n w`` and vanish for a
    tangent ``w``, and ``X_12 = I`` on the closure.  ``V_i`` partials are
    Pinocchio's analytic LOCAL joint-velocity derivatives, so no differencing is
    involved.  ``directions`` must lie in ``ker J_n`` and ``v`` in the closure
    tangent (``J_n v = 0``); both are required for the dropped terms to vanish.
    Raises ``ValueError`` on wrong shapes or non-tangent directions.
    Postcondition: finite ``(6, k)``.
    """
    pin = import_module("pinocchio")
    q_arr, v_arr = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
    w = np.asarray(directions, dtype=float)
    nv = plant.model.nv
    if w.ndim != 2 or w.shape[0] != nv or q_arr.shape != (nv,) or v_arr.shape != (nv,):
        raise ValueError(f"q, v need shape ({nv},) and directions ({nv}, k)")
    if w.shape[1] == 0:
        return np.zeros((6, 0))
    pin.computeJointJacobians(plant.model, plant.data, q_arr)
    pin.computeForwardKinematicsDerivatives(
        plant.model, plant.data, q_arr, v_arr, np.zeros(nv)
    )
    kernel = _native_jacobian(plant)
    if np.max(np.abs(kernel @ w)) > _TANGENT_TOL * max(1.0, float(np.abs(w).max())):
        raise ValueError("directions must be tangent to the closure (J w = 0)")
    constraint = plant.constraints[0]
    first = _frame_twist_partials(
        plant, int(constraint.joint1_id), constraint.joint1_placement, w
    )
    second = _frame_twist_partials(
        plant, int(constraint.joint2_id), constraint.joint2_placement, w
    )
    return np.asarray(first - second)


def _native_jacobian(plant: Any) -> Array:
    pin = import_module("pinocchio")
    return np.asarray(
        pin.getConstraintsJacobian(
            plant.model, plant.data, plant.constraints, plant.constraint_data
        ),
        dtype=float,
    )


class PinocchioAnalyticDerivatives(_DerivativeBase):
    """Analytic ``ClosedLoopDerivatives`` of the Pinocchio weld loop (``FULL``).

    ``adapter`` is a ``PinocchioForceAdapter``: it supplies the canonical
    ``M, h, J_c, Jdot_v`` through :class:`AdapterConstrainedProvider` (the basis,
    guards and admissibility check are the NSX-84 ones) and its native plant.
    ``step`` only sizes the rank/contact-mode guard probes; no derivative is
    differenced.  Preconditions and errors are those of
    :meth:`constrained_acceleration_derivatives` of the NSX-84 base.
    """

    scheme = "pinocchio_constraint_dynamics_derivatives"

    def __init__(self, adapter: Any, **options: Any) -> None:
        super().__init__(
            AdapterConstrainedProvider(adapter),
            EuclideanManifold(int(adapter.nv)),
            **options,
        )
        self._adapter = adapter
        self._plant = adapter.native_plant

    # -- closure convention (R8) ------------------------------------------
    def _require_identity_conversion(self, q: Array) -> None:
        adapter = self._adapter
        native_rot, native_point = adapter.native_closure_frame(q)
        grip_rot, grip_point = adapter.closure_wrench_frame(q)
        matrix = closure_conversion_matrix(
            native=adapter.native_closure_convention,
            native_frame_rotation=native_rot,
            grip_frame_rotation=grip_rot,
            native_reference_point=native_point,
            grip_reference_point=grip_point,
        )
        if np.max(np.abs(matrix - np.eye(6))) > _IDENTITY_TOL:
            raise ValueError(
                "native to canonical closure conversion is not the identity; the "
                "dC/dq term of the reaction derivative is not implemented"
            )

    # -- hooks of the NSX-84 base -----------------------------------------
    def _nominal(self, q: ArrayLike, v: ArrayLike, tau: ArrayLike) -> _Nominal:
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape == (self._manifold.nq,) and np.isfinite(q_arr).all():
            self._require_identity_conversion(q_arr)
        return super()._nominal(q, v, tau)

    def _solve(self, q: Array, v: Array, tau: Array) -> tuple[Array, Array]:
        native = native_constraint_derivatives(self._plant, q, v, tau)
        return np.asarray(native.acceleration), np.asarray(native.reaction)

    def _velocity_offset(self, q: Array, v: Array, basis: Array, jac: Array) -> Array:
        """``O`` with ``J_c O xi = -(dJ_c[P xi]) v`` from analytic kinematics."""
        nv, k = basis.shape
        if jac.shape[0] == 0 or k == 0:
            return np.zeros((nv, k))
        rate = weld_jacobian_directional_drift(self._plant, q, v, basis)
        return np.asarray(-np.linalg.pinv(jac, rcond=1e-12) @ rate)

    def _column_context(self, nom: _Nominal) -> NativeConstraintDerivatives:
        return native_constraint_derivatives(self._plant, nom.q, nom.v, nom.tau)

    def _column(
        self, nom: _Nominal, ctx: Any, dq: Array, dv: Array
    ) -> tuple[Array, Array]:
        d_acc = ctx.d_acc_d_q @ dq + ctx.d_acc_d_v @ dv
        d_lam = ctx.d_lambda_d_q @ dq + ctx.d_lambda_d_v @ dv
        return np.asarray(d_acc), np.asarray(d_lam)

    def _tau_columns(self, nom: _Nominal) -> tuple[Array, Array]:
        native = native_constraint_derivatives(self._plant, nom.q, nom.v, nom.tau)
        return np.asarray(native.d_acc_d_tau), np.asarray(native.d_lambda_d_tau)

    def constrained_acceleration_derivatives(
        self, q: ArrayLike, v: ArrayLike, tau: ArrayLike
    ) -> ClosedLoopDerivatives:
        """Tangent-bundle derivatives of ``(a, lambda)``, ``method=analytic``.

        Same preconditions, guards and postconditions as the NSX-84 base; the
        result is tagged ``derivative_method=analytic`` with no step.
        """
        result = super().constrained_acceleration_derivatives(q, v, tau)
        return dataclasses.replace(
            result, method=DerivativeMethod.ANALYTIC, scheme=self.scheme, step=None
        )
