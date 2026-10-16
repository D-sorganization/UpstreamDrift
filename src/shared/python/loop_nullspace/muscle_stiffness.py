"""Short-range musculotendon stiffness and its effect on the linearised loop (NSX-94).

Hypothesis ``H-MUSCLE-SHORT-RANGE-STIFFNESS`` (mechanism M3, citation
``CIT-CUI-2008``): the active stiffness of a muscle is force-dependent and acts
in series with a passive stiffness, so changing muscle force changes the
linearised dynamics.  The supporting evidence is feline hindlimb muscle and the
registry says "force-dependent", not "activation-dependent"; this module does
not claim the evidence transfers to a human golfer's arm, and no stiffness value
is adopted from it.

Per-muscle series stiffness.  The muscle-tendon unit is the fibre in series with
the tendon,

    k_MTU = k_m k_T / (k_m + k_T),

with ``k_T -> infinity`` (a rigid tendon, ``TendonKind.RIGID``) giving
``k_MTU = k_m``.  Two declared fibre-stiffness models:

* ``ENGINE_CURVE`` -- the engine's own ``dF/dl`` of the fibre along the tendon,
  tabulated against activation in the frame.  A negative value (the descending
  limb of the Gaussian active force-length curve) is *not* a stiffness and
  fails closed.
* ``FORCE_PROPORTIONAL`` -- the declared short-range-stiffness slope
  ``k_m = rho F(a)`` with ``rho`` in 1/m a hypothesis parameter supplied by the
  experiment.

Joint stiffness.  With ``tau_i = sum_k R_ki F_k`` and ``dF_k/dq_j = -k_k R_kj``
(a length increase is ``dL = -R dq``),

    K_ij = -d tau_i / d q_j = sum_k R_ki k_k R_kj  -  sum_k F_k dR_ki/dq_j .

The second (geometry) term needs the moment-arm Jacobian; when the frame does
not carry it the term is omitted *and flagged* (``geometric_term_included``).

Linearised dynamics.  A stiffness torque ``-K dq`` adds to the generalised
load, so the closed-loop response gains ``d a/d xi -= (d a/d tau) K P`` and
``d lambda/d xi -= (d lambda/d tau) K P`` (:func:`apply_muscle_stiffness`).  This
lifts, for the quasi-static activation, the ``muscle_activation`` entry of
:data:`adapter_derivatives.UNSUPPORTED_STATE_COMPONENTS`: the activation is a
held input of the experiment, entering the linearization through ``K(a)`` and the
noise covariance; the excitation-to-activation lag and fibre-length dynamics
are **not** differentiated.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.derivative_capability import (
    ClosedLoopDerivatives,
    DerivativeCapabilityUnavailable,
)
from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.muscle_frame import MuscleFrame, TendonKind

__all__ = [
    "MuscleStiffness",
    "StiffnessModel",
    "StiffnessNotQualified",
    "apply_muscle_stiffness",
    "muscle_stiffness",
    "series_stiffness",
]


class StiffnessNotQualified(DerivativeCapabilityUnavailable):
    """The model cannot supply a qualified musculotendon stiffness (fail closed)."""


class StiffnessModel(str, Enum):
    """How the fibre stiffness ``k_m`` is obtained."""

    ENGINE_CURVE = "engine_curve"
    FORCE_PROPORTIONAL = "force_proportional"


def series_stiffness(k_fiber: ArrayLike, k_tendon: ArrayLike) -> Array:
    """``k_m k_T / (k_m + k_T)`` elementwise; ``k_T = inf`` returns ``k_m``.

    Preconditions (``ValueError``): ``k_m`` finite and non-negative, ``k_T``
    positive (``inf`` allowed).  Postcondition: ``0 <= k_MTU <= min(k_m, k_T)``.
    """
    k_m = np.atleast_1d(np.asarray(k_fiber, dtype=np.float64))
    k_t = np.atleast_1d(np.asarray(k_tendon, dtype=np.float64))
    if not np.all(np.isfinite(k_m)) or np.any(k_m < 0.0):
        raise ValueError("fibre stiffness must be finite and non-negative")
    if np.any(np.isnan(k_t)) or np.any(k_t <= 0.0):
        raise ValueError("tendon stiffness must be positive (inf allowed)")
    out = np.where(np.isinf(k_t), k_m, k_m * k_t / (k_m + k_t))
    return np.asarray(np.where(k_m == 0.0, 0.0, out))


@dataclass(frozen=True)
class MuscleStiffness:
    """Per-muscle and joint stiffness at one activation vector.

    Attributes:
        fiber, tendon: ``k_m`` and ``k_T`` in N/m, ``(m,)`` (``tendon`` is
            ``inf`` for a rigid tendon).
        muscle: ``k_MTU`` in N/m, ``(m,)``.
        joint: ``K`` in N m/rad, ``(n, n)``, symmetric.
        geometric_term_included: Whether ``-sum F dR/dq`` is in ``joint``.
        model, tendon_kind: Provenance of the numbers.
    """

    fiber: Array
    tendon: Array
    muscle: Array
    joint: Array
    geometric_term_included: bool
    model: StiffnessModel
    tendon_kind: TendonKind


def _grid_values(frame: MuscleFrame, grid: Array, activation: Array) -> Array:
    return np.array(
        [
            np.interp(activation[k], frame.activation_grid, grid[:, k])
            for k in range(frame.n_muscles)
        ]
    )


def _fiber_stiffness(
    frame: MuscleFrame,
    activation: Array,
    model: StiffnessModel,
    force_gain: float | None,
) -> Array:
    if model is StiffnessModel.FORCE_PROPORTIONAL:
        if force_gain is None or not (math.isfinite(force_gain) and force_gain > 0.0):
            raise ValueError("force_gain (1/m) must be positive and finite")
        return np.asarray(force_gain * frame.engine_force(activation))
    if frame.fiber_stiffness_grid is None:
        raise StiffnessNotQualified(
            f"engine {frame.engine!r} exposes no fibre stiffness; use the declared "
            "FORCE_PROPORTIONAL model or leave the stiffness UNAVAILABLE"
        )
    k_fib = _grid_values(frame, frame.fiber_stiffness_grid, activation)
    bad = [frame.names[k] for k in range(k_fib.size) if k_fib[k] < 0.0]
    if bad:
        raise StiffnessNotQualified(
            f"negative engine fibre stiffness for {bad}: the muscle is on the "
            "descending limb of its active force-length curve, where dF/dl is not "
            "a stiffness and the series formula does not apply"
        )
    return k_fib


def _tendon_stiffness(frame: MuscleFrame, activation: Array) -> Array:
    if frame.tendon is TendonKind.RIGID:
        return np.full(frame.n_muscles, np.inf)
    if frame.tendon_stiffness_grid is None:
        raise StiffnessNotQualified(
            f"engine {frame.engine!r} has a compliant tendon but exposes no tendon "
            "stiffness; the series stiffness is unavailable"
        )
    return _grid_values(frame, frame.tendon_stiffness_grid, activation)


def muscle_stiffness(
    frame: MuscleFrame,
    activation: ArrayLike,
    model: StiffnessModel,
    *,
    force_gain: float | None = None,
) -> MuscleStiffness:
    """Series musculotendon and joint stiffness at ``activation`` (module doc).

    Preconditions (``ValueError``): ``activation`` in ``[0, 1]^m``; for
    ``FORCE_PROPORTIONAL`` a positive ``force_gain`` (1/m).  Raises
    :class:`StiffnessNotQualified` -- never returns a guess -- when the engine
    gives no fibre or tendon stiffness, or a negative fibre stiffness.
    Postconditions: ``0 <= muscle <= fiber``, ``joint`` symmetric, and
    ``joint = R^T diag(muscle) R`` exactly when no moment-arm Jacobian is
    carried.
    """
    a = np.asarray(activation, dtype=np.float64)
    if a.shape != (frame.n_muscles,) or np.any(a < 0.0) or np.any(a > 1.0):
        raise ValueError(
            f"activation must lie in [0, 1] with shape ({frame.n_muscles},)"
        )
    model = StiffnessModel(model)
    k_fib = _fiber_stiffness(frame, a, model, force_gain)
    k_ten = _tendon_stiffness(frame, a)
    k_mtu = series_stiffness(k_fib, k_ten)
    joint = frame.moment_arms.T @ (k_mtu[:, None] * frame.moment_arms)
    geometric = frame.moment_arm_jacobian is not None
    if geometric:
        force = frame.engine_force(a)
        joint = joint - np.einsum("k,kij->ij", force, frame.moment_arm_jacobian)
    return MuscleStiffness(
        fiber=k_fib,
        tendon=k_ten,
        muscle=k_mtu,
        joint=np.asarray(0.5 * (joint + joint.T)),
        geometric_term_included=geometric,
        model=model,
        tendon_kind=frame.tendon,
    )


def apply_muscle_stiffness(
    derivs: ClosedLoopDerivatives,
    joint_stiffness: ArrayLike,
    *,
    coordinate_indices: Sequence[int],
) -> ClosedLoopDerivatives:
    """Closed-loop derivatives with a joint stiffness ``K`` added to the load.

    ``joint_stiffness`` is the ``(n, n)`` matrix of the muscle coordinates;
    ``coordinate_indices`` places them among the ``nv`` velocity coordinates
    (others get zero stiffness).  Returns a copy with
    ``d_acc_d_xi - d_acc_d_tau K P`` and ``d_lambda_d_xi - d_lambda_d_tau K P``;
    the nominal motion, the load response and the velocity derivatives are
    unchanged.  The reaction is re-solved because ``d_*_d_tau`` already
    contains the KKT solve.

    Preconditions (``ValueError``): ``K`` finite and symmetric, indices unique
    and in ``[0, nv)`` with ``len == n``.  Postcondition: ``J_c @ basis = 0``
    is unaffected (the basis is shared).
    """
    k_mat = np.asarray(joint_stiffness, dtype=np.float64)
    idx = tuple(int(i) for i in coordinate_indices)
    n = len(idx)
    if k_mat.shape != (n, n) or not np.all(np.isfinite(k_mat)):
        raise ValueError(
            f"joint_stiffness must be finite with shape ({n}, {n}) (finite)"
        )
    if np.max(np.abs(k_mat - k_mat.T), initial=0.0) > 1e-9 * max(
        1.0, np.max(np.abs(k_mat))
    ):
        raise ValueError("joint_stiffness must be symmetric")
    nv = derivs.nv
    if len(set(idx)) != n or any(i < 0 or i >= nv for i in idx):
        raise ValueError(f"coordinate_indices must be unique and in [0, {nv})")
    full = np.zeros((nv, nv))
    full[np.ix_(idx, idx)] = k_mat
    shift = full @ derivs.basis
    return dataclasses.replace(
        derivs,
        d_acc_d_xi=derivs.d_acc_d_xi - derivs.d_acc_d_tau @ shift,
        d_lambda_d_xi=derivs.d_lambda_d_xi - derivs.d_lambda_d_tau @ shift,
    )
