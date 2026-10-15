"""Constrained ZTCF / ZVCF on the closed loop (NSX-30, #12570).

The unconstrained primitives in :mod:`.ztcf_zvcf` use ``M`` and ``h`` only,
which on a closed loop lets the reaction vanish.  For a
:class:`~.protocol.ConstrainedEquationsOfMotionProvider` the reaction is
re-solved for every counterfactual through the KKT projector
(:func:`~src.shared.python.loop_nullspace.constrained_dynamics.constrained_acceleration`),
never held::

    ZTCF: tau = 0, state (q, v) kept            -> a_ZTCF
    ZVCF: v = 0 and tau = 0, configuration kept -> a_ZVCF
    control-induced total: a(tau) - a_ZTCF = P_c tau

Names follow ``model_aware_matching.tex`` (Section "ZTCF and ZVCF as exact
projections"): both are *pointwise*; the control-preserved zero-velocity
acceleration is never called a ZVCF.  Allocation dependence follows
``loop_nullspace.tex``: the accelerations and the control-induced total depend
only on measured motion (``INVARIANT``); the per-actuator split
``P_c e_i tau_i`` depends on the individual torques and is
``FAMILY_DEPENDENT``.  The report is a measured-state diagnostic, not a
rollout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.constrained_dynamics import (
    constrained_acceleration,
)
from src.shared.python.loop_nullspace.evidence import AllocationDependence, Array
from src.shared.python.simulation_backends.protocol import (
    ConstrainedEquationsOfMotionProvider,
)

__all__ = [
    "ConstrainedCounterfactuals",
    "constrained_acceleration_at",
    "constrained_counterfactual_report",
    "is_constrained_provider",
]

_PER_ACTUATOR_HYPOTHESIS = (
    "per-actuator attribution P_c e_i tau_i depends on the individual torques, "
    "which vary across the admissible allocation family"
)
_GAUGE_HYPOTHESIS = (
    "closure reaction is unique only up to the null space of J_c^T when J_c "
    "is rank deficient"
)


def is_constrained_provider(provider: Any) -> bool:
    """Whether ``provider`` also supplies the closure rows."""
    return isinstance(provider, ConstrainedEquationsOfMotionProvider)


def _state(name: str, value: ArrayLike, n: int | None = None) -> Array:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    if n is not None and arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    return arr


def _require_constrained(provider: Any) -> None:
    if not is_constrained_provider(provider):
        raise TypeError(
            "provider must satisfy ConstrainedEquationsOfMotionProvider "
            f"(closure_jacobian, closure_drift); got {type(provider).__name__}"
        )


def constrained_acceleration_at(
    provider: ConstrainedEquationsOfMotionProvider,
    q: ArrayLike,
    v: ArrayLike,
    tau: ArrayLike,
) -> tuple[Array, Array]:
    """``(a, lambda)`` at ``(q, v)`` under generalised load ``tau``.

    One provider evaluation then the single KKT projector.  ``tau`` is the
    generalised load (``S^T tau_a``).  Raises ``TypeError`` for a provider
    without closure rows and ``ValueError`` for malformed state.
    """
    _require_constrained(provider)
    q_arr = _state("q", q)
    v_arr = _state("v", v)
    mass = np.asarray(provider.mass_matrix(q_arr), dtype=np.float64)
    n = mass.shape[0]
    tau_arr = _state("tau", tau, n)
    return constrained_acceleration(
        mass,
        provider.bias_forces(q_arr, v_arr),
        tau_arr,
        provider.closure_jacobian(q_arr),
        provider.closure_drift(q_arr, v_arr),
    )


@dataclass(frozen=True)
class ConstrainedCounterfactuals:
    """Pointwise constrained counterfactuals with their allocation dependence.

    Attributes:
        ztcf: ``a`` with ``tau = 0`` at the measured ``(q, v)``.
        zvcf: ``a`` with ``v = 0`` and ``tau = 0`` at the measured ``q``.
        control_total: ``a(tau) - a_ZTCF = P_c tau`` (``INVARIANT``).
        per_actuator: column ``i`` is ``P_c e_i tau_i`` (``FAMILY_DEPENDENT``);
            rows sum to ``control_total``.
        reaction_ztcf, reaction_zvcf: re-solved closure reactions.
        dependence: ``AllocationDependence`` per quantity name.
        hypotheses: the stated assumptions of each family-dependent quantity.
    """

    ztcf: Array
    zvcf: Array
    control_total: Array
    per_actuator: Array
    reaction_ztcf: Array
    reaction_zvcf: Array
    dependence: Mapping[str, AllocationDependence]
    hypotheses: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        for name in (
            "ztcf",
            "zvcf",
            "control_total",
            "per_actuator",
            "reaction_ztcf",
            "reaction_zvcf",
        ):
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite")
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)


def constrained_counterfactual_report(
    provider: ConstrainedEquationsOfMotionProvider,
    q: ArrayLike,
    v: ArrayLike,
    tau: ArrayLike,
) -> ConstrainedCounterfactuals:
    """ZTCF, ZVCF and the control-induced split at one measured state.

    Preconditions: ``provider`` is constrained (``TypeError``); ``q``, ``v``
    finite and ``tau`` a finite generalised load of length ``n`` (``ValueError``).
    Postconditions: ``ztcf + control_total`` equals the closed-loop acceleration
    under ``tau`` to rounding; ``per_actuator.sum(axis=1) == control_total``;
    every acceleration satisfies the closure at acceleration level.  The
    reactions are re-solved, so no quantity assumes a held ``lambda``.
    """
    _require_constrained(provider)
    q_arr = _state("q", q)
    v_arr = _state("v", v)
    n = np.asarray(provider.mass_matrix(q_arr)).shape[0]
    tau_arr = _state("tau", tau, n)
    zeros = np.zeros(n)
    ztcf, lam0 = constrained_acceleration_at(provider, q_arr, v_arr, zeros)
    zvcf, lam00 = constrained_acceleration_at(provider, q_arr, zeros, zeros)
    full, _ = constrained_acceleration_at(provider, q_arr, v_arr, tau_arr)
    control_total = full - ztcf
    columns = [
        constrained_acceleration_at(provider, q_arr, zeros, tau_arr[i] * np.eye(n)[i])[
            0
        ]
        - zvcf
        for i in range(n)
    ]
    dependence = {
        "ztcf": AllocationDependence.INVARIANT,
        "zvcf": AllocationDependence.INVARIANT,
        "control_total": AllocationDependence.INVARIANT,
        "per_actuator": AllocationDependence.FAMILY_DEPENDENT,
    }
    return ConstrainedCounterfactuals(
        ztcf=ztcf,
        zvcf=zvcf,
        control_total=control_total,
        per_actuator=np.stack(columns, axis=1),
        reaction_ztcf=lam0,
        reaction_zvcf=lam00,
        dependence=dependence,
        hypotheses={
            "per_actuator": (_PER_ACTUATOR_HYPOTHESIS,),
            "reaction": (_GAUGE_HYPOTHESIS,),
        },
    )
