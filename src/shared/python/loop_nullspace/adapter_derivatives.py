"""Hard-KKT finite-difference derivatives for snapshot adapters (NSX-87, #12716).

Gives a ``FULL`` closed-loop snapshot adapter (MuJoCo NSX-51, MyoSuite NSX-52)
the NSX-84 method ``constrained_acceleration_derivatives(q, v, tau)`` by driving
the single NSX-84 finite-difference routine
(:class:`~src.shared.python.loop_nullspace.tangent_linearization.FiniteDifferenceDerivatives`)
with the engine's canonical model functions ``M(q)``, ``h(q, v)``, ``J_c(q)`` and
``Jdot_v(q, v)``.  The derivatives are those of the *rigid* (hard-KKT) solve
``constrained_acceleration`` with the reaction re-solved, so the capability is
``PARTIAL`` with ``derivative_method=finite_difference``.

Non-claims.  ``mjd_transitionFD`` differentiates MuJoCo's *soft* constraint
stepping (solref/solimp compliance); it is a separately characterised registry
entry and its sensitivities are **not** conformance evidence for this seam and
are never used here.  Muscle activation states are not part of the differentiated
state (:data:`UNSUPPORTED_STATE_COMPONENTS`); NSX-94 adds only quasi-static
muscle stiffness on top (``muscle_stiffness.apply_muscle_stiffness``).  Agreement between
engines is not physical validation.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.configuration_manifold import EuclideanManifold
from src.shared.python.loop_nullspace.derivative_capability import (
    ClosedLoopDerivatives,
)
from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.snapshot import (
    canonical_closure,
    canonical_closure_jacobian,
    require_closed_loop_capability,
)
from src.shared.python.loop_nullspace.tangent_linearization import (
    FiniteDifferenceDerivatives,
    StepConvergence,
    fd_step_convergence,
)

__all__ = [
    "UNSUPPORTED_STATE_COMPONENTS",
    "AdapterClosedLoopModel",
    "adapter_closed_loop_derivatives",
    "adapter_step_convergence",
]

UNSUPPORTED_STATE_COMPONENTS: tuple[str, ...] = ("muscle_activation",)
"""State components outside the derivative (listed unsupported; NSX-94 adds only quasi-static stiffness)."""


_CACHE_SLOTS = 32


def _key(*arrays: ArrayLike) -> bytes:
    return b"|".join(np.asarray(a, dtype=np.float64).tobytes() for a in arrays)


class AdapterClosedLoopModel:
    """Canonical ``M, h, J_c, Jdot_v`` of a ``FULL`` snapshot adapter.

    The model-function view the NSX-84 routine differences.  Closure rows are
    converted by the NSX-02 converter (``snapshot.canonical_closure``), never
    re-derived.  Evaluations are memoised on the exact state bytes in a small
    LRU (``_CACHE_SLOTS`` entries) so the ``nv`` load-column solves, which share
    one ``(q, v)``, hit the engine once (NSX-88).

    ``order`` (NSX-86) optionally names the coordinate order of every caller
    vector and matrix axis; inputs are permuted into the adapter's
    ``coordinate_order`` by name and outputs back, so an engine's exact
    derivatives and the engine-free reference are evaluated on the identical
    problem.  The default is the adapter's own order (NSX-87/88 behaviour).
    Preconditions: ``adapter`` passes :func:`require_closed_loop_capability`
    (``CapabilityUnavailable`` otherwise); ``order`` is a permutation of the
    adapter's coordinate names (``ValueError``).
    """

    def __init__(self, adapter: Any, *, order: Sequence[str] | None = None) -> None:
        require_closed_loop_capability(adapter)
        own = tuple(adapter.coordinate_order) if order is not None else ()
        names = () if order is None else tuple(order)
        if order is not None and sorted(names) != sorted(own):
            raise ValueError("order must be a permutation of the adapter coordinates")
        self._adapter = adapter
        self._cache: OrderedDict[tuple[str, bytes], Any] = OrderedDict()
        identity = order is None or names == own
        self._to_adapter = None if identity else np.array([names.index(n) for n in own])
        self._to_caller = None if identity else np.array([own.index(n) for n in names])
        self._order = names if order is not None else None

    @property
    def adapter(self) -> Any:
        """The wrapped engine adapter."""
        return self._adapter

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        """Order of every vector and matrix axis of this model."""
        if self._order is not None:
            return self._order
        return tuple(self._adapter.coordinate_order)

    @property
    def nv(self) -> int:
        """Number of velocity coordinates."""
        return int(self._adapter.nv)

    def _own(self, x: ArrayLike) -> Array:
        arr = np.asarray(x, dtype=np.float64)
        return arr if self._to_adapter is None else arr[self._to_adapter]

    def _vec(self, x: Array) -> Array:
        return x if self._to_caller is None else np.asarray(x)[self._to_caller]

    def _memo(self, tag: str, key: bytes, compute: Any) -> Any:
        slot = (tag, key)
        if slot in self._cache:
            self._cache.move_to_end(slot)
            return self._cache[slot]
        value = compute()
        self._cache[slot] = value
        if len(self._cache) > _CACHE_SLOTS:
            self._cache.popitem(last=False)
        return value

    def _mass_bias(self, q: Array, v: Array) -> tuple[Array, Array]:
        return self._memo(
            "mb", _key(q, v), lambda: self._adapter.compute_mass_and_bias(q, v)
        )

    def mass_matrix(self, q: Array) -> Array:
        """``M(q)``, shape ``(nv, nv)``."""
        own = self._own(q)
        mass = self._memo(
            "m", _key(own), lambda: self._mass_bias(own, np.zeros(self.nv))[0]
        )
        if self._to_caller is None:
            return mass
        return np.asarray(mass)[np.ix_(self._to_caller, self._to_caller)]

    def bias_forces(self, q: Array, v: Array) -> Array:
        """``h(q, v)``, shape ``(nv,)``."""
        return self._vec(self._mass_bias(self._own(q), self._own(v))[1])

    def closure_jacobian(self, q: Array) -> Array:
        """Canonical ``J_c(q)``, shape ``(m, nv)``."""
        own = self._own(q)
        jac = self._memo(
            "j", _key(own), lambda: canonical_closure_jacobian(self._adapter, own)
        )
        return jac if self._to_caller is None else np.asarray(jac)[:, self._to_caller]

    def closure_drift(self, q: Array, v: Array) -> Array:
        """Canonical ``Jdot_v(q, v)``, shape ``(m,)``."""
        own_q, own_v = self._own(q), self._own(v)
        return self._memo(
            "d",
            _key(own_q, own_v),
            lambda: canonical_closure(self._adapter, own_q, own_v)[1],
        )


def adapter_closed_loop_derivatives(
    adapter: Any,
    q: ArrayLike,
    v: ArrayLike,
    tau: ArrayLike,
    *,
    step: float = 1e-5,
) -> ClosedLoopDerivatives:
    """Tangent-bundle derivatives of ``(a, lambda)`` for a ``FULL`` snapshot adapter.

    Central differences (step ``step``) of the hard-KKT solve along retractions
    of the closure tangent basis, by the NSX-84 routine.  ``tau`` is the
    generalised load (length ``nv``).  Preconditions (``ValueError``): finite
    ``q``, ``v``, ``tau`` of length ``nv``, ``J_c v = 0``, positive finite
    ``step``; the adapter must be ``FULL`` for the snapshot capability
    (``CapabilityUnavailable``).  Raises ``ClosureRankChangeError`` when the
    closure rank changes inside the step.  Postcondition: ``method`` is
    ``FINITE_DIFFERENCE`` and ``J_c @ basis = 0``.
    """
    model = AdapterClosedLoopModel(adapter)
    routine = FiniteDifferenceDerivatives(model, EuclideanManifold(model.nv), step=step)
    return routine.constrained_acceleration_derivatives(q, v, tau)


def adapter_step_convergence(
    adapter: Any,
    q: ArrayLike,
    v: ArrayLike,
    tau: ArrayLike,
    *,
    steps: tuple[float, ...] = (1e-3, 5e-4, 2.5e-4),
) -> StepConvergence:
    """NSX-84 step-convergence report of the adapter's finite differences (NSX-88)."""
    model = AdapterClosedLoopModel(adapter)
    return fd_step_convergence(
        model, EuclideanManifold(model.nv), q, v, tau, steps=steps
    )
