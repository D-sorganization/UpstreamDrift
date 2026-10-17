"""Constrained equations-of-motion provider over the NSX-13 engine seam (NSX-30).

``AdapterConstrainedProvider`` presents any adapter that declares
``Capability.CLOSED_LOOP_SNAPSHOT`` at ``CapabilityLevel.FULL`` as a
``ConstrainedEquationsOfMotionProvider``: ``M`` and ``h`` come from
``compute_mass_and_bias`` and the closure rows are converted to the canonical
grip-frame convention with the same NSX-02 path ``snapshot_from_adapter`` uses.
No engine is imported here; engine coverage is whatever the adapter provides.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.snapshot import (
    canonical_closure,
    require_closed_loop_capability,
)

__all__ = ["AdapterConstrainedProvider", "ReorderedProvider"]


class AdapterConstrainedProvider:
    """Canonical closed-loop provider backed by a ``FULL`` engine adapter.

    Preconditions: ``adapter`` declares ``CLOSED_LOOP_SNAPSHOT`` at ``FULL``
    (else ``CapabilityUnavailable``).  Every method validates its inputs as
    finite vectors of length ``adapter.nv`` (``ValueError``).
    """

    def __init__(self, adapter: Any) -> None:
        require_closed_loop_capability(adapter)
        self._adapter = adapter
        # Optional position-level closure error (NSX-31): present only when the
        # adapter offers one, so ``getattr(provider, "closure_error", None)``
        # tells a replay whether position drift can be reported.
        if callable(getattr(adapter, "closure_position_error", None)):
            self.closure_error = self._closure_error

    @property
    def nv(self) -> int:
        """Number of velocity coordinates."""
        return int(self._adapter.nv)

    def _vector(self, value: ArrayLike, name: str) -> Array:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (self.nv,):
            raise ValueError(f"{name} must have shape ({self.nv},), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
        return arr

    def mass_matrix(self, q: ArrayLike) -> Array:
        """``M(q)`` from the adapter, shape ``(nv, nv)``."""
        q_arr = self._vector(q, "q")
        mass, _ = self._adapter.compute_mass_and_bias(q_arr, np.zeros(self.nv))
        return np.asarray(mass, dtype=np.float64)

    def bias_forces(self, q: ArrayLike, v: ArrayLike) -> Array:
        """``h(q, v)`` from the adapter, shape ``(nv,)``."""
        _, bias = self._adapter.compute_mass_and_bias(
            self._vector(q, "q"), self._vector(v, "v")
        )
        return np.asarray(bias, dtype=np.float64)

    def closure_jacobian(self, q: ArrayLike) -> Array:
        """Canonical ``J_c(q)``, shape ``(m, nv)``."""
        q_arr = self._vector(q, "q")
        return canonical_closure(self._adapter, q_arr, np.zeros(self.nv))[0]

    def closure_drift(self, q: ArrayLike, v: ArrayLike) -> Array:
        """Canonical ``Jdot_v(q, v)``, shape ``(m,)``."""
        return canonical_closure(
            self._adapter, self._vector(q, "q"), self._vector(v, "v")
        )[1]

    def _closure_error(self, q: ArrayLike) -> Array:
        """Native position-level closure error ``Phi(q)`` (adapter-defined rows)."""
        return np.asarray(
            self._adapter.closure_position_error(self._vector(q, "q")),
            dtype=np.float64,
        )


class ReorderedProvider:
    """A constrained provider re-expressed in another coordinate order.

    ``native_order`` is the wrapped provider's coordinate order and
    ``order`` the caller's; both must hold the same names.  ``M``, ``h`` and
    ``J_c`` are permuted consistently, so every counterfactual is unchanged.
    """

    def __init__(
        self, provider: Any, native_order: tuple[str, ...], order: tuple[str, ...]
    ) -> None:
        if sorted(native_order) != sorted(order) or len(set(order)) != len(order):
            raise ValueError("native_order and order must name the same coordinates")
        self._provider = provider
        self._take = np.array([order.index(n) for n in native_order])
        self._back = np.array([native_order.index(n) for n in order])

    def _native(self, x: ArrayLike) -> Array:
        return np.asarray(x, dtype=np.float64)[self._take]

    def mass_matrix(self, q: ArrayLike) -> Array:
        """``M`` in the caller's order."""
        mass = self._provider.mass_matrix(self._native(q))
        return np.asarray(mass)[np.ix_(self._back, self._back)]

    def bias_forces(self, q: ArrayLike, v: ArrayLike) -> Array:
        """``h`` in the caller's order."""
        bias = self._provider.bias_forces(self._native(q), self._native(v))
        return np.asarray(bias)[self._back]

    def closure_jacobian(self, q: ArrayLike) -> Array:
        """``J_c`` with columns in the caller's order."""
        return np.asarray(self._provider.closure_jacobian(self._native(q)))[
            :, self._back
        ]

    def closure_drift(self, q: ArrayLike, v: ArrayLike) -> Array:
        """``Jdot_v`` (row space is unaffected by coordinate order)."""
        return np.asarray(
            self._provider.closure_drift(self._native(q), self._native(v))
        )
