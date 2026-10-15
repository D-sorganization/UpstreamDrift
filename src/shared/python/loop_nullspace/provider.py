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

__all__ = ["AdapterConstrainedProvider"]


class AdapterConstrainedProvider:
    """Canonical closed-loop provider backed by a ``FULL`` engine adapter.

    Preconditions: ``adapter`` declares ``CLOSED_LOOP_SNAPSHOT`` at ``FULL``
    (else ``CapabilityUnavailable``).  Every method validates its inputs as
    finite vectors of length ``adapter.nv`` (``ValueError``).
    """

    def __init__(self, adapter: Any) -> None:
        require_closed_loop_capability(adapter)
        self._adapter = adapter

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
