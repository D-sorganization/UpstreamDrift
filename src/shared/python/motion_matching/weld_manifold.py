"""Put a velocity on the dual-grip weld's constraint manifold (#11043).

The forward-dynamics KKT solve enforces the weld at the acceleration level
only (``J a = -J̇ q̇``), so it conserves whatever relative velocity the two
closure frames start with and the grip opens linearly in time. A replay
seeded with a finite-difference reference velocity must therefore start with
``J v = 0``. The projection used is the inelastic impulse the weld itself
would apply: the admissible velocity nearest ``v`` in kinetic energy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def project_onto_weld(mass: Array, jac: Array, rates: Array) -> Array:
    """Return ``v - M⁻¹Jᵀ (J M⁻¹ Jᵀ)⁺ J v``.

    Preconditions: ``mass`` is a finite, symmetric positive-definite
    ``(n, n)`` matrix, ``jac`` is finite ``(m, n)`` and ``rates`` is finite
    ``(n,)``, all in the same coordinate order.

    Postconditions: ``jac @ result`` is zero to solver precision; the removed
    velocity is ``M``-orthogonal to every ``w`` with ``jac @ w = 0``; a
    velocity that already satisfies the weld is returned unchanged.
    """
    m_mat = np.asarray(mass, dtype=float)
    j_mat = np.asarray(jac, dtype=float)
    v = np.asarray(rates, dtype=float)
    n = v.shape[0] if v.ndim == 1 else -1
    if v.ndim != 1 or j_mat.ndim != 2 or j_mat.shape[1] != n:
        raise ValueError("rates must be a vector matching the Jacobian columns")
    if m_mat.shape != (n, n):
        raise ValueError("mass must be square with one row per rate")
    if not (np.isfinite(m_mat).all() and np.isfinite(j_mat).all()):
        raise ValueError("mass and Jacobian must be finite")
    if not np.isfinite(v).all():
        raise ValueError("rates must be finite")
    inv_mass_jt = np.linalg.solve(m_mat, j_mat.T)
    multipliers = np.linalg.lstsq(j_mat @ inv_mass_jt, j_mat @ v, rcond=None)[0]
    return v - inv_mass_jt @ multipliers
