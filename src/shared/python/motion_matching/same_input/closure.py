"""Convention-free projection onto the dual-grip closure manifold.

Engines express the weld residual in different frames (world vs local, body-a
vs body-b point), and a violated weld is linearised differently by each.  At a
65 µm violation that alone moves constrained accelerations by up to 0.6 rad/s²
between otherwise identical models (#11606).  Two quantities do not depend on
the convention:

* the zero set of the pose residual (the closure manifold), and
* the row space of the rate map ``D`` (rate residual ``D v``), since every
  convention is ``T(q) J(q)`` for an invertible ``T``.

Off the manifold the engines' rate maps differ at first order in the
violation, so the position correction is the closest point on the manifold:
``q* - q`` lies in the row space of ``D(q*)``, evaluated on the manifold where
every engine agrees.  It is solved as a fixed point: search along
``q + D(q_k).T @ alpha`` for the zero of the engine's own pose residual, move
``q_k`` there, repeat.  The velocity correction removes the row-space component
at ``q*``, ``v - D^+ D v``.  Both give the same state in every engine up to
round-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class ClosurePlant(Protocol):
    """What the projection needs from an engine (see ``VectorPlant``)."""

    def closure_pose_residual(self, q: Array) -> Array: ...

    def closure_rate_matrix(self, q: Array) -> Array: ...


@dataclass(frozen=True)
class ClosureProjection:
    """Projected state and the residuals before and after."""

    q: Array
    v: Array
    pose_residual_before: float
    pose_residual_after: float
    rate_residual_before: float
    rate_residual_after: float
    iterations: int


def _residual_gain(
    plant: ClosurePlant, q0: Array, basis: Array, alpha: Array, step: float = 1e-7
) -> Array:
    """Central-difference Jacobian of the pose residual along ``basis``."""
    return np.column_stack(
        [
            (
                plant.closure_pose_residual(q0 + basis @ (alpha + step * e))
                - plant.closure_pose_residual(q0 + basis @ (alpha - step * e))
            )
            / (2.0 * step)
            for e in np.eye(alpha.size)
        ]
    )


def project_to_closure(
    plant: ClosurePlant,
    q: Array,
    v: Array,
    *,
    tolerance: float = 1e-13,
    max_iterations: int = 25,
) -> ClosureProjection:
    """Project ``(q, v)`` onto the closure manifold and its tangent space.

    Args:
        plant: Engine exposing the closure residuals in spec order.
        q, v: State to project (spec order).
        tolerance: Max-abs pose residual at which the Newton search stops.
        max_iterations: Newton iteration cap.

    Returns:
        The projected state.  Postcondition: the pose residual max-abs is at
        most ``tolerance`` (else ``ArithmeticError``) and ``D(q) v`` is zero to
        round-off.
    """
    q0 = np.asarray(q, dtype=float)
    v0 = np.asarray(v, dtype=float)
    if q0.ndim != 1 or q0.shape != v0.shape:
        raise ValueError("q and v must be 1-D vectors of equal length")
    if not (np.isfinite(q0).all() and np.isfinite(v0).all()):
        raise ValueError("State must be finite")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")

    residual = plant.closure_pose_residual(q0)
    before = float(np.abs(residual).max())
    q_proj = q0.copy()
    iterations = 0
    for _ in range(max_iterations):
        rows = plant.closure_rate_matrix(q_proj)
        basis = rows.T / np.linalg.norm(rows, axis=1)  # columns span row(D)
        alpha = np.linalg.lstsq(basis, q_proj - q0, rcond=None)[0]
        residual = plant.closure_pose_residual(q0 + basis @ alpha)
        while float(np.abs(residual).max()) > tolerance:
            if iterations == max_iterations:
                raise ArithmeticError(
                    "Closure projection did not converge: "
                    f"|r| = {np.abs(residual).max():.3e}"
                )
            alpha = alpha - np.linalg.solve(
                _residual_gain(plant, q0, basis, alpha), residual
            )
            residual = plant.closure_pose_residual(q0 + basis @ alpha)
            iterations += 1
        q_next = q0 + basis @ alpha
        moved = float(np.abs(q_next - q_proj).max())
        q_proj = q_next
        if moved <= 1e-15 * (1.0 + float(np.abs(q0).max())) or before <= tolerance:
            break

    rows = plant.closure_rate_matrix(q_proj)
    rate_before = float(np.abs(rows @ v0).max())
    v_proj = v0 - np.linalg.pinv(rows) @ (rows @ v0)
    return ClosureProjection(
        q=q_proj,
        v=v_proj,
        pose_residual_before=before,
        pose_residual_after=float(np.abs(residual).max()),
        rate_residual_before=rate_before,
        rate_residual_after=float(np.abs(rows @ v_proj).max()),
        iterations=iterations,
    )
