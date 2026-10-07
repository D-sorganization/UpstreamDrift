"""Orientation inverse kinematics on a MuJoCo model with coupled joints (#11644).

``solve_orientation`` finds the values of a few 1-DOF joints that bring one body
to a target world orientation, with the joint-equality couplings of
:mod:`couplings` substituted exactly and optional joint limits.  It is the unit
that the hierarchical spec-to-MyoFullBody mapping applies segment by segment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import require
from src.shared.python.myofullbody.couplings import JointCoupling

Array = np.ndarray
ACTIVE_TOL = 1e-9


@dataclass(frozen=True)
class OrientationFit:
    """Result of one group solve.

    Attributes:
        values: solved joint values (rad or m), in the order of ``qpos_adr``.
        errors_rad: geodesic angle between achieved and target, one per target body.
        at_bound: per joint, ``True`` if the solution sits on a limit.
    """

    values: Array
    errors_rad: tuple[float, ...]
    at_bound: Array

    @property
    def error_rad(self) -> float:
        """Largest per-body orientation error."""
        return max(self.errors_rad)


def rotation_error(target: Array, actual: Array) -> Array:
    """Rotation vector of ``target^T actual`` (3,), zero when they coincide."""
    return Rotation.from_matrix(target.T @ actual).as_rotvec()


def body_rotation(
    model: Any, data: Any, coupling: JointCoupling, q: Array, body: int
) -> Array:
    """World rotation matrix of ``body`` at ``q`` (dependent joints substituted)."""
    import mujoco

    data.qpos[:] = coupling.expand(q)
    mujoco.mj_kinematics(model, data)
    return np.asarray(data.xmat[body]).reshape(3, 3).copy()


class Starts(NamedTuple):
    """Extra initial guesses and the branch preference of :func:`solve_orientations`."""

    seeds: tuple[Array, ...] = ()
    prefer: Array | None = None


def solve_orientations(
    model: Any,
    data: Any,
    coupling: JointCoupling,
    q: Array,
    targets: Sequence[tuple[int, Array]],
    qpos_adr: tuple[int, ...],
    bounds: tuple[Array, Array] | None,
    starts: Starts = Starts(),
) -> OrientationFit:
    """Fit ``qpos_adr`` of ``q`` so that every ``(body, R)`` target is matched.

    Args:
        targets: ``(body index, world rotation)`` pairs, solved in one
            least-squares problem (equal weights).
        q: full coordinate vector; the entries at ``qpos_adr`` are the initial guess
            and every other entry is held fixed.
        bounds: ``(lower, upper)`` per joint, or ``None`` for unbounded.
        starts: ``seeds`` are extra initial guesses tried when the first one is not
            accurate.  When ``prefer`` is given, every start is run and, among the fits that reach the
            best cost (Euler-angle branches of one orientation), the one closest
            to ``prefer`` is returned.

    Returns:
        The best fit.  Postcondition: ``q`` is not modified.

    Raises:
        ValueError: if a target is not a 3x3 matrix or there is nothing to solve.
    """
    import mujoco

    require(len(targets) > 0 and len(qpos_adr) > 0, "need targets and joints")
    for _, rot in targets:
        require(rot.shape == (3, 3), "target must be a 3x3 rotation")
    idx = np.array(qpos_adr, dtype=int)
    lo, hi = (
        (np.full(len(idx), -np.inf), np.full(len(idx), np.inf))
        if bounds is None
        else (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
    )

    def residual(x: Array) -> Array:
        trial = q.copy()
        trial[idx] = x
        data.qpos[:] = coupling.expand(trial)
        mujoco.mj_kinematics(model, data)
        return np.concatenate(
            [
                rotation_error(rot, np.asarray(data.xmat[body]).reshape(3, 3))
                for body, rot in targets
            ]
        )

    best: tuple[float, Array, Array] | None = None
    seeds, prefer = starts.seeds, starts.prefer
    for start in (q[idx], *seeds):
        x0 = np.clip(np.asarray(start, float), lo, hi)
        result = least_squares(
            residual, x0, bounds=(lo, hi), xtol=1e-13, ftol=1e-13, gtol=1e-13
        )
        cost = float(np.linalg.norm(result.fun))
        if best is None:
            better = True
        elif prefer is not None and cost < 1e-8 and best[0] < 1e-8:
            better = bool(
                np.linalg.norm(result.x - prefer) < np.linalg.norm(best[1] - prefer)
            )
        else:
            better = cost < best[0]
        if better:
            best = (cost, result.x, result.fun)
        if cost < 1e-8 and prefer is None:
            break
    assert best is not None
    values = best[1]
    errors = tuple(
        float(np.linalg.norm(best[2][3 * i : 3 * i + 3])) for i in range(len(targets))
    )
    at_bound = (np.abs(values - lo) < ACTIVE_TOL) | (np.abs(values - hi) < ACTIVE_TOL)
    return OrientationFit(values, errors, at_bound)


def solve_orientation(
    model: Any,
    data: Any,
    coupling: JointCoupling,
    q: Array,
    body: int,
    qpos_adr: tuple[int, ...],
    target: Array,
    bounds: tuple[Array, Array] | None,
) -> OrientationFit:
    """Single-body form of :func:`solve_orientations` (no extra starts)."""
    return solve_orientations(
        model, data, coupling, q, [(body, target)], qpos_adr, bounds
    )
