"""Weld-consistent tracked reference (GCV-20, #11767).

The tracking low-pass (DESIGN_DECISIONS section 11) filters every coordinate
independently, so the filtered reference no longer satisfies the dual-grip
weld: on the driver capture it opens by up to 16.1 mm at 12 Hz and 10.7 mm
at 25 Hz (the unfiltered IK reference: 4.5 mm). The KKT replay enforces the
weld exactly, so it fights the inconsistent target with trail-arm effort
through the release and the clubhead peaks early. Each tracked sample is
therefore projected back onto the weld by a Gauss-Newton solve over the
trail-arm coordinates only (minimum-norm steps), leaving the lead arm, which
carries the club, and the body at their filtered values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching.pipeline.constants import (
    TRAIL_ARM_WELD_COORDINATES,
)

ResidualFn = Callable[[np.ndarray], np.ndarray]

WELD_TOL: float = 1e-6  # m and rad
WELD_MAX_ITER: int = 20
FD_STEP: float = 1e-6  # rad


def _check(q: np.ndarray, columns: Sequence[int]) -> list[int]:
    cols = [int(c) for c in columns]
    if q.ndim != 1 or not np.isfinite(q).all():
        raise ValueError("q must be a finite 1-D coordinate vector")
    if not cols or len(set(cols)) != len(cols):
        raise ValueError(f"columns must be non-empty and distinct, got {cols}")
    if min(cols) < 0 or max(cols) >= q.size:
        raise ValueError(f"columns must index q (size {q.size}), got {cols}")
    return cols


def project_onto_closure(
    q: np.ndarray,
    residual: ResidualFn,
    columns: Sequence[int],
    *,
    tol: float = WELD_TOL,
    max_iter: int = WELD_MAX_ITER,
) -> np.ndarray:
    """Return a copy of ``q`` whose ``residual`` is closed over ``columns``.

    Preconditions: ``q`` is a finite 1-D vector; ``columns`` are distinct
    indices into it; ``residual`` maps a coordinate vector to a finite 1-D
    residual.

    Postconditions: ``|residual(result)| < tol``; entries outside
    ``columns`` equal ``q``'s; a ``q`` already inside ``tol`` is returned
    unchanged (as a copy).

    Raises:
        ValueError: bad arguments, or the residual did not close within
            ``max_iter`` Gauss-Newton steps.
    """
    rows = np.array(q, dtype=float)
    cols = _check(rows, columns)
    if not (math.isfinite(tol) and tol > 0.0 and max_iter >= 1):
        raise ValueError("tol must be positive and max_iter at least 1")
    for _ in range(max_iter + 1):
        r = np.asarray(residual(rows), dtype=float)
        if np.linalg.norm(r) < tol:
            return rows
        jac = np.empty((r.size, len(cols)))
        for k, c in enumerate(cols):
            probe = rows.copy()
            probe[c] += FD_STEP
            jac[:, k] = (np.asarray(residual(probe), dtype=float) - r) / FD_STEP
        rows[cols] -= np.linalg.lstsq(jac, r, rcond=None)[0]
    raise ValueError(
        f"weld projection did not close: residual {np.linalg.norm(r):.3g} "
        f"after {max_iter} steps"
    )


def _homogeneous(pose: tuple[Any, Any]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = np.asarray(pose[0], dtype=float)
    out[:3, 3] = np.asarray(pose[1], dtype=float)
    return out


def _weld_residual(kin: Any, closure: Mapping[str, Any]) -> ResidualFn:
    body_a, body_b = closure["body_a"], closure["body_b"]
    place_a = np.asarray(closure["placement_a"], dtype=float)
    place_b = np.asarray(closure["placement_b"], dtype=float)

    def residual(q: np.ndarray) -> np.ndarray:
        poses = kin.body_poses(q, [body_a, body_b])
        rel = np.linalg.inv(_homogeneous(poses[body_a]) @ place_a) @ (
            _homogeneous(poses[body_b]) @ place_b
        )
        rotvec = Rotation.from_matrix(rel[:3, :3]).as_rotvec()
        return np.r_[rel[:3, 3], rotvec]

    return residual


def _max_position_mm(residual: ResidualFn, q: np.ndarray) -> float:
    return float(max(1e3 * np.linalg.norm(residual(row)[:3]) for row in q))


def weld_consistent_track(
    kin: Any,
    spec: Mapping[str, Any],
    q_track: np.ndarray,
    coordinates: Sequence[str] = TRAIL_ARM_WELD_COORDINATES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project every tracked sample onto the spec's dual-grip weld.

    Args:
        kin: Kinematics with ``coordinate_order`` and
            ``body_poses(q, bodies) -> {name: (R, t)}``.
        spec: Full-body spec; its ``closure`` names the welded frames.
        q_track: ``(n, nq)`` tracked reference in ``kin``'s coordinate order.
        coordinates: Coordinates the projection may move.

    Returns:
        The projected track and a receipt block. When the spec has no weld
        or the model lacks a listed coordinate, ``q_track`` itself is
        returned and the block says why (``applied`` is ``False``).

    Raises:
        ValueError: ``q_track`` is not a finite ``(n, nq)`` array, or a
            sample cannot be projected.
    """
    closure = spec.get("closure")
    if not closure:
        return q_track, {"applied": False, "source": "no closure in spec"}
    order = list(kin.coordinate_order)
    missing = [name for name in coordinates if name not in order]
    if missing:
        return q_track, {
            "applied": False,
            "source": f"missing trail-arm coordinates {missing}",
        }
    rows = np.asarray(q_track, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != len(order) or not np.isfinite(rows).all():
        raise ValueError("q_track must be a finite (n, nq) array in kin's order")
    residual = _weld_residual(kin, closure)
    cols = [order.index(name) for name in coordinates]
    out = np.array([project_onto_closure(row, residual, cols) for row in rows])
    return out, {
        "applied": True,
        "coordinates": list(coordinates),
        "max_position_mm_before": _max_position_mm(residual, rows),
        "max_position_mm_after": _max_position_mm(residual, out),
        "max_joint_change_rad": float(np.abs(out - rows).max()),
    }
