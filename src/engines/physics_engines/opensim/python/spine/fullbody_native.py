"""Native (OpenSim) support of the full-body coordinate map (SPN-9c, #12758).

Three jobs, each on an already-loaded ``opensim.Model``:

* observation: coordinate and body names, locked flags (names and flags only);
* sign probing: the cosine between a golf coordinate axis and the mapped target
  coordinate axis at the all-zero pose, by central finite differences of the
  child-body rotation (or origin displacement for a translation), so a sign in
  the map is observed rather than assumed;
* segment-orientation matching: where the golf and target joints parametrise
  one segment orientation with different Euler sequences (the shoulder is ZXY
  in the golf humanoid and an XYZ-like sequence in the target), the target
  coordinates are solved per frame so that the target segment reaches the golf
  segment's WORLD orientation, neutral-calibrated by the all-zero poses
  (:func:`neutral_alignment`).

Frames: ground X anterior, Y up, Z right; rotations are proper matrices
``R_world_body``; coordinates in radians and metres.

SWIG chains are wrapped in single-hop helpers so call chains stay at two
levels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
    transform_to_matrix,
)
from src.engines.physics_engines.opensim.python.spine.fullbody_map import (
    DirectEntry,
    OrientationGroup,
)
from src.shared.python.contracts import ensure, invariant
from src.shared.python.engine_core.opensim_runtime import import_opensim

__all__ = [
    "OrientationSolution",
    "body_names",
    "body_poses",
    "body_rotations",
    "body_mass_properties",
    "coordinate_locked_flags",
    "coordinate_names",
    "neutral_alignment",
    "probe_axis_dots",
    "select_continuous",
    "solve_orientation_group",
]

_PROBE_STEP = 1e-4
_ROT_TOL = 1e-8
_RESTARTS = 6
_FAR = 4.0

#: An unreachable frame keeps the previous frame's branch unless another
#: candidate lowers the orientation residual by more than this [rad] (#13056).
DEFAULT_CONTINUITY_SLACK_RAD = 0.1


# ------------------------------------------------------------------ adapters


def _coord_set(model: Any) -> Any:
    return model.getCoordinateSet()


def _body_set(model: Any) -> Any:
    return model.getBodySet()


def _constraint_set(model: Any) -> Any:
    return model.updConstraintSet()


def coordinate_names(model: Any) -> list[str]:
    """Names of every coordinate of ``model`` in model order."""
    coords = _coord_set(model)
    return [str(coords.get(i).getName()) for i in range(coords.getSize())]


def body_names(model: Any) -> list[str]:
    """Names of every body of ``model`` (ground excluded)."""
    bodies = _body_set(model)
    return [str(bodies.get(i).getName()) for i in range(bodies.getSize())]


def coordinate_locked_flags(model: Any) -> dict[str, bool]:
    """``{coordinate: locked in the file}`` (the ``locked`` property)."""
    coords = _coord_set(model)
    return {
        str(coords.get(i).getName()): bool(coords.get(i).get_locked())
        for i in range(coords.getSize())
    }


def _coordinate(model: Any, name: str) -> Any:
    coords = _coord_set(model)
    if coords.getIndex(name) < 0:
        raise ValueError(f"model has no coordinate {name!r}")
    return coords.get(name)


def _body(model: Any, name: str) -> Any:
    bodies = _body_set(model)
    if bodies.getIndex(name) < 0:
        raise ValueError(f"model has no body {name!r}")
    return bodies.get(name)


def _joint_of(coordinate: Any) -> Any:
    return coordinate.getJoint()


def _child_frame(joint: Any) -> Any:
    return joint.getChildFrame()


def _base_name(frame: Any) -> str:
    return str(frame.findBaseFrame().getName())


def _child_body_name(model: Any, name: str) -> str:
    """Name of the body driven by coordinate ``name``."""
    return _base_name(_child_frame(_joint_of(_coordinate(model, name))))


def _is_rotational(model: Any, name: str) -> bool:
    osim = import_opensim()
    return bool(_coordinate(model, name).getMotionType() == osim.Coordinate.Rotational)


def _zero_unlocked(model: Any, state: Any) -> None:
    coords = _coord_set(model)
    for i in range(coords.getSize()):
        coord = coords.get(i)
        if not coord.get_locked() and not coord.isDependent(state):
            coord.setValue(state, 0.0, False)


def _prepared(model: Any, *, zero_missing: bool, constraints_off: bool = False) -> Any:
    """Model state at defaults (or all-zero); constraints optionally disabled."""
    if constraints_off:
        constraints = _constraint_set(model)
        for i in range(constraints.getSize()):
            constraints.get(i).set_isEnforced(False)
    state = model.initSystem()
    if zero_missing:
        _zero_unlocked(model, state)
    return state


def _frame_count(table: Mapping[str, np.ndarray]) -> int:
    lengths = {int(np.asarray(v).shape[0]) for v in table.values()}
    if len(lengths) > 1:
        raise ValueError(f"coordinate series differ in length: {sorted(lengths)}")
    return lengths.pop() if lengths else 1


def _validated_table(
    model: Any, table: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, series in table.items():
        _coordinate(model, name)
        out[name] = np.asarray(series, dtype=float)
    return out


# ---------------------------------------------------------------- kinematics


def body_poses(
    model: Any,
    table: Mapping[str, np.ndarray],
    bodies: Sequence[str],
    *,
    zero_missing: bool = False,
) -> dict[str, np.ndarray]:
    """World poses ``(n, 4, 4)`` of ``bodies`` for a coordinate table.

    Coordinates absent from ``table`` keep the model defaults, or are zeroed
    when ``zero_missing``.  The state is realised to Position per frame;
    constraints are neither enforced nor changed.

    Raises:
        ValueError: unknown body or coordinate, series of unequal length.
    """
    for name in bodies:
        _body(model, name)
    values = _validated_table(model, table)
    n_frames = _frame_count(values)
    state = _prepared(model, zero_missing=zero_missing)
    out = {name: np.empty((n_frames, 4, 4)) for name in bodies}
    for k in range(n_frames):
        for name, series in values.items():
            _coordinate(model, name).setValue(state, float(series[k]), False)
        model.realizePosition(state)
        for name in bodies:
            out[name][k] = transform_to_matrix(
                _body(model, name).getTransformInGround(state)
            )
    return out


def body_rotations(
    model: Any,
    table: Mapping[str, np.ndarray],
    body: str,
    *,
    zero_missing: bool = False,
) -> np.ndarray:
    """World rotations ``(n, 3, 3)`` of ``body`` (see :func:`body_poses`)."""
    poses = body_poses(model, table, [body], zero_missing=zero_missing)[body]
    return np.array(poses[:, :3, :3])


def body_mass_properties(model: Any, name: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass [kg], centre of mass [m] and COM inertia ``(3, 3)`` [kg m^2] of a body.

    Read at runtime from the (non-licensed) golf model; all in the body frame.
    """
    body = _body(model, name)
    mass = float(body.getMass())
    com = np.array(body.getMassCenter().to_numpy(), dtype=float)
    inertia = body.getInertia()
    moments = np.array(inertia.getMoments().to_numpy(), dtype=float)
    products = np.array(inertia.getProducts().to_numpy(), dtype=float)
    tensor = np.diag(moments)
    for (i, j), value in zip(((0, 1), (0, 2), (1, 2)), products, strict=True):
        tensor[i, j] = tensor[j, i] = value
    return mass, com, tensor


def neutral_alignment(
    golf_model: Any, golf_body: str, target_model: Any, target_body: str
) -> np.ndarray:
    """``K = R_golf(0)^T R_target(0)``: the fixed frame difference of two segments.

    Both models are evaluated at the all-zero pose.  Mapping a golf world
    orientation ``R_g`` to the target uses ``R_target = R_g @ K``, so the two
    segments coincide at the neutral pose and differ afterwards only by the
    motion.  This assumes the zero poses are anatomically equivalent; any
    posture difference at zero becomes a constant orientation offset (stated in
    the documentation, quantified by the asset-gated check).
    """
    golf = body_rotations(golf_model, {}, golf_body, zero_missing=True)[0]
    target = body_rotations(target_model, {}, target_body, zero_missing=True)[0]
    return golf.T @ target


# --------------------------------------------------------------- sign probing


def _origin(state: Any, body: Any) -> np.ndarray:
    return transform_to_matrix(body.getTransformInGround(state))[:3, 3]


def _motion_direction(model: Any, state: Any, name: str) -> np.ndarray:
    """Unit rotation axis (or translation direction) of ``+d`` of coordinate ``name``."""
    coord = _coordinate(model, name)
    body = _body(model, _child_body_name(model, name))
    rotational = _is_rotational(model, name)
    value = coord.getValue(state)
    model.realizePosition(state)
    base = transform_to_matrix(body.getTransformInGround(state))
    coord.setValue(state, value + _PROBE_STEP, False)
    model.realizePosition(state)
    moved = transform_to_matrix(body.getTransformInGround(state))
    coord.setValue(state, value, False)
    model.realizePosition(state)
    if rotational:
        vec = Rotation.from_matrix(moved[:3, :3] @ base[:3, :3].T).as_rotvec()
    else:
        vec = moved[:3, 3] - base[:3, 3]
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError(f"coordinate {name!r} does not move its child body")
    return vec / norm


def _probe_state(model: Any) -> Any:
    return _prepared(model, zero_missing=True, constraints_off=True)


def probe_axis_dots(
    golf_model: Any, target_model: Any, entries: Sequence[DirectEntry]
) -> dict[str, float]:
    """Cosine of the golf and target motion directions for each entry.

    Both models are copied with constraints disabled and evaluated at the
    all-zero pose; a coordinate's direction is the rotation axis (rotational)
    or displacement direction (translational) of its child body for a small
    positive step.  A mapped ``sign`` is acceptable when ``sign * dot`` is at
    least the entry's ``min_axis_dot``.  The golf coordinate and its target
    must have the same motion type (``ValueError`` otherwise).  The inputs are
    not modified.
    """
    osim = import_opensim()
    golf = osim.Model(golf_model)
    target = osim.Model(target_model)
    golf_state, target_state = _probe_state(golf), _probe_state(target)
    out: dict[str, float] = {}
    for entry in entries:
        if _is_rotational(golf, entry.golf) != _is_rotational(target, entry.target):
            raise ValueError(
                f"{entry.golf!r} and {entry.target!r} have different motion types"
            )
        a = _motion_direction(golf, golf_state, entry.golf)
        b = _motion_direction(target, target_state, entry.target)
        out[entry.golf] = float(a @ b)
    ensure(len(out) == len(entries), "one cosine per entry", len(out))
    return out


# -------------------------------------------------------- orientation matching


@dataclass(frozen=True)
class OrientationSolution:
    """Per-frame target coordinates of one orientation group.

    ``coordinates`` maps each target coordinate name to an ``(n,)`` read-only
    array (radians); ``residual_rad`` is the geodesic angle between the
    achieved and the requested world orientation per frame (never repaired).
    """

    coordinates: Mapping[str, np.ndarray]
    residual_rad: np.ndarray

    @property
    def max_step_rad(self) -> float:
        """Largest frame-to-frame change of any group coordinate [rad]."""
        q = np.column_stack(list(self.coordinates.values()))
        return float(np.abs(np.diff(q, axis=0)).max()) if q.shape[0] > 1 else 0.0

    def __post_init__(self) -> None:
        coords = {k: np.array(v, dtype=float) for k, v in self.coordinates.items()}
        for arr in coords.values():
            arr.setflags(write=False)
        resid = np.array(self.residual_rad, dtype=float)
        resid.setflags(write=False)
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "residual_rad", resid)


def _check_targets(target_world: np.ndarray) -> np.ndarray:
    rots = np.asarray(target_world, dtype=float)
    if rots.ndim != 3 or rots.shape[1:] != (3, 3) or rots.shape[0] < 1:
        raise ValueError(f"target_world must have shape (n, 3, 3), got {rots.shape}")
    gram = np.einsum("nij,nkj->nik", rots, rots)
    if not np.all(np.isfinite(rots)) or not np.allclose(
        gram, np.eye(3), atol=_ROT_TOL, rtol=0
    ):
        raise ValueError("target_world must contain finite proper rotations")
    return rots


def _limits(
    model: Any, names: Sequence[str], bounds: Mapping[str, tuple[float, float]] | None
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(names), -_FAR)
    upper = np.full(len(names), _FAR)
    for k, name in enumerate(names):
        if bounds is not None and name in bounds:
            lower[k], upper[k] = bounds[name]
            continue
        coord = _coordinate(model, name)
        lo, hi = float(coord.getRangeMin()), float(coord.getRangeMax())
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            lower[k], upper[k] = max(lo, -_FAR), min(hi, _FAR)
    if np.any(lower >= upper):
        raise ValueError("lower bounds must be below upper bounds")
    return lower, upper


def select_continuous(
    candidates: Sequence[tuple[np.ndarray, float]],
    previous: np.ndarray | None,
    *,
    tolerance_rad: float,
    slack_rad: float,
) -> tuple[np.ndarray, float]:
    """Choose one ``(coordinates, residual)`` candidate, preferring continuity.

    When the best residual is within ``tolerance_rad`` the admissible
    candidates are the converged ones; otherwise (an unreachable orientation)
    they are those within ``slack_rad`` of the best residual.  The admissible
    candidate closest to ``previous`` (max-norm) is returned; without a
    previous frame the lowest residual wins.  A branch switch therefore
    happens only when it lowers the residual by more than ``slack_rad``.

    Raises:
        ValueError: no candidate, or a negative or non-finite slack.
    """
    if not candidates:
        raise ValueError("select_continuous needs at least one candidate")
    if not np.isfinite(slack_rad) or slack_rad < 0.0:
        raise ValueError(f"slack_rad must be finite and >= 0: {slack_rad}")
    best = min(angle for _, angle in candidates)
    if previous is None:
        return next((x, a) for x, a in candidates if a == best)
    limit = tolerance_rad if best <= tolerance_rad else best + slack_rad
    admissible = [(x, a) for x, a in candidates if a <= limit]
    return min(admissible, key=lambda c: float(np.abs(c[0] - previous).max()))


def _solve_frame(
    objective: Any,
    starts: Sequence[np.ndarray],
    limits: tuple[np.ndarray, np.ndarray],
    tolerance_rad: float,
) -> list[tuple[np.ndarray, float]]:
    """Local solutions from the warm starts, then deterministic restarts.

    Restarts stop once a warm start has converged.
    """
    lower, upper = limits
    rng = np.random.default_rng(0)
    pool = list(starts) + [rng.uniform(lower, upper) for _ in range(_RESTARTS)]
    found: list[tuple[np.ndarray, float]] = []
    for index, start in enumerate(pool):
        result = least_squares(
            objective, np.clip(start, lower, upper), bounds=(lower, upper)
        )
        found.append((np.array(result.x), float(np.linalg.norm(result.fun))))
        best = min(angle for _, angle in found)
        if best <= tolerance_rad and index + 1 >= len(starts):
            break
    invariant(
        all(np.isfinite(angle) for _, angle in found), "solver residual must be finite"
    )
    return found


def solve_orientation_group(
    model: Any,
    group: OrientationGroup,
    fixed: Mapping[str, np.ndarray],
    target_world: np.ndarray,
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    tolerance_rad: float = 1e-6,
    continuity_slack_rad: float = DEFAULT_CONTINUITY_SLACK_RAD,
) -> OrientationSolution:
    """Solve the group's target coordinates for requested world orientations.

    Per frame the ``fixed`` coordinates (trunk, pelvis, ...) are set, every
    other coordinate is zero, and the three group coordinates minimise the
    geodesic angle between the target body's world rotation and
    ``target_world[k]`` within the coordinate ranges (or ``bounds``).  Several
    deterministic starts are tried when the warm start does not reach
    ``tolerance_rad``.  An unreachable orientation gives a positive
    ``residual_rad`` and the closest in-range solution; nothing is rescaled.
    Among the local solutions the one nearest the previous frame is kept
    (:func:`select_continuous`, ``continuity_slack_rad``), so an Euler branch
    is not left for an equal or marginally better residual; the largest step
    is :attr:`OrientationSolution.max_step_rad`.

    Preconditions: ``target_world`` is ``(n, 3, 3)`` finite proper rotations;
    the group's coordinates and body exist; ``tolerance_rad`` is positive and
    finite; ``fixed`` series have length ``n`` (``ValueError``).
    Postcondition: one value per frame and coordinate, within the limits.
    """
    if not np.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
        raise ValueError(f"tolerance_rad must be positive and finite: {tolerance_rad}")
    if not np.isfinite(continuity_slack_rad) or continuity_slack_rad < 0.0:
        raise ValueError(
            f"continuity_slack_rad must be finite and >= 0: {continuity_slack_rad}"
        )
    rots = _check_targets(target_world)
    n_frames = rots.shape[0]
    names = list(group.target_coordinates)
    coords = [_coordinate(model, name) for name in names]
    body = _body(model, group.target_body)
    values = _validated_table(model, fixed)
    if values and _frame_count(values) != n_frames:
        raise ValueError(f"fixed coordinates must have {n_frames} frames")
    state = _prepared(model, zero_missing=True)
    limits = _limits(model, names, bounds)
    solved = np.zeros((n_frames, 3))
    residual = np.zeros(n_frames)

    for k in range(n_frames):
        for name, series in values.items():
            _coordinate(model, name).setValue(state, float(series[k]), False)

        def objective(x: np.ndarray, k: int = k) -> np.ndarray:
            for coord, value in zip(coords, x, strict=True):
                coord.setValue(state, float(value), False)
            model.realizePosition(state)
            rot = transform_to_matrix(body.getTransformInGround(state))[:3, :3]
            return Rotation.from_matrix(rot.T @ rots[k]).as_rotvec()

        starts = [solved[k - 1]] if k else []
        starts.append(np.zeros(3))
        found = _solve_frame(objective, starts, limits, tolerance_rad)
        solved[k], residual[k] = select_continuous(
            found,
            solved[k - 1] if k else None,
            tolerance_rad=tolerance_rad,
            slack_rad=continuity_slack_rad,
        )
    ensure(bool(np.all(np.isfinite(solved))), "orientation solution must be finite")
    return OrientationSolution(
        {name: solved[:, j] for j, name in enumerate(names)}, residual
    )
