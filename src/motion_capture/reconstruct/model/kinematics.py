"""Articulated forward kinematics for a tree of segments with typed joints.

The engine behind the model-registry fitting work (epic #9709, issue #9711):
a generic tree of segments, each attached to its parent by a joint with one
to three scalar DOFs — every DOF is a rotation about a fixed axis in the
joint's pre-rotation frame with inclusive limits, so a hinge is one DOF, a
universal joint two, and a spherical joint three (an Euler composition in
the declared axis order). Forward kinematics maps a ``q(t)`` batch to
landmark positions vectorised over all frames; :func:`jacobian` returns the
analytic ``dLandmarks/dq`` the continuous fit needs. Pure numpy: nothing
here reaches into detectors, GUIs or simulation backends.

Geometry conventions: the root segment's frame starts at the world origin
aligned with the world axes; each child adds its parent-frame ``offset``
*before* its joint rotation, so a DOF rotates everything attached at or
below that joint about the joint origin. All angles are radians, all
lengths metres (ADR-0041 world frame).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require

Array = npt.NDArray[np.float64]

_AXIS_TOL = 1e-6
_LIMIT_TOL = 1e-9

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class DOF:
    """One scalar rotation: a fixed unit axis plus inclusive limits (radians)."""

    axis: Vec3
    lower: float
    upper: float


@dataclass(frozen=True)
class Joint:
    """A named joint: one to three DOFs composed in the declared order."""

    name: str
    dofs: tuple[DOF, ...]


@dataclass(frozen=True)
class Segment:
    """A body in the tree: parent name (``None`` for the single root), the
    joint connecting it to that parent, and the parent-frame offset from the
    parent's frame origin to this joint's origin."""

    name: str
    parent: str | None
    joint: Joint
    offset: Vec3


@dataclass(frozen=True)
class Landmark:
    """A point detectors observe, fixed in one segment's frame."""

    name: str
    segment: str
    offset: Vec3


def _rodrigues(axis: np.ndarray, theta: Array) -> Array:
    """Rotation matrices (T, 3, 3) about *axis* by per-frame angles."""
    t = np.asarray(theta, dtype=np.float64)
    k = np.zeros((t.size, 3, 3))
    k[:, 0, 1] = -axis[2]
    k[:, 0, 2] = axis[1]
    k[:, 1, 0] = axis[2]
    k[:, 1, 2] = -axis[0]
    k[:, 2, 0] = -axis[1]
    k[:, 2, 1] = axis[0]
    angle = t[:, None, None]
    small = np.abs(t) < 1e-8
    sin_t = np.where(small, t - t**3 / 6.0, np.sin(t))[:, None, None]
    one_minus_cos = np.where(small, t * t / 2.0 - t**4 / 24.0, 1.0 - np.cos(t))[
        :, None, None
    ]
    eye = np.eye(3)[None, :, :]
    return eye + sin_t * k + one_minus_cos * (k @ k)


@dataclass(frozen=True)
class ModelSpec:
    """A validated articulated model: segments, landmarks and derived order.

    Build through :meth:`create`, which enforces the contracts and
    precomputes the topological segment order, the DOF count and, for every
    segment, its chain of ancestors (itself included) used by the Jacobian.
    """

    segments: Mapping[str, Segment]
    landmarks: Mapping[str, Landmark]
    order: tuple[str, ...] = field(default=())
    n_dof: int = field(default=0)
    ancestors: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def create(
        cls, segments: Mapping[str, Segment], landmarks: Mapping[str, Landmark]
    ) -> ModelSpec:
        require(bool(segments), "a model needs at least one segment")
        require(
            all(
                seg.parent is not None or seg.joint is not None
                for seg in segments.values()
            ),
            "every segment needs a joint",
        )
        for name, seg in segments.items():
            require(seg.name == name, "segment dict key must match its name", name)
            require(
                seg.parent is None or seg.parent in segments,
                "segment parent must exist in the model",
                seg.parent,
            )
            _validate_joint(seg.joint)
            _validate_vec(seg.offset, f"segment {name} offset")
        roots = [n for n, s in segments.items() if s.parent is None]
        require(len(roots) == 1, "exactly one root segment is required", roots)
        order = _topological_order(segments, roots[0])
        if order is None:
            require(False, "segment tree must not contain a cycle and must be rooted")
            raise ValueError("segment tree must not contain a cycle and must be rooted")
        for lm in landmarks.values():
            require(
                lm.segment in segments,
                "landmark must attach to an existing segment",
                lm.segment,
            )
            _validate_vec(lm.offset, f"landmark {lm.name} offset")
        n_dof = sum(len(s.joint.dofs) for s in segments.values())
        require(n_dof >= 1, "a model needs at least one DOF")
        ancestors = {name: frozenset(_chain(segments, name)) for name in order}
        return cls(
            segments=segments,
            landmarks=landmarks,
            order=order,
            n_dof=n_dof,
            ancestors=ancestors,
        )


def _validate_vec(value: Vec3, what: str) -> None:
    require(len(value) == 3, f"{what} must have three components", value)
    require(all(np.isfinite(v) for v in value), f"{what} must be finite", value)


def _validate_joint(joint: Joint) -> None:
    require(1 <= len(joint.dofs) <= 3, "a joint needs one to three DOFs", joint.name)
    for dof in joint.dofs:
        require(
            np.isfinite(dof.lower) and np.isfinite(dof.upper),
            "DOF limits must be finite",
            joint.name,
        )
        require(
            dof.lower <= dof.upper, "DOF lower limit must not exceed upper", joint.name
        )
        axis = np.asarray(dof.axis, dtype=np.float64)
        require(axis.size == 3, "DOF axis must have three components", joint.name)
        require(bool(np.isfinite(axis).all()), "DOF axis must be finite", joint.name)
        require(
            abs(float(np.linalg.norm(axis)) - 1.0) <= _AXIS_TOL,
            "DOF axis must be a unit vector",
            joint.name,
        )


def _topological_order(
    segments: Mapping[str, Segment], root: str
) -> tuple[str, ...] | None:
    """Parents-before-children order, or ``None`` on a cycle/disconnected node."""
    order: list[str] = []
    seen: set[str] = set()
    for name in segments:
        chain: list[str] = []
        visited: set[str] = set()
        current: str | None = name
        while current is not None:
            if current in visited:
                return None  # cycle
            visited.add(current)
            chain.append(current)
            current = segments[current].parent
        if chain[-1] != root:
            return None  # does not reach the root
        for node in reversed(chain):
            if node not in seen:
                seen.add(node)
                order.append(node)
    if len(order) != len(segments):  # pragma: no cover - keys are unique
        return None
    return tuple(order)


def _chain(segments: Mapping[str, Segment], name: str) -> list[str]:
    """Names from the root down to and including *name*."""
    chain: list[str] = []
    current: str | None = name
    while current is not None:
        chain.append(current)
        current = segments[current].parent
    return list(reversed(chain))


@dataclass(frozen=True)
class FKResult:
    """Batched kinematic state, frames first.

    ``landmarks`` is ``(T, L, 3)`` in :attr:`ModelSpec.landmarks` insertion
    order; ``positions`` and ``orientations`` are ``(T, B, 3)`` and
    ``(T, B, 3, 3)`` in :attr:`ModelSpec.order`.
    """

    landmarks: Array
    positions: Array
    orientations: Array


def limit_violations(
    spec: ModelSpec, q: Array, tol: float = _LIMIT_TOL
) -> npt.NDArray[np.bool_]:
    """Boolean mask ``(T, n_dof)`` of angles outside their inclusive limits."""
    q = _validated_q(spec, q, check_limits=False)
    lower = np.array([d.lower for s in spec.order for d in spec.segments[s].joint.dofs])
    upper = np.array([d.upper for s in spec.order for d in spec.segments[s].joint.dofs])
    return (q < lower - tol) | (q > upper + tol)


def _validated_q(spec: ModelSpec, q: Array, *, check_limits: bool) -> Array:
    arr = np.asarray(q, dtype=np.float64)
    require(arr.ndim == 2, "q must be a (T, n_dof) array", arr.shape)
    require(
        arr.shape[1] == spec.n_dof,
        "q width must match the model DOF count",
        (arr.shape[1], spec.n_dof),
    )
    require(arr.shape[0] >= 1, "q needs at least one frame")
    require(bool(np.isfinite(arr).all()), "q must be finite")
    if check_limits:
        violations = limit_violations(spec, arr)
        require(
            not bool(violations.any()),
            "q violates the declared joint limits",
            int(violations.sum()),
        )
    return arr


def forward_kinematics(
    spec: ModelSpec, q: Array, *, enforce_limits: bool = True
) -> FKResult:
    """Map a ``(T, n_dof)`` angle batch to landmark positions over all frames."""
    q_arr = _validated_q(spec, q, check_limits=enforce_limits)
    frames = q_arr.shape[0]
    positions = np.zeros((frames, len(spec.order), 3))
    orientations = np.zeros((frames, len(spec.order), 3, 3))
    for index, name in enumerate(spec.order):
        seg = spec.segments[name]
        if seg.parent is None:
            parent_pos: Array = np.zeros((frames, 3))
            parent_rot: Array = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
        else:
            parent_index = spec.order.index(seg.parent)
            parent_pos = positions[:, parent_index]
            parent_rot = orientations[:, parent_index]
        offset = np.asarray(seg.offset, dtype=np.float64)
        local = parent_pos + parent_rot @ offset
        rot: Array = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
        for dof_index, dof in enumerate(seg.joint.dofs):
            axis = np.asarray(dof.axis, dtype=np.float64)
            rot = rot @ _rodrigues(axis, q_arr[:, _dof_offset(spec, name) + dof_index])
        positions[:, index] = local
        orientations[:, index] = parent_rot @ rot
    landmarks = np.empty((frames, len(spec.landmarks), 3))
    for li, lm in enumerate(spec.landmarks.values()):
        index = spec.order.index(lm.segment)
        offset = np.asarray(lm.offset, dtype=np.float64)
        landmarks[:, li] = positions[:, index] + orientations[:, index] @ offset
    result = FKResult(
        landmarks=landmarks, positions=positions, orientations=orientations
    )
    ensure(
        result.landmarks.shape == (frames, len(spec.landmarks), 3)
        and bool(np.isfinite(result.landmarks).all()),
        "forward kinematics must produce finite (T, L, 3) landmarks",
    )
    return result


def _dof_offset(spec: ModelSpec, segment: str) -> int:
    """Index of the segment's first DOF in the flattened q vector."""
    base = 0
    for name in spec.order:
        if name == segment:
            return base
        base += len(spec.segments[name].joint.dofs)
    raise KeyError(segment)  # pragma: no cover - validated at spec creation


def jacobian(spec: ModelSpec, q: Array, *, enforce_limits: bool = True) -> Array:
    """Analytic ``dLandmarks/dq`` with shape ``(T, 3 * L, n_dof)``.

    Column *d* for landmark *l* is ``omega x (p_l - origin_k)`` for every DOF
    whose owning segment *k* is an ancestor-or-self of the landmark's
    segment, and zero otherwise — the standard rotational Jacobian about the
    joint origin, with ``omega`` the DOF's axis rotated into the world by
    the rotation factors preceding it.
    """
    q_arr = _validated_q(spec, q, check_limits=enforce_limits)
    frames = q_arr.shape[0]
    state = forward_kinematics(spec, q_arr, enforce_limits=False)
    n_landmarks = len(spec.landmarks)
    jac = np.zeros((frames, 3 * n_landmarks, spec.n_dof))
    landmark_list = list(spec.landmarks.values())
    for name in spec.order:
        seg = spec.segments[name]
        base = _dof_offset(spec, name)
        parent_index = 0 if seg.parent is None else spec.order.index(seg.parent)
        pre: Array = (
            np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
            if seg.parent is None
            else state.orientations[:, parent_index].copy()
        )
        for dof_index, dof in enumerate(seg.joint.dofs):
            axis = np.asarray(dof.axis, dtype=np.float64)
            omega = pre @ axis
            for li, lm in enumerate(landmark_list):
                if name not in spec.ancestors[lm.segment]:
                    continue
                row = slice(3 * li, 3 * li + 3)
                owner_index = spec.order.index(name)
                lever = state.landmarks[:, li] - state.positions[:, owner_index]
                jac[:, row, base + dof_index] = np.cross(omega, lever)
            pre = pre @ _rodrigues(axis, q_arr[:, base + dof_index])
    ensure(
        jac.shape == (frames, 3 * n_landmarks, spec.n_dof)
        and bool(np.isfinite(jac).all()),
        "jacobian must be finite with shape (T, 3L, n_dof)",
    )
    return jac
