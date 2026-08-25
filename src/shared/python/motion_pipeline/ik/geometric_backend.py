"""Geometric (analytic) backend for Inverse Kinematics.

Implements issue #7046: a real, dependency-free single-frame IK solver
using damped least squares (Levenberg-Marquardt) over a forward-kinematics
model derived directly from the :class:`SkeletonRig` contract.

Unlike the engine-specific backends (mujoco/drake/opensim/pinocchio) which
require heavy optional wheels, this backend needs only NumPy and is always
available, so it serves as the canonical fallback solver and the reference
for cross-backend parity.

Forward-kinematics model
-------------------------
Each joint contributes one revolute DOF per declared axis (in order). A
joint's world position is its parent's world transform applied to the
joint's ``tpose_offset``; the joint's own rotation (about its axes) is then
composed into the frame propagated to children. Markers are matched to
joints by name (or by ``semantic_label``), which mirrors how the canonical
marker sets label joints.

Performance (issue #8921)
--------------------------
Two structures are expensive to recompute and are hoisted out of the hot
path:

* Rig topology (:func:`_dof_layout`, :func:`_topological_order`, and the
  per-joint ancestor sets) depends only on the rig, never on ``q``. It is
  built once per rig via :func:`_get_rig_topology`, cached by rig identity.
* The Levenberg-Marquardt Jacobian is the analytic revolute-joint form
  (column ``k`` = ``axis_world_k x (p_i - p_pivot_k)``) computed from a
  single forward-kinematics pass (:func:`_forward_kinematics_full`), instead
  of ``2 * n_dof`` finite-difference forward-kinematics calls per iteration.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import (
    JointStateFrame,
    JointTrajectory,
    MarkerTrajectory,
    SkeletonRig,
)
from .base import BaseIKSolver, IKConfig, MarkerWeights

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

_Vec3 = tuple[float, float, float]

_AXIS_UNIT: dict[str, np.ndarray] = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    """Return the 3x3 rotation matrix for ``angle`` about a signed axis.

    ``axis`` is one of the canonical :data:`contracts.Axis` literals
    (e.g. ``"X"``, ``"+Y"``, ``"-Z"``). A leading ``-`` negates the angle.
    """
    letter = axis[-1].upper()
    sign = -1.0 if axis.startswith("-") else 1.0
    a = sign * angle
    c, s = float(np.cos(a)), float(np.sin(a))
    if letter == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if letter == "Y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if letter == "Z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unknown rotation axis: {axis!r}")


def _axis_local_vector(axis: str) -> np.ndarray:
    """Return the unit rotation axis (joint-local frame) for a signed axis.

    Mirrors the sign convention of :func:`_axis_rotation`: rotating by
    ``angle`` about a ``"-Z"``-declared axis is equivalent (for Jacobian
    purposes) to rotating by ``angle`` about the vector ``(0, 0, -1)``.
    """
    letter = axis[-1].upper()
    sign = -1.0 if axis.startswith("-") else 1.0
    try:
        unit = _AXIS_UNIT[letter]
    except KeyError as exc:
        raise ValueError(f"Unknown rotation axis: {axis!r}") from exc
    return sign * unit


def _dof_layout(rig: SkeletonRig) -> list[tuple[str, str]]:
    """Return the ordered ``(joint_name, axis)`` layout of the DOF vector.

    Matches the convention used across the pipeline: joints are walked in
    insertion order and each declared axis contributes one DOF.
    """
    layout: list[tuple[str, str]] = []
    for jname, jdef in rig.joints.items():
        for axis in jdef.axes:
            layout.append((jname, axis))
    return layout


def _topological_order(rig: SkeletonRig) -> list[str]:
    """Return joint names ordered so every parent precedes its children."""
    visited: list[str] = []
    seen: set[str] = set()

    def _visit(name: str) -> None:
        if name in seen:
            return
        jdef = rig.joints[name]
        if jdef.parent is not None and jdef.parent not in seen:
            _visit(jdef.parent)
        seen.add(name)
        visited.append(name)

    for name in rig.joints:
        _visit(name)
    return visited


@dataclass(frozen=True)
class _RigTopology:
    """Precomputed, ``q``-independent structure of a rig's kinematic tree.

    Building this requires the full recursive tree walk performed by
    :func:`_dof_layout` and :func:`_topological_order`; see
    :func:`_get_rig_topology` for why/how it is cached.
    """

    layout: tuple[tuple[str, str], ...]
    order: tuple[str, ...]
    dof_indices_by_joint: dict[str, tuple[int, ...]]
    # joint -> frozenset of strict ancestor joint names (root .. parent).
    ancestors: dict[str, frozenset[str]]


def _build_rig_topology(rig: SkeletonRig) -> _RigTopology:
    layout = tuple(_dof_layout(rig))
    order = tuple(_topological_order(rig))

    dof_indices_by_joint: dict[str, list[int]] = {}
    for idx, (jname, _axis) in enumerate(layout):
        dof_indices_by_joint.setdefault(jname, []).append(idx)

    ancestors: dict[str, frozenset[str]] = {}
    for jname in order:
        parent = rig.joints[jname].parent
        ancestors[jname] = (
            frozenset() if parent is None else frozenset({parent, *ancestors[parent]})
        )

    return _RigTopology(
        layout=layout,
        order=order,
        dof_indices_by_joint={k: tuple(v) for k, v in dof_indices_by_joint.items()},
        ancestors=ancestors,
    )


# Rig identity -> (weak reference to the rig, its cached topology).
#
# SkeletonRig (a Pydantic model) is not hashable, so this caches on object
# identity (`id(rig)`) rather than a value key. The guard reference check
# means a stale entry (from a garbage-collected rig whose id got reused by
# an unrelated object) is detected and rebuilt rather than silently
# returning the wrong topology. Bounded FIFO-ish eviction keeps long-running
# processes (e.g. batch IK over many distinct rigs) from growing this
# unboundedly.
_TOPOLOGY_CACHE: dict[int, tuple[weakref.ReferenceType[SkeletonRig], _RigTopology]] = {}
_TOPOLOGY_CACHE_MAX = 64


def _get_rig_topology(rig: SkeletonRig) -> _RigTopology:
    """Return the cached :class:`_RigTopology` for ``rig``, building it once.

    Avoids repeating the O(n_joints) recursive tree walk on every one of the
    millions of forward-kinematics calls a single IK trial can perform
    (issue #8921).
    """
    key = id(rig)
    cached = _TOPOLOGY_CACHE.get(key)
    if cached is not None and cached[0]() is rig:
        return cached[1]

    topo = _build_rig_topology(rig)
    if len(_TOPOLOGY_CACHE) >= _TOPOLOGY_CACHE_MAX:
        _TOPOLOGY_CACHE.pop(next(iter(_TOPOLOGY_CACHE)))
    _TOPOLOGY_CACHE[key] = (weakref.ref(rig), topo)
    return topo


def _forward_kinematics_full(
    rig: SkeletonRig,
    q: Sequence[float] | np.ndarray,
    topo: _RigTopology,
) -> tuple[np.ndarray, dict[str, int], list[np.ndarray]]:
    """Single forward-kinematics pass used by the IK hot path.

    Unlike :func:`forward_kinematics` (the public, dict-returning API) this
    returns plain NumPy arrays so callers (the LM residual and Jacobian) do
    not pay for a dict-of-Python-float-tuples round trip on every call, and
    it also returns the world-frame rotation axis of every DOF so the
    analytic Jacobian can be built from this one pass instead of
    ``2 * n_dof`` extra forward-kinematics evaluations.

    Returns:
        positions: ``(n_joints, 3)`` world positions, row order matches
            ``topo.order``.
        name_to_row: joint name -> row index into ``positions``.
        dof_axis_world: length ``len(topo.layout)`` list of ``(3,)``
            world-frame unit rotation axes, aligned with ``topo.layout``
            (and therefore with ``q``).

    Raises:
        ValueError: If ``len(q)`` does not match the rig DOF count.
    """
    layout = topo.layout
    if len(q) != len(layout):
        raise ValueError(f"q has {len(q)} entries, expected {len(layout)} DOFs for rig")

    n_joints = len(topo.order)
    positions = np.zeros((n_joints, 3))
    name_to_row = {name: i for i, name in enumerate(topo.order)}
    world_rot: dict[str, np.ndarray] = {}
    dof_axis_world: list[np.ndarray] = [np.zeros(3) for _ in layout]

    for jname in topo.order:
        jdef = rig.joints[jname]
        offset = np.asarray(jdef.tpose_offset, dtype=float) * rig.scale
        if jdef.parent is None:
            parent_rot = np.eye(3)
            parent_pos = np.zeros(3)
        else:
            parent_rot = world_rot[jdef.parent]
            parent_pos = positions[name_to_row[jdef.parent]]

        positions[name_to_row[jname]] = parent_pos + parent_rot @ offset

        # Walk this joint's own axes in declared order, recording each DOF's
        # world-frame rotation axis *before* that axis's own rotation is
        # applied (composed local rotations act to the right, so an axis's
        # world direction only sees rotations that precede it).
        local_partial = np.eye(3)
        for dof_idx in topo.dof_indices_by_joint.get(jname, ()):
            _, axis = layout[dof_idx]
            dof_axis_world[dof_idx] = (
                parent_rot @ local_partial @ _axis_local_vector(axis)
            )
            local_partial = local_partial @ _axis_rotation(axis, float(q[dof_idx]))

        world_rot[jname] = parent_rot @ local_partial

    return positions, name_to_row, dof_axis_world


def forward_kinematics(rig: SkeletonRig, q: list[float]) -> dict[str, _Vec3]:
    """Compute world positions of every joint for joint vector ``q``.

    This is the public, backend-agnostic entry point (kept dict-returning
    for API compatibility with existing callers/tests); the IK solver's hot
    loop uses :func:`_forward_kinematics_full` directly to avoid the
    dict-of-tuples <-> ndarray round trip on every LM iteration.

    Args:
        rig: Skeleton rig defining the kinematic tree.
        q: Joint angles (radians), one entry per DOF in :func:`_dof_layout`
            order.

    Returns:
        Mapping ``joint_name -> (x, y, z)`` world position.

    Raises:
        ValueError: If ``len(q)`` does not match the rig DOF count.
    """
    topo = _get_rig_topology(rig)
    positions, name_to_row, _ = _forward_kinematics_full(rig, q, topo)
    return {
        name: (
            float(positions[row, 0]),
            float(positions[row, 1]),
            float(positions[row, 2]),
        )
        for name, row in name_to_row.items()
    }


class GeometricIKSolver(BaseIKSolver):
    """Damped-least-squares (Levenberg-Marquardt) inverse-kinematics solver.

    Minimizes the weighted squared distance between observed markers and the
    forward-kinematics position of the joint each marker is associated with.
    Dependency-free: requires only NumPy.
    """

    def __init__(self, config: IKConfig | None = None):
        super().__init__(config)

    def solve(
        self,
        markers: MarkerTrajectory,
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
        config: IKConfig | None = None,
    ) -> JointTrajectory:
        """Solve IK frame-by-frame, warm-starting from the previous solution."""
        if not isinstance(markers, MarkerTrajectory):
            raise TypeError("markers must be a MarkerTrajectory")
        config = config or self.config

        frames: list[JointStateFrame] = []
        warm_start: list[float] | None = None
        for frame in markers.frames:
            marker_positions = {
                name: (m.x, m.y, m.z) for name, m in frame.markers.items()
            }
            q = self.solve_frame(
                marker_positions, rig, weights, initial_q=warm_start, config=config
            )
            warm_start = q
            frames.append(
                JointStateFrame(
                    timestamp=frame.timestamp,
                    q=q,
                    qdot=None,
                    qddot=None,
                    frame_index=frame.frame_index,
                )
            )

        return JointTrajectory(
            id=f"ik-geometric-{markers.id}",
            skeleton=rig,
            frames=frames,
            metadata={
                "backend": "geometric",
                "config": {
                    "max_iterations": config.max_iterations,
                    "tolerance": config.tolerance,
                },
            },
        )

    def solve_frame(
        self,
        markers: dict[str, tuple[float, float, float]],
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
        initial_q: list[float] | None = None,
        config: IKConfig | None = None,
    ) -> list[float]:
        """Solve single-frame IK via damped least squares.

        Args:
            markers: Mapping marker name -> (x, y, z) target position.
            rig: Skeleton rig.
            weights: Optional per-marker weights.
            initial_q: Optional warm-start joint vector.
            config: Optional per-call solver configuration. Falls back to
                ``self.config`` when omitted (previously this parameter did
                not exist and per-call configs passed to :meth:`solve` were
                silently ignored by the iteration loop; see issue #8921).

        Returns:
            Joint vector ``q`` (radians) in :func:`_dof_layout` order,
            clamped to joint limits.

        Raises:
            ValueError: If no marker maps to a joint in the rig (DbC: there
                must be at least one usable target).
        """
        config = config or self.config
        targets = self._match_targets(markers, rig)
        if not targets:
            raise ValueError(
                "No marker matches a joint in the rig; cannot solve IK frame"
            )

        topo = _get_rig_topology(rig)
        layout = topo.layout
        n_dof = len(layout)
        joint_names = [name for name, _ in targets]
        target_vec = np.concatenate([np.asarray(p, dtype=float) for _, p in targets])
        weight_vec = self._weight_vector(joint_names, weights)

        q = (
            np.zeros(n_dof)
            if initial_q is None
            else np.asarray(initial_q, dtype=float).copy()
        )
        q = self._clamp_array(q, rig, layout)

        lam = 1e-2  # Levenberg-Marquardt damping
        prev_cost = np.inf
        for _ in range(config.max_iterations):
            positions, name_to_row, dof_axis_world = _forward_kinematics_full(
                rig, q, topo
            )
            residual = self._residual_from_positions(
                positions, name_to_row, joint_names, target_vec, weight_vec
            )
            cost = float(residual @ residual)
            if cost < config.tolerance:
                break

            jac = self._analytic_jacobian(
                topo, positions, name_to_row, dof_axis_world, joint_names, weight_vec
            )
            # Damped normal equations: (JᵀJ + λI) dq = Jᵀ r
            jtj = jac.T @ jac
            jtr = jac.T @ residual
            try:
                dq = np.linalg.solve(jtj + lam * np.eye(n_dof), jtr)
            except np.linalg.LinAlgError:
                break
            q_new = self._clamp_array(q + dq, rig, layout)

            new_positions, new_name_to_row, _ = _forward_kinematics_full(
                rig, q_new, topo
            )
            new_residual = self._residual_from_positions(
                new_positions, new_name_to_row, joint_names, target_vec, weight_vec
            )
            new_cost = float(new_residual @ new_residual)
            if new_cost < cost:
                q = q_new
                lam = max(lam * 0.5, 1e-9)
                if abs(prev_cost - new_cost) < config.tolerance:
                    break
                prev_cost = new_cost
            else:
                lam = min(lam * 2.0, 1e6)

        result = q.tolist()
        if not self._validate_result(result, rig):
            raise RuntimeError("Geometric IK produced an invalid (non-finite) pose")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_targets(
        markers: dict[str, tuple[float, float, float]],
        rig: SkeletonRig,
    ) -> list[tuple[str, tuple[float, float, float]]]:
        """Associate markers with rig joints by name or semantic label."""
        label_to_joint: dict[str, str] = {}
        for jname, jdef in rig.joints.items():
            if jdef.semantic_label:
                label_to_joint[jdef.semantic_label] = jname
        targets: list[tuple[str, tuple[float, float, float]]] = []
        for marker_name, pos in markers.items():
            if marker_name in rig.joints:
                targets.append((marker_name, pos))
            elif marker_name in label_to_joint:
                targets.append((label_to_joint[marker_name], pos))
        return targets

    @staticmethod
    def _weight_vector(
        joint_names: list[str], weights: MarkerWeights | None
    ) -> np.ndarray:
        if weights is None:
            return np.ones(3 * len(joint_names))
        per_joint = [weights.get_weight(name) for name in joint_names]
        return np.repeat(np.asarray(per_joint, dtype=float), 3)

    @staticmethod
    def _residual_from_positions(
        positions: np.ndarray,
        name_to_row: dict[str, int],
        joint_names: list[str],
        target_vec: np.ndarray,
        weight_vec: np.ndarray,
    ) -> np.ndarray:
        """Weighted ``target - current`` residual from an FK positions array."""
        current = np.concatenate([positions[name_to_row[name]] for name in joint_names])
        return weight_vec * (target_vec - current)

    @staticmethod
    def _analytic_jacobian(
        topo: _RigTopology,
        positions: np.ndarray,
        name_to_row: dict[str, int],
        dof_axis_world: list[np.ndarray],
        joint_names: list[str],
        weight_vec: np.ndarray,
    ) -> np.ndarray:
        """Closed-form Jacobian of the weighted FK targets w.r.t. ``q``.

        Column ``k`` (DOF ``(jname, axis)``) is the standard revolute-joint
        velocity contribution ``axis_world_k x (p_i - p_pivot)`` for every
        target ``i`` downstream of ``jname`` (zero otherwise), where
        ``p_pivot`` is ``jname``'s own world position -- translation to a
        joint happens before that joint's own rotations are applied, so all
        of a joint's declared axes share the same pivot.

        Replaces the ``2 * n_dof`` finite-difference forward-kinematics
        calls of the previous implementation with zero extra
        forward-kinematics evaluations: ``positions`` and ``dof_axis_world``
        come from the single :func:`_forward_kinematics_full` pass already
        performed for the residual (issue #8921).
        """
        n_dof = len(topo.layout)
        n_targets = len(joint_names)
        jac = np.zeros((3 * n_targets, n_dof))

        target_rows = [name_to_row[name] for name in joint_names]
        target_positions = positions[target_rows]
        target_ancestors = [topo.ancestors[name] for name in joint_names]

        for k, (jname, _axis) in enumerate(topo.layout):
            pivot = positions[name_to_row[jname]]
            axis_world = dof_axis_world[k]
            for i in range(n_targets):
                if jname not in target_ancestors[i]:
                    continue
                lever = target_positions[i] - pivot
                jac[3 * i : 3 * i + 3, k] = np.cross(axis_world, lever)

        return weight_vec[:, np.newaxis] * jac

    @staticmethod
    def _clamp_array(
        q: np.ndarray, rig: SkeletonRig, layout: Iterable[tuple[str, str]]
    ) -> np.ndarray:
        """Clamp each DOF to its joint limit (if any), preserving order."""
        clamped = q.copy()
        # Per-joint axis index so we can pick the right limit entry.
        axis_index: dict[str, int] = {}
        for i, (jname, _axis) in enumerate(layout):
            idx = axis_index.get(jname, 0)
            axis_index[jname] = idx + 1
            jdef = rig.joints[jname]
            if jdef.limits and idx < len(jdef.limits):
                limit = jdef.limits[idx]
                if limit.lower is not None:
                    clamped[i] = max(clamped[i], limit.lower)
                if limit.upper is not None:
                    clamped[i] = min(clamped[i], limit.upper)
        return clamped
