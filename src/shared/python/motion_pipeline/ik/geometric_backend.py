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
"""

from __future__ import annotations

import logging

import numpy as np

from ..contracts import (
    JointStateFrame,
    JointTrajectory,
    MarkerTrajectory,
    SkeletonRig,
)
from .base import BaseIKSolver, IKConfig, MarkerWeights

logger = logging.getLogger(__name__)

_Vec3 = tuple[float, float, float]


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


def forward_kinematics(rig: SkeletonRig, q: list[float]) -> dict[str, _Vec3]:
    """Compute world positions of every joint for joint vector ``q``.

    Args:
        rig: Skeleton rig defining the kinematic tree.
        q: Joint angles (radians), one entry per DOF in :func:`_dof_layout`
            order.

    Returns:
        Mapping ``joint_name -> (x, y, z)`` world position.

    Raises:
        ValueError: If ``len(q)`` does not match the rig DOF count.
    """
    layout = _dof_layout(rig)
    if len(q) != len(layout):
        raise ValueError(f"q has {len(q)} entries, expected {len(layout)} DOFs for rig")

    # Per-joint local rotation (product of its axis rotations).
    angles_by_joint: dict[str, list[tuple[str, float]]] = {}
    for (jname, axis), angle in zip(layout, q, strict=True):
        angles_by_joint.setdefault(jname, []).append((axis, angle))

    local_rot: dict[str, np.ndarray] = {}
    for jname in rig.joints:
        rot = np.eye(3)
        for axis, angle in angles_by_joint.get(jname, []):
            rot = rot @ _axis_rotation(axis, angle)
        local_rot[jname] = rot

    # Propagate transforms root -> leaves in topological (parent-first) order.
    order = _topological_order(rig)
    world_rot: dict[str, np.ndarray] = {}
    world_pos: dict[str, np.ndarray] = {}
    for jname in order:
        jdef = rig.joints[jname]
        offset = np.asarray(jdef.tpose_offset, dtype=float) * rig.scale
        if jdef.parent is None:
            parent_rot = np.eye(3)
            parent_pos = np.zeros(3)
        else:
            parent_rot = world_rot[jdef.parent]
            parent_pos = world_pos[jdef.parent]
        pos = parent_pos + parent_rot @ offset
        world_pos[jname] = pos
        world_rot[jname] = parent_rot @ local_rot[jname]

    return {
        name: (float(p[0]), float(p[1]), float(p[2])) for name, p in world_pos.items()
    }


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
            q = self.solve_frame(marker_positions, rig, weights, initial_q=warm_start)
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
    ) -> list[float]:
        """Solve single-frame IK via damped least squares.

        Args:
            markers: Mapping marker name -> (x, y, z) target position.
            rig: Skeleton rig.
            weights: Optional per-marker weights.
            initial_q: Optional warm-start joint vector.

        Returns:
            Joint vector ``q`` (radians) in :func:`_dof_layout` order,
            clamped to joint limits.

        Raises:
            ValueError: If no marker maps to a joint in the rig (DbC: there
                must be at least one usable target).
        """
        targets = self._match_targets(markers, rig)
        if not targets:
            raise ValueError(
                "No marker matches a joint in the rig; cannot solve IK frame"
            )

        layout = _dof_layout(rig)
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
        for _ in range(self.config.max_iterations):
            residual = self._residual(rig, q, joint_names, target_vec, weight_vec)
            cost = float(residual @ residual)
            if cost < self.config.tolerance:
                break

            jac = self._numeric_jacobian(rig, q, joint_names, weight_vec, target_vec)
            # Damped normal equations: (JᵀJ + λI) dq = Jᵀ r
            jtj = jac.T @ jac
            jtr = jac.T @ residual
            try:
                dq = np.linalg.solve(jtj + lam * np.eye(n_dof), jtr)
            except np.linalg.LinAlgError:
                break
            q_new = self._clamp_array(q + dq, rig, layout)

            new_residual = self._residual(
                rig, q_new, joint_names, target_vec, weight_vec
            )
            new_cost = float(new_residual @ new_residual)
            if new_cost < cost:
                q = q_new
                lam = max(lam * 0.5, 1e-9)
                if abs(prev_cost - new_cost) < self.config.tolerance:
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
    def _residual(
        rig: SkeletonRig,
        q: np.ndarray,
        joint_names: list[str],
        target_vec: np.ndarray,
        weight_vec: np.ndarray,
    ) -> np.ndarray:
        positions = forward_kinematics(rig, q.tolist())
        current = np.concatenate(
            [np.asarray(positions[name], dtype=float) for name in joint_names]
        )
        return weight_vec * (target_vec - current)

    def _numeric_jacobian(
        self,
        rig: SkeletonRig,
        q: np.ndarray,
        joint_names: list[str],
        weight_vec: np.ndarray,
        target_vec: np.ndarray,
    ) -> np.ndarray:
        """Central-difference Jacobian of the residual w.r.t. ``q``."""
        eps = 1e-6
        n_dof = len(q)
        n_res = 3 * len(joint_names)
        jac = np.zeros((n_res, n_dof))
        for j in range(n_dof):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[j] += eps
            q_minus[j] -= eps
            r_plus = self._residual(rig, q_plus, joint_names, target_vec, weight_vec)
            r_minus = self._residual(rig, q_minus, joint_names, target_vec, weight_vec)
            jac[:, j] = (r_plus - r_minus) / (2.0 * eps)
        # d(residual)/dq = -d(current)/dq; residual = w*(target - fk(q)).
        return -jac

    @staticmethod
    def _clamp_array(
        q: np.ndarray, rig: SkeletonRig, layout: list[tuple[str, str]]
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
