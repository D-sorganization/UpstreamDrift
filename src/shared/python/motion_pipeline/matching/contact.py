"""
Ground-contact models for motion-matching swing/stance tracking.

Part of issue #4568. Pure-Python; no engine deps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol
from collections.abc import Iterable, Mapping

import numpy as np

from ..contracts import JointTrajectory, SkeletonRig

logger = logging.getLogger(__name__)


class ContactModel(Protocol):
    """Minimum interface for a ground-contact model."""

    def contact_points(self, rig: SkeletonRig) -> list[str]:
        """Return joint/segment names treated as contact points."""
        ...

    def contact_forces(self, state: Mapping[str, Any], time: float) -> np.ndarray:
        """
        Return a (n_points, 3) array of contact forces in world frame.

        ``state`` should provide ``positions`` (n_points, 3) and
        ``velocities`` (n_points, 3) of the contact points.
        """
        ...


@dataclass
class FlatGroundContact:
    """
    Spring-damper normal force at a flat ground (z=0 plane) plus
    Coulomb tangential friction.

    Attributes:
        points: Ordered list of contact-point joint names.
        stiffness: Normal spring stiffness, N/m.
        damping: Normal damping, N·s/m.
        friction: Coulomb friction coefficient.
        ground_height: Height of the ground plane along the up axis (m).
    """

    points: list[str] = field(default_factory=list)
    stiffness: float = 1.0e5
    damping: float = 1.0e3
    friction: float = 0.8
    ground_height: float = 0.0

    def __post_init__(self) -> None:
        if self.stiffness < 0:
            raise ValueError("stiffness must be >= 0")
        if self.damping < 0:
            raise ValueError("damping must be >= 0")
        if self.friction < 0:
            raise ValueError("friction must be >= 0")
        if not np.isfinite(self.ground_height):
            raise ValueError("ground_height must be finite")

    def contact_points(self, rig: SkeletonRig) -> list[str]:
        if rig is None:
            raise ValueError("rig must be provided")
        # Validate that every named point exists in the rig.
        bad = [p for p in self.points if p not in rig.joints]
        if bad:
            raise ValueError(f"Contact points not in rig: {bad}")
        return list(self.points)

    def contact_forces(self, state: Mapping[str, Any], time: float) -> np.ndarray:
        """
        Compute (n_points, 3) ground reaction forces.

        ``state`` requires ``positions`` shaped (n_points, 3) and
        optionally ``velocities`` shaped (n_points, 3). Up axis is z.
        """
        if state is None:
            raise ValueError("state must be provided")
        if not np.isfinite(time):
            raise ValueError("time must be finite")

        positions = np.asarray(state.get("positions"), dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (n,3), got shape {positions.shape}")
        velocities = state.get("velocities")
        if velocities is None:
            velocities = np.zeros_like(positions)
        else:
            velocities = np.asarray(velocities, dtype=float)
        if velocities.shape != positions.shape:
            raise ValueError("velocities shape must match positions shape")

        forces = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            penetration = self.ground_height - positions[i, 2]
            if penetration <= 0:
                continue  # no contact
            v_normal = -velocities[i, 2]
            f_n = self.stiffness * penetration + self.damping * v_normal
            f_n = max(0.0, f_n)
            forces[i, 2] = f_n

            v_tan = velocities[i, :2]
            v_tan_norm = float(np.linalg.norm(v_tan))
            if v_tan_norm > 1e-9 and f_n > 0:
                f_t = -self.friction * f_n * v_tan / v_tan_norm
                forces[i, 0] = f_t[0]
                forces[i, 1] = f_t[1]

        if not np.all(np.isfinite(forces)):
            raise ValueError("Computed non-finite contact force")
        return forces


@dataclass
class NoContactModel:
    """Pass-through contact model — always reports zero force."""

    points: list[str] = field(default_factory=list)

    def contact_points(self, rig: SkeletonRig) -> list[str]:
        if rig is None:
            raise ValueError("rig must be provided")
        return list(self.points)

    def contact_forces(self, state: Mapping[str, Any], time: float) -> np.ndarray:
        if state is None:
            raise ValueError("state must be provided")
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        positions = np.asarray(state.get("positions"), dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (n,3), got shape {positions.shape}")
        return np.zeros_like(positions)


def infer_contact_phases(
    traj: JointTrajectory,
    contact_points: list[str],
    rig: SkeletonRig,
    *,
    height_threshold: float = 0.05,
    min_phase_duration: float = 0.05,
) -> list[tuple[float, float]]:
    """
    Heuristic stance-phase detection from joint kinematics.

    Approximates each contact-point height by mapping the contact-point
    joint name to a flat index into the trajectory's q vector and
    treating that q value as a vertical offset (suitable for synthetic
    pendulum-style rigs and unit tests). For richer rigs the caller
    should swap in a forward-kinematics-aware variant.

    Args:
        traj: Joint trajectory.
        contact_points: Joint names whose stance phases we want.
        rig: Skeleton rig (used to resolve joint indices).
        height_threshold: Below this height the point is "in contact".
        min_phase_duration: Drop phases shorter than this (seconds).

    Returns:
        List of ``(t_start, t_end)`` stance intervals (sorted, non-overlapping).
    """
    if traj is None or not traj.frames:
        raise ValueError("trajectory must have frames")
    if rig is None:
        raise ValueError("rig must be provided")
    if not np.isfinite(height_threshold) or height_threshold < 0:
        raise ValueError("height_threshold must be finite and >= 0")
    if not np.isfinite(min_phase_duration) or min_phase_duration < 0:
        raise ValueError("min_phase_duration must be finite and >= 0")

    if not contact_points:
        return []

    joint_names = list(rig.joints.keys())
    indices: list[int] = []
    cursor = 0
    for jname in joint_names:
        n_axes = len(rig.joints[jname].axes)
        if jname in contact_points:
            indices.append(cursor)  # use first axis as proxy height
        cursor += n_axes

    if not indices:
        return []

    times = np.asarray([f.timestamp for f in traj.frames], dtype=float)
    q = np.asarray([list(f.q) for f in traj.frames], dtype=float)
    # min height across selected contact points per frame
    heights = np.min(q[:, indices], axis=1)

    in_contact = heights < height_threshold

    phases: list[tuple[float, float]] = []
    start_idx: int | None = None
    for i, c in enumerate(in_contact):
        if c and start_idx is None:
            start_idx = i
        elif not c and start_idx is not None:
            t0 = float(times[start_idx])
            t1 = float(times[i - 1])
            if t1 - t0 >= min_phase_duration or start_idx == i - 1:
                phases.append((t0, t1))
            start_idx = None
    if start_idx is not None:
        t0 = float(times[start_idx])
        t1 = float(times[-1])
        if t1 - t0 >= min_phase_duration or start_idx == len(times) - 1:
            phases.append((t0, t1))

    # Postcondition: sorted, finite
    for t0, t1 in phases:
        if not (np.isfinite(t0) and np.isfinite(t1)) or t1 < t0:
            raise ValueError(f"Invalid phase ({t0}, {t1})")
    return phases


def _validate_contact_model(model: ContactModel, rig: SkeletonRig) -> Iterable[str]:
    """Helper: returns the contact point list, raising on misconfiguration."""
    return model.contact_points(rig)


__all__ = [
    "ContactModel",
    "FlatGroundContact",
    "NoContactModel",
    "infer_contact_phases",
]
