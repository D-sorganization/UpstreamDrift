"""Engine-neutral reference for Drake/Pinocchio tree equivalence (#14756).

A small numpy reference for the quantities two engines must agree on when
they load the same URDF tree: DOF ordering, floating-base conventions,
rotational inertia about the link origin, gravity, joint damping, external
wrenches and the ADR-0059 trunk equalities. Revolute and fixed joints only.
Nothing here imports Drake or Pinocchio; engine adapters compare against it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.contracts import ensure, require
from src.shared.python.model_generation.core.types import Joint, JointType, Link

GRAVITY = np.array([0.0, 0.0, -9.81])
_MOVING = (JointType.REVOLUTE, JointType.CONTINUOUS)


def _rot(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(r),
        np.sin(r),
        np.cos(p),
        np.sin(p),
        np.cos(y),
        np.sin(y),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def dof_order(joints: Sequence[Joint], root: str) -> list[str]:
    """Moving-joint names, depth-first in declaration order (not canonical sort)."""
    require(any(j.parent == root for j in joints), "root must parent a joint", root)
    order: list[str] = []

    def visit(link: str) -> None:
        for j in joints:
            if j.parent == link:
                if j.joint_type in _MOVING:
                    order.append(j.name)
                visit(j.child)

    visit(root)
    ensure(len(order) == len(set(order)), "DOF names must be unique", order)
    return order


@dataclass(frozen=True)
class TreeState:
    """World-frame link frames and joint axes/origins for one configuration."""

    link_rot: Mapping[str, np.ndarray]
    link_pos: Mapping[str, np.ndarray]
    joint_axis: Mapping[str, np.ndarray]
    joint_pos: Mapping[str, np.ndarray]


def kinematics(joints: Sequence[Joint], root: str, q: Mapping[str, float]) -> TreeState:
    """Forward kinematics of the tree for joint angles ``q`` (rad)."""
    rot, pos = {root: np.eye(3)}, {root: np.zeros(3)}
    axes: dict[str, np.ndarray] = {}
    origins: dict[str, np.ndarray] = {}
    for j in joints:  # parents are declared before children in a valid tree
        require(j.parent in rot, "joints must be declared parent-first", j.name)
        r0 = rot[j.parent] @ _rot(j.origin.rpy)
        origins[j.name] = pos[j.parent] + rot[j.parent] @ np.asarray(j.origin.xyz)
        pos[j.child] = origins[j.name]
        rot[j.child] = r0
        if j.joint_type in _MOVING:
            axes[j.name] = r0 @ (np.asarray(j.axis) / np.linalg.norm(j.axis))
            rot[j.child] = r0 @ _axis_angle(j.axis, q.get(j.name, 0.0))
    return TreeState(rot, pos, axes, origins)


def inertia_about_link_origin(
    mass: float,
    com: ArrayLike,
    inertia_at_com: np.ndarray,
    inertial_rpy: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Rotational inertia about the link origin, in the link frame.

    ``inertia_at_com`` is in the inertial frame ``inertial_rpy`` (URDF
    ``<inertial><origin rpy>``); rotate into the link frame, then shift.
    """
    require(mass > 0.0, "mass must be positive", mass)
    rot = _rot(inertial_rpy)
    c = np.asarray(com, dtype=float)
    out = rot @ inertia_at_com @ rot.T + mass * (c @ c * np.eye(3) - np.outer(c, c))
    ensure(np.allclose(out, out.T), "inertia must stay symmetric")
    return out


def _com_world(link: Link, st: TreeState) -> np.ndarray:
    c = np.asarray(link.inertia.center_of_mass)
    return st.link_pos[link.name] + st.link_rot[link.name] @ c


def _mass(link: Link) -> float:
    return float(link.inertia.mass)


def potential_energy(
    links: Sequence[Link], joints: Sequence[Joint], root: str, q: Mapping[str, float]
) -> float:
    """Gravitational potential energy."""
    st = kinematics(joints, root, q)
    return float(
        -sum(
            _mass(k) * GRAVITY @ _com_world(k, st)
            for k in links
            if k.name in st.link_pos
        )
    )


def _descendants(joints: Sequence[Joint], link: str) -> set[str]:
    out = {link}
    for j in joints:
        if j.parent in out:
            out.add(j.child)
    return out


def gravity_torques(
    links: Sequence[Link], joints: Sequence[Joint], root: str, q: Mapping[str, float]
) -> np.ndarray:
    """Generalized gravity ``dU/dq`` in :func:`dof_order` order."""
    st = kinematics(joints, root, q)
    child = {j.name: j.child for j in joints}
    tau = []
    for name in dof_order(joints, root):
        sub = _descendants(joints, child[name])
        t = 0.0
        for k in links:
            if k.name in sub:
                arm = np.cross(
                    st.joint_axis[name], _com_world(k, st) - st.joint_pos[name]
                )
                t += float(-_mass(k) * (GRAVITY @ arm))
        tau.append(t)
    return np.array(tau)


def joint_damping_torques(
    joints: Sequence[Joint], root: str, qd: Mapping[str, float]
) -> np.ndarray:
    """Viscous torque ``-b qd`` in :func:`dof_order` order."""
    damping = {j.name: j.dynamics.damping for j in joints}
    return np.array([-damping[n] * qd.get(n, 0.0) for n in dof_order(joints, root)])


def external_wrench_torques(
    joints: Sequence[Joint],
    root: str,
    q: Mapping[str, float],
    link: str,
    point_in_link: ArrayLike,
    force: ArrayLike,
    torque: ArrayLike = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Joint torques ``J^T w`` of a world-frame wrench applied at a link point."""
    st = kinematics(joints, root, q)
    p = st.link_pos[link] + st.link_rot[link] @ np.asarray(point_in_link)
    child = {j.name: j.child for j in joints}
    f, m = np.asarray(force, dtype=float), np.asarray(torque, dtype=float)
    return np.array(
        [
            (
                st.joint_axis[n] @ (m + np.cross(p - st.joint_pos[n], f))
                if link in _descendants(joints, child[n])
                else 0.0
            )
            for n in dof_order(joints, root)
        ]
    )


def coupled_generalized_force(
    tau: np.ndarray,
    names: Sequence[str],
    q: Mapping[str, float],
    equalities: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, list[str]]:
    """Reduce ``tau`` onto the independent DOFs of ADR-0059 equalities.

    ``q[joint1] = polycoef . [1, q2, q2^2, ...]``; ``joint2`` gains
    ``tau[joint1] * dq1/dq2``. Returns the force and independent DOF names.
    """
    require(len(tau) == len(names), "tau must match names")
    out = dict(zip(names, map(float, tau), strict=True))
    for eq in equalities:
        j1, j2 = str(eq["joint1"]), str(eq["joint2"])
        coef = list(eq["polycoef"])  # type: ignore[call-overload]
        slope = sum(k * c * q[j2] ** (k - 1) for k, c in enumerate(coef) if k)
        out[j2] += out.pop(j1) * slope
    return np.array(list(out.values())), list(out)


def pinocchio_to_canonical_floating(q: Sequence[float], v: Sequence[float]):
    """Pinocchio free-flyer ``q=[p, qx qy qz qw]``, ``v=[lin, ang]`` -> canonical.

    Canonical is ``q=[p, qw qx qy qz]``, ``v=[lin, ang]`` (ordering only; the
    frame each velocity is expressed in is not converted).
    """
    require(len(q) == 7 and len(v) == 6, "free-flyer is 7 q and 6 v")
    return np.r_[q[:3], q[6], q[3:6]], np.asarray(v, dtype=float)


def drake_to_canonical_floating(q: Sequence[float], v: Sequence[float]):
    """Drake floating base ``q=[qw qx qy qz, p]``, ``v=[ang, lin]`` -> canonical."""
    require(len(q) == 7 and len(v) == 6, "floating base is 7 q and 6 v")
    return np.r_[q[4:7], q[:4]], np.r_[v[3:6], v[:3]]
