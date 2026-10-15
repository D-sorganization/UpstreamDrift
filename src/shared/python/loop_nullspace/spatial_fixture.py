"""Pinocchio spatial two-arm club chain with a WELD6 closure (NSX-12, fixture (c)).

Structure (matches the NSX-00 ``spatial_counts`` golden):

* coordinates ``root (6, optional) | lead arm (7) | trail arm (7)``;
* each arm is three links (upper arm, forearm, hand) with a 3-DOF shoulder,
  2-DOF elbow and 2-DOF wrist, all single-axis revolutes on generic unit axes;
* the optional floating root is a prismatic-xyz plus revolute z-y-x chain, so
  ``nq == nv`` and the snapshot ``q`` has exactly ``n`` entries;
* a rigid bare club is welded to the lead hand; the trail hand closes the loop
  onto the club grip frame (WELD6).  The grip frame is placed where the trail
  hand already is at the chosen posture, so the closure holds by construction;
* optional ground contact: two point feet (or four: two per foot) rigidly
  attached to the root.

``J_c`` follows ``closure_convention``: rows ``[lin; ang]`` of ``V_trail -
V_club`` in the grip frame.  Not a golf posture and not a golfer model.
"""

from __future__ import annotations

from dataclasses import dataclass

import importlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    FootContact,
    GripFrame,
    InputProvenance,
    LoopTopology,
    ModelProvenance,
    SpatialInertia,
)
from src.shared.python.loop_nullspace.fixtures import FixtureCase, load_golden

pin: Any = importlib.import_module("pinocchio")

Array = NDArray[np.float64]

_AXES = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (2 / 3, 2 / 3, 1 / 3),
    (1 / 3, -2 / 3, 2 / 3),
    (2 / 7, 3 / 7, 6 / 7),
    (6 / 7, -2 / 7, 3 / 7),
    (-3 / 7, 6 / 7, 2 / 7),
    (1 / 9, 4 / 9, 8 / 9),  # NSX-00 lists (2,-4,7)/9, which is not a unit vector
    (4 / 9, 1 / 9, -8 / 9),
]
_ELBOW = np.array([0.1, 0.0, -0.3])
_WRIST = np.array([0.2, 0.1, -0.25])
_SHOULDER_X = 0.1
_Q_LEAD = [0.3, -0.2, 0.4, -0.8, 0.5, 0.2, -0.3]
_Q_TRAIL = [-0.4, 0.3, 0.1, -0.9, -0.4, -0.2, 0.25]
_Q_ROOT = [0.05, -0.02, 0.01, 0.1, -0.05, 0.08]
_FEET = [np.array([0.2, 0.1, -1.0]), np.array([-0.3, -0.2, -1.0])]
_FOOT_SECOND_OFFSET = [np.array([0.2, 0.2, 0.0]), np.array([0.2, -0.1, 0.0])]
_FOOT_MU = 0.8
_CLUB_MASS, _CLUB_LENGTH, _CLUB_BUTT = 0.313, 1.0, 0.2
_HAND_MASS = 0.4
_SEED = 12557
_COORD_ARM = ("sh1", "sh2", "sh3", "el1", "el2", "wr1", "wr2")


def _dummy() -> pin.Inertia:
    return pin.Inertia(0.05, np.zeros(3), np.eye(3) * 1e-4)


def _link(mass: float, com: np.ndarray, length: float) -> pin.Inertia:
    side = mass * length**2 / 12.0
    return pin.Inertia(mass, com, np.eye(3) * (side + 1e-4))


@dataclass
class _Spatial:
    """A built Pinocchio model plus the ids needed to evaluate the closure."""

    model: pin.Model
    data: pin.Data
    lead_grip: int
    trail_grip: int
    feet_frames: list[int]
    q_star: Array
    n_root: int
    rot_grip: Array  # world <- grip rotation (fixed, from the closed posture)

    def closure_jacobian(self, q: Array) -> Array:
        """Canonical ``J_c`` (6, n) at ``q`` with the fixed grip rotation."""
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jac = [
            pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
            for fid in (self.trail_grip, self.lead_grip)
        ]
        return _to_grip(self.rot_grip) @ (jac[0] - jac[1])

    def closure_drift(self, q: Array, v: Array) -> Array:
        """``Jdot v`` from classical accelerations with ``a = 0``."""
        zero = np.zeros(self.model.nv)
        pin.forwardKinematics(self.model, self.data, q, v, zero)
        pin.updateFramePlacements(self.model, self.data)
        acc = []
        for fid in (self.trail_grip, self.lead_grip):
            motion = pin.getFrameClassicalAcceleration(
                self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED
            )
            acc.append(np.concatenate([motion.linear, motion.angular]))
        return _to_grip(self.rot_grip) @ (acc[0] - acc[1])


def _to_grip(rot: Array) -> Array:
    out = np.zeros((6, 6))
    out[:3, :3] = rot.T
    out[3:, 3:] = rot.T
    return out


def _add_arm(
    model: pin.Model,
    parent: int,
    shoulder: np.ndarray,
    axis_offset: int,
    hand_extra: tuple[pin.Inertia, pin.SE3] | None,
) -> int:
    """Append a 7-joint arm and return the wrist-2 (hand) joint id."""
    bodies = [
        _dummy(),
        _dummy(),
        _link(2.0, _ELBOW / 2.0, 0.3),
        _dummy(),
        _link(1.6, _WRIST / 2.0, 0.3),
        _dummy(),
        _link(_HAND_MASS, np.array([0.0, 0.0, -0.08]), 0.16),
    ]
    offsets = [
        shoulder,
        0 * shoulder,
        0 * shoulder,
        _ELBOW,
        0 * shoulder,
        _WRIST,
        0 * shoulder,
    ]
    joint = parent
    for k in range(7):
        axis = np.array(_AXES[(axis_offset + k) % len(_AXES)], dtype=float)
        placement = pin.SE3(np.eye(3), np.asarray(offsets[k], dtype=float))
        joint = model.addJoint(
            joint, pin.JointModelRevoluteUnaligned(axis), placement, f"j{joint}_{k}"
        )
        model.appendBodyToJoint(joint, bodies[k], pin.SE3.Identity())
    if hand_extra is not None:
        model.appendBodyToJoint(joint, hand_extra[0], hand_extra[1])
    return joint


def _club_inertia() -> pin.Inertia:
    """Bare club in the grip frame: butt ``_CLUB_BUTT`` behind the grip on z."""
    com = np.array([0.0, 0.0, _CLUB_LENGTH / 2.0 - _CLUB_BUTT])
    side = _CLUB_MASS * _CLUB_LENGTH**2 / 12.0
    return pin.Inertia(_CLUB_MASS, com, np.diag([side, side, 1e-4]))


def _foot_points(points_per_foot: int) -> list[Array]:
    """Rigid foot points: one per foot, or a second one offset in the ground plane.

    Matches ``FEET`` / ``FEET4`` of the NSX-00 ``spatial_counts`` derivation.
    """
    if points_per_foot == 1:
        return list(_FEET)
    return [
        _FEET[0],
        _FEET[0] + _FOOT_SECOND_OFFSET[0],
        _FEET[1],
        _FEET[1] + _FOOT_SECOND_OFFSET[1],
    ]


def _build(
    floating_root: bool, feet: bool, grip: pin.SE3 | None, points_per_foot: int = 1
) -> _Spatial:
    model = pin.Model()
    root = 0
    if floating_root:
        comp = pin.JointModelComposite()
        for joint in (
            pin.JointModelPX(),
            pin.JointModelPY(),
            pin.JointModelPZ(),
            pin.JointModelRZ(),
            pin.JointModelRY(),
            pin.JointModelRX(),
        ):
            comp.addJoint(joint)
        root = model.addJoint(0, comp, pin.SE3.Identity(), "root")
        model.appendBodyToJoint(
            root,
            pin.Inertia(30.0, np.array([0.0, 0.0, -0.2]), np.eye(3) * 1.0),
            pin.SE3.Identity(),
        )
    club = (_club_inertia(), grip) if grip is not None else None
    lead = _add_arm(model, root, np.array([-_SHOULDER_X, 0.0, 0.0]), 0, club)
    trail = _add_arm(model, root, np.array([_SHOULDER_X, 0.0, 0.0]), 3, None)
    lead_grip_pose = grip if grip is not None else pin.SE3.Identity()
    lead_grip = model.addFrame(
        pin.Frame("lead_grip", lead, 0, lead_grip_pose, pin.FrameType.OP_FRAME)
    )
    trail_grip = model.addFrame(
        pin.Frame("trail_grip", trail, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME)
    )
    feet_frames = []
    if feet:
        for k, pos in enumerate(_foot_points(points_per_foot)):
            feet_frames.append(
                model.addFrame(
                    pin.Frame(
                        f"foot_{k}",
                        root,
                        0,
                        pin.SE3(np.eye(3), pos),
                        pin.FrameType.OP_FRAME,
                    )
                )
            )
    n_root = 6 if floating_root else 0
    q_star = np.array([*(_Q_ROOT if floating_root else []), *_Q_LEAD, *_Q_TRAIL])
    return _Spatial(
        model,
        model.createData(),
        lead_grip,
        trail_grip,
        feet_frames,
        q_star,
        n_root,
        np.eye(3),
    )


def _build_closed(
    floating_root: bool, feet: bool, points_per_foot: int = 1
) -> _Spatial:
    """Two-pass build: place the grip frame where the trail hand already is."""
    probe = _build(floating_root, feet, None, points_per_foot)
    pin.forwardKinematics(probe.model, probe.data, probe.q_star)
    pin.updateFramePlacements(probe.model, probe.data)
    o_lead = probe.data.oMf[probe.lead_grip]
    o_trail = probe.data.oMf[probe.trail_grip]
    grip = o_lead.inverse() * o_trail
    sp = _build(floating_root, feet, grip, points_per_foot)
    pin.forwardKinematics(sp.model, sp.data, sp.q_star)
    pin.updateFramePlacements(sp.model, sp.data)
    sp.rot_grip = np.array(sp.data.oMf[sp.lead_grip].rotation)
    return sp


def _admissible_rates(sp: _Spatial, rng: np.random.Generator) -> tuple[Array, Array]:
    jc = sp.closure_jacobian(sp.q_star)
    pinv = np.linalg.pinv(jc, rcond=1e-12)
    v = rng.uniform(-1.0, 1.0, sp.model.nv)
    v = v - pinv @ (jc @ v)
    drift = sp.closure_drift(sp.q_star, v)
    a = rng.uniform(-1.0, 1.0, sp.model.nv)
    a = a - pinv @ (jc @ a + drift)
    return v, a


def _foot_contacts(sp: _Spatial) -> tuple[tuple[FootContact, ...], Array]:
    contacts, rows = [], []
    for k, fid in enumerate(sp.feet_frames):
        pos = np.array(sp.data.oMf[fid].translation)
        contacts.append(
            FootContact(f"foot_{k}", pos, np.array([0.0, 0.0, 1.0]), _FOOT_MU)
        )
        rows.append(
            pin.getFrameJacobian(sp.model, sp.data, fid, pin.LOCAL_WORLD_ALIGNED)[:3]
        )
    return tuple(contacts), np.vstack(rows)


def _welded_inertias() -> tuple[SpatialInertia, ...]:
    hand = SpatialInertia(_HAND_MASS, np.zeros(3), np.eye(3) * 2e-4)
    return (hand, hand)


def build_case(
    *, floating_root: bool = False, feet: bool = False, foot_points: int = 1
) -> FixtureCase:
    """Build the spatial snapshot and attach its NSX-00 ``spatial_counts`` golden.

    Raises:
        ValueError: If ``feet`` is requested without ``floating_root`` (the
            feet are rigidly attached to the root), or ``foot_points`` is not
            1 or 2 (points per foot; 2 gives the four-point NSX-60 variant).
    """
    if feet and not floating_root:
        raise ValueError("feet require floating_root=True (feet attach to the root)")
    if foot_points not in (1, 2):
        raise ValueError(f"foot_points must be 1 or 2, got {foot_points!r}")
    sp = _build_closed(floating_root, feet, foot_points)
    model, data, q = sp.model, sp.data, sp.q_star
    v, a = _admissible_rates(sp, np.random.default_rng(_SEED))
    jc = sp.closure_jacobian(q)
    drift = sp.closure_drift(q, v)
    mass = np.array(pin.crba(model, data, q))
    mass = np.triu(mass) + np.triu(mass, 1).T
    h = np.array(pin.nonLinearEffects(model, data, q, v))
    n, n_root = model.nv, sp.n_root
    names = (
        *(["root_x", "root_y", "root_z", "root_rz", "root_ry", "root_rx"][:n_root]),
        *(f"lead_{c}" for c in _COORD_ARM),
        *(f"trail_{c}" for c in _COORD_ARM),
    )
    actuated = tuple(range(n_root, n))
    topo = LoopTopology(
        ClosureKind.WELD6,
        "trail",
        "lead",
        actuated,
        tuple(range(n_root)),
        tuple(range(n_root, n)),
        names,
    )
    contacts, j_g = _foot_contacts(sp) if feet else ((), None)
    grip_rot = sp.rot_grip
    snap = ClosedLoopSnapshot(
        q=q,
        v=v,
        a=a,
        M=mass,
        h=h,
        S=np.eye(n)[list(actuated)],
        J_c=jc,
        Jdot_v=drift,
        grip_frame=GripFrame(grip_rot, np.array(data.oMf[sp.lead_grip].translation)),
        bare_club=SpatialInertia(
            _CLUB_MASS,
            np.array([0.0, 0.0, _CLUB_LENGTH / 2.0 - _CLUB_BUTT]),
            np.diag([_CLUB_MASS * _CLUB_LENGTH**2 / 12.0] * 2 + [1e-4]),
        ),
        welded_hand_inertias=_welded_inertias(),
        model_provenance=ModelProvenance(
            "spatial_two_arm_club_pinocchio", pin.__version__
        ),
        input_provenance=InputProvenance(
            "synthetic",
            "nsx-00-golden:spatial_counts",
            f"floating_root={floating_root} feet={feet} seed={_SEED}",
        ),
        topology=topo,
        J_g=j_g,
        foot_contacts=contacts,
    )
    variant = (
        (
            "floating_root_four_point_feet"
            if foot_points == 2
            else "floating_root_two_point_feet"
        )
        if feet
        else "floating_root_no_feet"
        if floating_root
        else "fixed_base_arms_only"
    )
    return FixtureCase(snap, load_golden("spatial_counts", variant))


def jdot_v_finite_difference_error(*, floating_root: bool) -> float:
    """Max abs error of ``Jdot v`` against a central difference of ``J_c(q+tv) v``.

    The grip rotation is held fixed, so this checks the drift exactly (it does
    not rely on ``J_c v = 0``).
    """
    sp = _build_closed(floating_root, False)
    rng = np.random.default_rng(_SEED)
    v = rng.uniform(-1.0, 1.0, sp.model.nv)
    q, eps = sp.q_star, 1e-6
    plus = sp.closure_jacobian(pin.integrate(sp.model, q, eps * v)) @ v
    minus = sp.closure_jacobian(pin.integrate(sp.model, q, -eps * v)) @ v
    return float(np.max(np.abs((plus - minus) / (2 * eps) - sp.closure_drift(q, v))))


def closure_jacobian_finite_difference_error(*, floating_root: bool) -> float:
    """Max abs error of ``J_c`` against a central difference of the weld error.

    The error is ``log6(lead_grip^-1 trail_grip)`` (translation, rotation vector
    in the lead grip frame); at the closed posture its derivative is the
    canonical ``J_c``, which is an independent route from the frame Jacobians.
    """
    sp = _build_closed(floating_root, False)
    q, eps = sp.q_star, 1e-6

    def weld_error(qq: Array) -> Array:
        pin.forwardKinematics(sp.model, sp.data, qq)
        pin.updateFramePlacements(sp.model, sp.data)
        rel = sp.data.oMf[sp.lead_grip].inverse() * sp.data.oMf[sp.trail_grip]
        return np.asarray(pin.log6(rel).vector)

    cols = []
    for j in range(sp.model.nv):
        step = np.zeros(sp.model.nv)
        step[j] = eps
        plus = weld_error(pin.integrate(sp.model, q, step))
        minus = weld_error(pin.integrate(sp.model, q, -step))
        cols.append((plus - minus) / (2 * eps))
    return float(np.max(np.abs(np.array(cols).T - sp.closure_jacobian(q))))
