"""Refit the arms of a pose to moved wrist joints with the club held still (#15255).

``skinned_hand_wrists.apply_skinned_hand_wrists`` moves the wrist joints on the
club. A pose fitted before the move then carries the club centimetres away and
leaves the trail hand off the grip. :func:`refit_arms_to_club` solves the
eighteen arm coordinates again, by bounded least squares on the MuJoCo model:

* the club frame returns to where the original pose had it;
* the trail hand's weld closes;
* with the freedom left over, each forearm turns toward the direction the
  skinned hand's wrist is posed in (:func:`skinned_club_hands.forearm_reach_m`),
  while each elbow's hinge axis is held near where the incoming pose had it, so
  an upper arm cannot roll over to line a forearm up (#15297).

Kinematics only. Nothing here says the refitted pose is a good address, and the
angle left between a forearm and its hand is reported, not hidden.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.skinned_club_hands import forearm_reach_m
from src.shared.python.motion_matching.wrist_scapula_rom import WRIST_SCAPULA_ROM

Array = NDArray[np.float64]

#: A coordinate is an arm coordinate when it starts with ``L`` or ``R`` and
#: carries one of these: elbow, forearm, scapula, shoulder, wrist.
ARM_COORDINATE_KEYS = ("EInput", "FInput", "ScapInput", "SInput", "WInput")
#: (wrist joint, elbow joint, upper-arm twist joint) of each arm. The anchors
#: of the first two give the forearm's line; the last turns about the upper arm.
_ARM_JOINTS = {
    "lead": ("LWInputX", "LEInput", "LSInputZ"),
    "trail": ("RWInputX", "REInput", "RSInputZ"),
}
_WRIST_KEY = "WInput"
_SITE_ON_HAND, _SITE_ON_CLUB = "native_closure_a", "native_closure_b"
#: Residual scales: a millimetre of position counts like 5 mrad of turn.
_POSITION_SCALE_M, _TURN_SCALE_RAD = 1e-3, 5e-3
#: Pull of each arm coordinate toward the pose it came from (per radian).
_STAY_WEIGHT = 0.05
#: The same for the shoulder blades, held much harder: left free, the solver
#: shrugs a shoulder up to line a forearm with its hand.
_SCAPULA_KEY, _SCAPULA_STAY_WEIGHT = "ScapInput", 2.0
_MAX_EVALUATIONS = 600
#: Factor on the club and weld errors in the second, closing pass.
_CLOSING_STIFFNESS = 1000.0
#: Address posture (deg, flexion positive), owner 2026-11-18 (#15277): the lower
#: back near neutral and a slight forward bend in the thoracic region, not an
#: arch in the lumbar joint. Pass ``address_posture()`` as ``posture``.
ADDRESS_POSTURE_DEG: dict[str, float] = {
    "lumbar_flexion": 2.0,
    "thoracic_flexion": 10.0,
}
#: Default pull of a posture coordinate toward its target (per radian): a
#: degree off counts like a third of a millimetre of club error, far more than
#: the forearm pull, far less than closing the hands.
POSTURE_WEIGHT = 20.0


@dataclass(frozen=True)
class ArmRefitSettings:
    """Weights and bounds of the refit that are not a call's own choice (#15297).

    ``alignment_weight`` is the default pull of a forearm toward its hand's
    (per radian, a millimetre of club error counting 1). ``twist_stay_weight``
    pulls each elbow hinge axis toward its direction in the incoming pose (per
    radian): at 2.0 the forearm pull moved an upper arm's twist by under 3
    degrees where, unheld, it rolled one over by 70. ``evidence_wrist_bounds``
    widens a wrist coordinate's range to the evidence range of #14795 where
    the spec carries a narrower (legacy) one.
    """

    alignment_weight: float = 1.0
    twist_stay_weight: float = 2.0
    evidence_wrist_bounds: bool = True

    def __post_init__(self) -> None:
        for name in ("alignment_weight", "twist_stay_weight"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and non-negative")


class _Choices(NamedTuple):
    """The keyword choices of one :func:`refit_arms_to_club` call."""

    alignment_weight: float | None = None
    hold_club: bool = True
    also_free: Sequence[str] = ()
    posture: Mapping[str, float] | None = None
    posture_weight: float = POSTURE_WEIGHT


@dataclass(frozen=True)
class ArmRefit:
    """A refitted pose and how well it holds.

    ``q`` is in the spec's ``coordinate_order`` (radians, metres). Offsets and
    closure are of the refitted pose; the ``*_before_*`` fields are the same
    quantities with the original coordinates on the new model.
    ``forearm_to_cuff_deg`` is the angle between each model forearm and the
    forearm direction of the skinned hand it carries. ``upper_arm_twist_deg``
    is each upper arm's turn about its own long axis from the incoming pose,
    positive in the sense of its twist coordinate (``*SInputZ``).
    """

    q: Array
    club_offset_m: float
    club_turn_deg: float
    closure_m: float
    closure_deg: float
    club_offset_before_m: float
    closure_before_m: float
    forearm_to_cuff_deg: dict[str, float]
    upper_arm_twist_deg: dict[str, float]


class _Model:
    """A spec's MuJoCo model with the few kinematic readings the refit needs."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        xml, _ = export_full_body_mjcf(json.dumps(spec).encode())
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.order = list(spec["coordinate_order"])
        self._address = [
            int(self.model.jnt_qposadr[self.model.joint(name).id])
            for name in self.order
        ]
        self._weld = np.asarray(spec["closure"]["placement_b"], dtype=float)

    def pose(self, q: Array) -> None:
        self.data.qpos[self._address] = q
        mujoco.mj_kinematics(self.model, self.data)

    def _site(self, name: str) -> tuple[Array, Array]:
        site = self.data.site(name)
        return np.array(site.xmat).reshape(3, 3), np.array(site.xpos)

    def club_frame(self) -> tuple[Array, Array]:
        """World rotation and origin of the spec's club frame."""
        rotation, point = self._site(_SITE_ON_CLUB)
        world = rotation @ self._weld[:3, :3].T
        return world, point - world @ self._weld[:3, 3]

    def closure(self) -> tuple[Array, Array]:
        """Weld error: hand point minus club point (m), and the turn (rotvec)."""
        hand_rotation, hand_point = self._site(_SITE_ON_HAND)
        club_rotation, club_point = self._site(_SITE_ON_CLUB)
        turn = Rotation.from_matrix(club_rotation.T @ hand_rotation).as_rotvec()
        return hand_point - club_point, turn

    def forearm_lines(self) -> dict[str, Array]:
        """Unit direction from wrist joint to elbow joint, per hand, world."""
        lines = {}
        for side, (wrist, elbow, _) in _ARM_JOINTS.items():
            line = self.data.joint(elbow).xanchor - self.data.joint(wrist).xanchor
            lines[side] = line / np.linalg.norm(line)
        return lines

    def upper_arms(self) -> dict[str, tuple[Array, Array]]:
        """Per hand: unit twist axis of the upper arm and elbow hinge axis, world."""
        return {
            side: (
                np.array(self.data.joint(twist).xaxis),
                np.array(self.data.joint(elbow).xaxis),
            )
            for side, (_, elbow, twist) in _ARM_JOINTS.items()
        }


def _angle_deg(a: Array, b: Array) -> float:
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def _twist_deg(before: tuple[Array, Array], after: tuple[Array, Array]) -> float:
    """Turn of an upper arm about its long axis between two poses, swing removed."""
    frames = [
        np.column_stack([long, hinge, np.cross(long, hinge)])
        for long, hinge in (before, after)
    ]
    quat = Rotation.from_matrix(frames[1] @ frames[0].T).as_quat()  # x, y, z, w
    angle = math.degrees(2.0 * math.atan2(float(quat[:3] @ before[0]), float(quat[3])))
    return (angle + 180.0) % 360.0 - 180.0


def _arm_indices(order: list[str]) -> list[int]:
    return [
        i
        for i, name in enumerate(order)
        if name[0] in "LR" and any(key in name for key in ARM_COORDINATE_KEYS)
    ]


def _checked(
    spec_from: Mapping[str, Any],
    spec_to: Mapping[str, Any],
    q: Array,
    choices: _Choices,
) -> tuple[Array, Any, dict[str, float]]:
    """The start pose, the club and the posture, after the refit's preconditions."""
    order = list(spec_to["coordinate_order"])
    start = np.asarray(q, dtype=float)
    posture = dict(choices.posture or {})
    if order != list(spec_from["coordinate_order"]):
        raise ValueError("the two specs do not share a coordinate order")
    if start.shape != (len(order),) or not np.isfinite(start).all():
        raise ValueError(f"q must hold {len(order)} finite coordinate values")
    if choices.alignment_weight is not None and not choices.alignment_weight >= 0.0:
        raise ValueError("alignment_weight must be non-negative")
    unknown = [name for name in choices.also_free if name not in order]
    if unknown:
        raise ValueError(f"also_free names unknown coordinates: {unknown}")
    unknown = [name for name in posture if name not in order]
    if unknown or not np.isfinite(list(posture.values())).all():
        raise ValueError(
            f"posture needs finite targets of known coordinates: {unknown}"
        )
    if not choices.posture_weight >= 0.0:
        raise ValueError("posture_weight must be non-negative")
    club = assembly_from_spec(spec_to)
    if club is None:
        raise ValueError("spec has no club to hold")
    return start, club, posture


def _free_coordinates(
    spec: Mapping[str, Any], also_free: Sequence[str], evidence_wrists: bool
) -> tuple[list[int], Array, Array, Array]:
    """Indices solved for, their bounds (rad) and their pull toward the start."""
    order = list(spec["coordinate_order"])
    free = sorted({*_arm_indices(order), *(order.index(n) for n in also_free)})
    # A trunk coordinate has no hard range; its soft limits bound it here.
    ranges = {
        **spec.get("soft_limits_deg", {}),
        **spec.get("coordinate_ranges_deg", {}),
    }
    for name, rom in WRIST_SCAPULA_ROM.items() if evidence_wrists else ():
        if _WRIST_KEY in name and name in ranges:
            low, high = ranges[name]
            ranges[name] = (min(low, rom.lower_deg), max(high, rom.upper_deg))
    low, high = np.radians(
        [ranges.get(order[i], (-180.0, 180.0)) for i in free], dtype=float
    ).T
    stay = np.array(
        [
            _SCAPULA_STAY_WEIGHT if _SCAPULA_KEY in order[i] else _STAY_WEIGHT
            for i in free
        ]
    )
    return free, low, high, stay


def _report(
    q: Array,
    after: tuple[Array, Array, Array, Array],
    before: tuple[Array, Array],
    target: _Model,
    aims: tuple[Mapping[str, Array], Mapping[str, tuple[Array, Array]]],
) -> ArmRefit:
    """The refit's result; ``target`` is still posed at the solution.

    ``aims`` holds the cuff directions and the incoming pose's upper arms.
    """
    offset, turn, gap, twist = after
    cuffs, incoming = aims
    lines, upper = target.forearm_lines(), target.upper_arms()
    return ArmRefit(
        q=q,
        club_offset_m=float(np.linalg.norm(offset)),
        club_turn_deg=float(np.degrees(np.linalg.norm(turn))),
        closure_m=float(np.linalg.norm(gap)),
        closure_deg=float(np.degrees(np.linalg.norm(twist))),
        club_offset_before_m=float(np.linalg.norm(before[0])),
        closure_before_m=float(np.linalg.norm(before[1])),
        forearm_to_cuff_deg={
            side: _angle_deg(lines[side], cuffs[side]) for side in cuffs
        },
        upper_arm_twist_deg={
            side: _twist_deg(incoming[side], upper[side]) for side in incoming
        },
    )


def address_posture() -> dict[str, float]:
    """``ADDRESS_POSTURE_DEG`` in radians, as ``refit_arms_to_club`` takes it."""
    return {name: math.radians(deg) for name, deg in ADDRESS_POSTURE_DEG.items()}


def refit_arms_to_club(
    spec_from: Mapping[str, Any],
    spec_to: Mapping[str, Any],
    q: Array,
    *,
    alignment_weight: float | None = None,
    hold_club: bool = True,
    also_free: Sequence[str] = (),
    posture: Mapping[str, float] | None = None,
    posture_weight: float = POSTURE_WEIGHT,
) -> ArmRefit:
    """Arm coordinates on ``spec_to`` that put the club where ``q`` had it.

    The club's target is its frame on ``spec_from`` at ``q``, carried by the
    lead arm. ``alignment_weight`` scales the pull of each forearm toward its
    skinned hand's forearm direction (per radian, against a millimetre of
    club error counted as 1; ``None`` takes the settings' default); ``0``
    leaves the spare freedom near ``q``. Either way each elbow's hinge axis is
    held near its direction at ``q`` (``ArmRefitSettings.twist_stay_weight``).
    ``hold_club=False`` drops the club target, so both arms may carry the club
    to wherever the trail hand closes nearest ``q``: the way to close a pose
    that was never closed, such as a spec's address seed.

    ``also_free`` names further coordinates solved with the arms, for
    example the trunk's flexion: the skinned wrists sit nearer the shoulders
    than the joints they replace, so arms alone must bend at the elbow.

    ``posture`` gives coordinates an angle (rad) to sit at (#15277): solved
    with the arms, pulled there by ``posture_weight`` in place of toward ``q``.

    Preconditions: both specs share ``coordinate_order`` and hold a club;
    ``q`` has one finite value per coordinate; ``alignment_weight >= 0``;
    every ``also_free`` and ``posture`` name is a coordinate, posture targets
    are finite and ``posture_weight >= 0`` (``ValueError`` otherwise).
    Postconditions: only arm, ``also_free`` and ``posture`` coordinates differ
    from ``q``; each lies inside its ``coordinate_ranges_deg`` (a wrist's
    widened to its evidence range, see :class:`ArmRefitSettings`) or, without
    one, its ``soft_limits_deg``.
    """
    choices = _Choices(alignment_weight, hold_club, also_free, posture, posture_weight)
    return _refit(ArmRefitSettings(), spec_from, spec_to, q, choices)


def refit_arms_with(
    settings: ArmRefitSettings,
    spec_from: Mapping[str, Any],
    spec_to: Mapping[str, Any],
    q: Array,
    **choices: Any,
) -> ArmRefit:
    """:func:`refit_arms_to_club`, with its keyword choices, under ``settings``."""
    return _refit(settings, spec_from, spec_to, q, _Choices(**choices))


def _refit(
    settings: ArmRefitSettings,
    spec_from: Mapping[str, Any],
    spec_to: Mapping[str, Any],
    q: Array,
    choices: _Choices,
) -> ArmRefit:
    start, club, posture = _checked(spec_from, spec_to, q, choices)
    alignment = choices.alignment_weight
    alignment = settings.alignment_weight if alignment is None else alignment
    source, target = _Model(spec_from), _Model(spec_to)
    source.pose(start)
    club_rotation, club_origin = source.club_frame()
    incoming = source.upper_arms()
    cuffs = {
        side: club_rotation @ along
        for side, (_, along) in forearm_reach_m(club).items()
    }
    arms, low, high, stay = _free_coordinates(
        spec_to, (*choices.also_free, *posture), settings.evidence_wrist_bounds
    )
    order = list(spec_to["coordinate_order"])
    rest = start[arms].copy()  # where each solved coordinate is pulled to
    for name, angle in posture.items():
        at = arms.index(order.index(name))
        rest[at], stay[at] = angle, choices.posture_weight

    def errors(x: Array) -> tuple[Array, Array, Array, Array]:
        pose = start.copy()
        pose[arms] = x
        target.pose(pose)
        rotation, origin = target.club_frame()
        turn = Rotation.from_matrix(club_rotation.T @ rotation).as_rotvec()
        gap, twist = target.closure()
        return origin - club_origin, turn, gap, twist

    held = 1.0 if choices.hold_club else 0.0

    def residuals(x: Array, stiffness: float) -> Array:
        offset, turn, gap, twist = errors(x)
        lines = target.forearm_lines()
        # The difference of unit vectors grows with the angle all the way to
        # 180 degrees; a cross product would fall again past 90.
        pull = [lines[side] - cuffs[side] for side in sorted(cuffs)]
        hinges = target.upper_arms()
        roll = [hinges[side][1] - incoming[side][1] for side in sorted(incoming)]
        return np.concatenate(
            [
                held * stiffness * offset / _POSITION_SCALE_M,
                held * stiffness * turn / _TURN_SCALE_RAD,
                stiffness * gap / _POSITION_SCALE_M,
                stiffness * twist / _TURN_SCALE_RAD,
                np.concatenate(pull) * alignment,
                np.concatenate(roll) * settings.twist_stay_weight,
                (x - rest) * stay,
            ]
        )

    first = np.clip(start[arms], low + 1e-9, high - 1e-9)
    offset_before, _, gap_before, _ = errors(start[arms])
    # Two passes: the first balances closing against the forearm pull, the
    # second holds that balance and closes the club and the weld tightly.
    solved = least_squares(
        residuals, first, bounds=(low, high), max_nfev=_MAX_EVALUATIONS, args=(1.0,)
    )
    solved = least_squares(
        residuals,
        np.clip(solved.x, low + 1e-9, high - 1e-9),
        bounds=(low, high),
        max_nfev=_MAX_EVALUATIONS,
        args=(_CLOSING_STIFFNESS,),
    )
    refitted = start.copy()
    refitted[arms] = solved.x
    after = errors(solved.x)
    before = (offset_before, gap_before)
    return _report(refitted, after, before, target, (cuffs, incoming))
