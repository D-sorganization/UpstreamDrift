"""Shared multibody model provider for swing optimization backends.

Part of epic #8390 (B1/#8396) — the prerequisite seam for engine-backed
swing solvers (#8397 Drake, #8398 CasADi, #8399 Crocoddyl, #8400 batch).

The flagship ``SwingOptimizer`` parameterizes the swing over the seven
``JOINTS`` of :mod:`._swing_kinematics` but historically carried no
multibody model (only a lumped scalar inertia). This module renders that
same seven-DOF chain as a canonical ``SkeletonRig`` — with segment offsets
and joint limits derived from :class:`GolferModel` anthropometrics — and
reuses the motion-pipeline URDF bridge so every engine consumes one model
source (no engine-specific loaders, per CROSS_ENGINE_PARITY_SPEC).

Scope note: segment geometry uses documented anthropometric fractions.
Segment *inertials* come in two flavours (#9755):

- ``"anthropometric"`` (default): per-link mass, COM and inertia derived
  from :class:`GolferModel` mass ratios and :class:`ClubModel` component
  masses by :func:`swing_link_inertials`. This is what the dynamic
  backends (CasADi RNEA, Crocoddyl DDP, bioptim OCP) must consume so that
  torque limits, injury surrogates and clubhead speeds are real-golfer
  numbers.
- ``"placeholder"``: the bridge's generic ``mass=1.0, I=1e-2`` conditioning
  values. Kept for kinematic matching and for behaviour-preserving tests of
  the symbolic RNEA against its original oracle.
"""

from __future__ import annotations

import tempfile
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.model_bridge import (
    LinkInertial,
    rig_root_link_name,
    rig_to_urdf,
)
from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import ClubModel, GolferModel

__all__ = [
    "SWING_RIG_ID",
    "InertialSource",
    "LinkInertial",
    "build_drake_plant",
    "build_mujoco_model",
    "build_pinocchio_model",
    "build_swing_rig",
    "placeholder_link_inertials",
    "swing_joint_limits",
    "swing_link_inertials",
    "swing_urdf",
]

InertialSource = Literal["anthropometric", "placeholder"]

SWING_RIG_ID = "golf_swing_7dof"

# Anthropometric segment fractions (of golfer height / arm length) used for
# joint placement along the chain. Documented, not tuned: they place joints
# at plausible segment boundaries so Jacobians are well-conditioned.
_PELVIS_HEIGHT_FRACTION = 0.53
_SHOULDER_OFFSET_FRACTION = 0.18
_UPPER_ARM_FRACTION = 0.55

# Rotation axes per swing DOF: axial rotations about Z, swing/hinge DOFs
# about Y, wrist cock about X — one revolute DOF per JOINTS entry.
_JOINT_AXES: dict[str, str] = {
    "hip_rotation": "Z",
    "trunk_rotation": "Z",
    "shoulder_horizontal": "Z",
    "shoulder_vertical": "Y",
    "elbow_flexion": "Y",
    "wrist_cock": "X",
    "wrist_rotation": "Z",
}


def swing_joint_limits(golfer: GolferModel) -> dict[str, tuple[float, float]]:
    """Per-DOF (lower, upper) limits in radians from golfer ROMs."""
    if golfer is None:
        raise ValueError("golfer must be provided")
    return {
        "hip_rotation": golfer.hip_rom,
        "trunk_rotation": golfer.trunk_rotation_rom,
        "shoulder_horizontal": golfer.shoulder_rom,
        "shoulder_vertical": golfer.shoulder_rom,
        "elbow_flexion": golfer.elbow_rom,
        "wrist_cock": golfer.wrist_rom,
        "wrist_rotation": golfer.wrist_rom,
    }


def build_swing_rig(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
) -> SkeletonRig:
    """Build the canonical seven-DOF swing chain as a ``SkeletonRig``.

    DOF order matches ``_swing_kinematics.JOINTS`` exactly, so trajectories
    produced by ``SwingOptimizer`` map 1:1 onto model coordinates.
    """
    golfer = golfer or GolferModel()
    club = club or ClubModel()
    limits = swing_joint_limits(golfer)

    pelvis_z = _PELVIS_HEIGHT_FRACTION * golfer.height
    shoulder_z = golfer.trunk_length
    upper_arm = _UPPER_ARM_FRACTION * golfer.arm_length
    forearm = golfer.arm_length - upper_arm

    # Offsets are from the parent joint, along the chain. The club length
    # extends the terminal wrist segment so end-of-chain kinematics see the
    # clubhead radius.
    offsets: dict[str, list[float]] = {
        "hip_rotation": [0.0, 0.0, pelvis_z],
        "trunk_rotation": [0.0, 0.0, shoulder_z],
        "shoulder_horizontal": [
            0.0,
            _SHOULDER_OFFSET_FRACTION * golfer.height / 2.0,
            0.0,
        ],
        "shoulder_vertical": [0.0, 0.0, 0.0],
        "elbow_flexion": [0.0, 0.0, -upper_arm],
        "wrist_cock": [0.0, 0.0, -forearm],
        "wrist_rotation": [0.0, 0.0, -club.total_length],
    }

    joints: dict[str, JointDef] = {}
    for i, name in enumerate(JOINTS):
        parent = JOINTS[i - 1] if i > 0 else None
        children = [JOINTS[i + 1]] if i < len(JOINTS) - 1 else []
        lower, upper = limits[name]
        joints[name] = JointDef(
            name=name,
            parent=parent,
            children=children,
            tpose_offset=offsets[name],
            axes=[_JOINT_AXES[name]],  # type: ignore[list-item]
            limits=[JointLimit(lower=float(lower), upper=float(upper))],
        )
    return SkeletonRig(id=SWING_RIG_ID, joints=joints, root_joint=JOINTS[0])


# Anthropometric inertial model (#9755). Fractions follow Winter (2009)
# segment tables, rounded: the whole arm is ~5 % of body mass split
# upper arm : forearm : hand = 0.56 : 0.32 : 0.12, and the trunk ratio is
# split evenly between the pelvis/abdomen link (hip DOF) and the thorax /
# shoulder-girdle link (trunk DOF). Segments are modelled as uniform
# cylinders (limbs, trunk) or a solid sphere (clubhead) so every diagonal
# moment is strictly positive -- MuJoCo rejects zero-inertia moving bodies.
_UPPER_ARM_MASS_FRACTION = 0.56
_FOREARM_MASS_FRACTION = 0.32
_HAND_MASS_FRACTION = 0.12
_PELVIS_TRUNK_SPLIT = 0.5
_TRUNK_RADIUS_FRACTION = 0.08  # of golfer height (~0.14 m)
_LIMB_RADIUS = 0.04  # m, upper arm / forearm
_HAND_LENGTH = 0.08  # m, along the club axis from the wrist
_GRIP_LENGTH = 0.25  # m, along the shaft from the butt
_CLUBHEAD_RADIUS = 0.045  # m, equivalent solid sphere
_CLUB_SHAFT_RADIUS = 0.006  # m
# Zero-length "joint carrier" links (co-located DOFs) keep the bridge's
# dummy conditioning inertials.
_CARRIER_MASS = 1e-2
_CARRIER_INERTIA = 1e-4


def _cylinder_inertia(
    mass: float, length: float, radius: float, axis: int
) -> tuple[float, float, float, float, float, float]:
    """Uniform solid cylinder about its COM, symmetry axis ``axis`` (0/1/2)."""
    axial = 0.5 * mass * radius**2
    transverse = mass * (3.0 * radius**2 + length**2) / 12.0
    diag = [transverse, transverse, transverse]
    diag[axis] = axial
    return (diag[0], diag[1], diag[2], 0.0, 0.0, 0.0)


def _sphere_inertia(
    mass: float, radius: float
) -> tuple[float, float, float, float, float, float]:
    moment = 0.4 * mass * radius**2
    return (moment, moment, moment, 0.0, 0.0, 0.0)


def _combine_point_and_rod_masses(
    parts: list[tuple[float, float, float]],
) -> LinkInertial:
    """Combine collinear parts along -Z into one link inertial.

    Each part is ``(mass, z_center, length)`` where ``length`` is the extent
    of a thin rod centred at ``z_center`` (0 for a point mass). The rod is
    given a small radius so its axial moment stays positive.
    """
    total = sum(m for m, _z, _l in parts)
    z_com = sum(m * z for m, z, _l in parts) / total
    ixx = iyy = izz = 0.0
    for mass, z_center, length in parts:
        rod = _cylinder_inertia(mass, length, _CLUB_SHAFT_RADIUS, axis=2)
        d = z_center - z_com
        ixx += rod[0] + mass * d**2
        iyy += rod[1] + mass * d**2
        izz += rod[2]
    return LinkInertial(
        mass=total, com=(0.0, 0.0, z_com), inertia=(ixx, iyy, izz, 0.0, 0.0, 0.0)
    )


def swing_link_inertials(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
) -> dict[str, LinkInertial]:
    """Per-joint physical inertials for the seven-DOF swing chain (#9755).

    Link ``i`` (named after joint ``i``) is the body between joint ``i`` and
    joint ``i + 1`` in :func:`build_swing_rig`'s geometry:

    - ``hip_rotation``: pelvis + abdomen, along +Z to the trunk joint.
    - ``trunk_rotation``: thorax + shoulder girdle, along +Y to the shoulder.
    - ``shoulder_horizontal``: zero-length carrier (dummy inertial).
    - ``shoulder_vertical``: upper arm, along -Z.
    - ``elbow_flexion``: forearm, along -Z.
    - ``wrist_cock``: hand + grip + shaft, along -Z to the clubhead.
    - ``wrist_rotation``: clubhead (solid sphere) at the chain tip.

    Postcondition: the masses sum to
    ``golfer.mass * (trunk_mass_ratio + arm_mass_ratio) + club.total_mass``
    plus one carrier dummy mass.
    """
    golfer = golfer or GolferModel()
    club = club or ClubModel()
    rig = build_swing_rig(golfer, club)
    offset = {name: rig.joints[name].tpose_offset for name in JOINTS}

    trunk_mass = golfer.mass * golfer.trunk_mass_ratio
    arm_mass = golfer.mass * golfer.arm_mass_ratio
    trunk_radius = _TRUNK_RADIUS_FRACTION * golfer.height

    pelvis_len = float(offset["trunk_rotation"][2])
    pelvis_mass = _PELVIS_TRUNK_SPLIT * trunk_mass
    thorax_len = float(offset["shoulder_horizontal"][1])
    thorax_mass = (1.0 - _PELVIS_TRUNK_SPLIT) * trunk_mass
    upper_len = -float(offset["elbow_flexion"][2])
    fore_len = -float(offset["wrist_cock"][2])

    hand_mass = _HAND_MASS_FRACTION * arm_mass
    shaft_parts = [
        (hand_mass, -0.5 * _HAND_LENGTH, _HAND_LENGTH),
        (club.grip_mass, -0.5 * _GRIP_LENGTH, _GRIP_LENGTH),
        (club.shaft_mass, -0.5 * club.shaft_length, club.shaft_length),
    ]

    return {
        "hip_rotation": LinkInertial(
            mass=pelvis_mass,
            com=(0.0, 0.0, 0.5 * pelvis_len),
            inertia=_cylinder_inertia(pelvis_mass, pelvis_len, trunk_radius, 2),
        ),
        "trunk_rotation": LinkInertial(
            mass=thorax_mass,
            com=(0.0, 0.5 * thorax_len, 0.0),
            inertia=_cylinder_inertia(thorax_mass, thorax_len, trunk_radius, 1),
        ),
        "shoulder_horizontal": LinkInertial(
            mass=_CARRIER_MASS,
            inertia=(_CARRIER_INERTIA,) * 3 + (0.0,) * 3,
        ),
        "shoulder_vertical": LinkInertial(
            mass=_UPPER_ARM_MASS_FRACTION * arm_mass,
            com=(0.0, 0.0, -0.5 * upper_len),
            inertia=_cylinder_inertia(
                _UPPER_ARM_MASS_FRACTION * arm_mass, upper_len, _LIMB_RADIUS, 2
            ),
        ),
        "elbow_flexion": LinkInertial(
            mass=_FOREARM_MASS_FRACTION * arm_mass,
            com=(0.0, 0.0, -0.5 * fore_len),
            inertia=_cylinder_inertia(
                _FOREARM_MASS_FRACTION * arm_mass, fore_len, _LIMB_RADIUS, 2
            ),
        ),
        "wrist_cock": _combine_point_and_rod_masses(shaft_parts),
        "wrist_rotation": LinkInertial(
            mass=club.head_mass,
            inertia=_sphere_inertia(club.head_mass, _CLUBHEAD_RADIUS),
        ),
    }


def placeholder_link_inertials() -> dict[str, LinkInertial]:
    """The bridge's generic conditioning inertials, as :class:`LinkInertial`.

    Equivalent to what :func:`rig_to_urdf` emits with no ``link_inertials``;
    exposed so the symbolic RNEA can be validated against the placeholder
    URDF as well as the anthropometric one.
    """
    return {
        name: LinkInertial(mass=1.0, inertia=(1e-2, 1e-2, 1e-2, 0.0, 0.0, 0.0))
        for name in JOINTS
    }


def resolve_link_inertials(
    golfer: GolferModel | None,
    club: ClubModel | None,
    inertials: InertialSource,
) -> dict[str, LinkInertial] | None:
    """Map an :data:`InertialSource` to the bridge's ``link_inertials``."""
    if inertials == "anthropometric":
        return swing_link_inertials(golfer, club)
    if inertials == "placeholder":
        return None
    raise ValueError(
        f"inertials must be 'anthropometric' or 'placeholder', got {inertials!r}"
    )


def swing_urdf(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    *,
    inertials: InertialSource = "anthropometric",
) -> str:
    """Render the swing chain as URDF text (single model source).

    ``inertials`` selects anthropometric (default) or placeholder link
    inertials; see the module docstring.
    """
    return rig_to_urdf(
        build_swing_rig(golfer, club),
        link_inertials=resolve_link_inertials(golfer, club, inertials),
    )


def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


def build_pinocchio_model(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    *,
    inertials: InertialSource = "anthropometric",
) -> Any:
    """Build a ``pin.Model`` of the swing chain.

    Raises:
        RuntimeError: When the pinocchio bindings are not installed.
    """
    if not _module_available("pinocchio"):
        raise RuntimeError(
            "pinocchio is not installed. Install the pinocchio extra: "
            "pip install 'upstream-drift[pinocchio]'"
        )
    pin = import_module("pinocchio")
    return pin.buildModelFromXML(swing_urdf(golfer, club, inertials=inertials))


def build_drake_plant(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    *,
    inertials: InertialSource = "anthropometric",
) -> Any:
    """Build a finalized continuous-time Drake ``MultibodyPlant``.

    The root link is welded to the world so ``num_positions() == 7``.

    Raises:
        RuntimeError: When pydrake is not installed.
    """
    if not _module_available("pydrake"):
        raise RuntimeError(
            "pydrake is not installed. Install the drake extra: "
            "pip install 'upstream-drift[drake]'"
        )
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant

    rig = build_swing_rig(golfer, club)
    plant = MultibodyPlant(time_step=0.0)
    urdf = rig_to_urdf(
        rig, link_inertials=resolve_link_inertials(golfer, club, inertials)
    )
    Parser(plant).AddModelsFromString(urdf, "urdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(rig_root_link_name(rig)))
    plant.Finalize()
    return plant


def build_mujoco_model(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    *,
    inertials: InertialSource = "anthropometric",
) -> Any:
    """Build a MuJoCo ``MjModel`` of the swing chain (URDF ingestion).

    Raises:
        RuntimeError: When the mujoco bindings are not installed.
    """
    if not _module_available("mujoco"):
        raise RuntimeError(
            "mujoco is not installed. Install the core package: "
            "pip install upstream-drift"
        )
    mujoco = import_module("mujoco")
    # MuJoCo selects its URDF parser by file extension.
    with tempfile.TemporaryDirectory() as tmp:
        urdf_path = Path(tmp) / f"{SWING_RIG_ID}.urdf"
        urdf_path.write_text(
            swing_urdf(golfer, club, inertials=inertials), encoding="utf-8"
        )
        return mujoco.MjModel.from_xml_path(str(urdf_path))
