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

Scope note: segment geometry uses documented anthropometric fractions and
generic inertials from the bridge. The chain is a *conditioning* model for
gradient-based solvers — DOF count, ordering, and joint limits are exact;
segment inertia is not a biomechanical claim.
"""

from __future__ import annotations

import tempfile
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.model_bridge import (
    rig_root_link_name,
    rig_to_urdf,
)
from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import ClubModel, GolferModel

__all__ = [
    "SWING_RIG_ID",
    "build_drake_plant",
    "build_mujoco_model",
    "build_pinocchio_model",
    "build_swing_rig",
    "swing_joint_limits",
    "swing_urdf",
]

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


def swing_urdf(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
) -> str:
    """Render the swing chain as URDF text (single model source)."""
    return rig_to_urdf(build_swing_rig(golfer, club))


def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


def build_pinocchio_model(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
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
    return pin.buildModelFromXML(swing_urdf(golfer, club))


def build_drake_plant(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
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
    Parser(plant).AddModelsFromString(rig_to_urdf(rig), "urdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(rig_root_link_name(rig)))
    plant.Finalize()
    return plant


def build_mujoco_model(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
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
        urdf_path.write_text(swing_urdf(golfer, club), encoding="utf-8")
        return mujoco.MjModel.from_xml_path(str(urdf_path))
