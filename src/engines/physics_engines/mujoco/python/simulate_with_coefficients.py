"""Compatibility facade and parity driver for the canonical MuJoCo forward simulator.

Matches ``CROSS_ENGINE_GOLF_EQUIVALENCE_SPEC.md`` (§4.2, §5) and
``MUJOCO_PARITY_SPEC.md`` (§2).

Public API:
    SimOptions -- forward-sim options.
    SimOut -- canonical output dataclass.
    EngineJointMap -- translation between canonical 27 coordinates and MuJoCo actuators.
    simulate_with_coefficients -- primary forward dynamics rollout function.
    synthesize_target_from_coefficients -- TDD target synthesis helper.
    PolynomialTorqueDriver -- contextual continuous torque actuator driver.
    polynomial_torque_bounds -- parameter bounds helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.engines.physics_engines.mujoco.python.motion_matching.simulate import (
    SimOptions,
    SimOut,
    simulate_with_coefficients,
    synthesize_target_from_coefficients,
)
from src.engines.physics_engines.mujoco.python.motion_matching.torque_driver import (
    POLY_BOUNDS,
    PolynomialTorqueDriver,
    polynomial_torque_bounds,
)

# Canonical site / frame names exposed in SimOut and cross-engine specs
GRIP_SITE_NAME: str = "mid_hands"
CLUBHEAD_SITE_NAME: str = "clubhead"

DEFAULT_GOLFER_XML: Path = (
    Path(__file__).resolve().parents[1] / "models" / "generated" / "golfer.xml"
)


def is_mujoco_available() -> bool:
    """Return whether a functional MuJoCo C++ runtime (mujoco) is available."""
    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        return False
    if type(mujoco).__module__ == "unittest.mock":
        return False
    model_cls = getattr(mujoco, "MjModel", None)
    return callable(model_cls) and type(model_cls).__module__ != "unittest.mock"


# Canonical 27 actuation channels from CROSS_ENGINE_GOLF_EQUIVALENCE_SPEC §3
CANONICAL_COORDINATE_NAMES: tuple[str, ...] = (
    "TranslationInputX",
    "TranslationInputY",
    "TranslationInputZ",
    "HipInputX",
    "HipInputY",
    "HipInputZ",
    "SpineInputX",
    "SpineInputY",
    "TorsoInput",
    "LEInput",
    "LFInput",
    "LScapInputX",
    "LScapInputY",
    "LSInputX",
    "LSInputY",
    "LSInputZ",
    "LWInputX",
    "LWInputY",
    "REInput",
    "RFInput",
    "RScapInputX",
    "RScapInputY",
    "RSInputX",
    "RSInputY",
    "RSInputZ",
    "RWInputX",
    "RWInputY",
)


@dataclass(frozen=True)
class EngineJointMap:
    """Translation between canonical 27 coordinates and native engine DOFs/actuators.

    Matches CROSS_ENGINE_GOLF_EQUIVALENCE_SPEC.md §4.2.
    """

    coordinate_names: tuple[str, ...]
    engine_dof_indices: tuple[int, ...]
    sign_flips: tuple[float, ...]


def get_mujoco_canonical_joint_map(model: Any) -> EngineJointMap:
    """Build the EngineJointMap for the canonical golfer MuJoCo model.

    Resolves the 27 canonical actuation channels into native actuator/DOF indices
    and sign conventions.
    """
    import mujoco

    # Mapping of canonical names to MuJoCo actuator name patterns or joint names
    mapping_spec: list[tuple[str, str, float]] = [
        # Pelvis root translation / orientation (unactuated or freejoint in standard MJCF,
        # mapped to actuators or generalized coordinates)
        ("TranslationInputX", "root_trans_x", 1.0),
        ("TranslationInputY", "root_trans_y", 1.0),
        ("TranslationInputZ", "root_trans_z", 1.0),
        ("HipInputX", "root_rot_x", 1.0),
        ("HipInputY", "root_rot_y", 1.0),
        ("HipInputZ", "root_rot_z", 1.0),
        # Spine / Lumbar
        ("SpineInputX", "act_lumbar1_dof0", 1.0),
        ("SpineInputY", "act_lumbar1_dof1", 1.0),
        ("TorsoInput", "act_thorax1", 1.0),
        # Left Arm
        ("LEInput", "act_forearm_left_dof0", 1.0),
        ("LFInput", "act_forearm_left_dof1", 1.0),
        ("LScapInputX", "act_scapula_left_dof0", 1.0),
        ("LScapInputY", "act_scapula_left_dof1", 1.0),
        ("LSInputX", "act_upper_arm_left_dof0", 1.0),
        ("LSInputY", "act_upper_arm_left_dof1", 1.0),
        ("LSInputZ", "act_upper_arm_left_dof2", 1.0),
        ("LWInputX", "act_hand_left_dof0", 1.0),
        ("LWInputY", "act_hand_left_dof1", 1.0),
        # Right Arm
        ("REInput", "act_forearm_right_dof0", 1.0),
        ("RFInput", "act_forearm_right_dof1", 1.0),
        ("RScapInputX", "act_scapula_right_dof0", 1.0),
        ("RScapInputY", "act_scapula_right_dof1", 1.0),
        ("RSInputX", "act_upper_arm_right_dof0", 1.0),
        ("RSInputY", "act_upper_arm_right_dof1", 1.0),
        ("RSInputZ", "act_upper_arm_right_dof2", 1.0),
        ("RWInputX", "act_hand_right_dof0", 1.0),
        ("RWInputY", "act_hand_right_dof1", 1.0),
    ]

    coord_names: list[str] = []
    dof_indices: list[int] = []
    signs: list[float] = []

    for cname, target_name, sign in mapping_spec:
        coord_names.append(cname)
        # Search actuators first
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, target_name)
        if aid >= 0:
            dof_indices.append(int(aid))
            signs.append(sign)
            continue

        # Try searching by joint name
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target_name)
        if jid >= 0:
            dofadr = int(model.jnt_dofadr[jid])
            dof_indices.append(dofadr)
            signs.append(sign)
            continue

        # If not found directly, map to -1 or index if within nu
        dof_indices.append(-1)
        signs.append(sign)

    return EngineJointMap(
        coordinate_names=tuple(coord_names),
        engine_dof_indices=tuple(dof_indices),
        sign_flips=tuple(signs),
    )


__all__ = [
    "CANONICAL_COORDINATE_NAMES",
    "CLUBHEAD_SITE_NAME",
    "DEFAULT_GOLFER_XML",
    "EngineJointMap",
    "GRIP_SITE_NAME",
    "POLY_BOUNDS",
    "PolynomialTorqueDriver",
    "SimOptions",
    "SimOut",
    "get_mujoco_canonical_joint_map",
    "is_mujoco_available",
    "polynomial_torque_bounds",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
]
