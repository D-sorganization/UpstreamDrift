"""Compatibility facade and parity driver for the canonical Pinocchio forward simulator.

Matches ``CROSS_ENGINE_GOLF_EQUIVALENCE_SPEC.md`` (§3, §4.2, §5) and
``PINOCCHIO_PARITY_SPEC.md`` (§2).

Public API:
    SimOptions -- forward-sim options.
    SimOut -- canonical output dataclass matching Simscape contract.
    EngineJointMap -- translation between canonical 27 coordinates and Pinocchio DOFs.
    simulate_with_coefficients -- primary forward dynamics rollout function.
    synthesize_target_from_coefficients -- target synthesis helper.
    get_pinocchio_canonical_joint_map -- canonical coordinate to DOF mapper.
    evaluate_polynomial_torque -- power-basis torque evaluator.
    evaluate_bernstein_torque -- Bernstein-basis torque evaluator.
    is_pinocchio_available -- runtime availability check.
    polynomial_torque_bounds -- parameter bounds helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (
    CLUBHEAD_FRAME_NAME,
    COEFFS_PER_JOINT,
    GRIP_FRAME_NAME,
    POLY_DEGREE,
    SimOptions,
    SimOut,
    evaluate_bernstein_torque,
    evaluate_polynomial_torque,
    is_pinocchio_available,
    simulate_with_coefficients,
)
from src.engines.physics_engines.pinocchio.python.motion_matching.synthesize import (
    SynthesizeOptions,
    synthesize_target_from_coefficients,
)

DEFAULT_GOLFER_URDF: Path = (
    Path(__file__).resolve().parents[1] / "models" / "generated" / "golfer.urdf"
)

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

# Coefficient bounds mirrored from shared spec
POLY_BOUNDS: tuple[float, float, float, float, float, float, float] = (
    1000.0,  # |A| / c0
    1000.0,  # |B| / c1
    500.0,  # |C| / c2
    500.0,  # |D| / c3
    100.0,  # |E| / c4
    100.0,  # |F| / c5
    25.0,  # |G| / c6
)


def polynomial_torque_bounds(
    n_joints: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (lb, ub) element-wise bounds for the flattened theta vector."""
    if n_joints <= 0:
        raise ValueError(f"n_joints must be > 0; got {n_joints}")
    per_joint = np.asarray(POLY_BOUNDS, dtype=np.float64)
    ub = np.tile(per_joint, n_joints)
    return -ub.copy(), ub.copy()


@dataclass(frozen=True)
class EngineJointMap:
    """Translation between canonical 27 coordinates and native engine DOFs/actuators.

    Matches CROSS_ENGINE_GOLF_EQUIVALENCE_SPEC.md §4.2.
    """

    coordinate_names: tuple[str, ...]
    engine_dof_indices: tuple[int, ...]
    sign_flips: tuple[float, ...]


def get_pinocchio_canonical_joint_map(model: Any) -> EngineJointMap:
    """Build the EngineJointMap for the canonical golfer Pinocchio model.

    Resolves the 27 canonical actuation channels into native DOF velocity indices
    (``idx_v``) and sign conventions.
    """
    mapping_spec: list[tuple[str, str, float]] = [
        ("TranslationInputX", "root_trans_x", 1.0),
        ("TranslationInputY", "root_trans_y", 1.0),
        ("TranslationInputZ", "root_trans_z", 1.0),
        ("HipInputX", "root_rot_x", 1.0),
        ("HipInputY", "root_rot_y", 1.0),
        ("HipInputZ", "root_rot_z", 1.0),
        ("SpineInputX", "pelvis_to_lumbar1_intermediate", 1.0),
        ("SpineInputY", "lumbar1_intermediate_to_lumbar1", 1.0),
        ("TorsoInput", "lumbar3_to_thorax1", 1.0),
        ("LEInput", "upper_arm_left_to_forearm_left_intermediate", 1.0),
        ("LFInput", "forearm_left_intermediate_to_forearm_left", 1.0),
        ("LScapInputX", "thorax3_to_scapula_left_intermediate", 1.0),
        ("LScapInputY", "scapula_left_intermediate_to_scapula_left", 1.0),
        ("LSInputX", "scapula_left_to_upper_arm_left_gimbal_z", 1.0),
        ("LSInputY", "upper_arm_left_gimbal_z_to_upper_arm_left_gimbal_y", 1.0),
        ("LSInputZ", "upper_arm_left_gimbal_y_to_upper_arm_left", 1.0),
        ("LWInputX", "forearm_left_to_hand_left_intermediate", 1.0),
        ("LWInputY", "hand_left_intermediate_to_hand_left", 1.0),
        ("REInput", "upper_arm_right_to_forearm_right_intermediate", 1.0),
        ("RFInput", "forearm_right_intermediate_to_forearm_right", 1.0),
        ("RScapInputX", "thorax3_to_scapula_right_intermediate", 1.0),
        ("RScapInputY", "scapula_right_intermediate_to_scapula_right", 1.0),
        ("RSInputX", "scapula_right_to_upper_arm_right_gimbal_z", 1.0),
        ("RSInputY", "upper_arm_right_gimbal_z_to_upper_arm_right_gimbal_y", 1.0),
        ("RSInputZ", "upper_arm_right_gimbal_y_to_upper_arm_right", 1.0),
        ("RWInputX", "forearm_right_to_hand_right_intermediate", 1.0),
        ("RWInputY", "hand_right_intermediate_to_hand_right", 1.0),
    ]

    coord_names: list[str] = []
    dof_indices: list[int] = []
    signs: list[float] = []

    has_exist_joint = hasattr(model, "existJointName") and callable(
        model.existJointName
    )

    for cname, target_name, sign in mapping_spec:
        coord_names.append(cname)
        signs.append(sign)

        if has_exist_joint and model.existJointName(target_name):
            jid = model.getJointId(target_name)
            idx_v = getattr(model.joints[jid], "idx_v", -1)
            dof_indices.append(int(idx_v))
        else:
            dof_indices.append(-1)

    return EngineJointMap(
        coordinate_names=tuple(coord_names),
        engine_dof_indices=tuple(dof_indices),
        sign_flips=tuple(signs),
    )


__all__ = [
    "CANONICAL_COORDINATE_NAMES",
    "CLUBHEAD_FRAME_NAME",
    "COEFFS_PER_JOINT",
    "DEFAULT_GOLFER_URDF",
    "EngineJointMap",
    "GRIP_FRAME_NAME",
    "POLY_BOUNDS",
    "POLY_DEGREE",
    "SimOptions",
    "SimOut",
    "SynthesizeOptions",
    "evaluate_bernstein_torque",
    "evaluate_polynomial_torque",
    "get_pinocchio_canonical_joint_map",
    "is_pinocchio_available",
    "polynomial_torque_bounds",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
]
