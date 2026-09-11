"""Compatibility facade and parity driver for the canonical Drake forward simulator.

Matches ``CROSS_ENGINE_GOLF_EQUIVALENCE_SPEC.md`` (§3, §4.2, §5) and
``DRAKE_PARITY_SPEC.md`` (§2).

Public API:
    SimOptions -- forward-sim options.
    SimOut -- canonical output dataclass matching Simscape contract.
    EngineJointMap -- translation between canonical 27 coordinates and Drake DOFs.
    simulate_with_coefficients -- primary forward dynamics rollout function.
    synthesize_target_from_coefficients -- target synthesis helper.
    get_drake_canonical_joint_map -- canonical coordinate to DOF mapper.
    evaluate_polynomial_torque -- power-basis torque evaluator.
    evaluate_bernstein_torque -- Bernstein-basis torque evaluator.
    evaluate_torque_polynomial -- general polynomial torque evaluator.
    is_drake_available -- runtime availability check.
    polynomial_torque_bounds -- parameter bounds helper.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.drake.python.motion_matching.simulate import (
    CLUBHEAD_FRAME_NAME,
    COEFFS_PER_JOINT,
    DEFAULT_GOLFER_URDF,
    GRIP_FRAME_NAME,
    POLY_DEGREE,
    SimOptions,
    SimOut,
    evaluate_bernstein_torque,
    evaluate_polynomial_torque,
    evaluate_torque_polynomial,
    is_drake_available,
    simulate_with_coefficients,
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

# Coefficient bounds mirrored from shared cross-engine spec
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


def get_drake_canonical_joint_map(plant: Any) -> EngineJointMap:
    """Build the EngineJointMap for the canonical golfer Drake model.

    Resolves the 27 canonical actuation channels into native Drake DOF velocity
    indices and sign conventions.
    """
    channel_to_joint: dict[str, str] = {
        "TranslationInputX": "root_trans_x",
        "TranslationInputY": "root_trans_y",
        "TranslationInputZ": "root_trans_z",
        "HipInputX": "root_rot_x",
        "HipInputY": "root_rot_y",
        "HipInputZ": "root_rot_z",
        "SpineInputX": "pelvis_to_lumbar1_intermediate",
        "SpineInputY": "lumbar1_intermediate_to_lumbar1",
        "TorsoInput": "lumbar3_to_thorax1",
        "LEInput": "upper_arm_left_to_forearm_left_intermediate",
        "LFInput": "forearm_left_intermediate_to_forearm_left",
        "LScapInputX": "thorax3_to_scapula_left_intermediate",
        "LScapInputY": "scapula_left_intermediate_to_scapula_left",
        "LSInputX": "scapula_left_to_upper_arm_left_gimbal_z",
        "LSInputY": "upper_arm_left_gimbal_z_to_upper_arm_left_gimbal_y",
        "LSInputZ": "upper_arm_left_gimbal_y_to_upper_arm_left",
        "LWInputX": "forearm_left_to_hand_left_intermediate",
        "LWInputY": "hand_left_intermediate_to_hand_left",
        "REInput": "upper_arm_right_to_forearm_right_intermediate",
        "RFInput": "forearm_right_intermediate_to_forearm_right",
        "RScapInputX": "thorax3_to_scapula_right_intermediate",
        "RScapInputY": "scapula_right_intermediate_to_scapula_right",
        "RSInputX": "scapula_right_to_upper_arm_right_gimbal_z",
        "RSInputY": "upper_arm_right_gimbal_z_to_upper_arm_right_gimbal_y",
        "RSInputZ": "upper_arm_right_gimbal_y_to_upper_arm_right",
        "RWInputX": "forearm_right_to_hand_right_intermediate",
        "RWInputY": "hand_right_intermediate_to_hand_right",
    }

    coord_names: list[str] = []
    dof_indices: list[int] = []
    signs: list[float] = []

    has_joint = hasattr(plant, "HasJointNamed") and callable(plant.HasJointNamed)
    has_actuator = hasattr(plant, "HasJointActuatorNamed") and callable(
        plant.HasJointActuatorNamed
    )

    for cname, target_name in channel_to_joint.items():
        sign = 1.0
        coord_names.append(cname)
        signs.append(sign)

        resolved_idx = -1
        if has_joint and plant.HasJointNamed(target_name):
            joint = plant.GetJointByName(target_name)
            if hasattr(joint, "velocity_start") and callable(joint.velocity_start):
                resolved_idx = int(joint.velocity_start())
            elif hasattr(joint, "velocity_start"):
                resolved_idx = int(joint.velocity_start)
            elif hasattr(joint, "idx_v"):
                resolved_idx = int(joint.idx_v)
        elif has_actuator and plant.HasJointActuatorNamed(target_name):
            actuator = plant.GetJointActuatorByName(target_name)
            if hasattr(actuator, "index") and callable(actuator.index):
                resolved_idx = int(actuator.index())
            elif hasattr(actuator, "index"):
                resolved_idx = int(actuator.index)

        dof_indices.append(resolved_idx)

    return EngineJointMap(
        coordinate_names=tuple(coord_names),
        engine_dof_indices=tuple(dof_indices),
        sign_flips=tuple(signs),
    )


@dataclass(frozen=True)
class SynthesizeOptions:
    """Engine-specific options for :func:`synthesize_target_from_coefficients`."""

    sim_options: SimOptions | None = None
    align: Any = None
    subject_id: str = "synthetic"
    trial_id: str | None = None
    initial_pose: dict[str, Any] | None = None


def synthesize_target_from_coefficients(
    theta: NDArray[np.float64],
    opts: SynthesizeOptions | None = None,
) -> Any:
    """Run simulate_with_coefficients and package into a canonical ClubTarget."""
    from src.shared.python.motion_matching.club_target import (
        ClubTarget,
        SourceProvenance,
    )
    from src.shared.python.motion_matching.loaders._quaternion import rotmat_to_quat

    options = opts if opts is not None else SynthesizeOptions()
    if options.sim_options is not None:
        sim_opts = options.sim_options
    elif options.align is not None:
        sim_opts = SimOptions(
            simulation_time_s=float(options.align.simulation_time_s),
            sample_rate_hz=float(options.align.sample_rate_hz),
        )
    else:
        sim_opts = SimOptions()

    sim_out = simulate_with_coefficients(
        theta, options=sim_opts, initial_pose=options.initial_pose
    )

    canonical_bytes = np.ascontiguousarray(theta, dtype=np.float64).tobytes()
    sha = hashlib.sha256(canonical_bytes).hexdigest()
    trial_id = options.trial_id if options.trial_id is not None else f"theta_{sha[:8]}"

    quat = rotmat_to_quat(sim_out.clubhead_rotation)
    diffs = np.asarray(np.diff(sim_out.clubhead_position, axis=0), dtype=np.float64)
    impact_idx = (
        int(np.argmax(np.einsum("ij,ij->i", diffs, diffs))) + 1 if len(diffs) > 0 else 1
    )

    source = SourceProvenance(
        filename="synthetic",
        format="synthetic",
        subject_id=options.subject_id,
        trial_id=trial_id,
        sha256=sha,
    )

    return ClubTarget(
        time=np.asarray(sim_out.time, dtype=np.float64),
        butt=np.asarray(sim_out.grip_position, dtype=np.float64),
        clubhead=np.asarray(sim_out.clubhead_position, dtype=np.float64),
        club_quat=quat,
        impact_idx=int(impact_idx),
        source=source,
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
    "evaluate_torque_polynomial",
    "get_drake_canonical_joint_map",
    "is_drake_available",
    "polynomial_torque_bounds",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
]
