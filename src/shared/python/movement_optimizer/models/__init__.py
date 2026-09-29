# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Body model and Lagrangian dynamics engine.

This package contains the anthropometric body model and the analytical
3-link planar dynamics that serve as the default physics backend.

Submodules:
    body_model: BodyModel, ChainGeometry, and joint-angle helpers.
    lagrangian_dynamics: LagrangianDynamics and balance_pose.
    bench_press_model: BenchPressModel and make_bench_press_config.
    exercise_configs: make_squat_config, make_full_squat_config, make_deadlift_config.

Design Principles:
    DBC -- every public method states and checks its preconditions.
    DRY -- mass/inertia setup is factored into private helpers.
    LoD -- callers interact only through the public API.
"""

from __future__ import annotations

# Re-export strength helpers that the original models.py exposed at package level.
from ..strength import (
    HillTorqueModel as HillTorqueModel,
)
from ..strength import (
    JointTorqueSet as JointTorqueSet,
)
from ..strength import (
    compute_max_load as compute_max_load,
)
from ..strength import (
    make_bench_press_torque_set as make_bench_press_torque_set,
)
from ..strength import (
    make_default_torque_set as make_default_torque_set,
)

# Re-export everything from submodules for backwards compatibility.
# All existing imports of the form ``from movement_optimizer.models import X``
# continue to work without modification.
from .bench_press_model import BenchPressModel as BenchPressModel
from .bench_press_model import make_bench_press_config as make_bench_press_config
from .bilateral_3d import Bilateral3DModel as Bilateral3DModel
from .bilateral_3d import Bilateral3DPose as Bilateral3DPose
from .body_model import BodyModel as BodyModel
from .body_model import ChainGeometry as ChainGeometry
from .body_model import clamp_joint_angles as clamp_joint_angles
from .body_model import joint_angles_within_limits as joint_angles_within_limits
from .chain_dynamics import ChainConfig as ChainConfig
from .chain_dynamics import ChainState as ChainState
from .chain_dynamics import initial_tip_kick_velocities as initial_tip_kick_velocities
from .chain_dynamics import random_wadded_chain_state as random_wadded_chain_state
from .chain_dynamics import simulate_chain as simulate_chain
from .chain_dynamics import simulate_chain_for_duration as simulate_chain_for_duration
from .chain_dynamics import steps_for_duration as steps_for_duration
from .chain_forces import ChainForceField as ChainForceField
from .chain_forces import ChainForceHistory as ChainForceHistory
from .chain_forces import chain_force_field as chain_force_field
from .chain_forces import chain_force_history as chain_force_history
from .chain_forces import link_accelerations as link_accelerations
from .exercise_configs import make_deadlift_config as make_deadlift_config
from .exercise_configs import make_full_squat_config as make_full_squat_config
from .exercise_configs import make_squat_config as make_squat_config
from .lagrangian_dynamics import LagrangianDynamics as LagrangianDynamics
from .lagrangian_dynamics import balance_pose as balance_pose
from .swingset import CyclicPolicyBounds as CyclicPolicyBounds
from .swingset import CyclicPolicySearchSpace as CyclicPolicySearchSpace
from .swingset import CyclicPolicyTraceSample as CyclicPolicyTraceSample
from .swingset import SwingSetConfig as SwingSetConfig
from .swingset import SwingSetState as SwingSetState
from .swingset import constrain_swing_pose as constrain_swing_pose
from .swingset import cyclic_policy_controls as cyclic_policy_controls
from .swingset import estimate_swingset_joint_torques as estimate_swingset_joint_torques
from .swingset import optimize_cyclic_policy as optimize_cyclic_policy
from .swingset import (
    optimize_cyclic_policy_iterative as optimize_cyclic_policy_iterative,
)
from .swingset import simulate_swingset as simulate_swingset
from .swingset import simulate_swingset_controls as simulate_swingset_controls
from .swingset_forces import SwingForceField as SwingForceField
from .swingset_forces import SwingForceHistory as SwingForceHistory
from .swingset_forces import swing_force_field as swing_force_field
from .swingset_forces import swing_force_history as swing_force_history

__all__ = [
    "BenchPressModel",
    "Bilateral3DModel",
    "Bilateral3DPose",
    "BodyModel",
    "ChainConfig",
    "ChainForceField",
    "ChainForceHistory",
    "ChainGeometry",
    "ChainState",
    "CyclicPolicyBounds",
    "CyclicPolicySearchSpace",
    "CyclicPolicyTraceSample",
    "HillTorqueModel",
    "JointTorqueSet",
    "LagrangianDynamics",
    "SwingForceField",
    "SwingForceHistory",
    "SwingSetConfig",
    "SwingSetState",
    "balance_pose",
    "chain_force_field",
    "chain_force_history",
    "clamp_joint_angles",
    "compute_max_load",
    "constrain_swing_pose",
    "cyclic_policy_controls",
    "estimate_swingset_joint_torques",
    "initial_tip_kick_velocities",
    "joint_angles_within_limits",
    "link_accelerations",
    "make_bench_press_config",
    "make_bench_press_torque_set",
    "make_deadlift_config",
    "make_default_torque_set",
    "make_full_squat_config",
    "make_squat_config",
    "optimize_cyclic_policy",
    "optimize_cyclic_policy_iterative",
    "random_wadded_chain_state",
    "simulate_chain",
    "simulate_chain_for_duration",
    "simulate_swingset",
    "simulate_swingset_controls",
    "steps_for_duration",
    "swing_force_field",
    "swing_force_history",
]
