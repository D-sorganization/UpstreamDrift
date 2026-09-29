"""Reinforcement Learning environments for robotics.

This module provides Gymnasium-compatible environments for training
RL agents on robotics tasks including humanoid locomotion, manipulation,
and bimanual coordination.
"""

from __future__ import annotations

from src.learning.rl.base_env import RoboticsGymEnv
from src.learning.rl.configs import (
    ActionConfig,
    ObservationConfig,
    RewardConfig,
    TaskConfig,
)
from src.learning.rl.humanoid_envs import HumanoidStandEnv, HumanoidWalkEnv
from src.learning.rl.manipulation_envs import (
    DualArmManipulationEnv,
    ManipulationPickPlaceEnv,
)

__all__ = [
    "RoboticsGymEnv",
    "HumanoidWalkEnv",
    "HumanoidStandEnv",
    "ManipulationPickPlaceEnv",
    "DualArmManipulationEnv",
    "ObservationConfig",
    "ActionConfig",
    "RewardConfig",
    "TaskConfig",
]
