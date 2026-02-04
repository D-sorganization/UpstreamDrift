"""Learning and Adaptation module for UpstreamDrift.

This module provides machine learning capabilities for robotics:
- Reinforcement Learning (RL) environments compatible with Gymnasium
- Imitation Learning algorithms (Behavior Cloning, DAgger, GAIL)
- Sim-to-Real transfer techniques (Domain Randomization, System Identification)
- Motion Retargeting between different embodiments

Phase 3 of the Robotics Expansion Proposal.
"""

from __future__ import annotations

from src.learning.imitation import (
    GAIL,
    BehaviorCloning,
    DAgger,
    Demonstration,
    DemonstrationDataset,
    ImitationLearner,
)
from src.learning.retargeting import (
    MotionRetargeter,
    SkeletonConfig,
)
from src.learning.rl import (
    ActionConfig,
    DualArmManipulationEnv,
    HumanoidStandEnv,
    HumanoidWalkEnv,
    ManipulationPickPlaceEnv,
    ObservationConfig,
    RewardConfig,
    RoboticsGymEnv,
    TaskConfig,
)
from src.learning.sim2real import (
    DomainRandomizationConfig,
    DomainRandomizer,
    SystemIdentifier,
)

__all__ = [
    # RL
    "RoboticsGymEnv",
    "HumanoidWalkEnv",
    "HumanoidStandEnv",
    "ManipulationPickPlaceEnv",
    "DualArmManipulationEnv",
    "ObservationConfig",
    "ActionConfig",
    "RewardConfig",
    "TaskConfig",
    # Imitation
    "Demonstration",
    "DemonstrationDataset",
    "ImitationLearner",
    "BehaviorCloning",
    "DAgger",
    "GAIL",
    # Sim2Real
    "DomainRandomizationConfig",
    "DomainRandomizer",
    "SystemIdentifier",
    # Retargeting
    "MotionRetargeter",
    "SkeletonConfig",
]
