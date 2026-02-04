"""Gait type definitions for locomotion.

This module defines enumerations and data structures for
representing different gait patterns and phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class GaitType(Enum):
    """Type of locomotion gait."""

    STAND = auto()  # Static standing
    WALK = auto()  # Normal walking
    TROT = auto()  # Trot gait (diagonal pairs)
    RUN = auto()  # Running with flight phase
    CRAWL = auto()  # Slow, stable crawl
    BOUND = auto()  # Bounding gait
    GALLOP = auto()  # Galloping


class GaitPhase(Enum):
    """Phase within a gait cycle."""

    DOUBLE_SUPPORT = auto()  # Both feet on ground
    LEFT_SUPPORT = auto()  # Left foot only on ground
    RIGHT_SUPPORT = auto()  # Right foot only on ground
    FLIGHT = auto()  # Both feet off ground (running)
    LEFT_SWING = auto()  # Left leg swinging
    RIGHT_SWING = auto()  # Right leg swinging


class LegState(Enum):
    """State of an individual leg."""

    STANCE = auto()  # Foot on ground, supporting
    SWING = auto()  # Foot in air, swinging
    EARLY_CONTACT = auto()  # Just touched down
    LATE_CONTACT = auto()  # About to lift off
    LOADING = auto()  # Weight shifting onto leg
    UNLOADING = auto()  # Weight shifting off leg


class SupportState(Enum):
    """Support configuration for bipedal robot."""

    DOUBLE_SUPPORT_LEFT_LEADING = auto()  # Both feet, weight on left
    DOUBLE_SUPPORT_RIGHT_LEADING = auto()  # Both feet, weight on right
    DOUBLE_SUPPORT_CENTERED = auto()  # Both feet, weight centered
    SINGLE_SUPPORT_LEFT = auto()  # Left foot only
    SINGLE_SUPPORT_RIGHT = auto()  # Right foot only
    FLIGHT = auto()  # No ground contact


@dataclass
class GaitParameters:
    """Parameters defining a gait pattern.

    Attributes:
        gait_type: Type of gait.
        step_length: Forward step length [m].
        step_width: Lateral step width [m].
        step_height: Foot lift height [m].
        step_duration: Duration of one step [s].
        double_support_ratio: Ratio of double support phase.
        swing_height_profile: Height profile during swing ('sine', 'trap').
        com_height: Target CoM height [m].
        max_foot_velocity: Maximum foot velocity [m/s].
        settling_time: Time to settle after step [s].
    """

    gait_type: GaitType = GaitType.WALK
    step_length: float = 0.3
    step_width: float = 0.2
    step_height: float = 0.05
    step_duration: float = 0.5
    double_support_ratio: float = 0.2
    swing_height_profile: str = "sine"
    com_height: float = 0.9
    max_foot_velocity: float = 1.0
    settling_time: float = 0.1

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.step_length < 0:
            raise ValueError("step_length must be non-negative")
        if self.step_width < 0:
            raise ValueError("step_width must be non-negative")
        if self.step_height < 0:
            raise ValueError("step_height must be non-negative")
        if self.step_duration <= 0:
            raise ValueError("step_duration must be positive")
        if not 0 <= self.double_support_ratio <= 1:
            raise ValueError("double_support_ratio must be in [0, 1]")
        if self.com_height <= 0:
            raise ValueError("com_height must be positive")

    @property
    def swing_duration(self) -> float:
        """Get swing phase duration."""
        return self.step_duration * (1 - self.double_support_ratio)

    @property
    def double_support_duration(self) -> float:
        """Get double support phase duration."""
        return self.step_duration * self.double_support_ratio

    @property
    def step_frequency(self) -> float:
        """Get step frequency [Hz]."""
        return 1.0 / self.step_duration if self.step_duration > 0 else 0.0


def create_walk_parameters(
    step_length: float = 0.3,
    step_duration: float = 0.5,
    com_height: float = 0.9,
) -> GaitParameters:
    """Create parameters for walking gait.

    Args:
        step_length: Forward step length [m].
        step_duration: Duration of one step [s].
        com_height: Target CoM height [m].

    Returns:
        GaitParameters configured for walking.
    """
    return GaitParameters(
        gait_type=GaitType.WALK,
        step_length=step_length,
        step_width=0.2,
        step_height=0.05,
        step_duration=step_duration,
        double_support_ratio=0.2,
        swing_height_profile="sine",
        com_height=com_height,
    )


def create_run_parameters(
    step_length: float = 0.6,
    step_duration: float = 0.3,
    com_height: float = 0.85,
) -> GaitParameters:
    """Create parameters for running gait.

    Args:
        step_length: Forward step length [m].
        step_duration: Duration of one step [s].
        com_height: Target CoM height [m].

    Returns:
        GaitParameters configured for running.
    """
    return GaitParameters(
        gait_type=GaitType.RUN,
        step_length=step_length,
        step_width=0.15,
        step_height=0.1,
        step_duration=step_duration,
        double_support_ratio=0.0,  # No double support in running
        swing_height_profile="sine",
        com_height=com_height,
        max_foot_velocity=2.0,
    )


def create_stand_parameters(
    step_width: float = 0.2,
    com_height: float = 0.9,
) -> GaitParameters:
    """Create parameters for standing.

    Args:
        step_width: Lateral foot spacing [m].
        com_height: Target CoM height [m].

    Returns:
        GaitParameters configured for standing.
    """
    return GaitParameters(
        gait_type=GaitType.STAND,
        step_length=0.0,
        step_width=step_width,
        step_height=0.0,
        step_duration=1.0,
        double_support_ratio=1.0,  # Always double support
        com_height=com_height,
    )
