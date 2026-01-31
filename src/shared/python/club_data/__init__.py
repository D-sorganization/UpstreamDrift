"""Club data management module.

Provides functionality for loading, managing, and displaying golf club data
from Excel files and other sources. This module serves as the single source
of truth for club specifications across all physics engines (Drake, Pinocchio, MuJoCo).
"""

from .loader import (
    ClubDataLoader,
    ClubSpecification,
    ProPlayerData,
    SwingMetrics,
    load_club_data,
    load_pro_player_data,
)
from .display import ClubDataDisplayWidget, ClubTargetOverlay
from .targets import ClubTargetManager, TargetTrajectory

__all__ = [
    "ClubDataLoader",
    "ClubSpecification",
    "ProPlayerData",
    "SwingMetrics",
    "load_club_data",
    "load_pro_player_data",
    "ClubDataDisplayWidget",
    "ClubTargetOverlay",
    "ClubTargetManager",
    "TargetTrajectory",
]
