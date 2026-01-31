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
from .targets import ClubTargetManager, TargetTrajectory

__all__ = [
    "ClubDataLoader",
    "ClubSpecification",
    "ProPlayerData",
    "SwingMetrics",
    "load_club_data",
    "load_pro_player_data",
    "ClubTargetManager",
    "TargetTrajectory",
]

# Conditionally import PyQt6-dependent widgets
# Note: AttributeError catches the case when QtWidgets is None and
# class definitions try to inherit from QtWidgets.QWidget
try:
    from .club_data_tab import ClubDataTab
    from .display import ClubDataDisplayWidget, ClubTargetOverlay

    __all__ += ["ClubDataDisplayWidget", "ClubTargetOverlay", "ClubDataTab"]
except (ImportError, AttributeError):
    # PyQt6 not available - widgets not exported
    pass
