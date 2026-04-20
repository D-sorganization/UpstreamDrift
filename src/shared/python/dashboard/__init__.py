"""Shared dashboard package.

Exports the launcher entry point and the primary dashboard window used by all
physics-engine launchers in the fleet.
"""

from .launcher import launch_dashboard
from .window import UnifiedDashboardWindow

__all__ = [
    "UnifiedDashboardWindow",
    "launch_dashboard",
]
