"""Shared dashboard package.

Exports the launcher entry point and the primary dashboard window used by all
physics-engine launchers in the fleet.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "UnifiedDashboardWindow",
    "launch_dashboard",
]


def __getattr__(name: str) -> Any:
    if name == "launch_dashboard":
        from .launcher import launch_dashboard

        return launch_dashboard
    if name == "UnifiedDashboardWindow":
        from .window import UnifiedDashboardWindow

        return UnifiedDashboardWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
