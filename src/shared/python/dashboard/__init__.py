"""Shared dashboard package.

GUI entry points (``launch_dashboard``, ``UnifiedDashboardWindow``) pull in
PyQt6 at import time. API-server codepaths only need the headless pieces
(e.g. ``recorder.GenericPhysicsRecorder``), so we load the Qt-backed names
lazily via ``__getattr__`` — clients that import them still get the same
objects, but ``import src.shared.python.dashboard`` on its own no longer
requires PyQt6 to be installed.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "UnifiedDashboardWindow",
    "launch_dashboard",
]


def __getattr__(name: str) -> Any:
    if name == "launch_dashboard":
        from .launcher import launch_dashboard as _fn

        return _fn
    if name == "UnifiedDashboardWindow":
        from .window import UnifiedDashboardWindow as _cls

        return _cls
    raise AttributeError(
        f"module 'src.shared.python.dashboard' has no attribute {name!r}"
    )


if TYPE_CHECKING:
    from .launcher import launch_dashboard
    from .window import UnifiedDashboardWindow
