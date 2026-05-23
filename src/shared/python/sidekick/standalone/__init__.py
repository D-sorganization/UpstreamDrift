"""Standalone Sidekick — window shell, session store, and headless run (T1-T5).

Submodules:
    session_store   Platformdirs-scoped profile persistence (T3).
    window          StandaloneSidekickWindow QMainWindow shell (T2).
    run             Headless calculator invoker (T4).
"""

from __future__ import annotations

__all__ = [
    "session_store",
    "window",
]
