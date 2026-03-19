"""GUI components for the double pendulum simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import MainWindow

__all__ = ["MainWindow"]


def __getattr__(name: str) -> object:
    """Lazy import for MainWindow to avoid PyQt6 import at package level."""
    if name == "MainWindow":
        from .main_window import MainWindow

        return MainWindow
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
