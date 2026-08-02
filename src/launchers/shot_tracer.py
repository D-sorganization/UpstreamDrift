"""Lazy launcher for the multi-model shot tracer GUI.

Importing this module is intentionally headless-safe. Qt and pyqtgraph are
loaded only when the GUI entry point or exported widget classes are requested.
"""

# ruff: noqa: F822

from __future__ import annotations

import sys  # noqa: F401
from typing import Any

_GUI_EXPORTS = {
    "FlightModelRegistry",
    "FlightModelType",
    "QApplication",
    "QMessageBox",
    "MultiModelShotTracerWidget",
    "MultiModelShotTracerWindow",
    "PYQTGRAPH_AVAILABLE",
    "TRAJECTORY_COLORS",
    "UnifiedLaunchConditions",
    "compare_models",
    "gl",
}

PYQTGRAPH_AVAILABLE: bool | None = None
gl: Any = None


def _load_gui_module() -> Any:
    from src.launchers import _shot_tracer_gui

    return _shot_tracer_gui


def __getattr__(name: str) -> Any:
    if name in _GUI_EXPORTS:
        return getattr(_load_gui_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """Launch the Multi-Model Shot Tracer application."""
    _load_gui_module().main()


if __name__ == "__main__":
    from pathlib import Path
    import sys

    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    main()


__all__ = [
    "MultiModelShotTracerWidget",
    "MultiModelShotTracerWindow",
    "PYQTGRAPH_AVAILABLE",
    "TRAJECTORY_COLORS",
    "FlightModelRegistry",
    "FlightModelType",
    "QApplication",
    "QMessageBox",
    "UnifiedLaunchConditions",
    "compare_models",
    "gl",
    "main",
    "sys",
]
