# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""GUI package for the Movement Optimizer.

The original monolithic ``gui.py`` has been split into a package with
focused sub-modules:

- ``main_window``  -- :class:`MainWindow`
- ``exercise_tab`` -- :class:`ExerciseTab`
- ``comparison_dialog`` -- :class:`ComparisonDialog`
- ``widgets`` -- :class:`LabelledSlider`, :class:`ParameterSidebar`,
  :class:`PlaybackControls`

All public names are re-exported here for backward compatibility so that
``from movement_optimizer.gui import MainWindow`` continues to work.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ComparisonDialog",
    "ExerciseTab",
    "LabelledSlider",
    "MainWindow",
    "ParameterSidebar",
    "PlaybackControls",
]

_EXPORT_MAP = {
    "ComparisonDialog": (".comparison_dialog", "ComparisonDialog"),
    "ExerciseTab": (".exercise_tab", "ExerciseTab"),
    "LabelledSlider": (".widgets", "LabelledSlider"),
    "MainWindow": (".main_window", "MainWindow"),
    "ParameterSidebar": (".widgets", "ParameterSidebar"),
    "PlaybackControls": (".widgets", "PlaybackControls"),
}


def __getattr__(name: str) -> Any:
    """Load GUI exports lazily so lightweight helpers can import safely."""
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
