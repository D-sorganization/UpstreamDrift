# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Helpers for collecting and restoring GUI session state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..trajectory import OptimizationResult

if TYPE_CHECKING:
    from .main_window import MainWindow
    from .widgets import ParameterSidebar


def collect_results(window: MainWindow) -> dict[str, OptimizationResult]:
    """Collect non-null exercise results keyed by exercise type.

    Briefly acquires ``window._opt_lock`` so the snapshot is consistent with
    the worker thread's writes to ``window.exercise_states``.
    """
    with window._opt_lock:
        snapshot = [state.result for state in window.exercise_states]
    results: dict[str, OptimizationResult] = {}
    for index, (_, exercise_type) in enumerate(window.EXERCISE_CONFIGS):
        result = snapshot[index]
        if result is not None:
            results[exercise_type] = result
    return results


def collect_slider_values(sidebar: ParameterSidebar) -> dict[str, float]:
    """Serialize the sidebar's user-adjustable values."""
    return {
        "body_mass": sidebar.mass_slider.value(),
        "height": sidebar.height_slider.value(),
        "lower_leg": sidebar.ll_slider.value(),
        "upper_leg": sidebar.ul_slider.value(),
        "torso": sidebar.to_slider.value(),
        "bar_mass": sidebar.bar_slider.value(),
        "duration": sidebar.dur_slider.value(),
        "smoothness": sidebar.smooth_slider.value(),
    }


def restore_slider_values(
    sidebar: ParameterSidebar, slider_values: dict[str, float]
) -> None:
    """Apply persisted slider values to the GUI sidebar."""
    slider_map = {
        "body_mass": sidebar.mass_slider,
        "height": sidebar.height_slider,
        "lower_leg": sidebar.ll_slider,
        "upper_leg": sidebar.ul_slider,
        "torso": sidebar.to_slider,
        "bar_mass": sidebar.bar_slider,
        "duration": sidebar.dur_slider,
        "smoothness": sidebar.smooth_slider,
    }
    for key, slider in slider_map.items():
        if key in slider_values:
            slider.set_value(slider_values[key])
