# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Reusable GUI widgets.

This module re-exports the three widget classes from their dedicated modules
for backward-compatible imports.  New code should import directly from the
focused modules:

- ``gui.labelled_slider.LabelledSlider``
- ``gui.parameter_sidebar.ParameterSidebar``
- ``gui.playback_controls.PlaybackControls``
"""

from .labelled_slider import LabelledSlider
from .parameter_sidebar import ParameterSidebar
from .playback_controls import PlaybackControls

__all__ = [
    "LabelledSlider",
    "ParameterSidebar",
    "PlaybackControls",
]
