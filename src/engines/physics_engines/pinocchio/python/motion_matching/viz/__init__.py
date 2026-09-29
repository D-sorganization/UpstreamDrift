"""Pinocchio per-engine visualisation entry points (issue #4133).

The three canonical views per VISUALIZATION_SPEC.md are exported here:

* :func:`plot_trajectory_overlay` — measured vs simulated club skeleton (View 1).
* :func:`plot_error_timecourse` — stacked error/torque panels (View 2).
* :func:`plot_fit_quality_card` — single-figure summary card (View 3).

The 3D Meshcat overlay (extension of ``MotionVisualizer``) is implemented
in :mod:`.meshcat_overlay`.

The high-level :func:`visualize_fit` entry point bundles all three plus
the Meshcat URL into a ``dict[str, Path | str]``.
"""

from __future__ import annotations

from .error_timecourse import plot_error_timecourse
from .fit_quality_card import plot_fit_quality_card
from .meshcat_overlay import meshcat_overlay
from .trajectory_overlay import plot_trajectory_overlay
from .visualize_fit import visualize_fit

__all__ = [
    "meshcat_overlay",
    "plot_error_timecourse",
    "plot_fit_quality_card",
    "plot_trajectory_overlay",
    "visualize_fit",
]
