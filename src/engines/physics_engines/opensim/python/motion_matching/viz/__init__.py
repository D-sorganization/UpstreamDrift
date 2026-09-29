"""Three canonical visualisation figures for the OpenSim motion-matching pipeline.

Per VISUALIZATION_SPEC.md every engine must produce three figures:

1. **Trajectory overlay** — measured vs simulated club path/skeleton.
   Engine-specific: prefers ``opensim.Visualizer`` for interactive use,
   falls back to a headless matplotlib 3D rendering.
2. **Error timecourse** — stacked 2D plots of position / orientation error,
   clubhead speed, and joint torques against simulation time.
3. **Fit quality card** — single-figure summary card suitable for
   dropping into a PR description.

All three entry points have a uniform signature so cross-engine comparison
plots can render any engine's output by importing the matching module.

Public API
----------

- :func:`plot_trajectory_overlay`
- :func:`plot_error_timecourse`
- :func:`plot_fit_quality_card`
- :func:`render_with_opensim_visualizer` — interactive wrapper, optional.

Headless safety
---------------

The 2D plotters and the matplotlib fallback for the 3D plot are headless-
safe (Agg-compatible) and emit no warnings under pytest. The
``opensim.Visualizer`` path is only invoked when explicitly requested
*and* the bindings are importable.
"""

from __future__ import annotations

__all__ = [
    "OpenSimVisualizerUnavailableError",
    "plot_error_timecourse",
    "plot_fit_quality_card",
    "plot_trajectory_overlay",
    "render_with_opensim_visualizer",
]


def __getattr__(name: str):
    if name in {
        "plot_error_timecourse",
        "plot_fit_quality_card",
        "plot_trajectory_overlay",
    }:
        from . import figures

        return getattr(figures, name)
    if name in {
        "OpenSimVisualizerUnavailableError",
        "render_with_opensim_visualizer",
    }:
        from . import native

        return getattr(native, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
