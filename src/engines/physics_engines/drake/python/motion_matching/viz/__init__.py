"""Drake motion-matching visualization parity package (issue #4126).

Three views per VISUALIZATION_SPEC.md (cross-engine §2.5):

* :mod:`render_trajectory_overlay` -- Meshcat 3D side-by-side overlay
  of the measured club trajectory and the simulated club trajectory
  produced by ``simulate_with_coefficients``. Saves an HTML scene file
  plus a screenshot PNG fallback.
* :mod:`render_error_timecourse` -- 2D matplotlib stacked plots of
  position / orientation / clubhead-speed / joint-torque errors versus
  simulation time.
* :mod:`render_fit_quality_card` -- 2D matplotlib summary card with the
  headline RMSE metrics, suitable for dropping into a PR or status
  update.

The 2D plotters are deliberately engine-agnostic (they accept the
generic ``DrakeFitResult`` + canonical ``ClubTarget`` only) so other
parity engines can reuse them via ``from src.engines.physics_engines.drake
.python.motion_matching.viz import render_error_timecourse`` until a
formal ``shared/python/motion_matching/plot_*.py`` package is split out.
"""

from __future__ import annotations

from .render_error_timecourse import render_error_timecourse
from .render_fit_quality_card import render_fit_quality_card
from .render_trajectory_overlay import (
    DrakeFitResult,
    OverlayArtifacts,
    render_trajectory_overlay,
)

__all__ = [
    "DrakeFitResult",
    "OverlayArtifacts",
    "render_error_timecourse",
    "render_fit_quality_card",
    "render_trajectory_overlay",
]
