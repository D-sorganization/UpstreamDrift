"""MuJoCo motion-matching visualization helpers.

Public surface (per VISUALIZATION_SPEC.md three required views):
    render_trajectory_overlay  -- View 1.
    render_error_timecourse    -- View 2.
    render_fit_quality_card    -- View 3.
    FitResult, VizOptions      -- minimal viz-side dataclasses used by the
                                  three render functions above.
"""

from __future__ import annotations

from .render_swing import (
    FitResult,
    VizOptions,
    render_error_timecourse,
    render_fit_quality_card,
    render_trajectory_overlay,
)

__all__ = [
    "FitResult",
    "VizOptions",
    "render_error_timecourse",
    "render_fit_quality_card",
    "render_trajectory_overlay",
]
