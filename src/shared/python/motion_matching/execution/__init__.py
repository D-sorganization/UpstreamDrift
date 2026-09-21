"""Motion matching packaged execution services and runners.

Issue #10520 / Parent Epic #10508.
Moves execution outside the documentation tree into the core motion matching package.
"""

from __future__ import annotations

from .assets import (
    get_candidate_geometry_spec,
    get_capture_c3d,
    get_native_geometry_spec,
    get_opensim_model,
    resolve_output_root,
)
from .downswing import condition, run_downswing_experiment
from .driver import run_pipeline
from .mjx_export import export_mjx_package, stiffen_weld
from .spec_builder import (
    build_anthropometric_spec,
    leg_extension,
    marker_seeds,
    read_osim,
)

__all__ = [
    "build_anthropometric_spec",
    "condition",
    "export_mjx_package",
    "get_candidate_geometry_spec",
    "get_capture_c3d",
    "get_native_geometry_spec",
    "get_opensim_model",
    "leg_extension",
    "marker_seeds",
    "read_osim",
    "resolve_output_root",
    "run_downswing_experiment",
    "run_pipeline",
    "stiffen_weld",
]
