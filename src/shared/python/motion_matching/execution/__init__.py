"""Motion matching packaged execution services and runners.

Issue #10520 / Parent Epic #10508.
Moves execution outside the documentation tree into the core motion matching package.

Importing this package does **not** load the downswing driver or pipeline — those pull
heavy optional dependencies (e.g. matplotlib).  Submodules such as
:mod:`.grip_selection` stay importable without that cost (#14748).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "build_anthropometric_spec",
    "condition",
    "export_mjx_package",
    "get_candidate_geometry_spec",
    "get_capture_c3d",
    "get_club_markers",
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

_SUBPACKAGE_NAMES = frozenset(
    {"assets", "downswing", "driver", "mjx_export", "spec_builder"}
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # assets
    "get_candidate_geometry_spec": (
        "src.shared.python.motion_matching.execution.assets",
        "get_candidate_geometry_spec",
    ),
    "get_capture_c3d": (
        "src.shared.python.motion_matching.execution.assets",
        "get_capture_c3d",
    ),
    "get_club_markers": (
        "src.shared.python.motion_matching.execution.assets",
        "get_club_markers",
    ),
    "get_native_geometry_spec": (
        "src.shared.python.motion_matching.execution.assets",
        "get_native_geometry_spec",
    ),
    "get_opensim_model": (
        "src.shared.python.motion_matching.execution.assets",
        "get_opensim_model",
    ),
    "resolve_output_root": (
        "src.shared.python.motion_matching.execution.assets",
        "resolve_output_root",
    ),
    # downswing
    "condition": (
        "src.shared.python.motion_matching.execution.downswing",
        "condition",
    ),
    "run_downswing_experiment": (
        "src.shared.python.motion_matching.execution.downswing",
        "run_downswing_experiment",
    ),
    # driver
    "run_pipeline": (
        "src.shared.python.motion_matching.execution.driver",
        "run_pipeline",
    ),
    # mjx_export
    "export_mjx_package": (
        "src.shared.python.motion_matching.execution.mjx_export",
        "export_mjx_package",
    ),
    "stiffen_weld": (
        "src.shared.python.motion_matching.execution.mjx_export",
        "stiffen_weld",
    ),
    # spec_builder
    "build_anthropometric_spec": (
        "src.shared.python.motion_matching.execution.spec_builder",
        "build_anthropometric_spec",
    ),
    "leg_extension": (
        "src.shared.python.motion_matching.execution.spec_builder",
        "leg_extension",
    ),
    "marker_seeds": (
        "src.shared.python.motion_matching.execution.spec_builder",
        "marker_seeds",
    ),
    "read_osim": (
        "src.shared.python.motion_matching.execution.spec_builder",
        "read_osim",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGE_NAMES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
