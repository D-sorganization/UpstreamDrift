"""bioptim optimal-control layer for the seven-DOF swing (epic #9762).

Everything that knows bioptim's API lives in this subpackage; nothing
outside it may import ``bioptim`` (enforced by
``tests/architecture/test_bioptim_isolation.py``). The package itself never
imports bioptim at import time -- every symbol below is resolved lazily so
``import src.shared.python.optimization.ocp`` succeeds with no optional
extra installed.

Modules:

- ``_compat`` -- availability probe and the ``biorbd_casadi`` / ``tkinter``
  shims (Phase 0).
- ``symbolic_model`` -- :class:`SymbolicSwingModel`: CasADi functions with
  the anthropometric inertials (Phase 1.1).
- ``bioptim_model`` -- :class:`SwingBioModel`, the ``StateDynamics``
  adapter (Phase 1.2).
- ``swing_ocp`` -- max-clubhead-speed OCP (Phase 2).
- ``tracking_ocp`` / ``keypoint_map`` -- keypoint tracking (Phase 3).
- ``parameter_ocp`` -- simultaneous state + parameter estimation (Phase 4).
- ``mhe`` -- moving-horizon variant (Phase 5).
- ``result`` -- adapters back to the flagship result types.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from src.shared.python.optimization.ocp._compat import (
    BIOPTIM_INSTALL_HINT,
    BioptimNotAvailableError,
    bioptim_available,
    bioptim_version,
    require_bioptim,
)

__all__ = [
    "BIOPTIM_INSTALL_HINT",
    "BioptimNotAvailableError",
    "SwingBioModel",
    "SymbolicSwingModel",
    "bioptim_available",
    "bioptim_version",
    "build_max_speed_ocp",
    "build_tracking_ocp",
    "require_bioptim",
    "solve_max_speed_swing",
]

_LAZY: dict[str, str] = {
    "SymbolicSwingModel": "src.shared.python.optimization.ocp.symbolic_model",
    "SwingBioModel": "src.shared.python.optimization.ocp.bioptim_model",
    "build_max_speed_ocp": "src.shared.python.optimization.ocp.swing_ocp",
    "solve_max_speed_swing": "src.shared.python.optimization.ocp.swing_ocp",
    "build_tracking_ocp": "src.shared.python.optimization.ocp.tracking_ocp",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
