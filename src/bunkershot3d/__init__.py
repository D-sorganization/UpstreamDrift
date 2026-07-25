"""BunkerShot3D: A 3-D simulation of a golf bunker shot.

Re-exports the public API from all subpackages so consumers can import
directly from ``bunkershot3d`` instead of reaching into submodules.

The re-exports are **lazy** (PEP 562 module ``__getattr__``).  Eagerly
importing ``.backends`` here pulled MuJoCo into every consumer of any
BunkerShot3D symbol — including the launcher's startup import chain via
``simulation_backends.wrench_extractor`` — so a broken MuJoCo wheel crashed the
whole launcher before its window appeared (#8084).  Lazy binding keeps
``import bunkershot3d`` free of native-library side effects while leaving the
public API identical.
"""

from __future__ import annotations

from typing import Any, Final

__version__ = "0.1.0"

#: Public name -> submodule that defines it.
_EXPORTS: Final[dict[str, str]] = {
    "ChronoDriver": ".backends",
    "LiggghtsDriver": ".backends",
    "MPMDriver": ".backends",
    "AngleOfReposeExperiment": ".calibration",
    "CalibrationOptimizer": ".calibration",
    "DrainedShearCellExperiment": ".calibration",
    "BackendNotImplementedError": ".exceptions",
    "ClubheadGenerator": ".geometry",
    "BunkerShotResultReader": ".io",
    "BunkerShotResultWriter": ".io",
    "CoSimulator": ".kinematics",
    "CoupledDoublePendulum": ".kinematics",
    "SwingTrajectory": ".kinematics",
    "generate_reference_trajectory": ".kinematics",
    "WrenchTrace": ".postproc",
}


def __getattr__(name: str) -> Any:
    """Resolve a public re-export on first access.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__: list[str] = [
    "AngleOfReposeExperiment",
    "BackendNotImplementedError",
    "BunkerShotResultReader",
    "BunkerShotResultWriter",
    "CalibrationOptimizer",
    "ChronoDriver",
    "ClubheadGenerator",
    "CoSimulator",
    "CoupledDoublePendulum",
    "DrainedShearCellExperiment",
    "LiggghtsDriver",
    "MPMDriver",
    "SwingTrajectory",
    "WrenchTrace",
    "__version__",
    "generate_reference_trajectory",
]
