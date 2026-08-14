"""BunkerShot3D: a multi-fidelity wedge-design tool (ADR-0032).

Given two wedge sole geometries, which one performs better, in what conditions,
and how confident are we? That is the question this package answers, so the
public surface is organised around the vocabulary of that question rather than
around the simulation backends.

**The subpackages are the API.** Each is re-exported here by name:

======================  ====================================================
:mod:`~bunkershot3d.geometry`     the parametric wedge, its meshes and mass
                                  properties, and the delivered geometry
:mod:`~bunkershot3d.sand`         what the sand is, and whether a requested
                                  bed can physically exist
:mod:`~bunkershot3d.domain`       the narrow value objects the solvers take
:mod:`~bunkershot3d.config`       the YAML loader that assembles them
:mod:`~bunkershot3d.units`        the SI convention and its conversions
:mod:`~bunkershot3d.study`        DOE, sensitivity, surrogates, optimisation
:mod:`~bunkershot3d.provenance`   config hashing, RNG discipline, manifests
:mod:`~bunkershot3d.io`           the versioned result schema
:mod:`~bunkershot3d.backends`     the F3 grain-scale drivers
:mod:`~bunkershot3d.calibration`  experiment-matching calibration
:mod:`~bunkershot3d.kinematics`   swing trajectories and co-simulation
:mod:`~bunkershot3d.postproc`     wrench traces
:mod:`~bunkershot3d.exceptions`   the package error hierarchy
======================  ====================================================

Alongside them, a *curated* set of entry-point names is promoted to the top
level: the value objects, the config, the result reader/writer, the drivers and
the errors. The subpackages are deliberately **not** star-flattened. They export
around two hundred names between them and at least one genuine collision --
``study.rng.SeedRecord`` and ``provenance.rng.SeedRecord`` are different types
with the same name -- so flattening would silently pick a winner. Reach into
the subpackage for anything not listed in ``__all__``.

``__all__`` is authoritative: it is the whole public surface, not a subset of
it, and ``tests/bunkershot3d/test_public_api_8608.py`` enforces that.
"""

# Underscore-aliased: ``__all__`` is authoritative here, so nothing lands in
# this namespace under a public name unless it is part of the public API.
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

__version__ = "0.1.0"

# Subpackages: the primary API surface.
from . import (
    backends,
    ball,
    calibration,
    config,
    domain,
    exceptions,
    geometry,
    io,
    kinematics,
    metrics,
    postproc,
    provenance,
    sand,
    solver,
    units,
)

# Ball model: promoted to the flat surface because a bunker shot's whole point
# is the ball, and callers should not have to reach into a subpackage for it.
from .ball import (
    BallLie,
    BallLieType,
    BallProperties,
    BunkerShotState,
    compute_bunker_launch,
    to_post_impact_state,
)

# Backend drivers
from .backends import ChronoDriver, LiggghtsDriver, MPMDriver

# Calibration
from .calibration import (
    AngleOfReposeExperiment,
    CalibrationOptimizer,
    DrainedShearCellExperiment,
)

# Configuration: the loader/assembler for the value objects below.
from .config import BunkerShotConfig

# Domain value objects (ADR-0032 decision 1)
from .domain import (
    BoundaryCondition,
    ContactMaterial,
    DomainBox,
    GrainPopulation,
    SolverSettings,
    SwingCondition,
    TrajectorySource,
)

# Exceptions
from .exceptions import (
    BackendNotImplementedError,
    BunkerShot3DError,
    BunkerShot3DStateError,
    BunkerShot3DValueError,
    ConfigurationInvalidError,
    DomainInvariantError,
)

# Geometry: the design vector and how it is delivered.
from .geometry import (
    ClubheadGenerator,
    DeliveredGeometry,
    DeliveryCondition,
    GrindPreset,
    MassProperties,
    TriangleMesh,
    WedgeGeometry,
    deliver_wedge,
    get_preset,
    preset_names,
)

# I/O
from .io import BunkerShotResultReader, BunkerShotResultWriter

# Kinematics
from .kinematics import (
    CoSimulator,
    CoupledDoublePendulum,
    SwingTrajectory,
    generate_reference_trajectory,
)

# Post-processing
from .postproc import WrenchTrace

# Provenance
from .provenance import RunManifest, Validity, config_hash, physics_hash

# Sand
from .sand import SandState, usga_reference_sand

if _TYPE_CHECKING:  # pragma: no cover - typing only
    from . import study

#: Subpackages attached on first access rather than at import time.
#:
#: ``study`` imports ``scipy.optimize`` and ``scipy.stats.qmc`` at module
#: level. Importing it eagerly would make ``import bunkershot3d`` -- and
#: therefore ``bunkershot3d.postproc``, which cross-engine code imports for
#: wrench traces -- require the optimisation extras.
#: ``tests/bunkershot3d/test_optimizer.py`` pins that boundary.
_LAZY_SUBMODULES = frozenset({"study"})


def __getattr__(name: str) -> _Any:
    """Attach a lazily-imported subpackage on first access (PEP 562).

    Args:
        name: Attribute being looked up.

    Returns:
        The imported submodule.

    Raises:
        AttributeError: ``name`` is not a lazy submodule of this package.
    """
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the public surface, including not-yet-imported submodules."""
    return sorted(set(globals()) | _LAZY_SUBMODULES)


__all__: list[str] = [
    "AngleOfReposeExperiment",
    "BackendNotImplementedError",
    "BallLie",
    "BallLieType",
    "BallProperties",
    "BoundaryCondition",
    "BunkerShot3DError",
    "BunkerShot3DStateError",
    "BunkerShot3DValueError",
    "BunkerShotConfig",
    "BunkerShotResultReader",
    "BunkerShotResultWriter",
    "BunkerShotState",
    "CalibrationOptimizer",
    "ChronoDriver",
    "ClubheadGenerator",
    "CoSimulator",
    "ConfigurationInvalidError",
    "ContactMaterial",
    "CoupledDoublePendulum",
    "DeliveredGeometry",
    "DeliveryCondition",
    "DomainBox",
    "DomainInvariantError",
    "DrainedShearCellExperiment",
    "GrainPopulation",
    "GrindPreset",
    "LiggghtsDriver",
    "MPMDriver",
    "MassProperties",
    "RunManifest",
    "SandState",
    "SolverSettings",
    "SwingCondition",
    "SwingTrajectory",
    "TrajectorySource",
    "TriangleMesh",
    "Validity",
    "WedgeGeometry",
    "WrenchTrace",
    "__version__",
    "backends",
    "ball",
    "calibration",
    "compute_bunker_launch",
    "config",
    "config_hash",
    "deliver_wedge",
    "domain",
    "exceptions",
    "generate_reference_trajectory",
    "geometry",
    "get_preset",
    "io",
    "kinematics",
    "metrics",
    "physics_hash",
    "postproc",
    "preset_names",
    "provenance",
    "sand",
    "solver",
    "study",
    "to_post_impact_state",
    "units",
    "usga_reference_sand",
]
