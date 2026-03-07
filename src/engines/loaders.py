"""Engine loader functions.

Canonical location for engine loader functions. Previously lived in
``src.shared.python.engine_loaders`` which created an inverted dependency
(shared -> engines). Now lives in ``src.engines.loaders`` which is the
correct dependency direction (engines layer).

Each loader function uses lazy imports to avoid importing concrete engine
implementations at module level.

Design by Contract
------------------
All loader functions enforce postconditions to guarantee the returned engine
is in a usable state. Callers may rely on these guarantees without defensive
checks.

DRY — Factory Pattern
---------------------
The 7 loader functions share an identical 8-step pattern. The shared logic
is factored into :func:`_load_engine_with_probe` to eliminate ~150 LOC of
duplication while preserving each loader's unique import paths and
diagnostic messages.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.python.data_io.common_utils import GolfModelingError
from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

__all__ = [
    "load_mujoco_engine",
    "load_drake_engine",
    "load_pinocchio_engine",
    "load_opensim_engine",
    "load_myosim_engine",
    "load_pendulum_engine",
    "load_putting_green_engine",
    "LOADER_MAP",
]


# ---------------------------------------------------------------------------
# DbC helpers
# ---------------------------------------------------------------------------


def _ensure_engine_loaded(engine: PhysicsEngine, engine_name: str) -> None:
    """DbC postcondition: verify the engine object is non-None after loading.

    Parameters
    ----------
    engine:
        The engine returned by a loader function.
    engine_name:
        Human-readable engine name for the error message.

    Raises
    ------
    GolfModelingError
        If the engine is None or otherwise falsy.
    """
    if engine is None:
        raise GolfModelingError(
            f"DbC postcondition violated: {engine_name} loader returned None. "
            "The engine constructor must not return None."
        )


# ---------------------------------------------------------------------------
# DRY — shared probe-based loading logic
# ---------------------------------------------------------------------------


def _load_engine_with_probe(
    *,
    engine_name: str,
    probe_factory: Callable[[Path], Any],
    engine_factory: Callable[[], PhysicsEngine],
    model_path_fn: Callable[[Path], Path] | None = None,
    load_model: bool = True,
    install_hint: str = "",
    suite_root: Path,
) -> PhysicsEngine:
    """Shared engine-loading scaffold used by all probe-based loaders.

    Preconditions (DbC)
    -------------------
    - ``suite_root`` must be a Path (caller responsibility via type hint).

    Postconditions (DbC)
    --------------------
    - Returned engine is non-None (enforced by :func:`_ensure_engine_loaded`).

    Parameters
    ----------
    engine_name:
        Human-readable engine name used in log and error messages.
    probe_factory:
        Callable(suite_root) -> Probe instance.
    engine_factory:
        Zero-argument callable that creates the engine instance.
    model_path_fn:
        Optional callable(suite_root) -> Path. If provided and the path
        exists, ``engine.load_from_path()`` is called.
    load_model:
        Whether to attempt model loading (default ``True``).
    install_hint:
        Install instructions appended to ImportError messages.
    suite_root:
        Repository root path forwarded to probe and model path resolution.

    Returns
    -------
    PhysicsEngine
        A non-None, probe-verified engine.

    Raises
    ------
    GolfModelingError
        On ImportError, failed probe, or failed DbC postcondition.
    """
    probe = probe_factory(suite_root)
    result = probe.probe()

    if not result.is_available():
        raise GolfModelingError(
            f"{engine_name} not ready:\n{result.diagnostic_message}\n"
            f"Fix: {result.get_fix_instructions()}"
        )

    engine = engine_factory()

    if load_model and model_path_fn is not None:
        model_path = model_path_fn(suite_root)
        if model_path.exists():
            logger.info(f"Loading default {engine_name} model: {model_path}")
            try:
                engine.load_from_path(str(model_path))
            except (ValueError, RuntimeError, AttributeError) as exc:
                logger.warning(
                    f"Failed to load default model into {engine_name} "
                    f"(expected if missing meshes): {exc}"
                )
        else:
            logger.warning(f"Default {engine_name} model not found at {model_path}")

    _ensure_engine_loaded(engine, engine_name)
    return engine  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------


def load_mujoco_engine(suite_root: Path) -> PhysicsEngine:
    """Load MuJoCo engine with full initialization.

    Postcondition: returned engine is non-None (DbC).
    """
    try:
        import mujoco  # noqa: F401

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )
        from src.shared.python.engine_core.engine_probes import MuJoCoProbe

        def _model_path(root: Path) -> Path:
            return (
                root
                / "engines"
                / "physics_engines"
                / "mujoco"
                / "models"
                / "simple_pendulum.xml"
            )

        return _load_engine_with_probe(
            engine_name="MuJoCo",
            probe_factory=MuJoCoProbe,
            engine_factory=lambda: MuJoCoPhysicsEngine(),  # type: ignore[abstract]
            model_path_fn=_model_path,
            install_hint="Install mujoco>=3.2.3",
            suite_root=suite_root,
        )

    except ImportError as e:
        raise GolfModelingError(
            "MuJoCo requirements not met. Install mujoco>=3.2.3"
        ) from e


def load_drake_engine(suite_root: Path) -> PhysicsEngine:
    """Load Drake engine with full initialization.

    Postcondition: returned engine is non-None (DbC).
    """
    try:
        import pydrake  # noqa: F401

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )
        from src.shared.python.engine_core.engine_probes import DrakeProbe

        def _model_path(root: Path) -> Path:
            return (
                root
                / "engines"
                / "physics_engines"
                / "pinocchio"
                / "models"
                / "generated"
                / "golfer.urdf"
            )

        return _load_engine_with_probe(
            engine_name="Drake",
            probe_factory=DrakeProbe,
            engine_factory=lambda: DrakePhysicsEngine(),  # type: ignore[abstract]
            model_path_fn=_model_path,
            install_hint="Install drake>=1.22.0",
            suite_root=suite_root,
        )

    except ImportError as e:
        raise GolfModelingError("Drake requirements not met.") from e


def load_pinocchio_engine(suite_root: Path) -> PhysicsEngine:
    """Load Pinocchio engine.

    Postcondition: returned engine is non-None (DbC).
    """
    try:
        import pinocchio  # noqa: F401

        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )
        from src.shared.python.engine_core.engine_probes import PinocchioProbe

        def _model_path(root: Path) -> Path:
            return (
                root
                / "engines"
                / "physics_engines"
                / "pinocchio"
                / "models"
                / "generated"
                / "golfer.urdf"
            )

        return _load_engine_with_probe(
            engine_name="Pinocchio",
            probe_factory=PinocchioProbe,
            engine_factory=lambda: PinocchioPhysicsEngine(),  # type: ignore[abstract]
            model_path_fn=_model_path,
            install_hint="Install pin>=2.6.0",
            suite_root=suite_root,
        )

    except ImportError as e:
        raise GolfModelingError("Pinocchio requirements not met.") from e


def load_opensim_engine(suite_root: Path) -> PhysicsEngine:
    """Load OpenSim engine.

    Postcondition: returned engine is non-None (DbC).
    """
    try:
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )
        from src.shared.python.engine_core.engine_probes import OpenSimProbe

        return _load_engine_with_probe(
            engine_name="OpenSim",
            probe_factory=OpenSimProbe,
            engine_factory=lambda: OpenSimPhysicsEngine(),  # type: ignore[abstract]
            model_path_fn=None,
            load_model=False,
            install_hint="Install opensim>=4.4.0",
            suite_root=suite_root,
        )

    except ImportError as e:
        raise GolfModelingError("OpenSim requirements not met.") from e


def load_myosim_engine(suite_root: Path) -> PhysicsEngine:
    """Load MyoSim engine.

    Postcondition: returned engine is non-None (DbC).
    """
    try:
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )
        from src.shared.python.engine_core.engine_probes import MyoSimProbe

        return _load_engine_with_probe(
            engine_name="MyoSim",
            probe_factory=MyoSimProbe,
            engine_factory=lambda: MyoSuitePhysicsEngine(),  # type: ignore[abstract,no-any-return]
            model_path_fn=None,
            load_model=False,
            install_hint="Install myosuite>=2.0.0",
            suite_root=suite_root,
        )

    except ImportError as e:
        raise GolfModelingError("MyoSim requirements not met.") from e


def load_pendulum_engine(suite_root: Path) -> PhysicsEngine:  # noqa: ARG001
    """Load Pendulum (double-pendulum) engine.

    Postcondition: returned engine is non-None (DbC).
    """
    try:
        from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
            PendulumPhysicsEngine,
        )

        engine = PendulumPhysicsEngine()
        logger.info("Pendulum engine loaded successfully")

        # DbC postcondition
        _ensure_engine_loaded(engine, "Pendulum")
        return engine  # type: ignore[return-value]

    except ImportError as e:
        raise GolfModelingError("Pendulum engine not found.") from e


def load_putting_green_engine(suite_root: Path) -> PhysicsEngine:  # noqa: ARG001
    """Load Putting Green engine.

    Postcondition: returned engine is non-None (DbC).
    """
    try:
        from src.engines.physics_engines.putting_green import PuttingGreenSimulator

        # Putting green doesn't need probing - it's always available as pure Python
        simulator = PuttingGreenSimulator()
        logger.info("Putting Green engine loaded successfully")

        # DbC postcondition
        _ensure_engine_loaded(simulator, "PuttingGreen")  # type: ignore[arg-type]
        return simulator  # type: ignore[return-value]

    except ImportError as e:
        raise GolfModelingError("Putting Green engine not found.") from e


# Helper for loaders map
LOADER_MAP: dict[EngineType, Callable[[Path], PhysicsEngine]] = {
    EngineType.MUJOCO: load_mujoco_engine,
    EngineType.DRAKE: load_drake_engine,
    EngineType.PINOCCHIO: load_pinocchio_engine,
    EngineType.OPENSIM: load_opensim_engine,
    EngineType.MYOSIM: load_myosim_engine,
    EngineType.PENDULUM: load_pendulum_engine,
    EngineType.PUTTING_GREEN: load_putting_green_engine,
}
