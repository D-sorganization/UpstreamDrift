"""
Engine Manager for Golf Modeling Suite.

This module provides unified management of different physics engines
including MuJoCo, Drake, Pinocchio, OpenSim, MATLAB models, and pendulum models.
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from src.shared.python.config.model_registry import ModelRegistry

from ..core.contracts import ContractChecker, PreconditionError, precondition
from ..core.error_utils import EngineLaunchError
from ..data_io.common_utils import (
    GolfModelingError,
    get_logger,
    setup_structured_logging,
)
from ..data_io.path_utils import get_repo_root, get_src_root
from .engine_availability import EngineStatus as RuntimeEngineStatus
from .engine_availability import get_engine_status as get_runtime_engine_status
from .engine_registry import (
    EngineRegistration,
    EngineStatus,
    EngineType,
    get_registry,
)
from .interfaces import PhysicsEngine

# Configure structured logging
setup_structured_logging()
logger = get_logger(__name__)


# Map runtime-backed engine types to the importable Python package that the
# availability layer probes. Engines absent from this map (pendulum-family,
# putting-green, MATLAB) have no importable runtime dependency and are gated by
# source/asset presence alone. Centralizing this (LOD/DRY) keeps the discovery
# guard in one place instead of repeating ad hoc imports per caller (#6880).
_RUNTIME_DEPENDENCY_NAMES: dict[EngineType, str] = {
    EngineType.MUJOCO: "mujoco",
    EngineType.DRAKE: "drake",
    EngineType.PINOCCHIO: "pinocchio",
    EngineType.JAXSIM: "jaxsim.api",
    EngineType.OPENSIM: "opensim",
    EngineType.MYOSIM: "myosuite",
}


def runtime_dependency_name(engine_type: EngineType) -> str | None:
    """Return the importable runtime package name for an engine, if any.

    Returns ``None`` for engines that have no importable runtime dependency
    (pendulum-family, putting-green, MATLAB), which remain gated on source or
    asset presence rather than a Python import.
    """
    if engine_type is None:
        raise ValueError("engine_type must be provided")
    return _RUNTIME_DEPENDENCY_NAMES.get(engine_type)


class EngineManager(ContractChecker):
    """Manages different physics engines for golf swing modeling.

    Refactored to use EngineRegistry (Decoupling Phase).

    Design by Contract:
        Invariants:
            - engine_status dict is never None
            - engine_paths dict is never None
            - suite_root is a valid Path object
    """

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for EngineManager."""
        return [
            (
                lambda: (
                    self.engine_status is not None
                    and isinstance(self.engine_status, dict)
                ),
                "engine_status must be a non-None dict",
            ),
            (
                lambda: (
                    self.engine_paths is not None
                    and isinstance(self.engine_paths, dict)
                ),
                "engine_paths must be a non-None dict",
            ),
            (
                lambda: (
                    self.provider_engine_paths is not None
                    and isinstance(self.provider_engine_paths, dict)
                ),
                "provider_engine_paths must be a non-None dict",
            ),
            (
                lambda: (
                    self.suite_root is not None and isinstance(self.suite_root, Path)
                ),
                "suite_root must be a valid Path",
            ),
        ]

    def __init__(self, suite_root: Path | None = None) -> None:
        """Initialize the engine manager.

        Args:
            suite_root: Root directory of the Golf Modeling Suite
        """
        if suite_root is None:
            suite_root = get_src_root()
        self.suite_root = Path(suite_root)
        # Engines live under src/engines. When suite_root is the repo root
        # (not src/), prefer src/engines which has the complete engine set.
        src_engines = self.suite_root / "src" / "engines"
        direct_engines = self.suite_root / "engines"
        if src_engines.exists():
            self.engines_root = src_engines
        else:
            self.engines_root = direct_engines

        self.current_engine: EngineType | None = None
        self.active_physics_engine: PhysicsEngine | None = None
        self.engine_status: dict[EngineType, EngineStatus] = {}
        self.provider_engine_paths: dict[EngineType, tuple[Path, ...]] = {}

        # Define engine paths (Legacy map - could be moved to registry objects eventually)
        self.engine_paths = {
            EngineType.MUJOCO: (self.engines_root / "physics_engines" / "mujoco"),
            EngineType.DRAKE: (self.engines_root / "physics_engines" / "drake"),
            EngineType.PINOCCHIO: (self.engines_root / "physics_engines" / "pinocchio"),
            EngineType.JAXSIM: (self.engines_root / "physics_engines" / "jaxsim"),
            EngineType.OPENSIM: (self.engines_root / "physics_engines" / "opensim"),
            EngineType.MYOSIM: (self.engines_root / "physics_engines" / "myosuite"),
            EngineType.MATLAB_2D: (
                self.engines_root / "Simscape_Multibody_Models" / "2D_Golf_Model"
            ),
            EngineType.MATLAB_3D: (
                self.engines_root / "Simscape_Multibody_Models" / "3D_Golf_Model"
            ),
            EngineType.PENDULUM: self.engines_root / "pendulum_models",
            EngineType.GOLF_SWING_PENDULUM: (
                self.engines_root / "physics_engines" / "pendulum"
            ),
            EngineType.PUTTING_GREEN: (
                self.engines_root / "physics_engines" / "putting_green"
            ),
        }

        # Initialize probes
        from .engine_probes import (
            DrakeProbe,
            JaxSimProbe,
            MatlabProbe,
            MuJoCoProbe,
            MyoSimProbe,
            OpenSimProbe,
            PendulumProbe,
            PinocchioProbe,
        )

        self.probes = {
            EngineType.MUJOCO: MuJoCoProbe(self.suite_root),
            EngineType.DRAKE: DrakeProbe(self.suite_root),
            EngineType.PINOCCHIO: PinocchioProbe(self.suite_root),
            EngineType.JAXSIM: JaxSimProbe(self.suite_root),
            EngineType.OPENSIM: OpenSimProbe(self.suite_root),
            EngineType.MYOSIM: MyoSimProbe(self.suite_root),
            EngineType.PENDULUM: PendulumProbe(self.suite_root),
            EngineType.MATLAB_2D: MatlabProbe(self.suite_root, is_3d=False),
            EngineType.MATLAB_3D: MatlabProbe(self.suite_root, is_3d=True),
        }
        self.probe_results: dict[EngineType, Any] = {}

        # Register standard loaders (lazy import to avoid shared -> engines
        # module-level dependency; loaders now live in src.engines.loaders)
        from src.engines.loaders import LOADER_MAP

        registry = get_registry()
        for engine_type, loader_func in LOADER_MAP.items():
            # Create a partial to bind suite_root (positional, not keyword)
            factory = partial(loader_func, self.suite_root)
            registry.register(
                EngineRegistration(
                    engine_type=engine_type,
                    factory=factory,
                    registration_path=self.engine_paths.get(engine_type),
                    probe_class=(
                        type(self.probes.get(engine_type))
                        if engine_type in self.probes
                        else None
                    ),
                )
            )

        # Initialize engine status
        self.provider_engine_paths = self._discover_provider_engine_paths()
        self._discover_engines()

        # Engine storage (Legacy / Specifics)
        self._matlab_engine: Any = None
        self._matlab_model_dir: Path | None = None
        self._pendulum_model_dir: Path | None = None

    def get_active_physics_engine(self) -> PhysicsEngine | None:
        """Get the currently active PhysicsEngine instance."""
        return self.active_physics_engine

    def get_available_engines(self) -> list[EngineType]:
        """Get list of available engines."""
        return [
            engine
            for engine, status in self.engine_status.items()
            if status == EngineStatus.AVAILABLE
        ]

    @precondition(
        lambda self, engine_type: engine_type is not None,
        "Engine type must not be None",
    )
    def switch_engine(self, engine_type: EngineType) -> bool:
        """Switch to a different physics engine."""
        if engine_type is None:
            raise ValueError("engine_type must be provided")
        if engine_type not in self.engine_status:
            logger.error(f"Unknown engine type: {engine_type}")
            return False

        if self.engine_status[engine_type] != EngineStatus.AVAILABLE:
            logger.error(f"Engine {engine_type} is not available")
            return False

        try:
            self._load_engine(engine_type)
            self.current_engine = engine_type
            logger.info(f"Successfully switched to engine: {engine_type.value}")
            return True
        except GolfModelingError as e:
            logger.error(f"Failed to switch to engine {engine_type}: {e}")
            self.engine_status[engine_type] = EngineStatus.ERROR
            return False

    @staticmethod
    def _runtime_ready(engine_type: EngineType) -> bool:
        """Return True when an engine's runtime dependency is importable.

        DbC: ``EngineStatus.AVAILABLE`` for a runtime-backed engine means both
        adapter/source presence *and* runtime dependency readiness (#6880,
        #6884). Engines with no importable runtime dependency
        (:func:`runtime_dependency_name` returns ``None``) are always
        considered runtime-ready and remain gated on source presence alone.
        """
        dependency = runtime_dependency_name(engine_type)
        if dependency is None:
            return True
        return get_runtime_engine_status(dependency) == RuntimeEngineStatus.AVAILABLE

    def _discover_engines(self) -> None:
        """Discover available engines by checking directories and runtime deps.

        A runtime-backed engine is only ``AVAILABLE`` when its adapter/source
        path (or provider path) exists *and* its runtime dependency imports
        successfully. Path presence with a missing runtime dependency is
        reported as ``UNAVAILABLE`` so callers cannot mistake source presence
        for a live engine (#6884).
        """
        for engine_type, engine_path in self.engine_paths.items():
            provider_paths = self.provider_engine_paths.get(engine_type, ())
            available_provider_path = next(
                (path for path in provider_paths if path.exists()),
                None,
            )
            source_present = engine_path.exists() or available_provider_path is not None
            if not source_present:
                self.engine_status[engine_type] = EngineStatus.UNAVAILABLE
                logger.warning(
                    "engine_not_found engine=%s path=%s status=unavailable",
                    engine_type.value,
                    engine_path,
                )
                continue

            if not self._runtime_ready(engine_type):
                self.engine_status[engine_type] = EngineStatus.UNAVAILABLE
                logger.info(
                    "engine_runtime_missing engine=%s dependency=%s status=unavailable",
                    engine_type.value,
                    runtime_dependency_name(engine_type),
                )
                continue

            self.engine_status[engine_type] = EngineStatus.AVAILABLE
            discovered_path = (
                engine_path if engine_path.exists() else (available_provider_path)
            )
            logger.info(
                "engine_discovered engine=%s path=%s status=available",
                engine_type.value,
                discovered_path,
            )

    def _load_engine(self, engine_type: EngineType) -> None:
        """Load a specific engine."""
        if engine_type is None:
            raise ValueError("engine_type must be provided")
        logger.info("engine_loading_started engine=%s", engine_type.value)
        self.engine_status[engine_type] = EngineStatus.LOADING
        self.active_physics_engine = None

        try:
            # Handle special cases (MATLAB) that don't conform to standard PhysicsEngine yet
            if engine_type in (EngineType.MATLAB_2D, EngineType.MATLAB_3D):
                self._load_matlab_engine(engine_type)
            else:
                # Standard Registry Loading
                registry = get_registry()
                registration = registry.get(engine_type)
                if not registration:
                    # Fallback or error
                    raise GolfModelingError(f"No registration found for {engine_type}")

                # Instantiate
                engine = registration.factory()
                self.active_physics_engine = engine

            self.engine_status[engine_type] = EngineStatus.LOADED
            logger.info(
                "engine_loaded_successfully engine=%s status=loaded",
                engine_type.value,
            )

        except GolfModelingError:
            self.engine_status[engine_type] = EngineStatus.ERROR
            raise
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as e:
            self.engine_status[engine_type] = EngineStatus.ERROR
            logger.error(
                "engine_load_failed engine=%s error=%s",
                engine_type.value,
                e,
                exc_info=True,
            )
            raise EngineLaunchError(engine_type.value, reason=str(e)) from e

    def _load_matlab_engine(self, engine_type: EngineType) -> None:
        """Load MATLAB engine type."""
        if engine_type is None:
            raise ValueError("engine_type must be provided")
        self.active_physics_engine = None
        try:
            import matlab.engine

            logger.info(
                "matlab_engine_starting engine=%s timeout_seconds=60 (This may take 30-60 seconds)",
                engine_type.value,
            )
            # REL-001: Add timeout to prevent infinite hangs
            engine = matlab.engine.start_matlab(timeout=60.0)

            model_dir = self.engine_paths[engine_type] / "matlab"
            if not model_dir.exists():
                raise GolfModelingError(
                    f"MATLAB model directory not found: {model_dir}"
                )

            engine.addpath(str(model_dir), nargout=0)
            self._matlab_engine = engine
            self._matlab_model_dir = model_dir
            logger.info(
                "matlab_engine_loaded engine=%s model_dir=%s",
                engine_type.value,
                model_dir,
            )

        except ImportError as e:
            logger.error(
                "matlab_engine_import_failed: MATLAB Engine for Python not installed",
                exc_info=True,
            )
            raise GolfModelingError("MATLAB Engine for Python not installed.") from e

    def cleanup(self) -> None:
        """Clean up loaded engines."""
        if self._matlab_engine is not None:
            try:
                self._matlab_engine.quit()
                logger.info("matlab_engine_shutdown status=success")
            except (RuntimeError, OSError) as e:
                logger.warning(
                    "matlab_engine_shutdown_failed error=%s", e, exc_info=True
                )
            self._matlab_engine = None

        self.active_physics_engine = None
        self.current_engine = None
        logger.info("engine_cleanup_complete")

    def get_current_engine(self) -> EngineType | None:
        """Return the currently active engine type."""
        return self.current_engine

    def get_engine_status(self, engine_type: EngineType) -> EngineStatus:
        """Return the availability status of the specified engine."""
        return self.engine_status.get(engine_type, EngineStatus.UNAVAILABLE)

    def get_engine_info(self) -> dict[str, Any]:
        """Return a summary dict of current engine, available engines, and statuses."""
        return {
            "current_engine": (
                self.current_engine.value if self.current_engine else None
            ),
            "available_engines": [e.value for e in self.get_available_engines()],
            "engine_status": {e.value: s.value for e, s in self.engine_status.items()},
        }

    def validate_engine_configuration(self, engine_type: EngineType) -> bool:
        """Validate engine configuration."""
        if engine_type is None:
            raise ValueError("engine_type must be provided")
        if engine_type not in self.engine_status:
            return False

        base_path = self.engine_paths.get(engine_type)
        if base_path is None:
            return False

        validation_paths = {
            EngineType.MUJOCO: base_path / "python",
            EngineType.DRAKE: base_path / "python",
            EngineType.PINOCCHIO: base_path / "python",
            EngineType.OPENSIM: base_path / "python",
            EngineType.MYOSIM: base_path / "python",
            EngineType.MATLAB_2D: base_path / "matlab",
            EngineType.MATLAB_3D: base_path / "matlab",
            EngineType.PENDULUM: base_path / "python",
            EngineType.PUTTING_GREEN: base_path / "python",
        }

        validation_path = validation_paths.get(engine_type, base_path)
        if validation_path.exists():
            return True

        provider_paths = self.provider_engine_paths.get(engine_type, ())
        return any(path.exists() for path in provider_paths)

    def _discover_provider_engine_paths(self) -> dict[EngineType, tuple[Path, ...]]:
        """Discover provider-backed engine roots from the shared model registry."""
        config_path = self._get_model_registry_path()
        if config_path is None:
            return {}

        registry = ModelRegistry(config_path)
        # Safely resolve repository root to include workspace parent for sibling provider paths
        approved_roots = [self.suite_root, self.suite_root.parent]
        try:
            repo_root = get_repo_root()
            approved_roots.extend([repo_root, repo_root.parent])
        except (OSError, IndexError) as exc:
            logger.debug("Could not resolve repo root for provider paths: %s", exc)

        try:
            grouped_paths = registry.get_engine_provider_paths(
                self.suite_root,
                approved_roots=tuple(approved_roots),
            )
        except PreconditionError as exc:
            logger.warning(
                "Skipping provider-backed engine path discovery after model "
                "registry path contract violation: %s",
                exc,
            )
            return {}
        provider_paths: dict[EngineType, tuple[Path, ...]] = {}
        for engine_name, paths in grouped_paths.items():
            try:
                engine_type = EngineType(engine_name)
            except ValueError:
                logger.debug(
                    "Skipping provider engine path for unknown engine type '%s'",
                    engine_name,
                )
                continue
            provider_paths[engine_type] = paths
        return provider_paths

    def _get_model_registry_path(self) -> Path | None:
        """Resolve the shared model registry path relative to suite_root."""
        if self.suite_root.name == "src":
            candidates = (
                self.suite_root / "config" / "models.yaml",
                self.suite_root.parent / "config" / "models.yaml",
            )
        else:
            candidates = (
                self.suite_root / "src" / "config" / "models.yaml",
                self.suite_root / "config" / "models.yaml",
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def probe_all_engines(self) -> dict[EngineType, Any]:
        """Probe every registered engine and cache the results."""
        for engine_type, probe in self.probes.items():
            self.probe_results[engine_type] = probe.probe()
        return self.probe_results

    def get_probe_result(self, engine_type: EngineType) -> Any:
        """Return the probe result for a specific engine, probing first if needed."""
        if engine_type is None:
            raise ValueError("engine_type must be provided")
        if not self.probe_results:
            self.probe_all_engines()
        return self.probe_results.get(engine_type)

    def get_diagnostic_report(self) -> str:
        """Generate a human-readable diagnostic report for all engines."""
        if not self.probe_results:
            self.probe_all_engines()

        lines = [
            "",
            "=" * 70,
            "Golf Modeling Suite - Engine Readiness Report",
            "=" * 70,
            "",
        ]

        for result in self.probe_results.values():
            status_icon = "✅" if result.is_available() else "❌"
            lines.append(f"{status_icon} {result.engine_name.upper()}")
            lines.append(f"   Status: {result.status.value}")

            if result.version:
                lines.append(f"   Version: {result.version}")

            if result.missing_dependencies:
                lines.append(f"   Missing: {', '.join(result.missing_dependencies)}")

            lines.append(f"   {result.diagnostic_message}")

            if not result.is_available():
                fix = result.get_fix_instructions()
                lines.append(f"   Fix: {fix}")

            lines.append("")

        lines.append("=" * 70)
        lines.append("")

        return "\n".join(lines)
