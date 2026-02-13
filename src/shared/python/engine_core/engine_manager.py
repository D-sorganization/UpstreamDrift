"""
Engine Manager for Golf Modeling Suite.

This module provides unified management of different physics engines
including MuJoCo, Drake, Pinocchio, OpenSim, MATLAB models, and pendulum models.

OBS-001: Migrated to structured logging with structlog for better observability.
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from ..core.contracts import ContractChecker
from ..data_io.common_utils import (
    GolfModelingError,
    get_logger,
    setup_structured_logging,
)
from ..data_io.path_utils import get_src_root
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
                lambda: self.engine_status is not None
                and isinstance(self.engine_status, dict),
                "engine_status must be a non-None dict",
            ),
            (
                lambda: self.engine_paths is not None
                and isinstance(self.engine_paths, dict),
                "engine_paths must be a non-None dict",
            ),
            (
                lambda: self.suite_root is not None
                and isinstance(self.suite_root, Path),
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
        self.engines_root = self.suite_root / "engines"

        self.current_engine: EngineType | None = None
        self.active_physics_engine: PhysicsEngine | None = None
        self.engine_status: dict[EngineType, EngineStatus] = {}

        # Define engine paths (Legacy map - could be moved to registry objects eventually)
        self.engine_paths = {
            EngineType.MUJOCO: (self.engines_root / "physics_engines" / "mujoco"),
            EngineType.DRAKE: (self.engines_root / "physics_engines" / "drake"),
            EngineType.PINOCCHIO: (self.engines_root / "physics_engines" / "pinocchio"),
            EngineType.OPENSIM: (self.engines_root / "physics_engines" / "opensim"),
            EngineType.MYOSIM: (self.engines_root / "physics_engines" / "myosim"),
            EngineType.MATLAB_2D: (
                self.engines_root / "Simscape_Multibody_Models" / "2D_Golf_Model"
            ),
            EngineType.MATLAB_3D: (
                self.engines_root / "Simscape_Multibody_Models" / "3D_Golf_Model"
            ),
            EngineType.PENDULUM: self.engines_root / "pendulum_models",
            EngineType.PUTTING_GREEN: (
                self.engines_root / "physics_engines" / "putting_green"
            ),
        }

        # Initialize probes
        from .engine_probes import (
            DrakeProbe,
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
            # Create a partial to bind suite_root
            factory = partial(loader_func, suite_root=self.suite_root)
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

    def switch_engine(self, engine_type: EngineType) -> bool:
        """Switch to a different physics engine."""
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

    def _discover_engines(self) -> None:
        """Discover available engines by checking their directories."""
        for engine_type, engine_path in self.engine_paths.items():
            if engine_path.exists():
                self.engine_status[engine_type] = EngineStatus.AVAILABLE
                logger.info(
                    "engine_discovered",
                    engine=engine_type.value,
                    path=str(engine_path),
                    status="available",
                )
            else:
                self.engine_status[engine_type] = EngineStatus.UNAVAILABLE
                logger.warning(
                    "engine_not_found",
                    engine=engine_type.value,
                    path=str(engine_path),
                    status="unavailable",
                )

    def _load_engine(self, engine_type: EngineType) -> None:
        """Load a specific engine."""
        logger.info("engine_loading_started", engine=engine_type.value)
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
                "engine_loaded_successfully",
                engine=engine_type.value,
                status="loaded",
            )

        except Exception as e:  # noqa: BLE001 - engine factories may raise anything
            self.engine_status[engine_type] = EngineStatus.ERROR
            logger.error(
                "engine_load_failed",
                engine=engine_type.value,
                error=str(e),
                exc_info=True,
            )
            raise GolfModelingError(
                f"Failed to load engine {engine_type.value}: {e}"
            ) from e

    def _load_matlab_engine(self, engine_type: EngineType) -> None:
        """Load MATLAB engine type."""
        self.active_physics_engine = None
        try:
            import matlab.engine

            logger.info(
                "matlab_engine_starting",
                engine=engine_type.value,
                timeout_seconds=60,
                note="This may take 30-60 seconds",
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
                "matlab_engine_loaded",
                engine=engine_type.value,
                model_dir=str(model_dir),
            )

        except ImportError as e:
            logger.error(
                "matlab_engine_import_failed",
                error="MATLAB Engine for Python not installed",
                exc_info=True,
            )
            raise GolfModelingError("MATLAB Engine for Python not installed.") from e

    def cleanup(self) -> None:
        """Clean up loaded engines."""
        if self._matlab_engine is not None:
            try:
                self._matlab_engine.quit()
                logger.info("matlab_engine_shutdown", status="success")
            except Exception as e:  # noqa: BLE001 - cleanup must not propagate
                logger.warning(
                    "matlab_engine_shutdown_failed", error=str(e), exc_info=True
                )
            self._matlab_engine = None

        self.active_physics_engine = None
        self.current_engine = None
        logger.info("engine_cleanup_complete")

    def get_current_engine(self) -> EngineType | None:
        return self.current_engine

    def get_engine_status(self, engine_type: EngineType) -> EngineStatus:
        return self.engine_status.get(engine_type, EngineStatus.UNAVAILABLE)

    def get_engine_info(self) -> dict[str, Any]:
        return {
            "current_engine": (
                self.current_engine.value if self.current_engine else None
            ),
            "available_engines": [e.value for e in self.get_available_engines()],
            "engine_status": {e.value: s.value for e, s in self.engine_status.items()},
        }

    def validate_engine_configuration(self, engine_type: EngineType) -> bool:
        """Validate engine configuration."""
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
        return validation_path.exists()

    def probe_all_engines(self) -> dict[EngineType, Any]:
        for engine_type, probe in self.probes.items():
            self.probe_results[engine_type] = probe.probe()
        return self.probe_results

    def get_probe_result(self, engine_type: EngineType) -> Any:
        if not self.probe_results:
            self.probe_all_engines()
        return self.probe_results.get(engine_type)

    def get_diagnostic_report(self) -> str:
        if not self.probe_results:
            self.probe_all_engines()

        lines = [
            "",
            "=" * 70,
            "Golf Modeling Suite - Engine Readiness Report",
            "=" * 70,
            "",
        ]

        for _engine_type, result in self.probe_results.items():
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
