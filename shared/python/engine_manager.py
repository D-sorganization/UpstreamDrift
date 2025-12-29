"""
Engine Manager for Golf Modeling Suite.

This module provides unified management of different physics engines
including MuJoCo, Drake, Pinocchio, OpenSim, MATLAB models, and pendulum models.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from .common_utils import GolfModelingError, setup_logging
from .interfaces import PhysicsEngine

logger = setup_logging(__name__)


class EngineType(Enum):
    """Available physics engine types."""

    MUJOCO = "mujoco"
    DRAKE = "drake"
    PINOCCHIO = "pinocchio"
    OPENSIM = "opensim"
    MATLAB_2D = "matlab_2d"
    MATLAB_3D = "matlab_3d"
    PENDULUM = "pendulum"


class EngineStatus(Enum):
    """Engine status types."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class EngineManager:
    """Manages different physics engines for golf swing modeling."""

    def __init__(self, suite_root: Path | None = None):
        """Initialize the engine manager.

        Args:
            suite_root: Root directory of the Golf Modeling Suite
        """
        if suite_root is None:
            suite_root = Path(__file__).parent.parent.parent
        self.suite_root = Path(suite_root)
        self.engines_root = self.suite_root / "engines"

        self.current_engine_type: EngineType | None = None
        self.active_physics_engine: PhysicsEngine | None = None
        self.engine_status: dict[EngineType, EngineStatus] = {}

        # Define engine paths
        self.engine_paths = {
            EngineType.MUJOCO: (self.engines_root / "physics_engines" / "mujoco"),
            EngineType.DRAKE: (self.engines_root / "physics_engines" / "drake"),
            EngineType.PINOCCHIO: (self.engines_root / "physics_engines" / "pinocchio"),
            EngineType.OPENSIM: (self.engines_root / "physics_engines" / "opensim"),
            EngineType.MATLAB_2D: (
                self.engines_root / "Simscape_Multibody_Models" / "2D_Golf_Model"
            ),
            EngineType.MATLAB_3D: (
                self.engines_root / "Simscape_Multibody_Models" / "3D_Golf_Model"
            ),
            EngineType.PENDULUM: self.engines_root / "pendulum_models",
        }

        # Initialize engine probes
        from .engine_probes import (
            DrakeProbe,
            MatlabProbe,
            MuJoCoProbe,
            PendulumProbe,
            PinocchioProbe,
        )

        # Note: OpenSimProbe not yet implemented, treating as optional/manual check for now
        self.probes = {
            EngineType.MUJOCO: MuJoCoProbe(self.suite_root),
            EngineType.DRAKE: DrakeProbe(self.suite_root),
            EngineType.PINOCCHIO: PinocchioProbe(self.suite_root),
            EngineType.PENDULUM: PendulumProbe(self.suite_root),
            EngineType.MATLAB_2D: MatlabProbe(self.suite_root, is_3d=False),
            EngineType.MATLAB_3D: MatlabProbe(self.suite_root, is_3d=True),
        }
        self.probe_results: dict[EngineType, Any] = {}

        # Initialize engine status (Discovery)
        self._discover_engines()

        # Engine storage (Legacy / Specifics)
        self._mujoco_module: Any = None
        self._mujoco_model_dir: Path | None = None
        self._drake_module: Any = None
        self._drake_meshcat: Any = None
        self._pinocchio_module: Any = None
        self._matlab_engine: Any = None
        self._matlab_model_dir: Path | None = None
        self._pendulum_model_dir: Path | None = None

        # Plugin Registry (Mapping Type -> Loader Function)
        self._loaders = {
            EngineType.MUJOCO: self._load_mujoco_engine,
            EngineType.DRAKE: self._load_drake_engine,
            EngineType.PINOCCHIO: self._load_pinocchio_engine,
            EngineType.OPENSIM: self._load_opensim_engine,
            EngineType.MATLAB_2D: lambda: self._load_matlab_engine(
                EngineType.MATLAB_2D
            ),
            EngineType.MATLAB_3D: lambda: self._load_matlab_engine(
                EngineType.MATLAB_3D
            ),
            EngineType.PENDULUM: self._load_pendulum_engine,
        }

    def get_active_physics_engine(self) -> PhysicsEngine | None:
        """Get the currently active PhysicsEngine instance."""
        return self.active_physics_engine

    def get_available_engines(self) -> list[EngineType]:
        """Get list of available engines.

        Returns:
            List of available engine types
        """
        return [
            engine
            for engine, status in self.engine_status.items()
            if status == EngineStatus.AVAILABLE
        ]

    def switch_engine(self, engine_type: EngineType) -> bool:
        """Switch to a different physics engine.

        Args:
            engine_type: The engine to switch to

        Returns:
            True if switch was successful, False otherwise
        """
        if engine_type not in self.engine_status:
            logger.error(f"Unknown engine type: {engine_type}")
            return False

        if self.engine_status[engine_type] != EngineStatus.AVAILABLE:
            logger.error(f"Engine {engine_type} is not available")
            return False

        try:
            self._load_engine(engine_type)
            self.current_engine_type = engine_type
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
                logger.info(f"Engine {engine_type.value} is available at {engine_path}")
            else:
                self.engine_status[engine_type] = EngineStatus.UNAVAILABLE
                logger.warning(f"Engine {engine_type.value} not found at {engine_path}")

    def _load_engine(self, engine_type: EngineType) -> None:
        """Load a specific engine using the plugin registry.

        Args:
            engine_type: The engine to load

        Raises:
            GolfModelingError: If engine loading fails
        """
        logger.info(f"Loading engine: {engine_type.value}")
        self.engine_status[engine_type] = EngineStatus.LOADING

        # Unload current engine logic if needed could go here
        self.active_physics_engine = None

        try:
            loader = self._loaders.get(engine_type)
            if not loader:
                raise GolfModelingError(f"No loader defined for {engine_type}")

            loader()

            self.engine_status[engine_type] = EngineStatus.LOADED
            logger.info(f"Successfully loaded engine: {engine_type.value}")

        except Exception as e:
            self.engine_status[engine_type] = EngineStatus.ERROR
            raise GolfModelingError(
                f"Failed to load engine {engine_type.value}: {e}"
            ) from e

    def _load_mujoco_engine(self) -> None:
        """Load MuJoCo engine with full initialization."""
        try:
            # 1. Run probe
            from .engine_probes import MuJoCoProbe

            probe = MuJoCoProbe(self.suite_root)
            result = probe.probe()

            if not result.is_available():
                raise GolfModelingError(
                    f"MuJoCo not ready:\n{result.diagnostic_message}\n"
                    f"Fix: {result.get_fix_instructions()}"
                )

            # 2. Instantiate Interface
            from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
                MuJoCoPhysicsEngine,
            )

            engine = MuJoCoPhysicsEngine()
            self.active_physics_engine = engine

            import mujoco

            self._mujoco_module = mujoco
            # Optional: preload a default model or verify model directory?
            model_dir = self.engine_paths[EngineType.MUJOCO] / "assets"
            self._mujoco_model_dir = model_dir
            logger.info("MuJoCo engine fully loaded and instantiation successful")

        except ImportError as e:
            raise GolfModelingError(
                "MuJoCo requirements not met. Install mujoco>=3.2.3"
            ) from e

    def _load_drake_engine(self) -> None:
        """Load Drake engine with full initialization."""
        try:
            from .engine_probes import DrakeProbe

            probe = DrakeProbe(self.suite_root)
            result = probe.probe()

            if not result.is_available():
                raise GolfModelingError(
                    f"Drake not ready:\n{result.diagnostic_message}\n"
                    f"Fix: {result.get_fix_instructions()}"
                )

            from engines.physics_engines.drake.python.drake_physics_engine import (
                DrakePhysicsEngine,
            )

            engine = DrakePhysicsEngine()
            self.active_physics_engine = engine

            import pydrake

            self._drake_module = pydrake
            logger.info("Drake engine fully loaded and instantiated")

        except ImportError as e:
            raise GolfModelingError("Drake requirements not met.") from e

    def _load_pinocchio_engine(self) -> None:
        """Load Pinocchio engine."""
        try:
            from .engine_probes import PinocchioProbe

            probe = PinocchioProbe(self.suite_root)
            result = probe.probe()

            if not result.is_available():
                raise GolfModelingError(
                    f"Pinocchio not ready:\n{result.diagnostic_message}\n"
                    f"Fix: {result.get_fix_instructions()}"
                )

            from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                PinocchioPhysicsEngine,
            )

            engine = PinocchioPhysicsEngine()
            self.active_physics_engine = engine

            import pinocchio

            self._pinocchio_module = pinocchio
            logger.info("Pinocchio engine fully loaded and instantiated")

        except ImportError as e:
            raise GolfModelingError("Pinocchio requirements not met.") from e

    def _load_opensim_engine(self) -> None:
        """Load OpenSim engine (Stub)."""
        logger.info("Loading OpenSim engine (Stub)...")
        from engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        engine = OpenSimPhysicsEngine()
        self.active_physics_engine = engine
        logger.info("OpenSim engine stub loaded")

    def _load_matlab_engine(self, engine_type: EngineType) -> None:
        """Load MATLAB engine type."""
        # MATLAB doesn't have a PhysicsEngine wrapper yet, so active_physics_engine remains None?
        # Or we should wrap it eventually. For now, adhere to old logic but nullify active_physics_engine
        # since it doesn't support the protocol.
        self.active_physics_engine = None

        try:
            import matlab.engine

            logger.info("Starting MATLAB Engine (this may take 30-60 seconds)...")
            engine = matlab.engine.start_matlab()

            model_dir = self.engine_paths[engine_type] / "matlab"
            if not model_dir.exists():
                raise GolfModelingError(
                    f"MATLAB model directory not found: {model_dir}"
                )

            engine.addpath(str(model_dir), nargout=0)
            self._matlab_engine = engine
            self._matlab_model_dir = model_dir
            logger.info(f"MATLAB engine loaded for {engine_type.value}")

        except ImportError as e:
            raise GolfModelingError("MATLAB Engine for Python not installed.") from e

    def _load_pendulum_engine(self) -> None:
        """Load pendulum models."""
        self.active_physics_engine = None  # No protocol wrapper yet
        model_dir = self.engine_paths[EngineType.PENDULUM]
        if not model_dir.exists():
            raise GolfModelingError(f"Pendulum models not found: {model_dir}")
        self._pendulum_model_dir = model_dir

    def cleanup(self) -> None:
        """Clean up loaded engines."""
        if self._matlab_engine is not None:
            try:
                self._matlab_engine.quit()
                logger.info("MATLAB engine shut down")
            except Exception as e:
                logger.warning(f"Error shutting down MATLAB: {e}")
            self._matlab_engine = None

        if self._drake_meshcat is not None:
            self._drake_meshcat = None

        self.active_physics_engine = None
        self.current_engine_type = None

        logger.info("Engine cleanup complete")

    def get_current_engine(self) -> EngineType | None:
        """Get the currently active engine type."""
        return self.current_engine_type

    def get_engine_status(self, engine_type: EngineType) -> EngineStatus:
        """Get status of a specific engine."""
        return self.engine_status.get(engine_type, EngineStatus.UNAVAILABLE)

    def get_engine_info(self) -> dict[str, Any]:
        """Get information about all engines."""
        return {
            "current_engine": (
                self.current_engine_type.value if self.current_engine_type else None
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
            EngineType.MATLAB_2D: base_path / "matlab",
            EngineType.MATLAB_3D: base_path / "matlab",
            EngineType.PENDULUM: base_path / "python",
        }

        validation_path = validation_paths.get(engine_type, base_path)
        return validation_path.exists()

    def probe_all_engines(self) -> dict[EngineType, Any]:
        """Probe all engines for detailed readiness checks.

        Returns:
            Dictionary mapping engine types to probe results
        """
        for engine_type, probe in self.probes.items():
            self.probe_results[engine_type] = probe.probe()
        return self.probe_results

    def get_probe_result(self, engine_type: EngineType) -> Any:
        """Get probe result for a specific engine."""
        if not self.probe_results:
            self.probe_all_engines()
        return self.probe_results.get(engine_type)

    def get_diagnostic_report(self) -> str:
        """Get human-readable diagnostic report for all engines.

        Returns:
            Formatted diagnostic report
        """
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

