"""
Engine Manager for Golf Modeling Suite.

This module provides unified management of different physics engines
including MuJoCo, Drake, Pinocchio, MATLAB models, and pendulum models.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from .common_utils import GolfModelingError, setup_logging

logger = setup_logging(__name__)


class EngineType(Enum):
    """Available physics engine types."""

    MUJOCO = "mujoco"
    DRAKE = "drake"
    PINOCCHIO = "pinocchio"
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

        self.current_engine: EngineType | None = None
        self.engine_status: dict[EngineType, EngineStatus] = {}

        # Define engine paths
        self.engine_paths = {
            EngineType.MUJOCO: (self.engines_root / "physics_engines" / "mujoco"),
            EngineType.DRAKE: (self.engines_root / "physics_engines" / "drake"),
            EngineType.PINOCCHIO: (self.engines_root / "physics_engines" / "pinocchio"),
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
            MuJoCoProbe,
            PendulumProbe,
            PinocchioProbe,
        )

        self.probes = {
            EngineType.MUJOCO: MuJoCoProbe(self.suite_root),
            EngineType.DRAKE: DrakeProbe(self.suite_root),
            EngineType.PINOCCHIO: PinocchioProbe(self.suite_root),
            EngineType.PENDULUM: PendulumProbe(self.suite_root),
        }
        self.probe_results: dict[EngineType, Any] = {}

        # Initialize engine status
        self._discover_engines()

        # Engine storage
        self._mujoco_module: Any = None
        self._mujoco_model_dir: Path | None = None
        self._drake_module: Any = None
        self._drake_meshcat: Any = None
        self._pinocchio_module: Any = None
        self._matlab_engine: Any = None
        self._matlab_model_dir: Path | None = None
        self._pendulum_model_dir: Path | None = None

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
            self.current_engine = engine_type
            logger.info(f"Successfully switched to engine: {engine_type.value}")
            return True
        except Exception as e:
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
        """Load a specific engine.

        Args:
            engine_type: The engine to load

        Raises:
            GolfModelingError: If engine loading fails
        """
        logger.info(f"Loading engine: {engine_type.value}")
        self.engine_status[engine_type] = EngineStatus.LOADING

        try:
            # Engine-specific loading logic would go here
            # For now, we just mark it as loaded
            if engine_type == EngineType.MUJOCO:
                self._load_mujoco_engine()
            elif engine_type == EngineType.DRAKE:
                self._load_drake_engine()
            elif engine_type == EngineType.PINOCCHIO:
                self._load_pinocchio_engine()
            elif engine_type in [EngineType.MATLAB_2D, EngineType.MATLAB_3D]:
                self._load_matlab_engine(engine_type)
            elif engine_type == EngineType.PENDULUM:
                self._load_pendulum_engine()

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
            # 1. Run probe to validate dependencies
            from .engine_probes import MuJoCoProbe

            probe = MuJoCoProbe(self.suite_root)
            result = probe.probe()

            if not result.is_available():
                raise GolfModelingError(
                    f"MuJoCo not ready:\n{result.diagnostic_message}\n"
                    f"Fix: {result.get_fix_instructions()}"
                )

            # 2. Import MuJoCo (will fail if not installed)
            import mujoco

            logger.info(f"MuJoCo version {mujoco.__version__} imported successfully")

            # 3. Verify model files exist
            model_dir = self.engine_paths[EngineType.MUJOCO] / "assets"
            if not model_dir.exists():
                raise GolfModelingError(
                    f"MuJoCo model directory not found: {model_dir}\n"
                    f"Expected: {self.engine_paths[EngineType.MUJOCO]}/assets/"
                )

            # 4. Find and validate at least one model file
            model_files = list(model_dir.glob("*.xml"))
            if not model_files:
                raise GolfModelingError(
                    f"No MuJoCo model files (.xml) found in {model_dir}"
                )

            logger.info(f"Found {len(model_files)} MuJoCo models in {model_dir}")

            # 5. Test load a model to verify MuJoCo works
            test_model = model_files[0]
            try:
                _ = mujoco.MjModel.from_xml_path(str(test_model))
                logger.info(
                    f"Successfully validated MuJoCo with test model: {test_model.name}"
                )
            except Exception as e:
                raise GolfModelingError(
                    f"MuJoCo model validation failed for {test_model.name}: {e}"
                ) from e

            # 6. Store loaded state
            self._mujoco_module = mujoco
            self._mujoco_model_dir = model_dir

            logger.info("MuJoCo engine fully loaded and validated")

        except ImportError as e:
            raise GolfModelingError(
                "MuJoCo not installed. Install with: pip install mujoco>=3.2.3"
            ) from e

    def _load_drake_engine(self) -> None:
        """Load Drake engine with full initialization."""
        try:
            # 1. Run probe
            from .engine_probes import DrakeProbe

            probe = DrakeProbe(self.suite_root)
            result = probe.probe()

            if not result.is_available():
                raise GolfModelingError(
                    f"Drake not ready:\n{result.diagnostic_message}\n"
                    f"Fix: {result.get_fix_instructions()}"
                )

            # 2. Import Drake
            import pydrake

            logger.info(f"Drake version {pydrake.__version__} imported successfully")

            # 3. Verify Drake can create systems
            from pydrake.systems.framework import DiagramBuilder

            builder = DiagramBuilder()
            _ = builder.Build()
            logger.info("Drake system creation validated")

            # 4. Check Meshcat availability (for visualization)
            try:
                from pydrake.geometry import Meshcat

                meshcat = Meshcat()
                meshcat_url = meshcat.web_url()
                logger.info(f"Drake Meshcat available at: {meshcat_url}")
                self._drake_meshcat = meshcat
            except Exception as e:
                logger.warning(f"Drake Meshcat unavailable: {e}")
                self._drake_meshcat = None

            # 5. Store loaded state
            self._drake_module = pydrake

            logger.info("Drake engine fully loaded and validated")

        except ImportError as e:
            raise GolfModelingError(
                "Drake not installed. Install with: pip install drake>=1.22.0"
            ) from e

    def _load_pinocchio_engine(self) -> None:
        """Load Pinocchio engine with full initialization."""
        try:
            # 1. Run probe
            from .engine_probes import PinocchioProbe

            probe = PinocchioProbe(self.suite_root)
            result = probe.probe()

            if not result.is_available():
                raise GolfModelingError(
                    f"Pinocchio not ready:\n{result.diagnostic_message}\n"
                    f"Fix: {result.get_fix_instructions()}"
                )

            # 2. Import Pinocchio
            import pinocchio

            logger.info(f"Pinocchio version {pinocchio.__version__} imported")

            # 3. Verify model directory
            model_dir = self.engine_paths[EngineType.PINOCCHIO] / "models"
            if model_dir.exists():
                logger.info(f"Pinocchio models available in {model_dir}")
            else:
                logger.warning(f"Pinocchio model directory not found: {model_dir}")

            # 4. Test basic Pinocchio functionality
            _ = pinocchio.buildSampleModelHumanoid()
            logger.info("Pinocchio humanoid model creation validated")

            # 5. Store loaded state
            self._pinocchio_module = pinocchio

            logger.info("Pinocchio engine fully loaded and validated")

        except ImportError as e:
            raise GolfModelingError(
                "Pinocchio not installed. Install with: pip install pin>=2.6.0"
            ) from e

    def _load_matlab_engine(self, engine_type: EngineType) -> None:
        """Load MATLAB engine with validation."""
        try:
            # 1. Check MATLAB installation
            import matlab.engine

            # 2. Start MATLAB Engine
            logger.info("Starting MATLAB Engine (this may take 30-60 seconds)...")
            engine = matlab.engine.start_matlab()

            # 3. Verify model directory
            model_dir = self.engine_paths[engine_type] / "matlab"
            if not model_dir.exists():
                raise GolfModelingError(
                    f"MATLAB model directory not found: {model_dir}"
                )

            # 4. Add to MATLAB path
            engine.addpath(str(model_dir), nargout=0)

            # 5. Store engine
            self._matlab_engine = engine
            self._matlab_model_dir = model_dir

            logger.info(f"MATLAB engine loaded for {engine_type.value}")

        except ImportError as e:
            raise GolfModelingError(
                "MATLAB Engine for Python not installed.\n"
                "Install from: matlabroot/extern/engines/python\n"
                "Run: python setup.py install"
            ) from e

    def _load_pendulum_engine(self) -> None:
        """Load pendulum models with validation."""
        model_dir = self.engine_paths[EngineType.PENDULUM]

        if not model_dir.exists():
            raise GolfModelingError(f"Pendulum models not found: {model_dir}")

        # Verify Python pendulum implementations exist
        python_dir = model_dir / "python"
        if not python_dir.exists():
            raise GolfModelingError(
                f"Pendulum Python directory not found: {python_dir}"
            )

        logger.info(f"Pendulum models available in {model_dir}")
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
            try:
                # Drake Meshcat cleanup if needed
                pass
            except Exception as e:
                logger.warning(f"Error cleaning up Drake Meshcat: {e}")
            self._drake_meshcat = None

        logger.info("Engine cleanup complete")

    def get_current_engine(self) -> EngineType | None:
        """Get the currently active engine.

        Returns:
            Current engine type or None if no engine is active
        """
        return self.current_engine

    def get_engine_status(self, engine_type: EngineType) -> EngineStatus:
        """Get the status of a specific engine.

        Args:
            engine_type: The engine to check

        Returns:
            Engine status
        """
        return self.engine_status.get(engine_type, EngineStatus.UNAVAILABLE)

    def get_engine_info(self) -> dict[str, Any]:
        """Get information about all engines.

        Returns:
            Dictionary with engine information
        """
        return {
            "current_engine": (
                self.current_engine.value if self.current_engine else None
            ),
            "available_engines": [e.value for e in self.get_available_engines()],
            "engine_status": {e.value: s.value for e, s in self.engine_status.items()},
        }

    def validate_engine_configuration(self, engine_type: EngineType) -> bool:
        """Validate that an engine is properly configured.

        Args:
            engine_type: The engine to validate

        Returns:
            True if engine is properly configured, False otherwise
        """
        if engine_type not in self.engine_status:
            return False

        # Use existing engine_paths to avoid duplication
        base_path = self.engine_paths.get(engine_type)
        if base_path is None:
            return False

        # Check for engine-specific subdirectories
        validation_paths = {
            EngineType.MUJOCO: base_path / "python",
            EngineType.DRAKE: base_path / "python",
            EngineType.PINOCCHIO: base_path / "python",
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
        """Get probe result for a specific engine.

        Args:
            engine_type: The engine to get results for

        Returns:
            Probe result or None if not probed
        """
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
