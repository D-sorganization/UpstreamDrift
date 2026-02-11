"""Engine loader functions.

Canonical location for engine loader functions. Previously lived in
``src.shared.python.engine_loaders`` which created an inverted dependency
(shared -> engines). Now lives in ``src.engines.loaders`` which is the
correct dependency direction (engines layer).

Each loader function uses lazy imports to avoid importing concrete engine
implementations at module level.
"""

from pathlib import Path

from src.shared.python.data_io.common_utils import GolfModelingError
from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def load_mujoco_engine(suite_root: Path) -> PhysicsEngine:
    """Load MuJoCo engine with full initialization."""
    try:
        import mujoco  # noqa: F401

        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )
        from src.shared.python.engine_core.engine_probes import MuJoCoProbe

        probe = MuJoCoProbe(suite_root)
        result = probe.probe()

        if not result.is_available():
            raise GolfModelingError(
                f"MuJoCo not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )

        engine = MuJoCoPhysicsEngine()  # type: ignore[abstract]

        # Load default model to verify engine readiness
        model_path = (
            suite_root
            / "engines"
            / "physics_engines"
            / "mujoco"
            / "models"
            / "simple_pendulum.xml"
        )
        if model_path.exists():
            logger.info(f"Loading default MuJoCo model: {model_path}")
            engine.load_from_path(str(model_path))
        else:
            logger.warning(f"Default MuJoCo model not found at {model_path}")

        return engine  # type: ignore[no-any-return]

    except ImportError as e:
        raise GolfModelingError(
            "MuJoCo requirements not met. Install mujoco>=3.2.3"
        ) from e


def load_drake_engine(suite_root: Path) -> PhysicsEngine:
    """Load Drake engine with full initialization."""
    try:
        import pydrake  # noqa: F401

        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )
        from src.shared.python.engine_core.engine_probes import DrakeProbe

        probe = DrakeProbe(suite_root)
        result = probe.probe()

        if not result.is_available():
            raise GolfModelingError(
                f"Drake not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )

        engine = DrakePhysicsEngine()  # type: ignore[abstract]

        # Try to load the shared golfer URDF if available
        urdf_path = (
            suite_root
            / "engines"
            / "physics_engines"
            / "pinocchio"
            / "models"
            / "generated"
            / "golfer.urdf"
        )
        if urdf_path.exists():
            logger.info(
                f"Attempting to load shared golfer URDF into Drake: {urdf_path}"
            )
            try:
                engine.load_from_path(str(urdf_path))
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning(
                    f"Failed to load default URDF into Drake (expected if missing meshes): {e}"
                )
        else:
            logger.warning(f"Default URDF not found at {urdf_path}")

        return engine  # type: ignore[no-any-return]

    except ImportError as e:
        raise GolfModelingError("Drake requirements not met.") from e


def load_pinocchio_engine(suite_root: Path) -> PhysicsEngine:
    """Load Pinocchio engine."""
    try:
        import pinocchio  # noqa: F401

        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )
        from src.shared.python.engine_core.engine_probes import PinocchioProbe

        probe = PinocchioProbe(suite_root)
        result = probe.probe()

        if not result.is_available():
            raise GolfModelingError(
                f"Pinocchio not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )

        engine = PinocchioPhysicsEngine()  # type: ignore[abstract]

        # Load default golfer URDF
        model_path = (
            suite_root
            / "engines"
            / "physics_engines"
            / "pinocchio"
            / "models"
            / "generated"
            / "golfer.urdf"
        )
        if model_path.exists():
            logger.info(f"Loading default Pinocchio model: {model_path}")
            engine.load_from_path(str(model_path))
        else:
            logger.warning(f"Default Pinocchio model not found at {model_path}")

        return engine  # type: ignore[no-any-return]

    except ImportError as e:
        raise GolfModelingError("Pinocchio requirements not met.") from e


def load_opensim_engine(suite_root: Path) -> PhysicsEngine:
    """Load OpenSim engine."""
    try:
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )
        from src.shared.python.engine_core.engine_probes import OpenSimProbe

        probe = OpenSimProbe(suite_root)
        result = probe.probe()

        if not result.is_available():
            raise GolfModelingError(
                f"OpenSim not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )

        engine = OpenSimPhysicsEngine()  # type: ignore[abstract]
        return engine  # type: ignore[no-any-return]

    except ImportError as e:
        raise GolfModelingError("OpenSim requirements not met.") from e


def load_myosim_engine(suite_root: Path) -> PhysicsEngine:
    """Load MyoSim engine."""
    try:
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )
        from src.shared.python.engine_core.engine_probes import MyoSimProbe

        probe = MyoSimProbe(suite_root)
        result = probe.probe()

        if not result.is_available():
            raise GolfModelingError(
                f"MyoSim not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )

        return MyoSuitePhysicsEngine()  # type: ignore[abstract,no-any-return]

    except ImportError as e:
        raise GolfModelingError("MyoSim requirements not met.") from e


def load_pendulum_engine(suite_root: Path) -> PhysicsEngine:
    """Load Pendulum (double-pendulum) engine."""
    try:
        from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
            PendulumPhysicsEngine,
        )

        engine = PendulumPhysicsEngine()
        logger.info("Pendulum engine loaded successfully")
        return engine  # type: ignore[return-value]

    except ImportError as e:
        raise GolfModelingError("Pendulum engine not found.") from e


def load_putting_green_engine(suite_root: Path) -> PhysicsEngine:
    """Load Putting Green engine."""
    try:
        from src.engines.physics_engines.putting_green import PuttingGreenSimulator

        # Putting green doesn't need probing - it's always available as pure Python
        simulator = PuttingGreenSimulator()
        logger.info("Putting Green engine loaded successfully")
        return simulator  # type: ignore[return-value]

    except ImportError as e:
        raise GolfModelingError("Putting Green engine not found.") from e


# Helper for loaders map
LOADER_MAP = {
    EngineType.MUJOCO: load_mujoco_engine,
    EngineType.DRAKE: load_drake_engine,
    EngineType.PINOCCHIO: load_pinocchio_engine,
    EngineType.OPENSIM: load_opensim_engine,
    EngineType.MYOSIM: load_myosim_engine,
    EngineType.PENDULUM: load_pendulum_engine,
    EngineType.PUTTING_GREEN: load_putting_green_engine,
}
