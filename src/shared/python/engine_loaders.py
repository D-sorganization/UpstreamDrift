"""Engine loader functions."""

from src.shared.python.logging_config import get_logger
from pathlib import Path

from .common_utils import GolfModelingError
from .engine_registry import EngineType
from .interfaces import PhysicsEngine

logger = get_logger(__name__)


def load_mujoco_engine(suite_root: Path) -> PhysicsEngine:
    """Load MuJoCo engine with full initialization."""
    try:
        import mujoco  # noqa: F401

        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        from .engine_probes import MuJoCoProbe

        probe = MuJoCoProbe(suite_root)
        result = probe.probe()

        if not result.is_available():
            raise GolfModelingError(
                f"MuJoCo not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )

        engine = MuJoCoPhysicsEngine()

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

        from engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        from .engine_probes import DrakeProbe

        probe = DrakeProbe(suite_root)
        result = probe.probe()

        if not result.is_available():
            raise GolfModelingError(
                f"Drake not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )

        engine = DrakePhysicsEngine()

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
            except Exception as e:
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

        from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        from .engine_probes import PinocchioProbe

        probe = PinocchioProbe(suite_root)
        result = probe.probe()

        if not result.is_available():
            raise GolfModelingError(
                f"Pinocchio not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )

        engine = PinocchioPhysicsEngine()

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
        from engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        from .engine_probes import OpenSimProbe

        probe = OpenSimProbe(suite_root)
        result = probe.probe()

        if not result.is_available():
            raise GolfModelingError(
                f"OpenSim not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )

        engine = OpenSimPhysicsEngine()
        return engine  # type: ignore[no-any-return]

    except ImportError as e:
        raise GolfModelingError("OpenSim requirements not met.") from e


def load_myosim_engine(suite_root: Path) -> PhysicsEngine:
    """Load MyoSim engine."""
    try:
        from engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )

        from .engine_probes import MyoSimProbe

        probe = MyoSimProbe(suite_root)
        result = probe.probe()

        if not result.is_available():
            raise GolfModelingError(
                f"MyoSim not ready:\n{result.diagnostic_message}\n"
                f"Fix: {result.get_fix_instructions()}"
            )

        return MyoSuitePhysicsEngine()  # type: ignore[no-any-return]

    except ImportError as e:
        raise GolfModelingError("MyoSim requirements not met.") from e


# Helper for loaders map
LOADER_MAP = {
    EngineType.MUJOCO: load_mujoco_engine,
    EngineType.DRAKE: load_drake_engine,
    EngineType.PINOCCHIO: load_pinocchio_engine,
    EngineType.OPENSIM: load_opensim_engine,
    EngineType.MYOSIM: load_myosim_engine,
}
