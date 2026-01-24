"""Physics validation for complex (multi-body) models."""

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_manager import EngineManager, EngineType
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

# Locate the repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def is_engine_available(engine_type: EngineType) -> bool:
    """Check if an engine is installed and importable."""
    manager = EngineManager()
    probe_result = manager.get_probe_result(engine_type)
    return bool(probe_result.is_available())


def test_pinocchio_golfer_stability():
    """Verify the Pinocchio golfer URDF loads and simulates without exploding."""
    if not is_engine_available(EngineType.PINOCCHIO):
        pytest.skip("Pinocchio not installed")

    import pinocchio

    # 1. Locate Model
    urdf_path = (
        REPO_ROOT
        / "engines"
        / "physics_engines"
        / "pinocchio"
        / "models"
        / "generated"
        / "golfer.urdf"
    )

    if not urdf_path.exists():
        pytest.skip(f"Golfer URDF not found at {urdf_path}")

    # 2. Load Model
    # Use just the URDF path.
    # Pinocchio might need package directories or environment configuration
    # (for example, ROS_PACKAGE_PATH or the package_dirs argument) if the URDF
    # references meshes. For this basic validation we assume either no meshes
    # are required or that relative paths in the URDF resolve correctly.
    try:
        model = pinocchio.buildModelFromUrdf(str(urdf_path))
    except Exception as e:
        pytest.fail(f"Failed to load Golfer URDF: {e}")

    data = model.createData()

    # 3. Initial State
    q = pinocchio.neutral(model)
    v = np.zeros(model.nv)

    # 4. Simulation Step (Zero torque, just gravity)
    # Check for NaN immediately
    assert not np.any(np.isnan(q)), "Initial configuration contains NaNs"

    try:
        # Compute forward dynamics
        tau = np.zeros(model.nv)
        a = pinocchio.aba(model, data, q, v, tau)
    except Exception as e:
        pytest.fail(f"Forward dynamics (ABA) failed: {e}")

    # 5. Stability Check
    assert not np.any(np.isnan(a)), "Computed acceleration contains NaNs"
    assert not np.any(np.isinf(a)), "Computed acceleration explodes to Infinity"

    logger.info("Golfer URDF stability check passed.")


def test_mujoco_myoarm_stability():
    """Verify the MuJoCo MyoArm XML loads and steps safely."""
    if not is_engine_available(EngineType.MUJOCO):
        pytest.skip("MuJoCo not installed")

    import mujoco

    # 1. Locate Model
    xml_path = (
        REPO_ROOT
        / "engines"
        / "physics_engines"
        / "mujoco"
        / "myo_sim"
        / "arm"
        / "myoarm_simple.xml"
    )

    if not xml_path.exists():
        pytest.skip(f"MyoArm XML not found at {xml_path}")

    # 2. Load Model
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
    except Exception as e:
        pytest.fail(f"Failed to load MyoArm XML: {e}")

    # 3. Step Simulation
    steps = 100
    try:
        for _ in range(steps):
            mujoco.mj_step(model, data)
    except Exception as e:
        pytest.fail(f"Simulation stepping failed: {e}")

    # 4. Check State
    # Check for NaNs in qpos or qvel
    assert not np.any(np.isnan(data.qpos)), "MyoArm qpos contains NaNs"
    assert not np.any(np.isnan(data.qvel)), "MyoArm qvel contains NaNs"

    # Check bounds (simple sanity check that it hasn't exploded to 1e10)
    # Using a generous threshold
    assert np.all(np.abs(data.qpos) < 1e4), "MyoArm qpos diverged significantly"

    logger.info("MyoArm stability check passed.")
