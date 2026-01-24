from pathlib import Path

from src.shared.python.engine_availability import (
    DM_CONTROL_AVAILABLE,
    MUJOCO_AVAILABLE,
    PINOCCHIO_AVAILABLE,
)
from src.shared.python.logging_config import get_logger, setup_logging

# Configure logging
setup_logging()
logger = get_logger(__name__)

# Import available engines
if DM_CONTROL_AVAILABLE:
    from dm_control import suite
else:
    logger.warning(
        "dm_control not found. Please install it via the Dockerfile updates."
    )

if PINOCCHIO_AVAILABLE:
    import pinocchio as pin
else:
    logger.warning("Pinocchio not available.")

if MUJOCO_AVAILABLE:
    import mujoco
else:
    logger.warning("Mujoco not available.")


def save_mjcf(xml_string: str) -> Path:
    """Save the MJCF XML to a file."""
    output_dir = Path("models/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "humanoid_cm.xml"

    with model_path.open("w") as f:
        f.write(xml_string)
    logger.info("Saved MJCF to %s", model_path)
    return model_path


def load_humanoid_cm() -> None:
    """Load the Humanoid CM (CMU) model from dm_control."""
    if not DM_CONTROL_AVAILABLE:
        return

    logger.info("Loading Humanoid CM (CMU) model from dm_control...")
    # Load the humanoid environment (CMU Mocap based)
    # domain_name="humanoid" uses the CMU humanoid model
    env = suite.load(domain_name="humanoid", task_name="stand")

    # Access the physics
    physics = env.physics

    # Get the MJCF XML string
    # dm_control constructs the model in memory, but we can export it
    xml_string = physics.model.get_xml()

    logger.info("Successfully loaded Humanoid CM model via dm_control.")

    # Save XML for use with Pinocchio and native Mujoco
    model_path = save_mjcf(xml_string)

    # 1. Load with native Mujoco (checking compatibility)
    if MUJOCO_AVAILABLE:
        test_native_mujoco(str(model_path))

    # 2. Load with Pinocchio
    if PINOCCHIO_AVAILABLE:
        test_pinocchio(str(model_path))


def test_native_mujoco(model_path: str) -> None:
    """Test loading the model with native Mujoco."""
    logger.info("\n--- Testing Native Mujoco Loading ---")
    try:
        mj_model = mujoco.MjModel.from_xml_path(model_path)
        mj_data = mujoco.MjData(mj_model)
        logger.info("Native Mujoco load successful.")
        logger.info("nV (degrees of freedom): %s", mj_model.nv)
        logger.info("nQ (generalized coordinates): %s", mj_model.nq)

        # Helper to step simulation
        mujoco.mj_step(mj_model, mj_data)
        logger.info("Basic simulation step successful.")

    except Exception:
        logger.exception("Native Mujoco load/step failed")


def test_pinocchio(model_path: str) -> None:
    """Test loading the model with Pinocchio."""
    logger.info("\n--- Testing Pinocchio Loading ---")
    try:
        # Pinocchio MJCF loader (Pinocchio 3.x support for MJCF)
        # Note: Complex MJCF features might not be fully supported in Pinocchio yet.
        model = pin.buildModelFromMJCF(model_path)
        data = model.createData()
        logger.info("Pinocchio load successful.")
        logger.info("Pinocchio nq: %s, nv: %s", model.nq, model.nv)

        # Perform a simple algorithm check (e.g., Forward Kinematics)
        q = pin.neutral(model)
        pin.forwardKinematics(model, data, q)
        logger.info("Pinocchio forward kinematics computation successful.")

    except AttributeError:
        logger.warning(
            "Your Pinocchio version might not support 'buildModelFromMJCF'. "
            "Ensure you are using Pinocchio 3.x."
        )
    except Exception:
        logger.exception("Pinocchio load failed")
        logger.info(
            "Note: Pinocchio's MJCF support is evolving. "
            "You might need to simplify the MJCF or convert to URDF."
        )


def load_mujoco_playground_humanoid() -> None:
    """Attempt to load a humanoid from the new Mujoco Playground if available."""
    logger.info("\n--- Checking Mujoco Playground ---")
    try:
        # Note: Import names depend on the exact package structure of mujoco_playground
        import mujoco_playground

        logger.info("Mujoco Playground found: %s", mujoco_playground.__file__)
        # Add specific playground loading code here if the API is known
    except ImportError:
        logger.warning(
            "Mujoco Playground package not found "
            "(might be installed as 'playground' or requires PYTHONPATH setup)."
        )


if __name__ == "__main__":
    load_humanoid_cm()
    load_mujoco_playground_humanoid()
