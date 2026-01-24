"""
MoCapAct Integration Demo.

This script verifies the installation of Microsoft's MoCapAct package and
demonstrates how to access its humanoid model and dataset tools.

MoCapAct uses a specialized version of the dm_control humanoid.
"""

from pathlib import Path

from src.shared.python.logging_config import get_logger, setup_logging

# Configure logging
setup_logging()
logger = get_logger(__name__)


def check_mocapact_installation() -> bool:
    """Check if MoCapAct is installed and print instructions."""
    logger.info("--- Checking MoCapAct Installation ---")
    try:
        import mocapact

        logger.info("✅ MoCapAct imported successfully: %s", mocapact.__file__)

        from mocapact import observables  # noqa: F401

        logger.info("✅ mocapact.observables available")

        # Check for dist/expert (usually where policies are)
        # Note: MoCapAct usually requires downloading the dataset to use the policies.
        logger.info(
            "\nNote: To use the full MoCapAct dataset (clips and experts), "
            "you need to download it."
        )
        logger.info(
            "Run the following python commands to download the dataset "
            "(WARNING: Large download):"
        )
        logger.info("  from mocapact.transfer.data import download_dataset")
        logger.info("  download_dataset(target_dir='./data/mocapact')")

    except ImportError:
        logger.exception("❌ MoCapAct import failed")
        logger.info("Ensure it is installed using the Dockerfile instructions.")
        return False
    except Exception:
        logger.exception("Error during inspection")
        return False

    return True


def inspect_mocapact_model() -> None:
    """
    Attempt to load the underlying physics model used by MoCapAct.

    This usually wraps dm_control.locomotion.walkers.cmu_humanoid
    """
    logger.info("\n--- Inspecting MoCapAct Model ---")
    try:
        from dm_control.locomotion import walkers

        cmu_humanoid = walkers.cmu_humanoid
        logger.info("✅ dm_control CMU Humanoid is available (Foundation for MoCapAct)")

        # MoCapAct uses specific modifications.
        # Without the dataset, we effectively use the standard CMU humanoid
        # but configured for the MoCapAct observation space.

        walker = cmu_humanoid.CMUHumanoidPositionControlled()
        logger.info("Walker instance created: %s", type(walker))
        logger.info("Control timestep: %s", walker.control_timestep)

        # We can export this model to XML as well
        xml = walker.mjcf_model.to_xml_string()
        output_path = Path("models/generated/cmu_humanoid_position.xml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(xml)
        logger.info("Saved CMU Humanoid XML to %s", output_path)

    except Exception:
        logger.exception("Failed to inspect walker model")


if __name__ == "__main__":
    if check_mocapact_installation():
        inspect_mocapact_model()
