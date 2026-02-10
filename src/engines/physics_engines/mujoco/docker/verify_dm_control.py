import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    """Verify dm_control installation."""
    try:
        from dm_control import suite
    except ImportError as e:
        logger.error("[FAILURE] Could not import dm_control: %s", e)
        sys.exit(1)

    logger.info("Verifying DeepMind Control Suite installation...")
    logger.info("[SUCCESS] dm_control imported successfully.")

    try:
        # Load the cartpole task
        env = suite.load(domain_name="cartpole", task_name="swingup")
        logger.info("[SUCCESS] Loaded cartpole:swingup task.")

        # Reset the environment
        _ = env.reset()
        logger.info("[SUCCESS] Environment reset.")

        # Step through the environment
        action_spec = env.action_spec()
        rng = np.random.default_rng()
        action = rng.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape
        )
        _ = env.step(action)
        logger.info("[SUCCESS] Stepped environment with random action.")

        # Render a frame (headless)
        pixels = env.physics.render(height=480, width=640, camera_id=0)
        logger.info("[SUCCESS] Rendered frame of shape: %s", pixels.shape)

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("[FAILURE] An error occurred during verification: %s", e)
        sys.exit(1)

    logger.info("\nDeepMind Control Suite is correctly installed and functioning!")


if __name__ == "__main__":
    main()
