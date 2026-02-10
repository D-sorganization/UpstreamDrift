import logging

from dm_control import suite

logger = logging.getLogger(__name__)


def main() -> None:
    """Inspect humanoid model details."""
    env = suite.load(domain_name="humanoid", task_name="stand")
    logger.info("Joint names:")
    for i in range(env.physics.model.njnt):
        logger.info(" - %s", env.physics.model.id2name(i, "joint"))

    logger.info("\nActuator names:")
    # Mujoco py bindings might differ, but let's try to list actuator names

    # Print named actuators from observation spec mostly
    logger.info("Observation spec keys:", env.observation_spec().keys())

    # We can also iterate over physics.named.data.qpos
    logger.info("\nNamed Joints (qpos):")
    try:
        logger.info("%s", env.physics.named.data.qpos.axes.row.names)
    except (RuntimeError, ValueError, OSError):
        logger.error("Could not access named qpos.")

    logger.info("\nNamed Controls (ctrl):")
    try:
        logger.info("%s", env.physics.named.data.ctrl.axes.row.names)
    except (RuntimeError, ValueError, OSError):
        logger.error("Could not access named ctrl.")


if __name__ == "__main__":
    main()
