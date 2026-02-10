import logging

from dm_control import suite

logger = logging.getLogger(__name__)


def main() -> None:
    """Dump geom and body names from the environment."""
    logger.info("Loading humanoid_CMU:stand...")
    env = suite.load(domain_name="humanoid_CMU", task_name="stand")

    logger.info("\n--- GEOM NAMES ---")
    ngeom = env.physics.model.ngeom
    for i in range(ngeom):
        name = env.physics.model.id2name(i, "geom")
        logger.info("%s: %s", i, name)

    logger.info("\n--- BODY NAMES ---")
    nbody = env.physics.model.nbody
    for i in range(nbody):
        name = env.physics.model.id2name(i, "body")
        logger.info("%s: %s", i, name)


if __name__ == "__main__":
    main()
