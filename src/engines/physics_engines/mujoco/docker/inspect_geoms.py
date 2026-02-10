import logging

from dm_control import suite

logger = logging.getLogger(__name__)


def main() -> None:
    """Inspect and list environment geoms."""
    logger.info("Loading humanoid_CMU:stand...")
    env = suite.load(domain_name="humanoid_CMU", task_name="stand")

    logger.info("\nGeom Names (for coloring):")
    # In dm_control, geom names might be in physics.model.id2name('geom', id)
    # or accessible via named indexing if they are distinct.

    # Try iterating through geoms
    try:
        n_geoms = env.physics.model.ngeom
        for i in range(n_geoms):
            name = env.physics.model.id2name(i, "geom")
            logger.info("ID %s: %s", i, name)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error listing geoms: %s", e)

    # Also check if we can see specific body parts
    logger.info("\nBody Names:")
    try:
        n_bodies = env.physics.model.nbody
        for i in range(n_bodies):
            name = env.physics.model.id2name(i, "body")
            logger.info("ID %s: %s", i, name)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error listing bodies: %s", e)


if __name__ == "__main__":
    main()
