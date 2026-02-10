import logging

from dm_control import suite

logger = logging.getLogger(__name__)


def main() -> None:
    """Measure the height of the humanoid."""
    logger.info("Loading humanoid_CMU:stand...")
    env = suite.load(domain_name="humanoid_CMU", task_name="stand")
    physics = env.physics

    # Get positions of head and feet (using body positions)
    # The 'head' body usually corresponds to the center of the head.
    # We might want the top of the head geom.

    try:
        head_pos = physics.named.data.xpos["head"]
        lfoot_pos = physics.named.data.xpos["lfoot"]
        rfoot_pos = physics.named.data.xpos["rfoot"]

        # Determine lowest foot point (ground level approx)
        min_z = min(lfoot_pos[2], rfoot_pos[2])

        # Determine head height
        # Note: xpos is body centroid. Head geom likely extends above.
        # Let's find the head geom(s) and check their max Z.
        max_z = head_pos[2]

        # Check specific geoms attached to head body if possible
        # Finding geoms belonging to 'head' body is tricky without iterating tree.
        # But we can iterate all geoms and check max Z of those named 'head'

        geom_names = physics.model.id2name
        for i in range(physics.model.ngeom):
            name = geom_names(i, "geom")
            if name and "head" in name:
                # get geom position
                g_pos = physics.data.geom_xpos[i]
                # we technically need the top of the bounding box of the geom
                # center + size (estimated)
                # assuming sphere or capsule
                g_size = physics.model.geom_size[i]  # array
                # usually index 0 is radius
                radius = g_size[0]
                top = g_pos[2] + radius
                max_z = max(max_z, top)

        height = max_z - min_z
        logger.info("\n--- MEASUREMENTS ---")
        logger.info("Head Top Z: %s m", max_z)
        logger.info("Foot Low Z: %s m", min_z)
        logger.info("Total Height: %s m (approx %s ft)", height, height * 3.28084)
        logger.info("--------------------")

    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error measuring: %s", e)


if __name__ == "__main__":
    main()
