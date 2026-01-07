from dm_control import suite


def main() -> None:
    """Inspect and list environment geoms."""
    print("Loading humanoid_CMU:stand...")
    env = suite.load(domain_name="humanoid_CMU", task_name="stand")

    print("\nGeom Names (for coloring):")
    # In dm_control, geom names might be in physics.model.id2name('geom', id)
    # or accessible via named indexing if they are distinct.

    # Try iterating through geoms
    try:
        n_geoms = env.physics.model.ngeom
        for i in range(n_geoms):
            name = env.physics.model.id2name(i, "geom")
            print(f"ID {i}: {name}")
    except Exception as e:
        print(f"Error listing geoms: {e}")

    # Also check if we can see specific body parts
    print("\nBody Names:")
    try:
        n_bodies = env.physics.model.nbody
        for i in range(n_bodies):
            name = env.physics.model.id2name(i, "body")
            print(f"ID {i}: {name}")
    except Exception as e:
        print(f"Error listing bodies: {e}")


if __name__ == "__main__":
    main()
