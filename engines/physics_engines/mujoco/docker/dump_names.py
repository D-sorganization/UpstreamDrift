from dm_control import suite


def main() -> None:
    """Dump geom and body names from the environment."""
    print("Loading humanoid_CMU:stand...")
    env = suite.load(domain_name="humanoid_CMU", task_name="stand")

    print("\n--- GEOM NAMES ---")
    ngeom = env.physics.model.ngeom
    for i in range(ngeom):
        name = env.physics.model.id2name(i, "geom")
        print(f"{i}: {name}")

    print("\n--- BODY NAMES ---")
    nbody = env.physics.model.nbody
    for i in range(nbody):
        name = env.physics.model.id2name(i, "body")
        print(f"{i}: {name}")


if __name__ == "__main__":
    main()
