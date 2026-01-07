from dm_control import suite


def main() -> None:
    """Inspect humanoid model details."""
    env = suite.load(domain_name="humanoid", task_name="stand")
    print("Joint names:")
    for i in range(env.physics.model.njnt):
        print(f" - {env.physics.model.id2name(i, 'joint')}")

    print("\nActuator names:")
    # Mujoco py bindings might differ, but let's try to list actuator names

    # Print named actuators from observation spec mostly
    print("Observation spec keys:", env.observation_spec().keys())

    # We can also iterate over physics.named.data.qpos
    print("\nNamed Joints (qpos):")
    try:
        print(env.physics.named.data.qpos.axes.row.names)
    except Exception:
        print("Could not access named qpos.")

    print("\nNamed Controls (ctrl):")
    try:
        print(env.physics.named.data.ctrl.axes.row.names)
    except Exception:
        print("Could not access named ctrl.")


if __name__ == "__main__":
    main()
