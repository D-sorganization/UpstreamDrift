import sys

import numpy as np


def main() -> None:
    """Verify dm_control installation."""
    try:
        from dm_control import suite
    except ImportError as e:
        print(f"[FAILURE] Could not import dm_control: {e}")
        sys.exit(1)

    print("Verifying DeepMind Control Suite installation...")
    print("[SUCCESS] dm_control imported successfully.")

    try:
        # Load the cartpole task
        env = suite.load(domain_name="cartpole", task_name="swingup")
        print("[SUCCESS] Loaded cartpole:swingup task.")

        # Reset the environment
        _ = env.reset()
        print("[SUCCESS] Environment reset.")

        # Step through the environment
        action_spec = env.action_spec()
        rng = np.random.default_rng()
        action = rng.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape
        )
        _ = env.step(action)
        print("[SUCCESS] Stepped environment with random action.")

        # Render a frame (headless)
        pixels = env.physics.render(height=480, width=640, camera_id=0)
        print(f"[SUCCESS] Rendered frame of shape: {pixels.shape}")

    except Exception as e:
        print(f"[FAILURE] An error occurred during verification: {e}")
        sys.exit(1)

    print("\nDeepMind Control Suite is correctly installed and functioning!")


if __name__ == "__main__":
    main()
