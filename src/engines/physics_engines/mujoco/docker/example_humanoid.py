import imageio
import numpy as np
from dm_control import suite


def main() -> None:
    """Run the humanoid walking example."""
    print("Loading humanoid:walk task...")
    # Load the environment
    env = suite.load(domain_name="humanoid", task_name="walk")

    # Reset the environment
    env.reset()

    # Create action specification
    action_spec = env.action_spec()

    print("Simulating and rendering...")
    frames = []

    # Simulate for a few seconds (Control timestep is usually 0.02s)
    # 200 steps = 4 seconds approx
    for _ in range(200):
        # Generate a random action
        action = np.random.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape
        )

        # Step the environment
        env.step(action)

        # Render the scene (camera_id=0 usually tracks the agent or gives a good view)
        pixels = env.physics.render(height=480, width=640, camera_id=0)
        frames.append(pixels)

    print(f"Captured {len(frames)} frames.")

    # Save the frames as a video
    video_filename = "humanoid_walk.mp4"
    print(f"Saving video to {video_filename}...")
    imageio.mimsave(video_filename, frames, fps=30)
    print("Done!")


if __name__ == "__main__":
    main()
