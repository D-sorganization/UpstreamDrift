"""
MyoSuite Elbow Control Training Example
========================================

This example demonstrates how to train a neural network policy
to control elbow muscles using reinforcement learning.

Use Case: Golf swing elbow mechanics

Requirements:
    - myosuite: pip install myosuite
    - stable-baselines3: pip install stable-baselines3

Usage:
    python train_elbow_policy.py --timesteps 100000
"""

import argparse
import sys

try:
    from myosuite.utils import gym
except ImportError:
    print("ERROR: MyoSuite not installed.")
    print("Installation: pip install myosuite")
    sys.exit(1)

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError:
    print("ERROR: stable-baselines3 not installed.")
    print("Installation: pip install stable-baselines3")
    sys.exit(1)


def create_elbow_env(env_id: str = "myoElbowPose1D6MRandom-v0") -> gym.Env:
    """Create MyoSuite elbow environment.

    Args:
        env_id: MyoSuite environment ID.

    Returns:
        Gym environment instance.
    """
    return gym.make(env_id)


def train_policy(
    env: gym.Env,
    total_timesteps: int = 100000,
    save_path: str = "elbow_policy",
) -> SAC:
    """Train elbow control policy using SAC algorithm.

    Args:
        env: Gym environment.
        total_timesteps: Number of training steps.
        save_path: Path to save trained model.

    Returns:
        Trained SAC model.
    """
    print(f"\nTraining SAC policy for {total_timesteps} timesteps...")
    print("=" * 60)

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
    )

    # Create evaluation callback
    eval_env = create_elbow_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./{save_path}_best/",
        log_path=f"./{save_path}_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # Train
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save final model
    model.save(save_path)
    print(f"\nModel saved to: {save_path}.zip")

    return model


def evaluate_policy(model: SAC, env: gym.Env, n_episodes: int = 5) -> None:
    """Evaluate trained policy with visualization.

    Args:
        model: Trained SAC model.
        env: Gym environment.
        n_episodes: Number of evaluation episodes.
    """
    print(f"\nEvaluating policy for {n_episodes} episodes...")
    print("=" * 60)

    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            # Visualize (if display available)
            try:
                env.mj_render()
            except Exception:
                pass  # Headless mode

        print(f"Episode {episode + 1}: Steps={step}, Reward={total_reward:.2f}")

    env.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train elbow control policy")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="myoElbowPose1D6MRandom-v0",
        help="MyoSuite environment ID",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="elbow_policy",
        help="Save path for trained model",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate after training",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MyoSuite Elbow Control Training")
    print("Golf Modeling Suite - Muscle Control Example")
    print("=" * 60)

    # Create environment
    print(f"\n1. Creating environment: {args.env}")
    env = create_elbow_env(args.env)
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")

    # Train policy
    print("\n2. Training policy...")
    model = train_policy(env, args.timesteps, args.save)

    # Evaluate
    if args.evaluate:
        print("\n3. Evaluating policy...")
        eval_env = create_elbow_env(args.env)
        evaluate_policy(model, eval_env)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.save}.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
