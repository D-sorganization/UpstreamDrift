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
    logger.info("ERROR: MyoSuite not installed.")
    logger.info("Installation: pip install myosuite")
    sys.exit(1)

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
import logging
except ImportError:

logger = logging.getLogger(__name__)
    logger.info("ERROR: stable-baselines3 not installed.")
    logger.info("Installation: pip install stable-baselines3")
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
    logger.info(f"\nTraining SAC policy for {total_timesteps} timesteps...")
    logger.info("=" * 60)

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
    logger.info(f"\nModel saved to: {save_path}.zip")

    return model


def evaluate_policy(model: SAC, env: gym.Env, n_episodes: int = 5) -> None:
    """Evaluate trained policy with visualization.

    Args:
        model: Trained SAC model.
        env: Gym environment.
        n_episodes: Number of evaluation episodes.
    """
    logger.info(f"\nEvaluating policy for {n_episodes} episodes...")
    logger.info("=" * 60)

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

        logger.info(f"Episode {episode + 1}: Steps={step}, Reward={total_reward:.2f}")

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

    logger.info("=" * 60)
    logger.info("MyoSuite Elbow Control Training")
    logger.info("Golf Modeling Suite - Muscle Control Example")
    logger.info("=" * 60)

    # Create environment
    logger.info(f"\n1. Creating environment: {args.env}")
    env = create_elbow_env(args.env)
    logger.info(f"   Observation space: {env.observation_space.shape}")
    logger.info(f"   Action space: {env.action_space.shape}")

    # Train policy
    logger.info("\n2. Training policy...")
    model = train_policy(env, args.timesteps, args.save)

    # Evaluate
    if args.evaluate:
        logger.info("\n3. Evaluating policy...")
        eval_env = create_elbow_env(args.env)
        evaluate_policy(model, eval_env)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {args.save}.zip")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
