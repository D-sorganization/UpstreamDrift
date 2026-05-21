"""DAgger imitation learning algorithm."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.learning.imitation._base import ImitationLearner, TrainingConfig
from src.learning.imitation._bc import BehaviorCloning
from src.learning.imitation.dataset import Demonstration, DemonstrationDataset


class DAgger(ImitationLearner):
    """Dataset Aggregation with expert queries.

    DAgger iteratively collects data using the current policy
    but labels with expert actions, addressing distribution shift.

    Requires access to an expert policy during training.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: TrainingConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize DAgger learner."""
        if observation_dim is None:
            raise ValueError("observation_dim must be provided")
        super().__init__(observation_dim, action_dim, config, device)
        self._bc = BehaviorCloning(observation_dim, action_dim, config, device)
        self._aggregated_dataset: DemonstrationDataset | None = None

    def train(
        self,
        dataset: DemonstrationDataset,
        validation_split: float = 0.1,
    ) -> dict[str, list[float]]:
        """Train initial policy with behavior cloning.

        For DAgger iterations, use train_online().

        Args:
            dataset: Initial demonstration dataset.
            validation_split: Fraction for validation.

        Returns:
            Training history.
        """
        if dataset is None:
            raise ValueError("dataset must be provided")
        self._aggregated_dataset = dataset
        return self._bc.train(dataset, validation_split)

    @staticmethod
    def _compute_beta(iteration: int, iterations: int, schedule: str) -> float:
        """Compute the expert-mixing probability for a DAgger iteration."""
        if iteration is None:
            raise ValueError("iteration must be provided")
        if schedule == "linear":
            return 1.0 - iteration / iterations
        return 0.5**iteration

    def _collect_trajectory(
        self,
        env: Any,
        expert: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        beta: float,
        max_steps: int,
    ) -> tuple[Demonstration, float]:
        """Roll out one trajectory, mixing policy and expert actions."""
        if expert is None:
            raise ValueError("expert must be provided")
        obs, info = env.reset()
        demo_timestamps = [0.0]
        demo_positions = [obs[: obs.shape[0] // 2]]
        demo_velocities = [obs[obs.shape[0] // 2 :]]
        demo_actions: list[NDArray[np.floating]] = []

        total_reward = 0.0
        step = 0
        terminated = False

        while step < max_steps:
            policy_action = self.predict(obs)
            expert_action = expert(obs)
            demo_actions.append(expert_action)

            action = expert_action if np.random.random() < beta else policy_action

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            demo_timestamps.append(step * 0.01)
            demo_positions.append(obs[: obs.shape[0] // 2])
            demo_velocities.append(obs[obs.shape[0] // 2 :])

            if terminated or truncated:
                break

        demo = Demonstration(
            timestamps=np.array(demo_timestamps[:-1]),
            joint_positions=np.array(demo_positions[:-1]),
            joint_velocities=np.array(demo_velocities[:-1]),
            actions=np.array(demo_actions),
            source="dagger",
            success=not terminated,
        )
        return demo, total_reward

    def train_online(
        self,
        env: Any,  # RoboticsGymEnv
        expert: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        iterations: int = 10,
        trajectories_per_iter: int = 10,
        max_steps: int = 500,
        beta_schedule: str = "linear",
    ) -> dict[str, Any]:
        """Online training with expert intervention.

        Args:
            env: Gymnasium environment.
            expert: Expert policy function.
            iterations: Number of DAgger iterations.
            trajectories_per_iter: Trajectories per iteration.
            max_steps: Max steps per trajectory.
            beta_schedule: Schedule for mixing policy and expert.

        Returns:
            Training results.
        """
        if self._aggregated_dataset is None:
            raise ValueError("Must call train() first with initial dataset")

        results: dict[str, list] = {
            "iteration_rewards": [],
            "dataset_size": [],
        }

        for iteration in range(iterations):
            beta = self._compute_beta(iteration, iterations, beta_schedule)

            new_demos = []
            iteration_rewards = []

            for _ in range(trajectories_per_iter):
                demo, reward = self._collect_trajectory(env, expert, beta, max_steps)
                new_demos.append(demo)
                iteration_rewards.append(reward)

            self._aggregated_dataset.extend(new_demos)
            self._bc.train(self._aggregated_dataset)

            results["iteration_rewards"].append(np.mean(iteration_rewards))
            results["dataset_size"].append(len(self._aggregated_dataset))

        return results

    def predict(
        self,
        observation: NDArray[np.floating],
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Predict action using trained policy.

        Args:
            observation: Current observation.
            deterministic: If True, return deterministic action.

        Returns:
            Predicted action.
        """
        if observation is None:
            raise ValueError("observation must be provided")
        return self._bc.predict(observation, deterministic)

    def save(self, path: str | Path) -> None:
        """Save policy."""
        if path is None:
            raise ValueError("path must be provided")
        self._bc.save(path)

    def load(self, path: str | Path) -> None:
        """Load policy."""
        if path is None:
            raise ValueError("path must be provided")
        self._bc.load(path)
