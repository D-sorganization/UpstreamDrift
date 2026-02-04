"""Tests for imitation learning module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestDemonstration:
    """Tests for Demonstration dataclass."""

    def test_demonstration_creation(self) -> None:
        """Test creating a demonstration."""
        from src.learning.imitation import Demonstration

        n_frames = 100
        n_joints = 7

        demo = Demonstration(
            timestamps=np.arange(n_frames) * 0.01,
            joint_positions=np.random.randn(n_frames, n_joints),
            joint_velocities=np.random.randn(n_frames, n_joints),
            task_id="test_task",
            success=True,
        )

        assert demo.n_frames == n_frames
        assert demo.n_joints == n_joints
        assert demo.duration == pytest.approx(0.99, rel=1e-3)

    def test_demonstration_with_actions(self) -> None:
        """Test demonstration with action data."""
        from src.learning.imitation import Demonstration

        n_frames = 50
        n_joints = 7
        n_actions = 7

        demo = Demonstration(
            timestamps=np.arange(n_frames) * 0.01,
            joint_positions=np.random.randn(n_frames, n_joints),
            joint_velocities=np.random.randn(n_frames, n_joints),
            actions=np.random.randn(n_frames, n_actions),
        )

        assert demo.actions is not None
        assert demo.actions.shape == (n_frames, n_actions)

    def test_demonstration_subsample(self) -> None:
        """Test subsampling demonstration."""
        from src.learning.imitation import Demonstration

        n_frames = 100
        n_joints = 7

        demo = Demonstration(
            timestamps=np.arange(n_frames) * 0.01,
            joint_positions=np.random.randn(n_frames, n_joints),
            joint_velocities=np.random.randn(n_frames, n_joints),
        )

        subsampled = demo.subsample(factor=5)
        assert subsampled.n_frames == 20

    def test_demonstration_get_frame(self) -> None:
        """Test getting a single frame."""
        from src.learning.imitation import Demonstration

        demo = Demonstration(
            timestamps=np.arange(10) * 0.01,
            joint_positions=np.random.randn(10, 7),
            joint_velocities=np.random.randn(10, 7),
        )

        frame = demo.get_frame(5)
        assert "timestamp" in frame
        assert "joint_positions" in frame
        assert "joint_velocities" in frame


class TestDemonstrationDataset:
    """Tests for DemonstrationDataset."""

    def create_demo(self, n_frames: int = 50, n_joints: int = 7) -> "Demonstration":
        """Helper to create a demonstration."""
        from src.learning.imitation import Demonstration

        return Demonstration(
            timestamps=np.arange(n_frames) * 0.01,
            joint_positions=np.random.randn(n_frames, n_joints),
            joint_velocities=np.random.randn(n_frames, n_joints),
            actions=np.random.randn(n_frames, n_joints),
            success=True,
        )

    def test_dataset_creation(self) -> None:
        """Test creating an empty dataset."""
        from src.learning.imitation import DemonstrationDataset

        dataset = DemonstrationDataset()
        assert len(dataset) == 0

    def test_dataset_add(self) -> None:
        """Test adding demonstrations."""
        from src.learning.imitation import DemonstrationDataset

        dataset = DemonstrationDataset()
        dataset.add(self.create_demo())
        dataset.add(self.create_demo())

        assert len(dataset) == 2

    def test_dataset_to_transitions(self) -> None:
        """Test converting to transitions."""
        from src.learning.imitation import DemonstrationDataset

        dataset = DemonstrationDataset()
        dataset.add(self.create_demo(n_frames=50))
        dataset.add(self.create_demo(n_frames=50))

        states, actions, next_states = dataset.to_transitions()

        # 49 transitions per demo * 2 demos
        assert len(states) == 98
        assert len(actions) == 98
        assert len(next_states) == 98

    def test_dataset_augment(self) -> None:
        """Test data augmentation."""
        from src.learning.imitation import DemonstrationDataset

        dataset = DemonstrationDataset()
        dataset.add(self.create_demo())

        augmented = dataset.augment(noise_std=0.01, num_augmentations=3)

        # Original + 3 augmented
        assert len(augmented) == 4

    def test_dataset_save_load(self) -> None:
        """Test saving and loading dataset."""
        from src.learning.imitation import DemonstrationDataset

        dataset = DemonstrationDataset()
        dataset.add(self.create_demo())
        dataset.add(self.create_demo())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            dataset.save(path)

            loaded = DemonstrationDataset.load(path)

        assert len(loaded) == 2
        assert loaded[0].n_frames == 50

    def test_dataset_statistics(self) -> None:
        """Test computing statistics."""
        from src.learning.imitation import DemonstrationDataset

        dataset = DemonstrationDataset()
        dataset.add(self.create_demo())
        dataset.add(self.create_demo())

        stats = dataset.get_statistics()

        assert stats["n_demonstrations"] == 2
        assert stats["success_rate"] == 1.0
        assert "position_mean" in stats


class TestBehaviorCloning:
    """Tests for Behavior Cloning learner."""

    def test_bc_creation(self) -> None:
        """Test creating BC learner."""
        from src.learning.imitation import BehaviorCloning

        bc = BehaviorCloning(observation_dim=14, action_dim=7)
        assert bc.observation_dim == 14
        assert bc.action_dim == 7

    def test_bc_predict_before_training(self) -> None:
        """Test prediction before training."""
        from src.learning.imitation import BehaviorCloning

        bc = BehaviorCloning(observation_dim=14, action_dim=7)
        obs = np.random.randn(14)
        action = bc.predict(obs)

        assert action.shape == (7,)

    def test_bc_train(self) -> None:
        """Test training BC."""
        from src.learning.imitation import (
            BehaviorCloning,
            Demonstration,
            DemonstrationDataset,
            TrainingConfig,
        )

        # Create small dataset
        dataset = DemonstrationDataset()
        for _ in range(5):
            demo = Demonstration(
                timestamps=np.arange(20) * 0.01,
                joint_positions=np.random.randn(20, 7),
                joint_velocities=np.random.randn(20, 7),
                actions=np.random.randn(20, 7),
            )
            dataset.add(demo)

        config = TrainingConfig(epochs=5, batch_size=16)
        bc = BehaviorCloning(observation_dim=14, action_dim=7, config=config)

        history = bc.train(dataset)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 5

    def test_bc_save_load(self) -> None:
        """Test saving and loading BC policy."""
        from src.learning.imitation import BehaviorCloning

        bc = BehaviorCloning(observation_dim=14, action_dim=7)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.npz"
            bc.save(path)

            bc2 = BehaviorCloning(observation_dim=14, action_dim=7)
            bc2.load(path)

        # Verify same predictions
        obs = np.random.randn(14)
        action1 = bc.predict(obs)
        action2 = bc2.predict(obs)

        np.testing.assert_array_almost_equal(action1, action2)


class TestGAIL:
    """Tests for GAIL learner."""

    def test_gail_creation(self) -> None:
        """Test creating GAIL learner."""
        from src.learning.imitation import GAIL

        gail = GAIL(observation_dim=14, action_dim=7)
        assert gail.observation_dim == 14
        assert gail.action_dim == 7

    def test_gail_get_reward(self) -> None:
        """Test getting GAIL reward."""
        from src.learning.imitation import GAIL

        gail = GAIL(observation_dim=14, action_dim=7)

        state = np.random.randn(14)
        action = np.random.randn(7)

        reward = gail.get_reward(state, action)
        assert isinstance(reward, float)
