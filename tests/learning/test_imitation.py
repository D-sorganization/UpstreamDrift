"""Tests for imitation learning module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from src.learning.imitation import Demonstration

_TEST_RNG_SEED = 0

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _seed_numpy_rng() -> None:
    """Seed the legacy global RNG so ``np.random.randn`` draws are deterministic."""
    np.random.seed(_TEST_RNG_SEED)


def test_numpy_rng_seed_is_reset_for_deterministic_fixtures() -> None:
    expected = np.random.RandomState(_TEST_RNG_SEED).randn(3)
    np.testing.assert_allclose(np.random.randn(3), expected)


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

    def create_demo(self, n_frames: int = 50, n_joints: int = 7) -> Demonstration:
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

    def test_bc_default_config_learns_linear_expert(self) -> None:
        """Issue #13012: default TrainingConfig must learn a linear expert."""
        from src.learning.imitation import (
            BehaviorCloning,
            Demonstration,
            DemonstrationDataset,
        )

        rng = np.random.default_rng(42)
        n_frames = 1000
        q = rng.standard_normal((n_frames, 3))
        v = rng.standard_normal((n_frames, 3))
        obs = np.hstack([q, v])
        W_expert = rng.standard_normal((6, 2))
        actions = obs @ W_expert

        demo = Demonstration(
            timestamps=np.arange(n_frames) * 0.01,
            joint_positions=q,
            joint_velocities=v,
            actions=actions,
        )
        dataset = DemonstrationDataset([demo])

        bc = BehaviorCloning(observation_dim=6, action_dim=2)  # all defaults
        history = bc.train(dataset)

        action_var = float(np.var(actions))
        final_loss = history["train_loss"][-1]
        assert final_loss < 0.1 * action_var, (
            f"BC failed to learn: final loss {final_loss} >= 0.1 * var {0.1 * action_var}"
        )

    def test_dagger_default_config_learns_linear_expert(self) -> None:
        """Issue #13012: default TrainingConfig must learn a linear expert in DAgger."""
        from src.learning.imitation import (
            DAgger,
            Demonstration,
            DemonstrationDataset,
        )

        rng = np.random.default_rng(42)
        n_frames = 1000
        q = rng.standard_normal((n_frames, 3))
        v = rng.standard_normal((n_frames, 3))
        obs = np.hstack([q, v])
        W_expert = rng.standard_normal((6, 2))
        actions = obs @ W_expert

        demo = Demonstration(
            timestamps=np.arange(n_frames) * 0.01,
            joint_positions=q,
            joint_velocities=v,
            actions=actions,
        )
        dataset = DemonstrationDataset([demo])

        dagger = DAgger(observation_dim=6, action_dim=2)  # all defaults
        history = dagger.train(dataset)

        action_var = float(np.var(actions))
        final_loss = history["train_loss"][-1]
        assert final_loss < 0.1 * action_var, (
            f"DAgger failed to learn: final loss {final_loss} >= 0.1 * var {0.1 * action_var}"
        )

    def test_bc_train_warns_when_loss_does_not_decrease(self) -> None:
        """Issue #13012: BC must warn if training does not reduce loss."""
        from src.learning.imitation import (
            BehaviorCloning,
            Demonstration,
            DemonstrationDataset,
            TrainingConfig,
        )

        demo = Demonstration(
            timestamps=np.arange(10) * 0.01,
            joint_positions=np.random.randn(10, 3),
            joint_velocities=np.random.randn(10, 3),
            actions=np.random.randn(10, 2),
        )
        dataset = DemonstrationDataset([demo])

        # A vanishing (but contract-valid, > 0) learning rate cannot cut the loss
        # by 1 % in two epochs, triggering the warning (#13013 rejects lr == 0).
        cfg = TrainingConfig(epochs=2, learning_rate=1e-12)
        bc = BehaviorCloning(6, 2, config=cfg)

        with pytest.warns(
            UserWarning, match="Training did not decrease loss significantly"
        ):
            bc.train(dataset)

    @pytest.mark.parametrize("split", [-0.1, 1.0, 1.5])
    def test_bc_validation_split_contract(self, split: float) -> None:
        """DbC: invalid validation_split must be rejected."""
        from src.learning.imitation import (
            BehaviorCloning,
            Demonstration,
            DemonstrationDataset,
            TrainingConfig,
        )

        demo = Demonstration(
            timestamps=np.arange(10) * 0.01,
            joint_positions=np.random.randn(10, 3),
            joint_velocities=np.random.randn(10, 3),
            actions=np.random.randn(10, 2),
        )
        dataset = DemonstrationDataset([demo])
        bc = BehaviorCloning(6, 2, config=TrainingConfig(epochs=1))

        with pytest.raises(ValueError, match="validation_split"):
            bc.train(dataset, validation_split=split)


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


class TestMLPUtilities:
    """Tests for shared MLP and optimization utilities in _mlp.py."""

    def test_build_mlp_structure_and_initialization(self) -> None:
        from src.learning.imitation._mlp import _build_mlp

        layers = _build_mlp(input_dim=8, output_dim=3, hidden_sizes=[32, 16], init="he")
        assert len(layers) == 3
        assert layers[0]["W"].shape == (8, 32)
        assert layers[0]["b"].shape == (32,)
        assert layers[1]["W"].shape == (32, 16)
        assert layers[1]["b"].shape == (16,)
        assert layers[2]["W"].shape == (16, 3)
        assert layers[2]["b"].shape == (3,)

        # All biases should be initialized to zero
        for layer in layers:
            np.testing.assert_allclose(layer["b"], 0.0)

        # Legacy small initialization
        small_layers = _build_mlp(
            input_dim=8, output_dim=3, hidden_sizes=[16], init="small"
        )
        assert len(small_layers) == 2
        assert np.max(np.abs(small_layers[0]["W"])) < 0.1

    def test_build_mlp_invalid_inputs(self) -> None:
        from src.learning.imitation._mlp import _build_mlp

        with pytest.raises(ValueError, match="input_dim must be positive"):
            _build_mlp(0, 2, [16])
        with pytest.raises(ValueError, match="output_dim must be positive"):
            _build_mlp(4, -1, [16])
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            _build_mlp(4, 2, [16, 0])
        with pytest.raises(ValueError, match="Unknown initialization method"):
            _build_mlp(4, 2, [16], init="unknown")  # type: ignore[arg-type]

    def test_mlp_forward_and_backward_contracts(self) -> None:
        from src.learning.imitation._mlp import _build_mlp, _mlp_backward, _mlp_forward

        layers = _build_mlp(4, 2, [8])
        with pytest.raises(ValueError, match="x must be provided"):
            _mlp_forward(layers, None)  # type: ignore[arg-type]

        x = np.ones((5, 4))
        out, acts = _mlp_forward(layers, x)
        assert out.shape == (5, 2)
        assert len(acts) == len(layers)

        with pytest.raises(ValueError, match="activations length"):
            _mlp_backward(layers, acts[:1], np.ones((5, 2)))

    def test_adam_step_contracts(self) -> None:
        from src.learning.imitation._mlp import (
            _adam_step,
            _build_mlp,
            _init_adam_state,
            _mlp_backward,
            _mlp_forward,
        )

        layers = _build_mlp(4, 2, [8])
        state = _init_adam_state(layers)
        assert len(state) == len(layers)

        x = np.ones((3, 4))
        _, acts = _mlp_forward(layers, x)
        grads, _ = _mlp_backward(layers, acts, np.ones((3, 2)))

        with pytest.raises(ValueError, match="step must be positive"):
            _adam_step(layers, grads, state, lr=1e-3, step=0)

    @pytest.mark.parametrize("activation", ["relu", "tanh", "identity"])
    @pytest.mark.parametrize("dropout", [0.0, 0.3])
    def test_mlp_backward_matches_finite_differences(
        self, activation: str, dropout: float
    ) -> None:
        """Analytic gradients agree with central differences for every
        supported activation, with and without fixed dropout masks."""
        from src.learning.imitation._mlp import (
            _build_mlp,
            _dropout_masks,
            _mlp_backward,
            _mlp_forward,
        )

        rng = np.random.default_rng(3)
        layers = _build_mlp(3, 2, [5, 4], init="he", rng=rng)
        x = rng.standard_normal((6, 3))
        target = rng.standard_normal((6, 2))
        masks = _dropout_masks(layers, len(x), dropout, rng)

        def loss() -> float:
            out, _ = _mlp_forward(layers, x, activation=activation, masks=masks)
            return float(0.5 * np.sum((out - target) ** 2))

        out, acts = _mlp_forward(layers, x, activation=activation, masks=masks)
        grads, _ = _mlp_backward(
            layers, acts, out - target, activation=activation, masks=masks
        )

        eps = 1e-6
        for layer, grad in zip(layers, grads, strict=True):
            w = layer["W"]
            for idx in [(0, 0), (w.shape[0] - 1, w.shape[1] - 1)]:
                original = w[idx]
                w[idx] = original + eps
                up = loss()
                w[idx] = original - eps
                down = loss()
                w[idx] = original
                np.testing.assert_allclose(
                    grad["W"][idx], (up - down) / (2 * eps), rtol=1e-5, atol=1e-7
                )

    def test_dropout_masks_contract(self) -> None:
        from src.learning.imitation._mlp import _build_mlp, _dropout_masks

        layers = _build_mlp(4, 2, [8, 6])
        rng = np.random.default_rng(0)
        assert _dropout_masks(layers, 5, 0.0, rng) is None
        masks = _dropout_masks(layers, 5, 0.5, rng)
        assert masks is not None
        assert [m.shape for m in masks] == [(5, 8), (5, 6)]
        assert set(np.unique(np.concatenate([m.ravel() for m in masks]))) <= {0.0, 2.0}
        with pytest.raises(ValueError, match="dropout"):
            _dropout_masks(layers, 5, 1.0, rng)
