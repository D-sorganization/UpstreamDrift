"""Wave 6 coverage: src.learning.imitation BC / DAgger / GAIL pure-Python paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.learning.imitation._base import TrainingConfig
from src.learning.imitation._bc import BehaviorCloning
from src.learning.imitation._dagger import DAgger
from src.learning.imitation._gail import GAIL
from src.learning.imitation.dataset import Demonstration, DemonstrationDataset


def _toy_dataset(
    n_demos: int = 2, n_frames: int = 6, n_joints: int = 2
) -> DemonstrationDataset:
    demos = []
    rng = np.random.default_rng(0)
    for _ in range(n_demos):
        demos.append(
            Demonstration(
                timestamps=np.arange(n_frames) * 0.01,
                joint_positions=rng.standard_normal((n_frames, n_joints)),
                joint_velocities=rng.standard_normal((n_frames, n_joints)),
                actions=rng.standard_normal((n_frames, n_joints)),
            )
        )
    return DemonstrationDataset(demos)


def _tiny_cfg() -> TrainingConfig:
    return TrainingConfig(epochs=2, batch_size=4, learning_rate=1e-3, hidden_sizes=[8])


class TestBehaviorCloning:
    def test_forward_shape(self) -> None:
        bc = BehaviorCloning(observation_dim=4, action_dim=2, config=_tiny_cfg())
        x = np.zeros((3, 4))
        out = bc._forward(x)
        assert out.shape == (3, 2)

    def test_predict_1d(self) -> None:
        bc = BehaviorCloning(4, 2, _tiny_cfg())
        out = bc.predict(np.zeros(4))
        assert out.shape == (2,)

    def test_predict_batch(self) -> None:
        bc = BehaviorCloning(4, 2, _tiny_cfg())
        out = bc.predict(np.zeros((5, 4)))
        assert out.shape == (5, 2)

    def test_train_history(self) -> None:
        bc = BehaviorCloning(4, 2, _tiny_cfg())
        ds = _toy_dataset()
        hist = bc.train(ds, validation_split=0.2)
        assert "train_loss" in hist
        assert len(hist["train_loss"]) == 2

    def test_train_empty_raises(self) -> None:
        bc = BehaviorCloning(4, 2, _tiny_cfg())
        with pytest.raises(ValueError, match="no state-action pairs"):
            bc.train(DemonstrationDataset())

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        bc = BehaviorCloning(4, 2, _tiny_cfg())
        out = tmp_path / "bc.npz"
        bc.save(out)
        # numpy appends .npz; handle both cases
        path = out if out.exists() else out.with_suffix(".npz")
        bc2 = BehaviorCloning(4, 2, _tiny_cfg())
        bc2.load(path)
        assert bc2.observation_dim == 4
        assert bc2.action_dim == 2

    def test_get_training_history(self) -> None:
        bc = BehaviorCloning(4, 2, _tiny_cfg())
        bc.train(_toy_dataset())
        hist = bc.get_training_history()
        assert "train_loss" in hist


class TestDAgger:
    def test_compute_beta_linear(self) -> None:
        assert DAgger._compute_beta(0, 5, "linear") == pytest.approx(1.0)
        assert DAgger._compute_beta(5, 5, "linear") == pytest.approx(0.0)

    def test_compute_beta_exponential(self) -> None:
        assert DAgger._compute_beta(2, 5, "exp") == pytest.approx(0.25)

    def test_train_invokes_bc(self) -> None:
        d = DAgger(4, 2, _tiny_cfg())
        hist = d.train(_toy_dataset())
        assert "train_loss" in hist

    def test_train_online_without_initial_raises(self) -> None:
        d = DAgger(4, 2, _tiny_cfg())
        with pytest.raises(ValueError, match="train\\(\\) first"):
            d.train_online(env=None, expert=lambda o: o, iterations=1)

    def test_predict(self) -> None:
        d = DAgger(4, 2, _tiny_cfg())
        d.train(_toy_dataset())
        out = d.predict(np.zeros(4))
        assert out.shape == (2,)

    def test_save_load(self, tmp_path: Path) -> None:
        d = DAgger(4, 2, _tiny_cfg())
        out = tmp_path / "dagger.npz"
        d.save(out)
        path = out if out.exists() else out.with_suffix(".npz")
        d2 = DAgger(4, 2, _tiny_cfg())
        d2.load(path)

    def test_train_online_with_fake_env(self) -> None:
        """Run train_online against a fake env to cover trajectory collection."""

        class FakeEnv:
            def __init__(self) -> None:
                self.step_count = 0

            def reset(self):
                self.step_count = 0
                return np.zeros(4), {}

            def step(self, action):
                self.step_count += 1
                obs = np.zeros(4)
                terminated = self.step_count >= 3
                return obs, 1.0, terminated, False, {}

        d = DAgger(4, 2, _tiny_cfg())
        d.train(_toy_dataset())
        results = d.train_online(
            env=FakeEnv(),
            expert=lambda obs: np.zeros(2),
            iterations=2,
            trajectories_per_iter=1,
            max_steps=5,
            beta_schedule="linear",
        )
        assert "iteration_rewards" in results
        assert len(results["iteration_rewards"]) == 2


class TestGAIL:
    def test_forward_policy_and_disc(self) -> None:
        g = GAIL(4, 2, _tiny_cfg())
        action = g._forward_policy(np.zeros((3, 4)))
        assert action.shape == (3, 2)
        out = g._forward_discriminator(np.zeros((3, 4)), np.zeros((3, 2)))
        # sigmoid output in [0,1]
        assert out.shape == (3, 1)
        assert np.all((out >= 0) & (out <= 1))

    def test_train_history(self) -> None:
        g = GAIL(4, 2, _tiny_cfg())
        hist = g.train(_toy_dataset())
        assert "discriminator_loss" in hist
        assert "policy_loss" in hist
        assert len(hist["discriminator_loss"]) == 2

    def test_train_empty_raises(self) -> None:
        g = GAIL(4, 2, _tiny_cfg())
        with pytest.raises(ValueError, match="no state-action pairs"):
            g.train(DemonstrationDataset())

    def test_predict_1d(self) -> None:
        g = GAIL(4, 2, _tiny_cfg())
        out = g.predict(np.zeros(4))
        assert out.shape == (2,)

    def test_predict_stochastic(self) -> None:
        g = GAIL(4, 2, _tiny_cfg())
        out = g.predict(np.zeros(4), deterministic=False)
        assert out.shape == (2,)

    def test_get_reward(self) -> None:
        g = GAIL(4, 2, _tiny_cfg())
        r = g.get_reward(np.zeros(4), np.zeros(2))
        assert isinstance(r, float)

    def test_save_load(self, tmp_path: Path) -> None:
        g = GAIL(4, 2, _tiny_cfg())
        out = tmp_path / "gail.npz"
        g.save(out)
        path = out if out.exists() else out.with_suffix(".npz")
        g2 = GAIL(4, 2, _tiny_cfg())
        g2.load(path)
        assert g2.observation_dim == 4
