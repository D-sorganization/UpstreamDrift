"""TDD contract and determinism tests for imitation learning (Issue #13013).

Covers:
1. BehaviorCloning.train validation_split range validation [0, 1).
2. TrainingConfig activation and dropout validation and execution.
3. Seed threading and determinism across BC, DAgger, and GAIL.
4. DAgger dataset mutation immutability.
5. DAgger trajectory collection observation shape check and success attribution.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from src.learning.imitation._base import TrainingConfig
from src.learning.imitation._bc import BehaviorCloning
from src.learning.imitation._dagger import DAgger
from src.learning.imitation._gail import GAIL
from src.learning.imitation.dataset import Demonstration, DemonstrationDataset

pytestmark = pytest.mark.unit


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
                success=True,
            )
        )
    return DemonstrationDataset(demos)


class TestValidationSplitContract:
    """Item 1: validation_split contract and range checks."""

    def test_bc_validation_split_one_raises(self) -> None:
        bc = BehaviorCloning(4, 2, TrainingConfig(epochs=1, batch_size=4))
        ds = _toy_dataset()
        with pytest.raises(ValueError, match="validation_split"):
            bc.train(ds, validation_split=1.0)

    def test_bc_validation_split_negative_raises(self) -> None:
        bc = BehaviorCloning(4, 2, TrainingConfig(epochs=1, batch_size=4))
        ds = _toy_dataset()
        with pytest.raises(ValueError, match="validation_split"):
            bc.train(ds, validation_split=-0.1)

    def test_bc_validation_split_leaves_no_training_samples_raises(self) -> None:
        bc = BehaviorCloning(4, 2, TrainingConfig(epochs=1, batch_size=4))
        # 1 transition only
        ds = DemonstrationDataset(
            [
                Demonstration(
                    timestamps=np.array([0.0, 0.01]),
                    joint_positions=np.zeros((2, 2)),
                    joint_velocities=np.zeros((2, 2)),
                    actions=np.zeros((2, 2)),
                )
            ]
        )
        with pytest.raises(ValueError, match="training samples"):
            bc.train(ds, validation_split=0.99)


class TestConfigActivationAndDropout:
    """Item 2: activation and dropout config validation & handling."""

    def test_unsupported_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="activation"):
            TrainingConfig(activation="unsupported_fn")

    def test_invalid_dropout_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            TrainingConfig(dropout=1.0)
        with pytest.raises(ValueError, match="dropout"):
            TrainingConfig(dropout=-0.1)

    def test_bc_tanh_activation_execution(self) -> None:
        cfg = TrainingConfig(
            epochs=2,
            batch_size=4,
            activation="tanh",
            seed=42,
        )
        bc = BehaviorCloning(4, 2, cfg)
        ds = _toy_dataset()
        hist = bc.train(ds, validation_split=0.2)
        assert "train_loss" in hist
        assert len(hist["train_loss"]) == 2

    def test_bc_dropout_training_and_eval(self) -> None:
        cfg = TrainingConfig(
            epochs=2,
            batch_size=4,
            dropout=0.2,
            seed=42,
        )
        bc = BehaviorCloning(4, 2, cfg)
        ds = _toy_dataset()
        hist = bc.train(ds, validation_split=0.2)
        assert "train_loss" in hist
        # Predict uses eval mode (no dropout)
        pred = bc.predict(np.zeros(4))
        assert pred.shape == (2,)


class TestSeedDeterminism:
    """Item 3: seed threading produces deterministic training histories."""

    def test_bc_identical_seed_reproducibility(self) -> None:
        cfg1 = TrainingConfig(epochs=3, batch_size=4, seed=12345)
        cfg2 = TrainingConfig(epochs=3, batch_size=4, seed=12345)
        bc1 = BehaviorCloning(4, 2, cfg1)
        bc2 = BehaviorCloning(4, 2, cfg2)
        ds = _toy_dataset()

        h1 = bc1.train(ds, validation_split=0.2)
        h2 = bc2.train(ds, validation_split=0.2)

        assert h1 == h2
        obs = np.array([0.5, -0.2, 0.1, 0.9])
        np.testing.assert_allclose(bc1.predict(obs), bc2.predict(obs))

    def test_gail_identical_seed_reproducibility(self) -> None:
        cfg1 = TrainingConfig(epochs=2, batch_size=4, seed=999)
        cfg2 = TrainingConfig(epochs=2, batch_size=4, seed=999)
        g1 = GAIL(4, 2, cfg1)
        g2 = GAIL(4, 2, cfg2)
        ds = _toy_dataset()

        h1 = g1.train(ds, validation_split=0.2)
        h2 = g2.train(ds, validation_split=0.2)

        assert h1["discriminator_loss"] == h2["discriminator_loss"]
        assert h1["policy_loss"] == h2["policy_loss"]
        obs = np.array([0.5, -0.2, 0.1, 0.9])
        np.testing.assert_allclose(g1.predict(obs), g2.predict(obs))


class TestDAggerImmutabilityAndTrajectory:
    """Items 4 & 5: DAgger caller dataset protection and trajectory contracts."""

    def test_dagger_train_does_not_mutate_caller_dataset(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.step_count = 0

            def reset(self):
                self.step_count = 0
                return np.zeros(4), {}

            def step(self, action):
                self.step_count += 1
                return np.zeros(4), 1.0, self.step_count >= 2, False, {"success": True}

        ds = _toy_dataset(n_demos=1)
        initial_len = len(ds)

        cfg = TrainingConfig(epochs=1, batch_size=4, seed=777)
        d = DAgger(4, 2, cfg)
        d.train(ds)

        d.train_online(
            env=FakeEnv(),
            expert=lambda obs: np.zeros(2),
            iterations=2,
            trajectories_per_iter=2,
            max_steps=5,
        )

        assert len(ds) == initial_len
        assert d._aggregated_dataset is not None
        assert len(d._aggregated_dataset) == initial_len + 4

    def test_dagger_collect_trajectory_odd_obs_dim_raises(self) -> None:
        class OddObsEnv:
            def reset(self):
                return np.zeros(5), {}

            def step(self, action):
                return np.zeros(5), 0.0, True, False, {}

        d = DAgger(5, 2, TrainingConfig(epochs=1, batch_size=4))
        with pytest.raises(ValueError, match="even"):
            d._collect_trajectory(
                OddObsEnv(), expert=lambda o: np.zeros(2), beta=0.5, max_steps=5
            )

    def test_dagger_collect_trajectory_success_not_blindly_true_on_truncation(
        self,
    ) -> None:
        class TruncatingEnv:
            def reset(self):
                return np.zeros(4), {}

            def step(self, action):
                # Truncated without success, not terminated
                return np.zeros(4), 0.0, False, True, {}

        d = DAgger(4, 2, TrainingConfig(epochs=1, batch_size=4))
        demo, _ = d._collect_trajectory(
            TruncatingEnv(), expert=lambda o: np.zeros(2), beta=0.5, max_steps=5
        )
        # Previously success=not terminated would evaluate to True on truncated rollouts!
        assert demo.success is False
