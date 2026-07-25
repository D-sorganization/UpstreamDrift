"""Regression tests for issue #7973: GAIL.train() must actually train.

Before the fix, ``GAIL.train()`` never referenced ``self._policy`` for any
update, and the "discriminator update" was pure multiplicative weight decay
(``W -= lr * 0.01 * W``) independent of the data. The result was a randomly
initialised MLP returned as a trained imitation policy, with a smooth
converged-looking loss curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.imitation._base import TrainingConfig
from src.learning.imitation.dataset import Demonstration, DemonstrationDataset
from src.learning.imitation.learners import GAIL

pytestmark = pytest.mark.unit


def _cfg(epochs: int) -> TrainingConfig:
    """Small, fast network config for these regression tests."""
    return TrainingConfig(hidden_sizes=[32, 32], batch_size=32, epochs=epochs)


def _linear_expert_dataset(
    n: int = 128, obs_dim: int = 6, act_dim: int = 3, seed: int = 0
) -> tuple[DemonstrationDataset, np.ndarray, np.ndarray]:
    """Build a dataset whose action is a known linear function of the state."""
    rng = np.random.default_rng(seed)
    n_q = obs_dim // 2
    q = rng.normal(size=(n, n_q))
    qd = rng.normal(size=(n, n_q))
    states = np.concatenate([q, qd], axis=1)
    actions = states @ (rng.normal(size=(obs_dim, act_dim)) * 0.2)
    demo = Demonstration(
        timestamps=np.arange(n) * 0.01,
        joint_positions=q,
        joint_velocities=qd,
        actions=actions,
    )
    return DemonstrationDataset(demonstrations=[demo]), states, actions


class TestGailActuallyTrains:
    """#7973: weights must move, and move because of the data."""

    def test_policy_weights_change(self) -> None:
        """The policy is updated - it is not the random initialisation."""
        np.random.seed(0)
        dataset, _, _ = _linear_expert_dataset()
        gail = GAIL(observation_dim=6, action_dim=3, config=_cfg(epochs=20))

        before = [layer["W"].copy() for layer in gail._policy]
        gail.train(dataset, validation_split=0.0)
        after = [layer["W"] for layer in gail._policy]

        delta = max(
            float(np.max(np.abs(a - b))) for a, b in zip(before, after, strict=True)
        )
        assert delta > 1e-4, f"policy weights unchanged (max delta={delta})"

    def test_policy_output_scale_approaches_expert(self) -> None:
        """Predictions leave the ~1e-4 noise floor of the initialisation."""
        np.random.seed(1)
        dataset, states, actions = _linear_expert_dataset()
        gail = GAIL(observation_dim=6, action_dim=3, config=_cfg(epochs=40))

        before_scale = float(np.std(gail.predict(states)))
        gail.train(dataset, validation_split=0.0)
        after_scale = float(np.std(gail.predict(states)))
        expert_scale = float(np.std(actions))

        assert before_scale < 0.01 * expert_scale
        assert after_scale > 0.1 * expert_scale

    def test_discriminator_learns_to_separate_expert_from_random(self) -> None:
        """A trained discriminator must score expert pairs above random ones."""
        np.random.seed(2)
        dataset, states, actions = _linear_expert_dataset()
        gail = GAIL(observation_dim=6, action_dim=3, config=_cfg(epochs=40))
        gail.train(dataset, validation_split=0.0)

        rng = np.random.default_rng(3)
        random_actions = rng.normal(size=actions.shape) * float(np.std(actions))
        d_expert = float(np.mean(gail._forward_discriminator(states, actions)))
        d_random = float(np.mean(gail._forward_discriminator(states, random_actions)))

        assert d_expert > d_random + 0.02, f"D(expert)={d_expert} D(random)={d_random}"

    def test_discriminator_loss_decreases(self) -> None:
        """The reported discriminator loss must fall, not sit at 2*log(2)."""
        np.random.seed(4)
        dataset, _, _ = _linear_expert_dataset()
        gail = GAIL(observation_dim=6, action_dim=3, config=_cfg(epochs=30))
        hist = gail.train(dataset, validation_split=0.0)

        first = hist["discriminator_loss"][0]
        last = hist["discriminator_loss"][-1]
        assert last < first - 1e-3, f"disc loss flat: {first} -> {last}"
        # 2*log(2) == 1.3863 is the "D collapsed to 0.5" fixed point.
        assert last < 2.0 * np.log(2.0) - 1e-3

    def test_update_depends_on_the_data(self) -> None:
        """Two different expert datasets must produce different policies."""
        np.random.seed(5)
        ds_a, _, _ = _linear_expert_dataset(seed=10)
        ds_b, _, _ = _linear_expert_dataset(seed=11)

        np.random.seed(6)
        gail_a = GAIL(observation_dim=6, action_dim=3, config=_cfg(epochs=15))
        np.random.seed(6)
        gail_b = GAIL(observation_dim=6, action_dim=3, config=_cfg(epochs=15))
        np.testing.assert_allclose(gail_a._policy[0]["W"], gail_b._policy[0]["W"])

        np.random.seed(7)
        gail_a.train(ds_a, validation_split=0.0)
        np.random.seed(7)
        gail_b.train(ds_b, validation_split=0.0)

        delta = float(np.max(np.abs(gail_a._policy[-1]["W"] - gail_b._policy[-1]["W"])))
        assert delta > 1e-6, "training is independent of the demonstrations"

    def test_validation_split_reports_held_out_loss(self) -> None:
        """A non-zero validation split yields a val_policy_loss series."""
        np.random.seed(8)
        dataset, _, _ = _linear_expert_dataset()
        gail = GAIL(observation_dim=6, action_dim=3, config=_cfg(epochs=3))
        hist = gail.train(dataset, validation_split=0.25)

        assert len(hist["val_policy_loss"]) == 3
        assert all(np.isfinite(hist["val_policy_loss"]))

    @pytest.mark.parametrize("split", [-0.1, 1.0, 1.5])
    def test_invalid_validation_split_rejected(self, split: float) -> None:
        """DbC precondition on validation_split."""
        dataset, _, _ = _linear_expert_dataset(n=16)
        gail = GAIL(observation_dim=6, action_dim=3, config=_cfg(epochs=1))
        with pytest.raises(ValueError, match="validation_split"):
            gail.train(dataset, validation_split=split)
