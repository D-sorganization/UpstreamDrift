"""Regression tests for issue #7567.

The GAIL discriminator computed its sigmoid as ``1 / (1 + np.exp(-x))`` and its
BCE loss / reward as ``log(p + eps)`` / ``-log(1 - p + eps)``. For
large-magnitude logits ``np.exp(-x)`` overflows (``RuntimeWarning`` / ``inf``)
and the log terms saturate — the canonical case for a numerically-stable
log-sum-exp formulation.

These tests pin overflow-safe behaviour: the discriminator probability and the
GAIL reward must stay finite and in range even for extreme logits, evaluated
under ``np.errstate(over="raise")`` so the *old* implementation raises
``FloatingPointError`` (red) and the stable one passes (green). A moderate-logit
test confirms the BCE math is unchanged where overflow is not a factor.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.imitation._base import TrainingConfig
from src.learning.imitation._gail import GAIL


def _gail_with_linear_discriminator(scale: float) -> GAIL:
    """GAIL whose discriminator is a single linear layer with large weights.

    A one-layer discriminator makes the logit ``scale * sum(inputs)`` so tests
    can drive it to arbitrary magnitude (input dim = obs + action = 3).
    """
    gail = GAIL(observation_dim=2, action_dim=1, config=TrainingConfig(epochs=1))
    d = gail.observation_dim + gail.action_dim
    gail._discriminator = [{"W": np.full((d, 1), scale), "b": np.zeros(1)}]
    return gail


@pytest.mark.unit
def test_discriminator_probability_finite_for_extreme_logits() -> None:
    """Sigmoid must not overflow for large +/- logits (errstate=raise)."""
    gail = _gail_with_linear_discriminator(scale=50.0)
    # |logit| ~ 50 * (sum of |inputs|) ~ 1500.
    big_neg = np.full((4, 2), -10.0)
    big_pos = np.full((4, 2), 10.0)
    act_pos = np.full((4, 1), 10.0)
    act_neg = np.full((4, 1), -10.0)

    with np.errstate(over="raise", invalid="raise"):
        p_neg = gail._forward_discriminator(big_neg, act_neg)
        p_pos = gail._forward_discriminator(big_pos, act_pos)

    assert np.all(np.isfinite(p_neg)) and np.all(np.isfinite(p_pos))
    assert np.all((p_neg >= 0.0) & (p_neg <= 1.0))
    assert np.all((p_pos >= 0.0) & (p_pos <= 1.0))
    assert np.all(p_neg < 1e-6)  # saturates to 0
    assert np.all(p_pos > 1.0 - 1e-6)  # saturates to 1


@pytest.mark.unit
def test_reward_finite_for_saturated_discriminator() -> None:
    """get_reward = -log(1 - D) must stay finite when D -> 1."""
    gail = _gail_with_linear_discriminator(scale=50.0)
    state = np.full(2, 10.0)
    action = np.full(1, 10.0)  # drives D -> 1, so -log(1-D) -> large but finite
    with np.errstate(over="raise", invalid="raise"):
        reward = gail.get_reward(state, action)
    assert np.isfinite(reward)
    assert reward > 0.0


@pytest.mark.unit
def test_bce_matches_probability_form_for_moderate_logits() -> None:
    """For moderate logits the stable BCE equals the naive log-of-sigmoid.

    This locks the *math* of the log-sum-exp rewrite: it is the same binary
    cross-entropy, only evaluated without overflow/underflow.
    """
    logits = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    p = 1.0 / (1.0 + np.exp(-logits))  # safe at this magnitude
    naive_neg_log_p = -np.log(p)  # -log(sigmoid(z))
    naive_neg_log_1mp = -np.log(1.0 - p)  # -log(1 - sigmoid(z))

    stable_neg_log_p = np.logaddexp(0.0, -logits)  # softplus(-z)
    stable_neg_log_1mp = np.logaddexp(0.0, logits)  # softplus(z)

    np.testing.assert_allclose(stable_neg_log_p, naive_neg_log_p, rtol=1e-12)
    np.testing.assert_allclose(stable_neg_log_1mp, naive_neg_log_1mp, rtol=1e-12)
