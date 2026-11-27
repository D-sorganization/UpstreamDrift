"""Toy tests for NP-03 rank-1 probe and edit (#14478)."""

from __future__ import annotations

import numpy as np
import pytest

from src.research.neural_policy.edits import (
    apply_low_rank_weight_edit,
    probe_activation_direction,
    reverse_low_rank_edit,
)

pytestmark = pytest.mark.unit


def test_probe_recovers_planted_direction() -> None:
    h = np.array([0.0, 0.0, 1.0])
    g = np.array([0.0, 2.0, 0.0])
    u = probe_activation_direction(h, g)
    np.testing.assert_allclose(u, np.array([0.0, 1.0, 0.0]))


def test_rank1_masked_edit_reversible_and_bit_exact_unmasked_rows() -> None:
    w = np.ones((2, 3), dtype=float)
    u = np.array([1.0, 0.0])
    w_prime, recipe = apply_low_rank_weight_edit(
        w, u, alpha=1.0, mask_rows=np.array([0])
    )
    np.testing.assert_allclose(w_prime[1, :], w[1, :], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(w_prime[0, :], np.zeros(3))
    restored = reverse_low_rank_edit(w_prime, recipe)
    np.testing.assert_allclose(restored, w, atol=0.0, rtol=0.0)
    assert recipe.as_dict()["intervention"].startswith("rank1")
    assert recipe.mask_rows == (0,)


def test_full_matrix_edit_reversible() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 5))
    u = rng.standard_normal(4)
    u = u / np.linalg.norm(u)
    w_p, recipe = apply_low_rank_weight_edit(w, u, alpha=0.3, mask_rows=None)
    assert recipe.mask_rows is None
    back = reverse_low_rank_edit(w_p, recipe)
    np.testing.assert_allclose(back, w, rtol=1e-14, atol=1e-14)
