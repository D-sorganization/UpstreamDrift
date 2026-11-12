"""Toy tests for NP-04 LoRA / constrained projector comparator (#14479)."""

from __future__ import annotations

import numpy as np
import pytest

from src.research.neural_policy.lora_adapter import LoRAAdapter, project_box

pytestmark = pytest.mark.unit


def test_lora_rank_zero_matches_linear() -> None:
    rng = np.random.default_rng(1)
    w = rng.standard_normal((3, 4))
    x = rng.standard_normal(4)
    adapter = LoRAAdapter(w, rank=0, rng=rng)
    np.testing.assert_allclose(adapter.forward(x), w @ x)


def test_lora_rank_positive_zero_b_equals_baseline() -> None:
    rng = np.random.default_rng(2)
    w = rng.standard_normal((2, 3))
    x = rng.standard_normal(3)
    adapter = LoRAAdapter(w, rank=2, rng=rng)
    np.testing.assert_allclose(adapter.forward(x), w @ x)


def test_lora_same_seed_reproduces_a() -> None:
    seed = 42
    w = np.ones((2, 2))
    a1 = LoRAAdapter(w, rank=1, rng=np.random.default_rng(seed)).a
    a2 = LoRAAdapter(w, rank=1, rng=np.random.default_rng(seed)).a
    np.testing.assert_allclose(a1, a2, atol=0.0, rtol=0.0)


def test_project_box_keeps_torque_within_caps() -> None:
    x = np.array([-10.0, 0.5, 2.0])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    y = project_box(x, lo, hi)
    np.testing.assert_allclose(y, np.array([-1.0, 0.5, 1.0]))
