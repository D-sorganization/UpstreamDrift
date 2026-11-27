"""Tests for offline SVD-direction ablation vs high-frequency jitter (#14501)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.research.neural_policy.edits import apply_low_rank_weight_edit
from src.research.neural_policy.svd_jitter_ablation import (
    ablate_top_r_directions,
    compute_high_frequency_power,
    run_jitter_ablation_benchmark,
)

pytestmark = pytest.mark.unit


def test_deterministic_high_frequency_power_metric() -> None:
    t = np.linspace(0.0, 1.0, 100, endpoint=False)
    p_5hz = compute_high_frequency_power(np.sin(10.0 * np.pi * t), 100.0, 15.0)
    assert p_5hz == pytest.approx(0.0, abs=1e-12)

    p_25hz = compute_high_frequency_power(np.sin(50.0 * np.pi * t), 100.0, 15.0)
    assert p_25hz > 0.1
    assert compute_high_frequency_power(
        np.sin(50.0 * np.pi * t), 100.0, 30.0
    ) == pytest.approx(0.0, abs=1e-12)


def test_planted_direction_ablation_removes_jitter_random_does_not() -> None:
    t = np.linspace(0.0, 1.0, 100, endpoint=False)
    x = 0.5 * np.sin(60.0 * np.pi * t)[:, None] * np.array([[1.0, 0.0]])
    w = 5.0 * np.outer(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0]))

    p_before = compute_high_frequency_power(x @ w.T, 100.0, 15.0)
    assert p_before > 0.1

    w_abl, recipes = ablate_top_r_directions(w, r=1)
    assert len(recipes) == 1
    assert compute_high_frequency_power(x @ w_abl.T, 100.0, 15.0) == pytest.approx(
        0.0, abs=1e-12
    )

    w_rand, _ = apply_low_rank_weight_edit(w, np.array([0.0, 1.0, 0.0]), alpha=1.0)
    p_rand = compute_high_frequency_power(x @ w_rand.T, 100.0, 15.0)
    assert p_rand == pytest.approx(p_before, rel=1e-6)


def test_fixed_seed_reproduces_report_bit_for_bit() -> None:
    r1 = run_jitter_ablation_benchmark(seeds=(10, 11, 12, 13, 14), k_max_rank=2)
    r2 = run_jitter_ablation_benchmark(seeds=(10, 11, 12, 13, 14), k_max_rank=2)
    assert r1 == r2 and len(r1["rows"]) == 5
    assert "ablation_beat_low_pass" in r1
    assert "no claim is made about real markerless mocap" in r1["disclaimer"].lower()
    assert r1["provenance"]["issue"] == 14501


def test_lexical_guardrail_clean() -> None:
    banned = [
        "imped" + "ance",
        "stiff" + "ness",
        "syn" + "ergy",
        "co-con" + "traction",
        "mus" + "cle",
        "ana" + "tomy",
        "control-" + "affine",
        "ZT" + "CF",
        "ZV" + "CF",
    ]
    root = Path(__file__).resolve().parents[2]
    files = [
        root / "src" / "research" / "neural_policy" / "svd_jitter_ablation.py",
        Path(__file__),
    ]
    for fp in files:
        text = fp.read_text(encoding="utf-8")
        for term in banned:
            assert not re.findall(rf"\b{re.escape(term)}\b", text, re.IGNORECASE), (
                f"{term} in {fp}"
            )
