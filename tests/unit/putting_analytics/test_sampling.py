"""Sampler moments, reproducibility and per-channel stream independence."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.putting_analytics.sampling import sample_execution, sample_read
from src.shared.python.putting_analytics.skill import (
    PuttingSkillProfile,
    ReadDispersion,
    StrokeDispersion,
)

N = 200_000
STROKE = StrokeDispersion(1.0, 2.0, 0.1, 3.0, face_bias_deg=0.5, speed_bias=0.05)

pytestmark = pytest.mark.unit


def test_moments_match_the_profile() -> None:
    sample = sample_execution(STROKE, N, seed=1)
    assert sample.face_deg.mean() == pytest.approx(0.5, abs=0.01)
    assert sample.path_deg.std() == pytest.approx(2.0, rel=0.01)
    assert sample.speed_ratio.mean() == pytest.approx(1.05, rel=0.002)
    cv = sample.speed_ratio.std() / sample.speed_ratio.mean()
    assert cv == pytest.approx(0.1, rel=0.02)
    assert np.all(sample.speed_ratio > 0)
    read = sample_read(ReadDispersion(0.2, 1.5, 0.1, 0.1), N, seed=1)
    assert read.grade_ratio.mean() == pytest.approx(1.1, rel=0.003)
    assert read.aspect_error_deg.std() == pytest.approx(1.5, rel=0.01)


def test_same_seed_reproduces_and_new_seed_differs() -> None:
    first, again = (sample_execution(STROKE, 10, seed=7) for _ in range(2))
    assert np.array_equal(first.face_deg, again.face_deg)
    assert not np.array_equal(first.face_deg, sample_execution(STROKE, 10, 8).face_deg)


def test_ablating_one_channel_leaves_other_draws_identical() -> None:
    profile = PuttingSkillProfile("p", STROKE)
    full = sample_execution(profile.stroke, 50, seed=3)
    ablated = sample_execution(profile.without("path").stroke, 50, seed=3)
    assert np.array_equal(full.face_deg, ablated.face_deg)
    assert np.array_equal(full.speed_ratio, ablated.speed_ratio)
    assert np.all(ablated.path_deg == 0.0)


def test_start_line_blends_face_and_path() -> None:
    sample = sample_execution(STROKE, 5, seed=0)
    blended = sample.start_line_error_deg(0.8)
    assert np.allclose(blended, 0.8 * sample.face_deg + 0.2 * sample.path_deg)


def test_bad_sample_size_or_seed_is_rejected() -> None:
    with pytest.raises(ValueError):
        sample_execution(STROKE, 0, seed=1)
    with pytest.raises(ValueError):
        sample_read(ReadDispersion(), 5, seed=-1)
