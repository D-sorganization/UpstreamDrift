"""Paired comparison uncertainty contracts."""

import numpy as np
import pytest

from src.tools.shot_pattern_analysis.uncertainty import paired_variance_ratio

pytestmark = pytest.mark.unit


def test_identical_shots_have_exact_unit_ratio_interval():
    values = np.arange(20, dtype=float)
    result = paired_variance_ratio(values, values, resamples=100, seed=12)
    assert result == {"estimate": 1.0, "lower_95": 1.0, "upper_95": 1.0}


def test_paired_scaling_preserves_known_variance_ratio():
    values = np.linspace(-5, 5, 40)
    result = paired_variance_ratio(values * 2 + 10, values, resamples=100)
    assert result["estimate"] == pytest.approx(4)
    assert result["lower_95"] == pytest.approx(4)
    assert result["upper_95"] == pytest.approx(4)


@pytest.mark.parametrize(
    "a,b", [([1, 2], [1]), ([1, np.nan], [1, 2]), ([1, 2], [0, 0])]
)
def test_invalid_pairs_rejected(a, b):
    with pytest.raises(ValueError):
        paired_variance_ratio(a, b, resamples=100)


def test_bootstrap_seed_is_reproducible():
    values = np.linspace(-3, 3, 50)
    first = paired_variance_ratio(values**2, values, resamples=100, seed=19)
    second = paired_variance_ratio(values**2, values, resamples=100, seed=19)
    assert first == second
    assert first["lower_95"] < first["estimate"] < first["upper_95"]
