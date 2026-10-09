"""Correlated delivery assumptions stay paired and mathematically valid."""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.shot_pattern_analysis.noise import (
    NOISE_FIELDS,
    NoiseSpec,
    sample_paired_noise,
)

pytestmark = pytest.mark.unit


def test_seed_replays_same_joint_noise_and_zero_sd_axes() -> None:
    spec = NoiseSpec(
        standard_deviations=(1.0, 0.5, 0.44704, 0.00635, 0.00635, 0.0, 0.0)
    )
    first = sample_paired_noise(spec, n_shots=4000, seed=123)
    second = sample_paired_noise(spec, n_shots=4000, seed=123)
    assert first.shape == (4000, len(NOISE_FIELDS))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 5:], 0.0)
    np.testing.assert_allclose(
        first.std(axis=0)[:5], spec.standard_deviations[:5], rtol=0.04
    )


def test_selected_correlation_is_reflected_in_samples() -> None:
    matrix = np.eye(len(NOISE_FIELDS))
    matrix[0, 1] = matrix[1, 0] = 0.8
    spec = NoiseSpec(
        standard_deviations=(1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        correlation=tuple(tuple(float(v) for v in row) for row in matrix),
    )
    samples = sample_paired_noise(spec, n_shots=10_000, seed=9)
    assert np.corrcoef(samples[:, 0], samples[:, 1])[0, 1] == pytest.approx(
        0.8, abs=0.025
    )


@pytest.mark.parametrize(
    "bad_correlation",
    [
        ((1.0, 2.0), (2.0, 1.0)),
        ((1.0, 0.5), (0.4, 1.0)),
        ((1.0, 0.0), (0.0, -1.0)),
    ],
)
def test_invalid_correlation_is_rejected(bad_correlation) -> None:
    with pytest.raises(ValueError, match="correlation"):
        NoiseSpec(standard_deviations=(1.0,) * 7, correlation=bad_correlation)


def test_non_psd_correlation_is_rejected() -> None:
    matrix = np.eye(len(NOISE_FIELDS))
    matrix[:3, :3] = [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]
    with pytest.raises(ValueError, match="positive semidefinite"):
        NoiseSpec(
            standard_deviations=(1.0,) * 7,
            correlation=tuple(tuple(float(v) for v in row) for row in matrix),
        )


def test_invalid_sd_and_sample_count_are_rejected() -> None:
    with pytest.raises(ValueError, match="standard deviations"):
        NoiseSpec(standard_deviations=(1.0, -1.0, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="standard deviations"):
        NoiseSpec(standard_deviations=(1.0,) * 6)
    with pytest.raises(ValueError, match="n_shots"):
        sample_paired_noise(NoiseSpec((0.0,) * 7), n_shots=0, seed=1)
