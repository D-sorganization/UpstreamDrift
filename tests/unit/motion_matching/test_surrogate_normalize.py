"""Tests for per-feature z-score normalization helpers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from src.shared.python.motion_matching.surrogate._normalize import (
    NormalizationStats,
    denormalize_positions,
    fit_stats,
    zscore_coeffs,
    zscore_positions,
)


@pytest.fixture
def fake_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (coeffs, butt, clubhead) arrays for stats fitting."""
    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=(8, 14)).astype(np.float32)
    butt = rng.normal(size=(8, 30, 3)).astype(np.float32)
    head = rng.normal(size=(8, 30, 3)).astype(np.float32)
    return coeffs, butt, head


@pytest.mark.unit
def test_normalize_stats_only_use_train_split(
    fake_split: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """fit_stats must produce mean/std purely from the split it was given."""
    coeffs, butt, head = fake_split
    train_idx = np.arange(0, 5)
    val_idx = np.arange(5, 8)

    train_stats = fit_stats(coeffs[train_idx], butt[train_idx], head[train_idx])
    full_stats = fit_stats(coeffs, butt, head)

    # Adding val samples must shift the stats — proves we did not leak.
    assert not np.allclose(train_stats.coeffs_mean, full_stats.coeffs_mean)
    # And the train-only mean must equal np.mean of train slice exactly.
    np.testing.assert_allclose(
        train_stats.coeffs_mean, coeffs[train_idx].mean(axis=0), atol=1.0e-6
    )
    # val_idx exists; just sanity-check shape parity.
    assert val_idx.shape[0] > 0


@pytest.mark.unit
def test_normalize_round_trip(
    fake_split: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """denormalize(normalize(x)) == x within float32 tolerance."""
    _, butt, _ = fake_split
    stats = fit_stats(*fake_split)
    x = torch.from_numpy(butt[:2]).float()
    z = zscore_positions(x, stats.butt_mean, stats.butt_std)
    back = denormalize_positions(z, stats.butt_mean, stats.butt_std)
    torch.testing.assert_close(back, x, atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.unit
def test_zscore_coeffs_zero_mean_unit_std(
    fake_split: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Applying zscore_coeffs to the train data yields ~zero mean, ~unit std."""
    coeffs, butt, head = fake_split
    stats = fit_stats(coeffs, butt, head)
    x = torch.from_numpy(coeffs).float()
    z = zscore_coeffs(x, stats)
    assert abs(float(z.mean())) < 1.0e-5
    assert abs(float(z.std()) - 1.0) < 0.1  # small N => loose tolerance


@pytest.mark.unit
def test_fit_stats_floors_zero_std() -> None:
    """A constant feature must not produce a zero std (would div-by-zero later)."""
    coeffs = np.ones((4, 3), dtype=np.float32) * 3.14
    butt = np.zeros((4, 5, 3), dtype=np.float32)
    head = np.zeros((4, 5, 3), dtype=np.float32)
    stats = fit_stats(coeffs, butt, head)
    assert (stats.coeffs_std > 0).all()
    assert (stats.butt_std > 0).all()


@pytest.mark.unit
def test_fit_stats_rejects_empty() -> None:
    """Empty input must raise rather than producing garbage stats."""
    with pytest.raises(ValueError):
        fit_stats(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 5, 3), dtype=np.float32),
            np.zeros((0, 5, 3), dtype=np.float32),
        )


@pytest.mark.unit
def test_normalization_stats_is_frozen() -> None:
    """NormalizationStats is a frozen dataclass — assignment must raise."""
    stats = NormalizationStats(
        coeffs_mean=np.zeros(2, dtype=np.float32),
        coeffs_std=np.ones(2, dtype=np.float32),
        butt_mean=np.zeros(3, dtype=np.float32),
        butt_std=np.ones(3, dtype=np.float32),
        clubhead_mean=np.zeros(3, dtype=np.float32),
        clubhead_std=np.ones(3, dtype=np.float32),
    )
    # FrozenInstanceError is a subclass of AttributeError.
    with pytest.raises(AttributeError):
        stats.coeffs_mean = np.ones(2)  # type: ignore[misc]
