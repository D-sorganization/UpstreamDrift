"""Tests for the synthetic sweep generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.shared.python.motion_matching.dataset import make_synthetic_sweep


@pytest.mark.unit
def test_make_synthetic_sweep_writes_valid_parquet(tmp_path: Path) -> None:
    folder = make_synthetic_sweep(
        tmp_path / "ds", n_trials=4, n_joints=6, n_timesteps=20, seed=1
    )
    trials = pd.read_parquet(folder / "trials.parquet")
    timesteps = pd.read_parquet(folder / "timesteps.parquet")

    assert len(trials) == 4
    assert len(timesteps) == 4 * 20
    assert set(trials["trial_id"].unique()) == {0, 1, 2, 3}
    assert len(trials["coefficients"].iloc[0]) == 6 * 7
    assert len(timesteps["q"].iloc[0]) == 6


@pytest.mark.unit
def test_synthetic_coefficient_bounds_documented_ranges(tmp_path: Path) -> None:
    """Sampled coefficients fall in the per-slot A..G bounds."""
    folder = make_synthetic_sweep(
        tmp_path / "ds", n_trials=8, n_joints=4, n_timesteps=10, seed=42
    )
    trials = pd.read_parquet(folder / "trials.parquet")
    half_ranges = [1000, 1000, 500, 500, 100, 100, 25]
    for coeffs in trials["coefficients"]:
        for i, value in enumerate(coeffs):
            half = half_ranges[i % 7]
            assert (
                -half <= value <= half
            ), f"slot {i % 7} value {value} outside [-{half}, {half}]"
