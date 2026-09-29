"""Unit tests for the realistic-speed mask + timestep filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from src.shared.python.motion_matching.inverse_timestep.filter import (
    filter_timesteps_by_speed,
    realistic_speed_mask,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# realistic_speed_mask
# ---------------------------------------------------------------------------


def test_realistic_speed_mask_inclusive_bounds() -> None:
    speeds = np.array([49.9, 50.0, 100.0, 150.0, 150.1, 0.0, 200.0])
    mask = realistic_speed_mask(speeds, lo=50.0, hi=150.0)
    expected = np.array([False, True, True, True, False, False, False])
    np.testing.assert_array_equal(mask, expected)


def test_realistic_speed_mask_default_window() -> None:
    speeds = np.array([10.0, 50.0, 100.0, 150.0, 200.0])
    mask = realistic_speed_mask(speeds)
    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_realistic_speed_mask_custom_window() -> None:
    speeds = np.array([60.0, 80.0, 100.0])
    mask = realistic_speed_mask(speeds, lo=70.0, hi=90.0)
    np.testing.assert_array_equal(mask, [False, True, False])


def test_realistic_speed_mask_rejects_non_array() -> None:
    with pytest.raises(TypeError, match="np.ndarray"):
        realistic_speed_mask([50.0, 100.0], lo=50.0, hi=150.0)  # type: ignore[arg-type]


def test_realistic_speed_mask_rejects_2d_array() -> None:
    with pytest.raises(ValueError, match="1-D"):
        realistic_speed_mask(np.zeros((2, 2)), lo=50.0, hi=150.0)


def test_realistic_speed_mask_rejects_nan() -> None:
    speeds = np.array([50.0, np.nan, 100.0])
    with pytest.raises(ValueError, match="non-finite"):
        realistic_speed_mask(speeds)


def test_realistic_speed_mask_rejects_inf() -> None:
    speeds = np.array([50.0, np.inf, 100.0])
    with pytest.raises(ValueError, match="non-finite"):
        realistic_speed_mask(speeds)


def test_realistic_speed_mask_rejects_negative_lo() -> None:
    speeds = np.array([10.0])
    with pytest.raises(ValueError, match="lo must be >= 0"):
        realistic_speed_mask(speeds, lo=-1.0, hi=10.0)


def test_realistic_speed_mask_rejects_lo_eq_hi() -> None:
    speeds = np.array([10.0])
    with pytest.raises(ValueError, match="lo < hi"):
        realistic_speed_mask(speeds, lo=10.0, hi=10.0)


def test_realistic_speed_mask_rejects_lo_gt_hi() -> None:
    speeds = np.array([10.0])
    with pytest.raises(ValueError, match="lo < hi"):
        realistic_speed_mask(speeds, lo=200.0, hi=100.0)


# ---------------------------------------------------------------------------
# filter_timesteps_by_speed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeDataset:
    trials: Any
    timesteps: Any
    joint_names: tuple = ()
    coefficient_letters: tuple = ("A", "B", "C", "D", "E", "F", "G")
    schema_version: str = "compact-1.0"


def _make_dataset(speeds: list[float]) -> _FakeDataset:
    timesteps = pd.DataFrame(
        {
            "trial_id": [0] * len(speeds),
            "clubhead_speed_mph": speeds,
            "extra": list(range(len(speeds))),
        }
    )
    trials = pd.DataFrame({"trial_id": [0]})
    return _FakeDataset(trials=trials, timesteps=timesteps)


def test_filter_timesteps_basic() -> None:
    ds = _make_dataset([10.0, 60.0, 120.0, 200.0, 80.0])
    out = filter_timesteps_by_speed(ds, lo=50.0, hi=150.0)
    assert len(out.timesteps) == 3
    speeds = out.timesteps["clubhead_speed_mph"].tolist()
    assert speeds == [60.0, 120.0, 80.0]
    assert out.timesteps["extra"].tolist() == [1, 2, 4]


def test_filter_timesteps_preserves_trials_field() -> None:
    ds = _make_dataset([10.0, 60.0])
    out = filter_timesteps_by_speed(ds)
    assert out.trials is ds.trials
    assert out.schema_version == ds.schema_version


def test_filter_timesteps_empty_window_yields_zero_rows() -> None:
    ds = _make_dataset([200.0, 300.0, 400.0])
    out = filter_timesteps_by_speed(ds, lo=50.0, hi=150.0)
    assert len(out.timesteps) == 0


def test_filter_timesteps_requires_pandas() -> None:
    ds = _FakeDataset(trials=None, timesteps={"clubhead_speed_mph": [50.0]})
    with pytest.raises(TypeError, match="pandas DataFrame"):
        filter_timesteps_by_speed(ds)


def test_filter_timesteps_requires_speed_column() -> None:
    ts = pd.DataFrame({"trial_id": [0], "other": [1.0]})
    ds = _FakeDataset(trials=pd.DataFrame({"trial_id": [0]}), timesteps=ts)
    with pytest.raises(ValueError, match="clubhead_speed_mph"):
        filter_timesteps_by_speed(ds)


def test_filter_timesteps_validates_bounds() -> None:
    ds = _make_dataset([60.0])
    with pytest.raises(ValueError, match="lo < hi"):
        filter_timesteps_by_speed(ds, lo=100.0, hi=50.0)
