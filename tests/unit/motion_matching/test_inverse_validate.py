"""Unit tests for the round-trip validator (issue #4003 / #034)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from src.shared.python.motion_matching.inverse._validate import (
    ValidationReport,
    round_trip_validate,
)

from ._fixtures import make_target

_N = 21


def _perfect_forward(target_butt: NDArray[np.float64], target_ch: NDArray[np.float64]):
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (target_butt.shape[0], 1))

    def _fn(
        _c: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return target_butt.copy(), target_ch.copy(), quat

    return _fn


@pytest.mark.unit
def test_round_trip_validate_returns_report_shape() -> None:
    target = make_target(n=_N)
    fn = _perfect_forward(target.butt, target.clubhead)
    samples = [np.zeros(7), np.ones(7)]

    report = round_trip_validate(samples, target, fn, rmse_threshold_m=0.01)

    assert isinstance(report, ValidationReport)
    assert report.rmses_m.shape == (2,)
    assert report.accepted.shape == (2,)
    assert report.threshold_m == 0.01


@pytest.mark.unit
def test_round_trip_validate_perfect_forward_all_accepted() -> None:
    target = make_target(n=_N)
    fn = _perfect_forward(target.butt, target.clubhead)
    samples = [np.zeros(7) for _ in range(5)]

    report = round_trip_validate(samples, target, fn, rmse_threshold_m=0.005)

    assert bool(report.accepted.all())
    assert np.all(report.rmses_m < 1e-9)


@pytest.mark.unit
def test_round_trip_validate_rejects_offset_trajectory() -> None:
    target = make_target(n=_N)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (_N, 1))

    def _bad(
        _c: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return target.butt + 0.5, target.clubhead + 0.5, quat

    report = round_trip_validate([np.zeros(7)], target, _bad, rmse_threshold_m=0.01)

    assert not bool(report.accepted.any())
    assert report.rmses_m[0] > 0.5


@pytest.mark.unit
def test_round_trip_validate_best_index_is_lowest_rmse() -> None:
    target = make_target(n=_N)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (_N, 1))
    counter = {"i": 0}
    offsets = [0.5, 0.001, 0.2]

    def _fn(
        _c: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        i = counter["i"]
        counter["i"] = i + 1
        return target.butt + offsets[i], target.clubhead + offsets[i], quat

    report = round_trip_validate(
        [np.zeros(7) for _ in range(3)],
        target,
        _fn,
        rmse_threshold_m=0.01,
    )

    assert report.best_index == 1


@pytest.mark.unit
def test_round_trip_validate_empty_samples_raises() -> None:
    target = make_target(n=_N)
    fn = _perfect_forward(target.butt, target.clubhead)
    with pytest.raises(ValueError, match="at least one"):
        round_trip_validate([], target, fn, rmse_threshold_m=0.01)


@pytest.mark.unit
def test_round_trip_validate_nonpositive_threshold_raises() -> None:
    target = make_target(n=_N)
    fn = _perfect_forward(target.butt, target.clubhead)
    with pytest.raises(ValueError, match="positive"):
        round_trip_validate([np.zeros(3)], target, fn, rmse_threshold_m=0.0)


@pytest.mark.unit
def test_round_trip_validate_bad_forward_shape_raises() -> None:
    target = make_target(n=_N)

    def _wrong_shape(
        _c: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return np.zeros((5, 3)), np.zeros((5, 3)), np.zeros((5, 4))

    with pytest.raises(ValueError, match="shape mismatch"):
        round_trip_validate([np.zeros(7)], target, _wrong_shape, rmse_threshold_m=0.01)
