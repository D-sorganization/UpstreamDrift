"""Contracts for sampled holonomic trajectory validation and collocation."""

import numpy as np
import pytest

from src.shared.python.motion_matching.constrained_trajectory import (
    collocate_positions,
    evaluate_trajectory_closure,
)

pytestmark = pytest.mark.unit


def _position(value: np.ndarray) -> np.ndarray:
    return np.array([value[0] + value[1]])


def _rate(_: np.ndarray, rate: np.ndarray) -> np.ndarray:
    return np.array([rate[0] + rate[1]])


def _acceleration(
    _: np.ndarray, __: np.ndarray, acceleration: np.ndarray
) -> np.ndarray:
    return np.array([acceleration[0] + acceleration[1]])


def test_evaluate_trajectory_closure_accepts_a_closed_smooth_path() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    positions = np.column_stack((times**2, -(times**2)))

    report = evaluate_trajectory_closure(
        times,
        positions,
        _position,
        _rate,
        _acceleration,
    )

    assert report.position_max_abs == pytest.approx(0.0)
    assert report.rate_max_abs == pytest.approx(0.0)
    assert report.acceleration_max_abs == pytest.approx(0.0)
    assert report.rates.shape == positions.shape
    assert report.accelerations.shape == positions.shape


def test_collocate_positions_corrects_noisy_closed_path() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    seed = np.column_stack((times**2, -(times**2), times, -times))
    seed[2, 1] += 0.1

    result = collocate_positions(
        times,
        seed,
        _position,
        _rate,
        _acceleration,
        max_iterations=100,
        closure_tolerance=1e-8,
    )

    assert result.closure.position_max_abs <= 1e-8
    assert result.closure.rate_max_abs <= 1e-8
    assert result.closure.acceleration_max_abs <= 1e-8
    assert result.coordinates.shape == seed.shape
    assert result.optimizer_converged


@pytest.mark.parametrize(
    "times,coordinates",
    [
        (np.array([0.0, 1.0, 1.0, 2.0]), np.zeros((4, 2))),
        (np.array([0.0, 1.0, 2.0]), np.zeros((3, 2))),
        (np.array([0.0, 1.0, 2.0, 3.0]), np.full((4, 2), np.nan)),
    ],
)
def test_invalid_trajectory_inputs_are_rejected(
    times: np.ndarray, coordinates: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        evaluate_trajectory_closure(times, coordinates, _position, _rate, _acceleration)
