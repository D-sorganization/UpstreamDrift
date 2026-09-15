"""Diagnostic contracts for the ST-01 qualification experiment (#10124)."""

import numpy as np
import pytest

from scripts.shadow_tracker.pilot_metrics import closure_metrics, replay_difference

pytestmark = pytest.mark.unit


def test_closure_keeps_metres_and_radians_separate() -> None:
    residuals = np.array([[0.003, 0.004, 0, 0, 0, 2.5]])
    result = closure_metrics(residuals)
    assert result == {
        "max_grip_translation_m": pytest.approx(0.005),
        "max_grip_rotation_rad": pytest.approx(2.5),
    }


@pytest.mark.parametrize(
    "residuals",
    [np.zeros((0, 6)), np.zeros((2, 3)), np.ones(6), np.full((2, 6), np.nan)],
)
def test_closure_rejects_missing_or_invalid_evidence(residuals: np.ndarray) -> None:
    with pytest.raises(ValueError):
        closure_metrics(residuals)


def test_replay_difference_separates_scalar_translation_and_rotation() -> None:
    reference = np.zeros((2, 5))
    candidate = np.array([[0, 0, 0, 0, 0], [0.01, 0, 0, 0.02, 0]])
    result = replay_difference(reference, candidate, (0, 1, 2))
    assert result == {
        "max_translation_difference": pytest.approx(0.01),
        "max_rotation_difference": pytest.approx(0.02),
    }
    np.testing.assert_array_equal(reference, np.zeros((2, 5)))


@pytest.mark.parametrize(
    "candidate, indices",
    [
        (np.zeros((3, 5)), (0, 1, 2)),
        (np.full((2, 5), np.inf), (0, 1, 2)),
        (np.zeros((2, 5)), (0, 0, 1)),
        (np.zeros((2, 5)), (0, 1, 9)),
        (np.zeros((2, 5)), (0, 1, -1)),
        (np.zeros((2, 5)), (0, 1, True)),
    ],
)
def test_replay_difference_rejects_incompatible_input(
    candidate: np.ndarray, indices: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        replay_difference(np.zeros((2, 5)), candidate, indices)


def test_replay_difference_rejects_empty_evidence() -> None:
    with pytest.raises(ValueError):
        replay_difference(np.zeros((0, 5)), np.zeros((0, 5)), (0, 1, 2))
