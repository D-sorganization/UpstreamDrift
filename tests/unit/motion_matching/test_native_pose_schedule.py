"""Pose continuation must use observed clocks without shifting endpoints."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def schedule(*args: object) -> np.ndarray:
    path = Path(__file__).resolve().parents[3] / (
        "docs/development/simscape_tour_matching/native_evidence/"
        "reproduction/check_native_pose_feasibility.py"
    )
    spec = importlib.util.spec_from_file_location("pose_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pose_schedule(*args)


def test_backward_schedule_preserves_actual_capture_indices() -> None:
    np.testing.assert_array_equal(
        schedule([0, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8], 0.8, "backward"),
        [3, 2, 1],
    )


@pytest.mark.parametrize(
    "clock,requested,duration",
    [
        ([0, 0.6, 0.8], [0.6, 0.7, 0.8], 0.8),
        ([0, 0.6, 0.6, 0.8], [0.6, 0.8], 0.8),
        ([0, 0.6, 0.8], [0.8, 0.6], 0.8),
        ([0, 0.6, 0.8], [0.6], 0.8),
        ([0, 0.6, 0.8], [float("nan"), 0.8], 0.8),
    ],
)
def test_invalid_schedule_rejected(
    clock: list, requested: list, duration: float
) -> None:
    with pytest.raises(ValueError):
        schedule(clock, requested, duration, "forward")
