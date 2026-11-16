"""Resampled timestamps must sit on the requested ``target_fps`` grid."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from src.shared.python.motion_pipeline.preprocessing import (
    _resample_pure_python as pure,
)

from ._local_fixtures import make_keypoint_sequence, make_marker_trajectory

pytestmark = pytest.mark.unit

dispatch = importlib.import_module(
    "src.shared.python.motion_pipeline.preprocessing.resample"
)

_IMPLS = [
    pytest.param(dispatch.resample, id="dispatch"),
    pytest.param(pure.resample, id="pure_python"),
]


def _times(out) -> np.ndarray:
    return np.array([f.timestamp for f in out.frames])


@pytest.mark.parametrize("fn", _IMPLS)
def test_keypoint_grid_step_is_one_over_target_fps(fn) -> None:
    # 30 frames at 30 Hz span 29/30 s; at 100 Hz that is 96.67 sample
    # intervals, so the grid holds 97 samples 10 ms apart (0 .. 0.96 s).
    out = fn(make_keypoint_sequence(num_frames=30, num_kp=1, fps=30.0), 100.0)
    t = _times(out)
    assert len(t) == 97
    np.testing.assert_allclose(np.diff(t), 0.01, atol=1e-12)
    assert t[-1] <= 29 / 30 + 1e-12


@pytest.mark.parametrize("fn", _IMPLS)
def test_marker_grid_includes_exact_multiple_despite_float_error(fn) -> None:
    # 0.29 s * 100 Hz evaluates to 28.999999999999996; the 30th sample at
    # exactly 0.29 s must not be dropped.
    traj = make_marker_trajectory(num_frames=30, fps=100.0)
    out = fn(traj, 100.0, source_fps=100.0)
    t = _times(out)
    assert len(t) == 30
    np.testing.assert_allclose(t, np.arange(30) / 100.0, atol=1e-12)


@pytest.mark.parametrize("fn", _IMPLS)
@pytest.mark.parametrize("bad", [0.0, -30.0, float("nan"), float("inf")])
def test_target_fps_must_be_positive_and_finite(fn, bad) -> None:
    seq = make_keypoint_sequence(num_frames=5, num_kp=1, fps=30.0)
    with pytest.raises(ValueError, match="target_fps"):
        fn(seq, bad)
    traj = make_marker_trajectory(num_frames=5, fps=30.0)
    with pytest.raises(ValueError, match="target_fps"):
        fn(traj, bad)
