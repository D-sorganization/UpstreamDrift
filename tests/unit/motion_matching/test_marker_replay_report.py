"""Marker reports preserve observed samples and reject ambiguous trajectories."""

import numpy as np
import pytest

from src.shared.python.motion_matching.marker_replay_report import (
    marker_errors,
    plot_marker_replay,
)

pytestmark = pytest.mark.unit


def test_errors_use_euclidean_distance_and_keep_gaps() -> None:
    target = np.zeros((3, 2, 3))
    prediction = np.zeros_like(target)
    prediction[:, 0] = [0.003, 0.004, 0]
    valid = np.array([[True, True], [False, True], [False, False]])
    target[1, 0] = np.nan
    errors = marker_errors(target, prediction, valid)
    np.testing.assert_allclose(errors[0], [0.005, 0])
    assert errors[1, 1] == 0
    assert np.isnan(errors[1, 0])
    assert np.isnan(errors[2]).all()
    assert np.isnan(target[1, 0]).all()


@pytest.mark.parametrize("fault", ["shape", "mask", "observed_nan", "failed_replay"])
def test_report_rejects_invalid_replay(fault: str) -> None:
    target = np.zeros((3, 2, 3))
    prediction = np.zeros_like(target)
    valid = np.ones((3, 2), dtype=bool)
    if fault == "shape":
        prediction = prediction[:2]
    elif fault == "mask":
        valid = valid[:2]
    elif fault == "observed_nan":
        target[0, 0, 0] = np.nan
    else:
        prediction[0, 0, 0] = np.nan
        valid[0, 0] = False
    with pytest.raises(ValueError):
        marker_errors(target, prediction, valid)


def test_plot_exposes_observed_rms_in_mm_and_missing_frames() -> None:
    import matplotlib.pyplot as plt

    target = np.zeros((3, 2, 3))
    prediction = np.zeros_like(target)
    prediction[:, :, 0] = 0.01
    valid = np.array([[True, True], [False, False], [True, False]])
    figure = plot_marker_replay(
        np.array([0.0, 0.4, 0.8]),
        target,
        prediction,
        valid,
        ["A", "B"],
        candidate_sha256="a" * 64,
    )
    rms_line = figure.axes[0].lines[0]
    np.testing.assert_allclose(rms_line.get_ydata(), [10, np.nan, 10], equal_nan=True)
    assert "aaaaaaaaaaaa" in figure._suptitle.get_text()
    plt.close(figure)


def test_observed_rms_distinguishes_missing_from_zero() -> None:
    from src.shared.python.motion_matching.marker_replay_report import observed_rms

    assert observed_rms(np.array([np.nan]), np.array([False])) is None
    assert observed_rms(np.zeros(1), np.array([True])) == 0.0
    assert observed_rms(np.array([0.1, 0.3]), np.array([True, True])) == pytest.approx(
        np.sqrt(0.05)
    )
    with pytest.raises(ValueError):
        observed_rms(np.array([np.nan]), np.array([True]))
