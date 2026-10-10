"""Independent marker scoring must not learn from withheld observations."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.motion_matching import marker_calibration as calibration
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit


def _inputs() -> tuple[
    TourCapture, calibration.Offsets, list[dict[str, calibration.Pose]]
]:
    capture = TourCapture(
        np.array([0.0, 0.01]),
        ("withheld",),
        np.array([[[0.3, 0.0, 0.0]], [[0.3, 0.0, 0.0]]]),
        np.ones((2, 1), dtype=bool),
    )
    offsets = {"withheld": ("body", (0.1, 0.0, 0.0))}
    poses = [{"body": (np.eye(3), np.zeros(3))} for _ in range(2)]
    return capture, offsets, poses


def test_frozen_offsets_preserve_error_erased_by_exploratory_refit() -> None:
    capture, offsets, poses = _inputs()
    refitted = calibration.static_marker_offsets(capture, {"withheld": "body"}, poses)
    assert calibration._marker_rms(capture, refitted, poses) == pytest.approx(0.0)
    before = deepcopy(offsets)
    rms, per_marker = calibration.score_frozen_marker_offsets(capture, offsets, poses)
    assert rms == pytest.approx(0.2)
    assert per_marker == pytest.approx({"withheld": 0.2})
    assert offsets == before


def test_scoring_has_no_calibration_or_ik_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Withheld observations reached a calibration operation")

    monkeypatch.setattr(calibration, "static_marker_offsets", forbidden)
    monkeypatch.setattr(calibration, "calibrate_marker_offsets", forbidden)
    capture, offsets, poses = _inputs()
    rms, _ = calibration.score_frozen_marker_offsets(capture, offsets, poses)
    assert rms == pytest.approx(0.2)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-offset",
        "nan-offset",
        "missing-pose",
        "nan-pose",
        "reflection",
        "scaled-rotation",
        "short-poses",
        "empty-support",
    ],
)
def test_independent_score_rejects_unqualified_inputs(fault: str) -> None:
    capture, offsets, poses = _inputs()
    if fault == "missing-offset":
        offsets.clear()
    elif fault == "nan-offset":
        offsets["withheld"] = ("body", (float("nan"), 0.0, 0.0))
    elif fault == "missing-pose":
        poses[1].clear()
    elif fault == "nan-pose":
        poses[1]["body"][1][0] = float("nan")
    elif fault == "reflection":
        poses[1]["body"][0][0, 0] = -1.0
    elif fault == "scaled-rotation":
        poses[1]["body"][0][0, 0] = 2.0
    elif fault == "short-poses":
        poses.pop()
    else:
        capture = replace(capture, valid=np.zeros_like(capture.valid))
    with pytest.raises(ValueError):
        calibration.score_frozen_marker_offsets(capture, offsets, poses)


def test_masked_samples_are_excluded_and_observation_clock_is_unchanged() -> None:
    capture, offsets, poses = _inputs()
    valid = capture.valid.copy()
    valid[1, 0] = False
    points = capture.points_m.copy()
    points[1, 0] = 100.0
    capture = replace(capture, valid=valid, points_m=points)
    before = capture.time_s.copy()
    rms, _ = calibration.score_frozen_marker_offsets(capture, offsets, poses)
    assert rms == pytest.approx(0.2)
    np.testing.assert_array_equal(capture.time_s, before)


def test_score_uses_body_rotation_translation_and_pooled_sample_weighting() -> None:
    capture = TourCapture(
        np.array([0.0, 0.01]),
        ("a", "b"),
        np.array(
            [
                [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 3.0, 0.0], [np.nan, np.nan, np.nan]],
            ]
        ),
        np.array([[True, True], [True, False]]),
    )
    offsets = {"a": ("body", (1.0, 0.0, 0.0)), "b": ("body", (0.0, 0.0, 0.0))}
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    poses = [{"body": (rotation, np.array([1.0, 0.0, 0.0]))}] * 2
    rms, per_marker = calibration.score_frozen_marker_offsets(capture, offsets, poses)
    assert rms == pytest.approx(np.sqrt(4.0 / 3.0))
    assert per_marker == pytest.approx({"a": np.sqrt(2.0), "b": 0.0})
