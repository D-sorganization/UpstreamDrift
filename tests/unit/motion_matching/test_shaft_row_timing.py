"""Fixed row-time diagnostics preserve legacy geometry and reject unsupported roles."""

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.estimation import CubicHermiteSplineTrajectory
from src.shared.python.motion_matching.historical_fit import ImageFitResult
from src.shared.python.motion_matching.historical_fit.shaft_row_timing import (
    ShaftRowTiming,
    assess_row_timed_shaft,
)
from tests.unit.motion_matching.test_shaft_residuals import setup

pytestmark = pytest.mark.unit


def motion(native, speed=2.0):
    times = np.array([0.0, 1.0])
    q = np.array([[0.0, 0.0], [0.0, speed]])
    trajectory = CubicHermiteSplineTrajectory(times, 2)
    return ImageFitResult(
        source_times=times,
        q=q,
        rms_pixels=0,
        initial_rms_pixels=0,
        pixel_errors=np.zeros((2, 1, 2)),
        observed_point_count=2,
        model_sha=native.plant_sha,
        coordinate_order=native.coordinate_order,
        converged=False,
        optimizer_message="Synthetic Motion",
        knot_times=times,
        spline_coefficients=trajectory.initial_coefficients_from_samples(times, q),
        free_coordinates=native.coordinate_order,
    )


def timing(term, duration=0.1, direction=1):
    return ShaftRowTiming(
        term.source_clock_sha256, term.evidence.sha256, duration, direction
    )


def test_zero_readout_is_exact_legacy_assessment():
    native, camera, term, _ = setup()
    fitted = motion(native)
    legacy = term.assess(
        native, camera, fitted.evaluate_source_times(np.array(term.source_times))
    )
    result = assess_row_timed_shaft(native, camera, fitted, term, timing(term, 0))
    assert result.perpendicular_errors_pixels == legacy.perpendicular_errors_pixels
    assert result.raw_rms_pixels == legacy.raw_rms_pixels
    assert result.row_timing.physical_time_qualified is False


def test_static_motion_matches_for_nonzero_readout():
    native, camera, term, _ = setup()
    fitted = motion(native, 0)
    baseline = assess_row_timed_shaft(native, camera, fitted, term, timing(term, 0))
    shifted = assess_row_timed_shaft(native, camera, fitted, term, timing(term))
    assert shifted.perpendicular_errors_pixels == baseline.perpendicular_errors_pixels
    assert shifted.endpoint_source_times != baseline.endpoint_source_times


@pytest.mark.parametrize("direction", [-1, 1])
def test_rotating_axis_uses_each_observed_row_time(direction):
    native, camera, term, _ = setup()
    result = assess_row_timed_shaft(
        native, camera, motion(native), term, timing(term, direction=direction)
    )
    expected = []
    for x, y in term.evidence.frames[0].segment.points_px:
        time = term.source_times[0] + direction * 0.1 * (y / 79 - 0.5)
        expected.append(-x * np.sin(2 * time) + y * np.cos(2 * time))
    np.testing.assert_allclose(
        result.perpendicular_errors_pixels[0], expected, atol=1e-12
    )
    assert result.raw_rms_pixels == pytest.approx(np.sqrt(np.mean(np.square(expected))))


def test_complete_role_rejected_before_motion_evaluation(monkeypatch):
    native, camera, term, _ = setup()
    fitted = motion(native)
    calls = []
    monkeypatch.setattr(
        ImageFitResult, "evaluate_source_times", lambda *args: calls.append(args)
    )
    with pytest.raises(ValueError, match="support"):
        assess_row_timed_shaft(native, camera, fitted, term, timing(term, 10))
    assert calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("readout_source_seconds", True),
        ("readout_source_seconds", -1),
        ("readout_source_seconds", float("nan")),
        ("readout_source_seconds", float("inf")),
        ("scan_direction", True),
        ("scan_direction", 0),
        ("scan_direction", "1"),
        ("reference_row_fraction", 1.1),
        ("reference_row_fraction", False),
        ("row_mapping", "measured_sensor"),
        ("source_clock_sha256", "wrong"),
    ],
)
def test_invalid_timing_assumptions_rejected(field, value):
    _, _, term, _ = setup()
    with pytest.raises(ValueError):
        replace(timing(term), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_clock_sha256", "sha256:" + "f" * 64),
        ("evidence_sha256", "sha256:" + "f" * 64),
    ],
)
def test_foreign_clock_or_evidence_rejected(field, value):
    native, camera, term, _ = setup()
    with pytest.raises(ValueError, match="identity"):
        assess_row_timed_shaft(
            native,
            camera,
            motion(native),
            term,
            replace(timing(term), **{field: value}),
        )


def test_foreign_native_motion_rejected():
    native, camera, term, _ = setup()
    with pytest.raises(ValueError, match="model|order"):
        assess_row_timed_shaft(
            native,
            camera,
            replace(motion(native), model_sha="f" * 64),
            term,
            timing(term),
        )


def test_degenerate_projected_axis_rejected():
    native, camera, term, _ = setup()
    bad = replace(term, axis=replace(term.axis, point_b_m=(0, 0, 5)))
    with pytest.raises(ValueError, match="degenerate"):
        assess_row_timed_shaft(native, camera, motion(native), bad, timing(bad))


def test_abstention_remains_absent_without_fabricated_times():
    native, camera, term, _ = setup()
    frame = term.evidence.frames[0]
    segment = replace(
        frame.segment,
        status="ambiguous",
        points_px=None,
        confidence=None,
        visibility=None,
        sigma_px=None,
    )
    ev = replace(term.evidence, frames=(replace(frame, segment=segment),))
    abstained = replace(term, evidence=ev)
    result = assess_row_timed_shaft(
        native, camera, motion(native), abstained, timing(abstained)
    )
    assert result.raw_rms_pixels is None
    assert result.perpendicular_errors_pixels == (None,)
    assert result.endpoint_source_times == (None,)
