"""Stored spline extrema assess authored bounds without creating observations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import numpy as np
import pytest

from src.shared.python.estimation import CubicHermiteSplineTrajectory
from src.shared.python.motion_matching.historical_fit import ImageFitResult

pytestmark = pytest.mark.unit


def _fit(knots, positions, velocities, *, locked=0.3):
    times = np.asarray(knots, dtype=float)
    trajectory = CubicHermiteSplineTrajectory(times, 1)
    coefficients = trajectory.pack(
        np.asarray(positions)[:, None], np.asarray(velocities)[:, None]
    )
    q = np.column_stack([positions, np.full(len(times), locked)])
    return ImageFitResult(
        times,
        q,
        2.5,
        3.5,
        np.zeros((len(times), 1)),
        len(times),
        "model-sha",
        ("joint", "locked"),
        False,
        "research fixture",
        times,
        coefficients,
        ("joint",),
    )


def test_overshoot_is_detected_even_when_every_knot_is_bounded():
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    fit = _fit([10, 12], [0, 0], [2, -2])
    original = fit.q.copy(), fit.spline_coefficients.copy(), fit.pixel_errors.copy()
    result = assess_spline_bounds(fit, {"joint": (-0.5, 0.5)})
    coordinate = result.coordinates[0]
    assert coordinate.minimum == pytest.approx(0)
    assert coordinate.minimum_source_time == pytest.approx(10)
    assert coordinate.maximum == pytest.approx(1)
    assert coordinate.maximum_source_time == pytest.approx(11)
    assert coordinate.upper_violation == pytest.approx(0.5)
    assert coordinate.lower_violation == 0
    assert coordinate.within_authored_bounds is False
    assert result.bounded_coordinates == ("joint",)
    assert result.unbounded_coordinates == ("locked",)
    assert result.violating_coordinates == ("joint",)
    assert result.continuous_certified is False
    assert result.grip_assessment == result.ground_assessment == "not_assessed"
    assert fit.rms_pixels == 2.5 and fit.observed_point_count == 2
    for before, after in zip(
        original, (fit.q, fit.spline_coefficients, fit.pixel_errors), strict=True
    ):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize(
    "positions,velocities,minimum,maximum,min_time,max_time",
    [
        ([0, 1], [1, 1], 0, 1, 0, 1),
        ([0, 1], [0, 2], 0, 1, 0, 1),
        ([0.25, 0.25], [0, 0], 0.25, 0.25, 0, 0),
        (
            [0, 0],
            [1, 1],
            -np.sqrt(3) / 18,
            np.sqrt(3) / 18,
            (3 + np.sqrt(3)) / 6,
            (3 - np.sqrt(3)) / 6,
        ),
    ],
)
def test_linear_quadratic_constant_and_cubic_segments(
    positions, velocities, minimum, maximum, min_time, max_time
):
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    result = assess_spline_bounds(_fit([0, 1], positions, velocities), {})
    coordinate = result.coordinates[0]
    assert coordinate.minimum == pytest.approx(minimum)
    assert coordinate.maximum == pytest.approx(maximum)
    assert coordinate.minimum_source_time == pytest.approx(min_time)
    assert coordinate.maximum_source_time == pytest.approx(max_time)
    assert coordinate.within_authored_bounds is None


def test_multisegment_nonuniform_times_and_locked_bounds():
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    result = assess_spline_bounds(
        _fit([5, 6, 9], [0, 0, 0], [0, 0, 4], locked=-0.2),
        {"joint": (-1, 1), "locked": (0, 1)},
    )
    moving, locked = result.coordinates
    assert moving.minimum == pytest.approx(-16 / 9)
    assert moving.minimum_source_time == pytest.approx(8)
    assert moving.lower_violation == pytest.approx(7 / 9)
    assert locked.minimum == locked.maximum == -0.2
    assert locked.minimum_source_time == locked.maximum_source_time == 5
    assert result.violating_coordinates == ("joint", "locked")
    with pytest.raises(FrozenInstanceError):
        moving.minimum = 123
    with pytest.raises(FrozenInstanceError):
        result.coordinates = ()


@pytest.mark.parametrize(
    "bounds",
    [
        {"unknown": (0, 1)},
        {"joint": (1, 0)},
        {"joint": (0, float("inf"))},
        {"joint": (0,)},
        {"joint": (False, 1)},
    ],
)
def test_invalid_authored_bounds_fail(bounds):
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    with pytest.raises(ValueError):
        assess_spline_bounds(_fit([0, 1], [0, 0], [0, 0]), bounds)


@pytest.mark.parametrize(
    "change",
    [
        {"coordinate_order": ("joint", "joint")},
        {"free_coordinates": ("unknown",)},
        {"q": np.zeros((2, 1))},
        {"knot_times": np.array([1.0, 0.0])},
        {"spline_coefficients": np.zeros(3)},
    ],
)
def test_invalid_fit_shape_or_coordinate_identity_fails(change):
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    with pytest.raises(ValueError):
        assess_spline_bounds(replace(_fit([0, 1], [0, 0], [0, 0]), **change), {})


def test_empty_free_coordinates_are_rejected_explicitly():
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    fit = replace(
        _fit([2, 5], [0.2, 0.2], [0, 0]),
        free_coordinates=(),
        spline_coefficients=np.empty(0),
    )
    with pytest.raises(ValueError, match="free coordinate"):
        assess_spline_bounds(fit, {"joint": (-0.5, 0.5)})


def test_nonconstant_locked_samples_are_rejected():
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    fit = _fit([0, 1], [0, 0], [0, 0])
    changed = fit.q.copy()
    changed[1, 1] = 0.8
    with pytest.raises(ValueError, match="Locked coordinates"):
        assess_spline_bounds(replace(fit, q=changed), {})


def test_public_dtos_validate_names_and_detach_mutable_bound_inputs():
    from src.shared.python.motion_matching.historical_fit.assessment import (
        CoordinateExtrema,
    )

    bounds = [-1.0, 1.0]
    coordinate = CoordinateExtrema("joint", 0.0, 0.0, 0.5, 1.0, bounds)
    bounds[1] = 123
    assert coordinate.bounds == (-1.0, 1.0)
    with pytest.raises(ValueError, match="name"):
        CoordinateExtrema(123, 0.0, 0.0, 0.5, 1.0)


def test_source_samples_must_match_the_preserved_spline():
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    fit = _fit([0, 1], [0, 1], [1, 1])
    inconsistent = fit.q.copy()
    inconsistent[1, 0] = 0.8
    with pytest.raises(ValueError, match="source samples.*spline"):
        assess_spline_bounds(replace(fit, q=inconsistent), {})
    close = fit.q.copy()
    close[1, 0] += 1e-10
    result = assess_spline_bounds(replace(fit, q=close), {})
    assert result.source_sample_atol == 1e-10
    assert result.source_sample_rtol == 1e-8


def test_equal_authored_limits_are_valid_for_coordinate_bounds():
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    result = assess_spline_bounds(
        _fit([0, 1], [0.25, 0.25], [0, 0]),
        {"joint": (0.25, 0.25), "locked": (0.2, 0.2)},
    )
    assert result.coordinates[0].within_authored_bounds is True
    assert result.coordinates[1].upper_violation == pytest.approx(0.1)


@pytest.mark.parametrize(
    "bounds",
    [
        {123: (0, 1)},
        {None: (0, 1)},
        {("joint",): (0, 1)},
        {"joint": (-float("inf"), 1)},
    ],
)
def test_nonstring_names_and_one_sided_bounds_are_rejected(bounds):
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    with pytest.raises(ValueError):
        assess_spline_bounds(_fit([0, 1], [0, 0], [0, 0]), bounds)


def test_string_equal_objects_do_not_impersonate_coordinate_names():
    from collections import UserString
    from src.shared.python.motion_matching.historical_fit.assessment import (
        assess_spline_bounds,
    )

    with pytest.raises(ValueError, match="coordinate"):
        assess_spline_bounds(
            _fit([0, 1], [0, 0], [0, 0]), {UserString("joint"): (0, 1)}
        )
