"""Preserved Hermite starts retain physical derivatives and parent identities."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.estimation import CubicHermiteSplineTrajectory
from src.shared.python.motion_matching.historical_fit import (
    ImageSplineStart,
    ImageFitConfig,
    fit_image_trajectory,
    initialize_image_trajectory,
)
from tests.unit.motion_matching.test_historical_image_fit import native_problem

pytestmark = pytest.mark.unit


def _start(native, inputs):
    return ImageSplineStart.from_coefficients(
        inputs.knot_times,
        np.array([-0.3, -0.8, 1.0, -2.0]),
        tuple(native.coordinate_order),
        inputs.free_coordinates,
        native.plant_sha,
    )


def test_exact_start_skips_resampling_and_authored_initialization(monkeypatch):
    from src.shared.python.motion_matching.historical_fit import solver

    native, attachments, camera, inputs = native_problem()
    start = _start(native, inputs)
    captured = []
    monkeypatch.setattr(
        CubicHermiteSplineTrajectory,
        "initial_coefficients_from_samples",
        lambda *args: pytest.fail("resampled"),
    )
    monkeypatch.setattr(
        solver, "initialize_authored_hermite", lambda *args: pytest.fail("reauthored")
    )

    def solve(problem):
        captured.append(problem)
        return SimpleNamespace(
            coefficients=problem.initial_coefficients,
            success=False,
            message="wiring only",
        )

    monkeypatch.setattr(solver, "solve_single_trial_map", solve)
    config = ImageFitConfig(closure_weight=0, max_iterations=1)
    result = fit_image_trajectory(native, attachments, camera, inputs, config, start)
    np.testing.assert_array_equal(
        captured[0].initial_coefficients, start.spline_coefficients
    )
    assert result.initialization is None and result.observed_point_count == 6
    assert result.initial_rms_pixels < 1e-8
    trajectory = CubicHermiteSplineTrajectory(np.asarray(start.knot_times), 1)
    times = np.array([110.0, 110.25, 110.7, 111.0])
    before = trajectory.evaluate(np.asarray(start.spline_coefficients), times)
    after = trajectory.evaluate(result.spline_coefficients, times)
    for attribute in ("q", "v", "a"):
        np.testing.assert_array_equal(
            getattr(before, attribute), getattr(after, attribute)
        )
    evaluated = initialize_image_trajectory(
        native, attachments, camera, inputs, config, start
    )
    assert evaluated.optimizer_ran is False and evaluated.initialization is None
    np.testing.assert_array_equal(
        evaluated.spline_coefficients, start.spline_coefficients
    )
    assert len(captured) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"model_sha": "wrong-model"},
        {"coordinate_order": ("REInput",)},
        {"free_coordinates": ("LEInput",)},
        {"knot_times": (110.0, 110.5)},
    ],
)
def test_incompatible_exact_start_is_rejected(change):
    native, attachments, camera, inputs = native_problem()
    start = replace(_start(native, inputs), **change)
    with pytest.raises(ValueError, match="model|coordinate|knot|interval"):
        fit_image_trajectory(
            native, attachments, camera, inputs, ImageFitConfig(), start
        )


def test_exact_start_requires_strict_policy_and_matching_knot_grid():
    native, attachments, camera, inputs = native_problem()
    start = _start(native, inputs)
    with pytest.raises(ValueError, match="strict"):
        fit_image_trajectory(
            native,
            attachments,
            camera,
            inputs,
            ImageFitConfig(
                coordinate_bounds=(("REInput", -1.0, 0.0),),
                initialization_policy="authored_range_project_zero_slopes",
            ),
            start,
        )
    different = replace(inputs, knot_times=np.array([110.0, 110.5, 111.0]))
    with pytest.raises(ValueError, match="knot"):
        fit_image_trajectory(
            native, attachments, camera, different, ImageFitConfig(), start
        )


def test_start_copies_input_arrays_and_rejects_changed_coefficients():
    knots = np.array([1.0, 2.0])
    coefficients = np.array([0.0, 1.0, 0.4, -0.2])
    start = ImageSplineStart.from_coefficients(
        knots, coefficients, ("joint",), ("joint",), "model"
    )
    knots[:] = 9.0
    coefficients[:] = 9.0
    assert start.knot_times == (1.0, 2.0) and start.spline_coefficients == (
        0.0,
        1.0,
        0.4,
        -0.2,
    )
    with pytest.raises(ValueError, match="hash"):
        replace(start, spline_coefficients=(0.0, 1.0, 0.0, 0.0))


def test_nonuniform_exact_start_retains_derivatives_without_optimizer(monkeypatch):
    native, attachments, camera, inputs = native_problem()
    inputs = replace(inputs, knot_times=np.array([110.0, 110.2, 111.0]))
    coefficients = np.array([-0.3, -0.6, -0.8, 1.0, 0.2, -2.0])
    start = ImageSplineStart.from_coefficients(
        inputs.knot_times,
        coefficients,
        tuple(native.coordinate_order),
        inputs.free_coordinates,
        native.plant_sha,
    )
    monkeypatch.setattr(
        CubicHermiteSplineTrajectory,
        "initial_coefficients_from_samples",
        lambda *args: pytest.fail("resampled"),
    )
    result = initialize_image_trajectory(
        native, attachments, camera, inputs, ImageFitConfig(closure_weight=0), start
    )
    assert result.initial_spline == start and result.initialization is None
    trajectory = CubicHermiteSplineTrajectory(inputs.knot_times, 1)
    times = np.array([110.0, 110.1, 110.2, 110.8, 111.0])
    original = trajectory.evaluate(coefficients, times)
    restarted = trajectory.evaluate(result.spline_coefficients, times)
    for name in ("q", "v", "a"):
        np.testing.assert_array_equal(getattr(original, name), getattr(restarted, name))


def test_snapshot_json_roundtrip_rejects_malformed_declared_records():
    start = ImageSplineStart.from_coefficients(
        (1.0, 2.0), (0.0, 1.0, 0.4, -0.2), ("joint",), ("joint",), "model"
    )
    record = start.to_record()
    assert ImageSplineStart.from_record(record) == start
    record["spline_coefficients"][0] = 9.0
    with pytest.raises(ValueError, match="hash"):
        ImageSplineStart.from_record(record)
    with pytest.raises(ValueError, match="six"):
        ImageSplineStart.from_record({"knot_times": [1.0, 2.0]})
    with pytest.raises(ValueError, match="six"):
        ImageSplineStart.from_record({**start.to_record(), "extra": True})


@pytest.mark.parametrize(
    "changes",
    [
        {"knot_times": (1.0, 1.0)},
        {"spline_coefficients": (0.0,)},
        {"free_coordinates": ("joint", "joint")},
        {"coordinate_order": ("",)},
        {"model_sha": ""},
        {"coefficient_sha256": "wrong"},
    ],
)
def test_invalid_start_contracts_fail(changes):
    start = ImageSplineStart.from_coefficients(
        (1.0, 2.0), (0.0, 1.0, 0.0, 0.0), ("joint",), ("joint",), "model"
    )
    with pytest.raises(ValueError):
        replace(start, **changes)
