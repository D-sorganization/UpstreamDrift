"""Authored spline probes add constraints without fabricating image evidence."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    finite_difference_jacobian,
)
from src.shared.python.motion_matching.historical_fit import (
    CameraProjection,
    ImageFitConfig,
    ImageFitInputs,
    fit_image_trajectory,
)
from src.shared.python.motion_matching.historical_fit.solver import _Fit

pytestmark = pytest.mark.unit


def options():
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane

    return ConstraintOptions(GroundPlane((0, 0, 1), 0), 1, 1, 1, 1, 1, 1)


class FakePlant:
    coordinate_order = ("joint",)
    plant_sha = "fake-unit-fixture"

    def __init__(self):
        self.ik_calls = 0

    def marker_positions(self, q, attachments):
        return np.array([[q[0], 0, 1]])

    def closure_residuals(self, q):
        return np.array([q[0]])

    def create_ik(self, attachments):
        self.ik_calls += 1
        return self

    def constraint_residual_jacobian(self, q, config):
        from src.shared.python.motion_matching.constraint_kinematics import (
            ConstraintLinearization,
        )

        value = q[0]
        return ConstraintLinearization(
            np.array([value**2, np.sin(value), 0, 0, 0, 0, min(value, 0), 0]),
            np.array(
                [
                    [2 * value],
                    [np.cos(value)],
                    [0],
                    [0],
                    [0],
                    [0],
                    [float(value < 0)],
                    [0],
                ]
            ),
            tuple(self.coordinate_order),
            (
                "grip_position:x",
                "grip_position:y",
                "grip_position:z",
                "grip_rotation:x",
                "grip_rotation:y",
                "grip_rotation:z",
                "ground:left",
                "ground:right",
            ),
        )


def problem(config):
    native = FakePlant()
    camera = CameraProjection(np.eye(3), np.eye(3), np.zeros(3))
    inputs = ImageFitInputs(
        np.array([0.0, 1.0]),
        np.zeros((2, 1, 2)),
        np.ones((2, 1)),
        np.zeros(1),
        np.ones(1),
        ("joint",),
    )
    attachments = {"marker": ("body", (0, 0, 0))}
    return (
        native,
        attachments,
        camera,
        inputs,
        _Fit(native, attachments, camera, inputs, config),
    )


def test_interior_probes_expose_closure_without_adding_observations():
    config = ImageFitConfig(constraint_options=options(), interior_fractions=(0.5,))
    native, _, _, inputs, fit = problem(config)
    trajectory = CubicHermiteSplineTrajectory(inputs.knot_times, 1)
    coefficients = trajectory.pack(np.zeros((2, 1)), np.array([[4.0], [-4.0]]))
    evaluation = trajectory.evaluate(coefficients, fit.evaluation_times)
    residual = fit.residual(evaluation, {})
    np.testing.assert_array_equal(fit.evaluation_times, [0, 0.5, 1])
    assert native.ik_calls == 1
    # Source image/prior/speed rows: 4 + 2 + 2. Three fixed eight-row constraints.
    assert residual.shape == (32,)
    assert residual[8 + 8] == pytest.approx(1.0)
    assert inputs.observed_pixels.shape == (2, 1, 2)


def test_constraint_coefficient_derivatives_follow_exact_spline_chain():
    config = ImageFitConfig(
        prior_weight=0.2,
        smoothness_weight=0.3,
        constraint_options=options(),
        interior_fractions=(0.25, 0.5, 0.75),
    )
    _, _, _, inputs, fit = problem(config)
    trajectory = CubicHermiteSplineTrajectory(inputs.knot_times, 1)
    coefficients = trajectory.pack(np.array([[-0.4], [0.6]]), np.array([[0.2], [-0.1]]))
    evaluation = trajectory.evaluate(coefficients, fit.evaluation_times)
    chained = fit.jacobian(evaluation, {}, None)
    direct = finite_difference_jacobian(
        lambda value: fit.residual(
            trajectory.evaluate(value, fit.evaluation_times), {}
        ),
        coefficients,
    )
    np.testing.assert_allclose(chained, direct, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "fractions",
    [(0,), (1,), (-0.1,), (float("nan"),), (0.5, 0.5), (0.75, 0.25), (True,)],
)
def test_invalid_probe_fractions_fail_before_solver(fractions):
    with pytest.raises(ValueError, match="fraction"):
        ImageFitConfig(constraint_options=options(), interior_fractions=fractions)


def test_fractions_require_explicit_options_and_json_roundtrip_is_narrow():
    with pytest.raises(ValueError, match="constraint"):
        ImageFitConfig(interior_fractions=(0.5,))
    config = ImageFitConfig(constraint_options=options(), interior_fractions=(0.5,))
    restored = ImageFitConfig.from_record(json.loads(json.dumps(asdict(config))))
    assert restored == config
    assert (
        ImageFitConfig.from_record(json.loads(json.dumps(asdict(ImageFitConfig()))))
        == ImageFitConfig()
    )
    with pytest.raises(ValueError, match="constraint"):
        ImageFitConfig(constraint_options={})
    with pytest.raises(ValueError):
        ImageFitConfig.from_record({"constraint_options": {"ground": {}}})


def test_result_counts_only_source_observations_and_retains_probe_report():
    config = ImageFitConfig(
        max_iterations=1, constraint_options=options(), interior_fractions=(0.5,)
    )
    native, attachments, camera, inputs, _ = problem(config)
    result = fit_image_trajectory(native, attachments, camera, inputs, config)
    assert result.observed_point_count == 2
    assert result.q.shape == (2, 1)
    assert result.pixel_errors.shape == (2, 1)
    np.testing.assert_array_equal(result.constraint_times, [0, 0.5, 1])
    assert result.constraint_residuals.shape == (3, 8)
    assert len(result.constraint_row_labels) == 8
    assert not result.constraint_residuals.flags.writeable
    assert result.physical_time_qualified is False


def test_default_fit_never_requests_optional_constraint_capability():
    native, attachments, camera, inputs, fit = problem(ImageFitConfig())
    result = fit_image_trajectory(native, attachments, camera, inputs)
    assert native.ik_calls == 0
    np.testing.assert_array_equal(fit.evaluation_times, inputs.source_times)
    assert result.constraint_times.size == 0


def test_unsupported_constraint_provider_fails_explicitly():
    native, attachments, camera, inputs, _ = problem(ImageFitConfig())
    native.constraint_residual_jacobian = lambda *args: (_ for _ in ()).throw(
        NotImplementedError("unsupported")
    )
    with pytest.raises(NotImplementedError, match="unsupported"):
        fit_image_trajectory(
            native,
            attachments,
            camera,
            inputs,
            ImageFitConfig(constraint_options=options()),
        )


@pytest.mark.parametrize(
    "times",
    [
        np.array([[0.5]]),
        np.array([0.7, 0.3]),
        np.array([0.5, 0.5]),
        np.array([-0.1]),
        np.array([1.1]),
    ],
)
def test_constraint_report_rejects_invalid_tested_times(times):
    native, attachments, camera, inputs, _ = problem(ImageFitConfig())
    result = fit_image_trajectory(native, attachments, camera, inputs)
    with pytest.raises(ValueError, match="Constraint report"):
        replace(
            result,
            constraint_times=times,
            constraint_residuals=np.zeros((len(times), 1)),
            constraint_row_labels=("grip_position:x",),
        )


@pytest.mark.parametrize("labels", [["x"], ("x", "x"), ("",), (" ",), (1,)])
def test_constraint_report_rejects_mutable_or_ambiguous_labels(labels):
    native, attachments, camera, inputs, _ = problem(ImageFitConfig())
    result = fit_image_trajectory(native, attachments, camera, inputs)
    with pytest.raises(ValueError, match="Constraint report"):
        replace(
            result,
            constraint_times=np.array([0.5]),
            constraint_residuals=np.zeros((1, len(labels))),
            constraint_row_labels=labels,
        )


def test_probes_use_knot_intervals_and_deduplicate_source_overlap():
    native, attachments, camera, inputs, _ = problem(ImageFitConfig())
    inputs = replace(
        inputs,
        source_times=np.array([0, 0.25, 1]),
        observed_pixels=np.zeros((3, 1, 2)),
        confidence=np.ones((3, 1)),
        knot_times=np.array([0, 0.5, 1]),
    )
    config = ImageFitConfig(constraint_options=options(), interior_fractions=(0.5,))
    fit = _Fit(native, attachments, camera, inputs, config)
    np.testing.assert_array_equal(fit.evaluation_times, [0, 0.25, 0.75, 1])
    trajectory = CubicHermiteSplineTrajectory(inputs.knot_times, 1)
    coefficients = trajectory.pack(np.zeros((3, 1)), np.zeros((3, 1)))
    evaluation = trajectory.evaluate(coefficients, fit.evaluation_times)
    assert fit.residual(evaluation, {}).shape == (6 + 3 + 3 + 4 * 8,)
    np.testing.assert_array_equal(
        fit.source_evaluation(evaluation).times, inputs.source_times
    )


def test_malformed_constraint_provider_record_fails_explicitly():
    native, attachments, camera, inputs, _ = problem(ImageFitConfig())
    native.constraint_residual_jacobian = lambda *args: object()
    fit = _Fit(
        native,
        attachments,
        camera,
        inputs,
        ImageFitConfig(constraint_options=options()),
    )
    with pytest.raises(ValueError, match="Constraint linearization"):
        fit.constraint_linearizations(np.zeros((2, 1)))


@pytest.mark.parametrize("changed", ["coordinates", "labels"])
def test_constraint_provider_must_retain_native_order_and_stable_rows(changed):
    native, attachments, camera, inputs, _ = problem(ImageFitConfig())
    original = native.constraint_residual_jacobian

    def inconsistent(q, config):
        row = original(q, config)
        if changed == "coordinates":
            return replace(row, coordinate_order=("other",))
        if q[0] > 0:
            return replace(row, row_labels=tuple(reversed(row.row_labels)))
        return row

    native.constraint_residual_jacobian = inconsistent
    fit = _Fit(
        native,
        attachments,
        camera,
        inputs,
        ImageFitConfig(constraint_options=options()),
    )
    with pytest.raises(ValueError, match="Constraint linearization"):
        fit.constraint_linearizations(np.array([[0.0], [1.0]]))


def test_ordered_list_native_coordinates_are_supported():
    native, attachments, camera, inputs, _ = problem(ImageFitConfig())
    native.coordinate_order = ["joint"]
    fit = _Fit(
        native,
        attachments,
        camera,
        inputs,
        ImageFitConfig(constraint_options=options()),
    )
    rows = fit.constraint_linearizations(np.zeros((2, 1)))
    assert rows[0].coordinate_order == ("joint",)


def test_missing_source_evidence_stays_missing_with_interior_probes():
    native, attachments, camera, inputs, _ = problem(ImageFitConfig())
    inputs = replace(
        inputs,
        source_times=np.array([0, 0.4, 1]),
        knot_times=None,
        observed_pixels=np.zeros((3, 1, 2)),
        confidence=np.array([[1.0], [0.0], [1.0]]),
    )
    result = fit_image_trajectory(
        native,
        attachments,
        camera,
        inputs,
        ImageFitConfig(
            max_iterations=1, constraint_options=options(), interior_fractions=(0.5,)
        ),
    )
    assert result.observed_point_count == 2
    assert result.pixel_errors[1, 0] is None
    np.testing.assert_array_equal(result.source_times, [0, 0.4, 1])
    np.testing.assert_allclose(result.constraint_times, [0, 0.2, 0.4, 0.7, 1])
    assert result.maximum_constraint_residual == pytest.approx(0)
