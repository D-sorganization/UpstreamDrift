"""Sparse shaft terms preserve body evidence and use independent spline bases."""

from dataclasses import replace
import numpy as np
import pytest
from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    finite_difference_jacobian,
)
from src.shared.python.motion_matching.historical_fit import (
    CameraProjection,
    ImageFitInputs,
    ImageFitConfig,
)
from src.shared.python.motion_matching.historical_fit.solver import _Fit
from src.shared.python.motion_matching.historical_fit.shaft_residuals import (
    AdditionalImageResiduals,
    ImageSourceIdentity,
    ShaftAxisResidualTerm,
)
from src.shared.python.motion_matching.historical_fit.shaft_geometry import (
    AuthoredShaftAxis,
)
from tests.unit.motion_matching.test_shaft_observations import evidence

pytestmark = pytest.mark.unit


class Native:
    coordinate_order = ("x", "angle")
    plant_sha = "a" * 64

    def marker_positions(self, q, attachments):
        c, s = np.cos(q[1]), np.sin(q[1])
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return np.array(
            [r @ np.array(p) + [q[0], 0, 0] for _, p in attachments.values()]
        )


def setup():
    ev = evidence()
    identity = ImageSourceIdentity(
        ev.capture_id,
        ev.capture_sha256,
        ev.source_sha256,
        "camera",
        "sha256:" + "e" * 64,
    )
    axis = AuthoredShaftAxis("club", (0, 0, 4), (1, 0, 4), Native.plant_sha, "b" * 64)
    term = ShaftAxisResidualTerm(ev, axis, identity.source_clock_sha256, 0.5)
    bundle = AdditionalImageResiduals(identity, (term,))
    camera = CameraProjection(np.diag([100.0, 100.0, 1.0]), np.eye(3), np.zeros(3))
    return Native(), camera, term, bundle


def test_perpendicular_pixels_uncertainty_and_unoriented_axis():
    native, camera, term, _ = setup()
    # Horizontal native line at y=0: distances are precisely observed y pixels.
    q = np.zeros((1, 2))
    np.testing.assert_allclose(
        term.residual(native, camera, q), np.array([20, 40]) * np.sqrt(0.8 * 0.5) / 2
    )
    assessment = term.assess(native, camera, q)
    assert assessment.raw_rms_pixels == pytest.approx(np.sqrt((20**2 + 40**2) / 2))
    reversed_axis = replace(
        term,
        axis=replace(
            term.axis, point_a_m=term.axis.point_b_m, point_b_m=term.axis.point_a_m
        ),
    )
    np.testing.assert_allclose(
        reversed_axis.residual(native, camera, q), -term.residual(native, camera, q)
    )
    assert (
        reversed_axis.assess(native, camera, q).angular_errors_deg
        == assessment.angular_errors_deg
    )


def test_sparse_basis_preserves_legacy_prefix_and_body_geometry_times():
    native, camera, term, bundle = setup()
    attachments = {"body": ("club", (0, 1, 4))}
    inputs = ImageFitInputs(
        np.array([0.0, 1.0]),
        np.array([[[0.0, 25.0]], [[0.0, 25.0]]]),
        np.ones((2, 1)),
        np.zeros(2),
        np.ones(2),
        ("x", "angle"),
    )
    config = ImageFitConfig(closure_weight=0)
    legacy = _Fit(native, attachments, camera, inputs, config)
    extended = _Fit(native, attachments, camera, inputs, config, bundle)
    trajectory = CubicHermiteSplineTrajectory(inputs.source_times, 2)
    coefficients = trajectory.initial_coefficients_from_samples(
        inputs.source_times, np.array([[0.1, 0.2], [0.3, 0.4]])
    )
    old_eval = trajectory.evaluate(coefficients, legacy.evaluation_times)
    new_eval = trajectory.evaluate(coefficients, extended.map_evaluation_times)
    old = legacy.residual(old_eval, {})
    new = extended.residual(new_eval, {})
    np.testing.assert_array_equal(extended.evaluation_times, legacy.evaluation_times)
    assert len(extended.map_evaluation_times) == 3
    np.testing.assert_array_equal(new[: len(old)], old)
    assert len(new) == len(old) + 2
    np.testing.assert_allclose(
        extended.jacobian(new_eval, {}, None),
        finite_difference_jacobian(
            lambda c: extended.residual(
                trajectory.evaluate(c, extended.map_evaluation_times), {}
            ),
            coefficients,
        ),
        rtol=2e-5,
        atol=2e-6,
    )
    assert legacy.rms(
        legacy.image_residuals(legacy.expand(old_eval.q))
    ) == extended.rms(extended.image_residuals(extended.expand(new_eval.q[[0, 2]])))


@pytest.mark.parametrize(
    "field,value",
    [
        ("capture_id", "other"),
        ("source_clock_sha256", "sha256:" + "f" * 64),
        ("camera_id", "other"),
    ],
)
def test_context_mismatch_rejects(field, value):
    native, _, _, bundle = setup()
    with pytest.raises(ValueError, match="identity|clock|camera"):
        replace(
            bundle, source_identity=replace(bundle.source_identity, **{field: value})
        ).validate(native)


def test_wrong_model_and_degenerate_projection_reject():
    native, camera, term, bundle = setup()
    native.plant_sha = "f" * 64
    with pytest.raises(ValueError, match="model"):
        bundle.validate(native)
    native.plant_sha = term.axis.native_model_sha
    with pytest.raises(ValueError, match="degenerate"):
        replace(term, axis=replace(term.axis, point_b_m=(0, 0, 5))).residual(
            native, camera, np.zeros((1, 2))
        )
    with pytest.raises(ValueError, match="camera|depth|behind"):
        term.residual(
            native,
            replace(camera, translation=np.array([0, 0, -5.0])),
            np.zeros((1, 2)),
        )


def test_abstention_has_zero_rows_and_no_raw_observation():
    from src.shared.python.motion_matching.historical_fit.shaft_observations import (
        ShaftAxisSegment,
    )

    native, camera, term, _ = setup()
    frame = replace(
        term.evidence.frames[0],
        segment=ShaftAxisSegment(
            "ambiguous", None, "reviewer", "Blur", None, None, None
        ),
    )
    term = replace(term, evidence=replace(term.evidence, frames=(frame,)))
    np.testing.assert_array_equal(
        term.residual(native, camera, np.zeros((1, 2))), np.zeros(2)
    )
    assessment = term.assess(native, camera, np.zeros((1, 2)))
    assert assessment.observed_segment_count == 0
    assert assessment.raw_rms_pixels is None


def test_native_sparse_term_does_not_increase_geometry_mass_or_body_counts():
    from tests.unit.motion_matching.test_historical_image_fit import native_problem
    from src.shared.python.motion_matching.historical_fit import (
        ImageSplineStart,
        initialize_image_trajectory,
        resolve_authored_shaft_axis,
    )
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane
    from pathlib import Path

    native, attachments, camera, inputs = native_problem()
    original = evidence()
    frame = replace(original.frames[0].frame, pts_ticks=3315)
    ev = replace(original, frames=(replace(original.frames[0], frame=frame),))
    definition = (
        Path(__file__).resolve().parents[3]
        / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_bytes()
    axis = resolve_authored_shaft_axis(definition, native.plant_sha)
    context = ImageSourceIdentity(
        ev.capture_id,
        ev.capture_sha256,
        ev.source_sha256,
        frame.camera_id,
        "sha256:" + "e" * 64,
    )
    bundle = AdditionalImageResiduals(
        context, (ShaftAxisResidualTerm(ev, axis, context.source_clock_sha256, 0.5),)
    )
    constraints = ConstraintOptions(GroundPlane((0, 0, 1), 0), 1, 1, 1, 0.1, 0.1, 0.1)
    config = ImageFitConfig(closure_weight=0, constraint_options=constraints)
    trajectory = CubicHermiteSplineTrajectory(inputs.source_times, 1)
    coefficients = trajectory.initial_coefficients_from_samples(
        inputs.source_times, np.array([[-0.3], [-0.8]])
    )
    start = ImageSplineStart.from_coefficients(
        inputs.source_times,
        coefficients,
        tuple(native.coordinate_order),
        inputs.free_coordinates,
        native.plant_sha,
    )
    old = initialize_image_trajectory(
        native, attachments, camera, inputs, config, start
    )
    new = initialize_image_trajectory(
        native, attachments, camera, inputs, config, start, bundle
    )
    assert new.observed_point_count == old.observed_point_count == 6
    assert new.rms_pixels == old.rms_pixels
    np.testing.assert_array_equal(new.q, old.q)
    np.testing.assert_array_equal(new.constraint_times, old.constraint_times)
    np.testing.assert_array_equal(new.constraint_residuals, old.constraint_residuals)
    assert old.additional_image_assessments == ()
    assert len(new.additional_image_assessments) == 1
    fit = _Fit(native, attachments, camera, inputs, config, bundle)
    evaluation = trajectory.evaluate(coefficients, fit.map_evaluation_times)
    np.testing.assert_allclose(
        fit.jacobian(evaluation, {}, None),
        finite_difference_jacobian(
            lambda c: fit.residual(
                trajectory.evaluate(c, fit.map_evaluation_times), {}
            ),
            coefficients,
        ),
        rtol=3e-5,
        atol=1e-5,
    )


def test_outside_interval_and_unknown_visibility_boolean_rejected():
    native, camera, term, bundle = setup()
    with pytest.raises(ValueError, match="visibility"):
        replace(term, unknown_visibility_weight=True)
    inputs = ImageFitInputs(
        np.array([1.0, 2.0]),
        np.zeros((2, 1, 2)),
        np.ones((2, 1)),
        np.zeros(2),
        np.ones(2),
        ("x", "angle"),
    )
    with pytest.raises(ValueError, match="interval"):
        _Fit(
            native,
            {"body": ("club", (0, 1, 4))},
            camera,
            inputs,
            ImageFitConfig(closure_weight=0),
            bundle,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"evidence_sha256": "bad"},
        {"source_times": (float("nan"),)},
        {"source_times": (0.5, 0.4)},
        {"frame_indices": (-1,)},
        {"perpendicular_errors_pixels": ((1, float("inf")),)},
        {"angular_errors_deg": (-1.0,)},
        {"angular_errors_deg": (91.0,)},
        {"observed_segment_count": True},
        {"observed_segment_count": 0},
        {"raw_rms_pixels": -1},
        {"raw_rms_pixels": float("nan")},
        {"raw_rms_pixels": 999},
        {"angular_errors_deg": ()},
    ],
)
def test_assessment_rejects_malformed(updates):
    native, camera, term, _ = setup()
    assessment = term.assess(native, camera, np.zeros((1, 2)))
    with pytest.raises(ValueError):
        replace(assessment, **updates)


def test_assessment_copies_nested_mutable_values():
    native, camera, term, _ = setup()
    old = term.assess(native, camera, np.zeros((1, 2)))
    errors = [[20.0, 40.0]]
    new = replace(
        old,
        perpendicular_errors_pixels=errors,
        frame_indices=[0],
        source_times=[1 / 3],
        angular_errors_deg=[45.0],
    )
    errors[0][0] = 999
    assert new.perpendicular_errors_pixels == ((20.0, 40.0),)
    assert new.frame_indices == (0,)


def test_direct_residual_rejects_wrong_native_model():
    native, camera, term, _ = setup()
    native.plant_sha = "f" * 64
    with pytest.raises(ValueError, match="model"):
        term.residual(native, camera, np.zeros((1, 2)))


class BadTerm:
    def __init__(self, times, mode="valid"):
        self.source_times = times
        self.mode = mode
        self.calls = 0

    def validate(self, native, identity):
        pass

    def residual(self, native, camera, poses):
        self.calls += 1
        if self.mode == "matrix":
            return np.zeros((1, 2))
        if self.mode == "nan":
            return np.array([np.nan])
        if self.mode == "changing":
            return np.zeros(self.calls)
        return np.zeros(1)

    def assess(self, native, camera, poses):
        raise AssertionError("Not an accepted diagnostic")


@pytest.mark.parametrize(
    "times", [(), (float("nan"),), (0.4, 0.3), (0.3, 0.3), (float("inf"),)]
)
def test_bad_protocol_times_rejected_before_map(times):
    native, camera, _, bundle = setup()
    inputs = ImageFitInputs(
        np.array([0.0, 1.0]),
        np.zeros((2, 1, 2)),
        np.ones((2, 1)),
        np.zeros(2),
        np.ones(2),
        ("x", "angle"),
    )
    with pytest.raises(ValueError, match="times"):
        _Fit(
            native,
            {"body": ("club", (0, 1, 4))},
            camera,
            inputs,
            ImageFitConfig(closure_weight=0),
            replace(bundle, terms=(BadTerm(times),)),
        )


@pytest.mark.parametrize("mode", ["matrix", "nan", "changing"])
def test_bad_protocol_residual_rejected(mode):
    native, camera, _, bundle = setup()
    inputs = ImageFitInputs(
        np.array([0.0, 1.0]),
        np.zeros((2, 1, 2)),
        np.ones((2, 1)),
        np.zeros(2),
        np.ones(2),
        ("x", "angle"),
    )
    trajectory = CubicHermiteSplineTrajectory(inputs.source_times, 2)
    coefficients = trajectory.initial_coefficients_from_samples(
        inputs.source_times, np.zeros((2, 2))
    )
    with pytest.raises(ValueError, match="residual"):
        fit = _Fit(
            native,
            {"body": ("club", (0, 1, 4))},
            camera,
            inputs,
            ImageFitConfig(closure_weight=0),
            replace(bundle, terms=(BadTerm((0.5,), mode),)),
        )
        fit.residual(trajectory.evaluate(coefficients, fit.map_evaluation_times), {})
