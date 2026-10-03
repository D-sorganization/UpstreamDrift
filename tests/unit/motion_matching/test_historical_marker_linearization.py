"""Analytic marker and camera chains preserve the canonical image objective."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    HermiteBoundsDomain,
    finite_difference_jacobian,
)
from src.shared.python.motion_matching.historical_fit import (
    CameraProjection,
    ImageFitConfig,
    ImageFitInputs,
)
from src.shared.python.motion_matching.historical_fit.solver import _Fit
from src.shared.python.motion_matching.pipeline import (
    MarkerLinearization,
)
from src.shared.python.motion_matching.pipeline.plant import get_plant

pytestmark = pytest.mark.unit


def camera():
    return CameraProjection(
        np.array([[500.0, 37.0, 320.0], [11.0, 400.0, 240.0], [0.0, 0.0, 1.0]]),
        Rotation.from_rotvec([0.2, -0.1, 0.15]).as_matrix(),
        np.array([0.4, -0.3, 5.0]),
    )


def test_camera_derivative_matches_independent_world_point_differences():
    current = camera()
    points = np.array([[0.3, -0.2, 0.8], [-0.4, 0.6, 1.2]])
    derivative = current.project_jacobian(points)
    numeric = finite_difference_jacobian(
        lambda value: current.project(value.reshape(-1, 3)).ravel(), points.ravel()
    )
    for index in range(2):
        np.testing.assert_allclose(
            derivative[index],
            numeric[index * 2 : index * 2 + 2, index * 3 : index * 3 + 3],
            atol=1e-7,
            rtol=2e-7,
        )


@pytest.mark.parametrize(
    "points", [np.zeros(3), np.ones((1, 4)), [[0, 0, float("nan")]]]
)
def test_camera_derivative_rejects_malformed_points(points):
    with pytest.raises(ValueError):
        camera().project_jacobian(points)


def test_camera_derivative_retains_projection_depth_rejection():
    current = CameraProjection(np.eye(3), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="front"):
        current.project_jacobian(np.array([[0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("change", ["positions", "jacobian", "labels", "order"])
def test_linearization_rejects_malformed_shape_identity_and_values(change):
    values = {
        "positions": np.zeros((1, 3)),
        "jacobian": np.zeros((1, 3, 2)),
        "marker_labels": ("point",),
        "coordinate_order": ("a", "b"),
    }
    values[
        change
        if change not in {"labels", "order"}
        else {"labels": "marker_labels", "order": "coordinate_order"}[change]
    ] = {
        "positions": np.full((1, 3), np.nan),
        "jacobian": np.zeros((1, 3, 1)),
        "labels": ("",),
        "order": ("a", "a"),
    }[change]
    with pytest.raises(ValueError):
        MarkerLinearization(**values)


@pytest.mark.parametrize("names", [None, {"point"}, "point"])
def test_linearization_requires_ordered_name_sequences(names):
    with pytest.raises(ValueError, match="ordered sequence"):
        MarkerLinearization(np.zeros((1, 3)), np.zeros((1, 3, 1)), names, ("a",))


def test_linearization_is_defensively_immutable():
    points, derivative = np.ones((1, 3)), np.ones((1, 3, 1))
    record = MarkerLinearization(points, derivative, ["point"], ["joint"])
    points[:] = 5
    derivative[:] = 9
    assert record.marker_labels == ("point",) and record.coordinate_order == ("joint",)
    np.testing.assert_array_equal(record.positions, 1)
    np.testing.assert_array_equal(record.jacobian, 1)
    with pytest.raises(ValueError):
        record.jacobian[0, 0, 0] = 2


class FinitePlant:
    coordinate_order = ("b", "a")
    plant_sha = "fixture"

    def __init__(self):
        self.position_calls = 0

    def marker_positions(self, q, attachments):
        self.position_calls += 1
        return np.array([[q[0], q[1] ** 2, 1.0 + 0.1 * q[0]]])


class AnalyticPlant(FinitePlant):
    def __init__(self):
        super().__init__()
        self.factory_calls = self.linearization_calls = 0

    def create_marker_linearizer(self, attachments):
        self.factory_calls += 1
        return self

    def marker_linearization(self, q):
        self.linearization_calls += 1
        return MarkerLinearization(
            np.array([[q[0], q[1] ** 2, 1.0 + 0.1 * q[0]]]),
            np.array([[[1.0, 0.0], [0.0, 2 * q[1]], [0.1, 0.0]]]),
            ("point",),
            self.coordinate_order,
        )


def fixture(native):
    times = np.array([0.0, 0.3, 1.0])
    inputs = ImageFitInputs(
        times,
        np.zeros((3, 1, 2)),
        np.array([[0.25], [0.0], [0.7]]),
        np.array([0.1, 0.2]),
        np.array([0.5, 2.0]),
        ("a", "b"),
    )
    trajectory = CubicHermiteSplineTrajectory(np.array([0.0, 0.5, 1.0]), 2)
    coefficients = trajectory.pack(
        np.array([[0.2, 0.1], [0.4, 0.3], [0.6, 0.5]]), np.zeros((3, 2))
    )
    fit = _Fit(
        native,
        {"point": ("body", (0, 0, 0))},
        camera(),
        inputs,
        ImageFitConfig(closure_weight=0, prior_weight=0.01, smoothness_weight=0.02),
    )
    return fit, trajectory, coefficients


def test_analytic_path_uses_one_owned_provider_and_one_call_per_source_pose():
    native = AnalyticPlant()
    fit, trajectory, coefficients = fixture(native)
    evaluation = trajectory.evaluate(coefficients, fit.map_evaluation_times)
    fit.jacobian(evaluation, {}, None)
    assert native.factory_calls == 1
    assert native.linearization_calls == 3
    assert native.position_calls == 0
    fit.jacobian(evaluation, {}, None)
    assert native.factory_calls == 1 and native.linearization_calls == 6


def test_analytic_and_finite_fallback_preserve_residual_and_full_chain():
    exact, trajectory, coefficients = fixture(AnalyticPlant())
    fallback, _, _ = fixture(FinitePlant())
    evaluation = trajectory.evaluate(coefficients, exact.map_evaluation_times)
    np.testing.assert_array_equal(
        exact.residual(evaluation, {}), fallback.residual(evaluation, {})
    )
    np.testing.assert_allclose(
        exact.jacobian(evaluation, {}, None),
        fallback.jacobian(evaluation, {}, None),
        atol=1e-7,
    )
    np.testing.assert_array_equal(exact.jacobian(evaluation, {}, None)[2:4], 0)
    assert fallback.native.position_calls == 3 + 3 * (1 + 2 * 2)


@pytest.mark.parametrize("change", ["order", "labels", "wrong_record", "none"])
def test_analytic_provider_identity_and_record_must_match_fit(change):
    native = AnalyticPlant()
    original = native.marker_linearization

    def changed(q):
        row = original(q)
        if change == "none":
            return None
        if change == "wrong_record":
            return object()
        return replace(
            row,
            **(
                {"coordinate_order": ("a", "b")}
                if change == "order"
                else {"marker_labels": ("other",)}
            ),
        )

    native.marker_linearization = changed
    fit, trajectory, coefficients = fixture(native)
    with pytest.raises(ValueError, match="linearization"):
        fit.jacobian(
            trajectory.evaluate(coefficients, fit.map_evaluation_times), {}, None
        )


def test_explicit_unsupported_linearizer_retains_finite_fallback():
    native = AnalyticPlant()

    def unsupported(q):
        raise NotImplementedError("Fixture lacks analytic capability")

    native.marker_linearization = unsupported
    fit, trajectory, coefficients = fixture(native)
    fallback, _, _ = fixture(FinitePlant())
    evaluation = trajectory.evaluate(coefficients, fit.map_evaluation_times)
    np.testing.assert_array_equal(
        fit.jacobian(evaluation, {}, None), fallback.jacobian(evaluation, {}, None)
    )


def test_factory_returning_no_public_provider_rejects():
    native = AnalyticPlant()
    native.create_marker_linearizer = lambda attachments: object()
    with pytest.raises(ValueError, match="factory"):
        fixture(native)


class ConstrainedPlant(AnalyticPlant):
    def create_ik(self, attachments):
        self.ik_calls = getattr(self, "ik_calls", 0) + 1
        return self

    def constraint_residual_jacobian(self, q, options):
        from src.shared.python.motion_matching.constraint_kinematics import (
            ConstraintLinearization,
        )

        return ConstraintLinearization(
            np.array([q[0] ** 2, np.sin(q[1]), 0.0, 0.0, 0.0, 0.0]),
            np.array(
                [
                    [2 * q[0], 0.0],
                    [0.0, np.cos(q[1])],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
            self.coordinate_order,
            tuple(str(i) for i in range(6)),
        )


def test_separate_constraint_adapter_uses_declared_marker_factory():
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane

    class SeparatePlant(AnalyticPlant):
        def create_ik(self, attachments):
            return ConstrainedPlant()

    native = SeparatePlant()
    original, trajectory, coefficients = fixture(native)
    native.factory_calls = 0
    config = replace(
        original.config,
        constraint_options=ConstraintOptions(
            GroundPlane((0, 0, 1), 0), 1, 1, 1, 1, 1, 1
        ),
    )
    # The constraint adapter deliberately has no optional marker capability.
    native.create_ik = lambda attachments: type(
        "ConstraintOnly",
        (),
        {
            "constraint_residual_jacobian": ConstrainedPlant.constraint_residual_jacobian,
            "coordinate_order": native.coordinate_order,
        },
    )()
    fit = _Fit(native, original.attachments, original.camera, original.inputs, config)
    fit.jacobian(trajectory.evaluate(coefficients, fit.map_evaluation_times), {}, None)
    assert native.factory_calls == 1 and native.linearization_calls == 3


def test_camera_derivative_empty_and_near_front_shapes_match_projection():
    current = CameraProjection(np.eye(3), np.eye(3), np.zeros(3))
    assert current.project(np.empty((0, 3))).shape == (0, 2)
    assert current.project_jacobian(np.empty((0, 3))).shape == (0, 2, 3)
    points = np.array([[1e-9, -2e-9, 2e-8]])
    assert np.isfinite(current.project(points)).all()
    np.testing.assert_allclose(
        current.project_jacobian(points)[0],
        [[5e7, 0, -2.5e6], [0, 5e7, 5e6]],
    )


def test_bounded_chain_retains_interior_constraints_and_one_shared_adapter():
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane

    native = ConstrainedPlant()
    original, trajectory, _ = fixture(native)
    native.factory_calls = 0
    config = replace(
        original.config,
        interior_fractions=(0.25, 0.75),
        constraint_options=ConstraintOptions(
            GroundPlane((0, 0, 1), 0), 1, 1, 1, 1, 1, 1
        ),
    )
    fit = _Fit(native, original.attachments, original.camera, original.inputs, config)
    assert native.ik_calls == 1 and native.factory_calls == 0
    domain = HermiteBoundsDomain((0.0, 0.5, 1.0), ((-0.8, 0.8), (-0.7, 0.7)))
    x = np.array([0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.2, -0.1, 0.1, -0.2, -0.1, 0.2])
    evaluation = trajectory.evaluate(domain.decode(x), fit.map_evaluation_times)
    actual = fit.jacobian(evaluation, {}, None) @ domain.decode_jacobian(x)
    numeric = finite_difference_jacobian(
        lambda decision: fit.residual(
            trajectory.evaluate(domain.decode(decision), fit.map_evaluation_times), {}
        ),
        x,
    )
    assert len(fit.evaluation_times) > len(fit.inputs.source_times)
    np.testing.assert_allclose(actual, numeric, atol=2e-6, rtol=2e-5)


def test_bounded_hermite_camera_chain_matches_internal_decision_differences():
    fit, trajectory, _ = fixture(AnalyticPlant())
    domain = HermiteBoundsDomain((0.0, 0.5, 1.0), ((-0.8, 0.8), (-0.7, 0.7)))
    x = np.array([0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.2, -0.1, 0.1, -0.2, -0.1, 0.2])
    evaluation = trajectory.evaluate(domain.decode(x), fit.map_evaluation_times)
    actual = fit.jacobian(evaluation, {}, None) @ domain.decode_jacobian(x)
    numeric = finite_difference_jacobian(
        lambda decision: fit.residual(
            trajectory.evaluate(domain.decode(decision), fit.map_evaluation_times), {}
        ),
        x,
    )
    np.testing.assert_allclose(actual, numeric, atol=2e-6, rtol=2e-5)


def test_public_native_marker_derivatives_match_nonidentity_offset_fk():
    pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[3]
    native = get_plant(
        "mujoco",
        (
            root / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
        ).read_bytes(),
    )
    attachments = {
        "wrist": ("RF", (0.03, 0.01, -0.13)),
        "elbow": ("RS", (0.02, -0.04, -0.3)),
    }
    linearizer = native.create_marker_linearizer(attachments)
    q = np.zeros(len(native.coordinate_order))
    for name, value in {
        "TranslationInputX": 0.3,
        "HipInputZ": 0.2,
        "RSInputZ": -0.3,
        "REInput": -0.6,
    }.items():
        q[native.coordinate_order.index(name)] = value
    record = linearizer.marker_linearization(q)
    expected = native.marker_positions(q, attachments)
    numerical = finite_difference_jacobian(
        lambda pose: native.marker_positions(pose, attachments).ravel(), q
    )
    assert record.coordinate_order == tuple(native.coordinate_order)
    assert record.marker_labels == tuple(attachments)
    np.testing.assert_array_equal(record.positions, expected)
    np.testing.assert_allclose(
        record.jacobian.reshape(-1, len(q)), numerical, atol=2e-7, rtol=2e-5
    )
