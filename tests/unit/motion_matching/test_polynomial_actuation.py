"""Contract tests for global degree-six full-body actuation."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.shared.python.motion_matching.piecewise_polynomial import PolynomialSegment
from src.shared.python.motion_matching.polynomial_actuation import (
    ROOT_COORDINATES,
    FullBodyPolynomialControl,
)

pytestmark = pytest.mark.unit


def _spec() -> dict:
    return {
        "schema_version": "full-body-v1",
        "coordinate_order": [*ROOT_COORDINATES, "joint_b", "joint_a"],
        "joints": [
            {
                "parent": "world",
                "primitives": [
                    {"coordinate": name, "primitive": primitive}
                    for name, primitive in zip(
                        ROOT_COORDINATES,
                        ("Px", "Py", "Pz", "Rx", "Ry", "Rz"),
                        strict=True,
                    )
                ],
            },
            {
                "parent": "pelvis",
                "primitives": [
                    {"coordinate": "joint_a", "primitive": "Rx"},
                    {"coordinate": "joint_b", "primitive": "Ry"},
                ],
            },
        ],
    }


def test_from_spec_derives_root_exclusion_and_reordered_actuators() -> None:
    control = FullBodyPolynomialControl.from_spec(_spec(), duration_s=2.5)

    assert control.coordinate_names == (*ROOT_COORDINATES, "joint_b", "joint_a")
    assert control.actuated_names == ("joint_b", "joint_a")
    assert control.n_coordinates == 8
    assert control.n_parameters == 14


def test_efforts_and_jacobian_reuse_bernstein_segment_convention() -> None:
    names = (*ROOT_COORDINATES, "joint_b", "joint_a")
    control = FullBodyPolynomialControl(names, ("joint_b", "joint_a"), 2.5)
    parameters = np.arange(14, dtype=float) - 4.0
    time_s = 0.7

    efforts = control.efforts(parameters, time_s)
    expected = PolynomialSegment(
        0.0, 2.5, parameters.reshape(2, 7), is_bernstein=True
    ).evaluate(time_s)

    assert tuple(efforts) == names
    np.testing.assert_array_equal([efforts[name] for name in ROOT_COORDINATES], 0.0)
    np.testing.assert_allclose(
        [efforts["joint_b"], efforts["joint_a"]], expected, rtol=0.0, atol=0.0
    )

    jacobian = control.effort_jacobian(time_s)
    direction = np.linspace(-0.5, 0.8, 14)
    step = 1e-7
    plus = control.efforts(parameters + step * direction, time_s)
    minus = control.efforts(parameters - step * direction, time_s)
    measured = np.array([(plus[n] - minus[n]) / (2.0 * step) for n in names])
    np.testing.assert_allclose(jacobian @ direction, measured, rtol=2e-8, atol=2e-8)
    assert jacobian.flags.owndata
    assert not jacobian.flags.writeable


def test_export_is_ascending_physical_seconds_with_exact_root_zeros() -> None:
    control = FullBodyPolynomialControl.from_spec(_spec(), duration_s=1.7)
    parameters = np.linspace(-2.0, 3.0, control.n_parameters)
    exported = control.export_power_coefficients(parameters)

    assert exported.shape == (control.n_coordinates, 7)
    root_indices = [control.coordinate_names.index(name) for name in ROOT_COORDINATES]
    np.testing.assert_array_equal(exported[root_indices], 0.0)
    for time_s in (0.0, 0.63, control.duration_s):
        powers = time_s ** np.arange(7)
        reconstructed = exported @ powers
        efforts = control.efforts(parameters, time_s)
        np.testing.assert_allclose(
            reconstructed,
            [efforts[name] for name in control.coordinate_names],
            rtol=2e-13,
            atol=2e-13,
        )
    assert exported.flags.owndata
    assert not exported.flags.writeable


@pytest.mark.parametrize(
    ("coordinate_names", "actuated_names", "match"),
    [
        ((*ROOT_COORDINATES, "a", "a"), ("a",), "duplicate"),
        ((*ROOT_COORDINATES, "a"), ("missing",), "unknown"),
        ((*ROOT_COORDINATES, "a", "b"), ("a",), "all non-root"),
        ((*ROOT_COORDINATES, "a"), (ROOT_COORDINATES[0], "a"), "root"),
        (tuple(ROOT_COORDINATES[1:]) + ("a",), ("a",), "root"),
    ],
)
def test_direct_constructor_rejects_inconsistent_names(
    coordinate_names: tuple[str, ...],
    actuated_names: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        FullBodyPolynomialControl(coordinate_names, actuated_names, 1.0)


@pytest.mark.parametrize("duration", [0.0, -1.0, np.nan, np.inf, True])
def test_duration_must_be_finite_positive_real(duration: float) -> None:
    with pytest.raises(ValueError, match="duration_s"):
        FullBodyPolynomialControl((*ROOT_COORDINATES, "a"), ("a",), duration)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda spec: spec.update(schema_version="wrong"), "full-body-v1"),
        (lambda spec: spec["joints"].pop(0), "world root"),
        (
            lambda spec: spec["joints"][0]["primitives"].pop(),
            "six-coordinate",
        ),
        (
            lambda spec: spec["coordinate_order"].append("joint_a"),
            "duplicate",
        ),
        (
            lambda spec: spec["joints"][1]["primitives"][0].update(
                primitive="Spherical"
            ),
            "scalar",
        ),
    ],
)
def test_from_spec_rejects_invalid_inventory(mutation, match: str) -> None:
    spec = deepcopy(_spec())
    mutation(spec)
    with pytest.raises(ValueError, match=match):
        FullBodyPolynomialControl.from_spec(spec, duration_s=1.0)


def test_parameter_and_time_validation_and_final_roundoff_tolerance() -> None:
    control = FullBodyPolynomialControl.from_spec(_spec(), duration_s=1.0)
    parameters = np.zeros(control.n_parameters)

    with pytest.raises(ValueError, match="shape"):
        control.efforts(np.zeros((2, 7)), 0.0)
    parameters[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        control.efforts(parameters, 0.0)
    parameters[0] = 0.0
    for invalid_time in (-1e-12, 1.0 + 1e-10, np.nan, np.inf):
        with pytest.raises(ValueError, match="time_s"):
            control.efforts(parameters, invalid_time)
    accepted = 1.0 + 4.0 * np.finfo(float).eps
    assert np.isfinite(tuple(control.efforts(parameters, accepted).values())).all()


def test_tiny_duration_export_rejects_nonfinite_power_conversion() -> None:
    control = FullBodyPolynomialControl(
        (*ROOT_COORDINATES, "a"),
        ("a",),
        np.nextafter(0.0, 1.0),
    )

    with pytest.raises(ValueError, match="non-finite"):
        control.export_power_coefficients(np.ones(control.n_parameters))
