"""Contracts for pointwise constrained-contact reaction attribution."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics.contact_reaction_decomposition import (
    ContactReactionInputs,
    decompose_contact_reaction,
    evaluate_reaction_prediction,
)

pytestmark = pytest.mark.unit


def _sample(**overrides: object) -> ContactReactionInputs:
    values: dict[str, object] = {
        "mass_matrix": np.diag([2.0, 3.0]),
        "static_bias": np.array([10.0, 6.0]),
        "velocity_bias": np.array([4.0, -3.0]),
        "contact_jacobian": np.array([[1.0, 0.0]]),
        "constraint_bias": np.array([1.0]),
        "actuation_matrix": np.eye(2),
        "control": np.array([2.0, 9.0]),
        "external_generalized_force": np.array([6.0, 0.0]),
    }
    values.update(overrides)
    return ContactReactionInputs(**values)


def test_static_vertical_support_reproduces_weight() -> None:
    mass = 80.0
    gravity = 9.81
    result = decompose_contact_reaction(
        ContactReactionInputs(
            mass_matrix=np.array([[mass]]),
            static_bias=np.array([mass * gravity]),
            velocity_bias=np.zeros(1),
            contact_jacobian=np.ones((1, 1)),
            constraint_bias=np.zeros(1),
            actuation_matrix=np.ones((1, 1)),
            control=np.zeros(1),
            external_generalized_force=np.zeros(1),
        )
    )

    np.testing.assert_allclose(result.configuration_reaction, [mass * gravity])
    np.testing.assert_allclose(result.total_reaction, [mass * gravity])
    np.testing.assert_allclose(result.velocity_reaction, [0.0])
    np.testing.assert_allclose(result.control_reaction, [0.0])


def test_components_close_and_counterfactuals_retain_external_load() -> None:
    result = decompose_contact_reaction(_sample())

    np.testing.assert_allclose(result.configuration_reaction, [10.0])
    np.testing.assert_allclose(result.velocity_reaction, [2.0])
    np.testing.assert_allclose(result.control_reaction, [-2.0])
    np.testing.assert_allclose(result.external_reaction, [-6.0])
    np.testing.assert_allclose(result.total_reaction, [4.0])
    np.testing.assert_allclose(
        result.total_reaction,
        result.configuration_reaction
        + result.velocity_reaction
        + result.control_reaction
        + result.external_reaction,
    )
    np.testing.assert_allclose(result.ztcf_reaction, [6.0])
    np.testing.assert_allclose(result.zvcf_reaction, [4.0])
    np.testing.assert_allclose(result.zero_velocity_control_preserved_reaction, [2.0])
    assert not np.allclose(
        result.ztcf_reaction + result.zvcf_reaction,
        result.total_reaction,
    )


def test_reaction_satisfies_acceleration_constraint() -> None:
    sample = _sample()
    result = decompose_contact_reaction(sample)
    applied = (
        sample.actuation_matrix @ sample.control
        + sample.external_generalized_force
        + sample.contact_jacobian.T @ result.total_reaction
        - sample.static_bias
        - sample.velocity_bias
    )
    acceleration = np.linalg.solve(sample.mass_matrix, applied)

    np.testing.assert_allclose(
        sample.contact_jacobian @ acceleration + sample.constraint_bias,
        np.zeros(1),
        atol=1e-12,
    )


def test_rank_deficient_contact_allocation_fails_closed() -> None:
    with pytest.raises(ValueError, match="full row rank"):
        decompose_contact_reaction(
            _sample(
                contact_jacobian=np.array([[1.0, 0.0], [2.0, 0.0]]),
                constraint_bias=np.zeros(2),
            )
        )


def test_non_si_or_ambiguous_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="units"):
        _sample(units="body_weight")
    with pytest.raises(ValueError, match="frame"):
        _sample(frame="")


def test_prediction_metrics_pin_componentwise_and_impulse_errors() -> None:
    time = np.array([0.0, 0.5, 1.0])
    measured = np.array([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]])
    predicted = measured + np.array([1.0, -2.0])
    metrics = evaluate_reaction_prediction(
        time,
        measured,
        predicted,
        normalization_scale=np.array([4.0, 8.0]),
        component_names=("horizontal", "vertical"),
    )

    np.testing.assert_allclose(metrics.bias, [1.0, -2.0])
    np.testing.assert_allclose(metrics.rmse, [1.0, 2.0])
    np.testing.assert_allclose(metrics.nrmse, [0.25, 0.25])
    np.testing.assert_allclose(metrics.impulse_error, [1.0, -2.0])
    np.testing.assert_allclose(metrics.r_squared, [0.625, 0.625])
    assert metrics.component_names == ("horizontal", "vertical")


def test_prediction_metrics_require_explicit_nonzero_scale() -> None:
    with pytest.raises(ValueError, match="normalization_scale"):
        evaluate_reaction_prediction(
            np.array([0.0, 1.0]),
            np.zeros((2, 1)),
            np.zeros((2, 1)),
            normalization_scale=np.zeros(1),
            component_names=("vertical",),
        )
