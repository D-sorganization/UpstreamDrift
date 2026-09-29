"""Velocity-representation convention tests for floating-base engines."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.engines.common import velocity_conventions as shim
from src.shared.python.engine_core.capabilities import SPATIAL_JACOBIAN_ORDER
from src.shared.python.engine_core.velocity_conventions import (
    CANONICAL_FLOATING_BASE_CONVENTION,
    CANONICAL_VELOCITY_REPRESENTATION,
    FloatingBaseConvention,
    VelocityRepresentation,
    convert_floating_base_velocity,
    convert_gravity_vector,
    normalize_floating_base_velocity,
    single_floating_body_h_g,
)


def _rotation_z_90() -> np.ndarray:
    return np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.mark.parametrize("source", list(VelocityRepresentation))
@pytest.mark.parametrize("target", list(VelocityRepresentation))
def test_velocity_representation_converters_are_mutually_inverse(
    source: VelocityRepresentation,
    target: VelocityRepresentation,
) -> None:
    rotation = _rotation_z_90()
    velocity = np.array([0.1, -0.2, 0.3, 1.0, 2.0, -3.0])

    converted = convert_floating_base_velocity(
        velocity,
        source=source,
        target=target,
        rotation_inertial_from_body=rotation,
    )
    round_tripped = convert_floating_base_velocity(
        converted,
        source=target,
        target=source,
        rotation_inertial_from_body=rotation,
    )

    np.testing.assert_allclose(round_tripped, velocity, atol=1e-12)


def test_mixed_representation_matches_jaxsim_default_contract() -> None:
    rotation = _rotation_z_90()
    mixed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    inertial = normalize_floating_base_velocity(
        mixed,
        source=VelocityRepresentation.MIXED,
        rotation_inertial_from_body=rotation,
    )

    np.testing.assert_allclose(inertial[:3], rotation @ mixed[:3])
    np.testing.assert_allclose(inertial[3:], mixed[3:])
    assert CANONICAL_VELOCITY_REPRESENTATION is VelocityRepresentation.INERTIAL


def test_gravity_vector_uses_linear_frame_of_representation() -> None:
    rotation = _rotation_z_90()
    gravity = np.array([1.0, 2.0, -9.0])

    body = convert_gravity_vector(
        gravity,
        target=VelocityRepresentation.BODY_FIXED,
        rotation_inertial_from_body=rotation,
    )
    mixed = convert_gravity_vector(
        gravity,
        target=VelocityRepresentation.MIXED,
        rotation_inertial_from_body=rotation,
    )

    np.testing.assert_allclose(body, rotation.T @ gravity)
    np.testing.assert_allclose(mixed, gravity)


@pytest.mark.parametrize(
    ("representation", "expected_torque", "expected_force"),
    [
        (
            VelocityRepresentation.BODY_FIXED,
            np.array([-4.0, 3.0, 2.0]),
            np.array([-19.62, 0.0, -98.0665]),
        ),
        (
            VelocityRepresentation.INERTIAL,
            np.array([-3.0, -4.0, 2.0]),
            np.array([0.0, -19.62, -98.0665]),
        ),
        (
            VelocityRepresentation.MIXED,
            np.array([-4.0, 3.0, 2.0]),
            np.array([0.0, -19.62, -98.0665]),
        ),
    ],
)
def test_single_floating_body_h_g_matches_analytic_case(
    representation: VelocityRepresentation,
    expected_torque: np.ndarray,
    expected_force: np.ndarray,
) -> None:
    rotation = _rotation_z_90()
    inertia = np.diag([2.0, 3.0, 5.0])
    omega_body = np.array([1.0, 2.0, -1.0])
    omega = (
        rotation @ omega_body
        if representation == VelocityRepresentation.INERTIAL
        else omega_body
    )

    terms = single_floating_body_h_g(
        mass_kg=10.0,
        inertia_body_kg_m2=inertia,
        angular_velocity=omega,
        representation=representation,
        rotation_inertial_from_body=rotation,
        gravity_inertial_mps2=np.array([0.0, -1.962, -9.80665]),
    )

    np.testing.assert_allclose(terms.h[:3], expected_torque, atol=1e-12)
    np.testing.assert_allclose(terms.h[3:], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(terms.g[:3], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(terms.g[3:], expected_force, atol=1e-12)


def test_convention_contract_integrates_spatial_jacobian_order() -> None:
    assert CANONICAL_FLOATING_BASE_CONVENTION.spatial_jacobian_order is (
        SPATIAL_JACOBIAN_ORDER
    )
    with pytest.raises(ValueError, match="SPATIAL_JACOBIAN_ORDER"):
        FloatingBaseConvention(
            velocity_representation=VelocityRepresentation.INERTIAL,
            spatial_jacobian_order=("linear", "angular"),
        )


def test_engine_common_shim_reexports_canonical_objects() -> None:
    assert shim.VelocityRepresentation is VelocityRepresentation
    assert shim.convert_floating_base_velocity is convert_floating_base_velocity


@pytest.mark.parametrize(
    "public_function",
    [
        convert_floating_base_velocity,
        normalize_floating_base_velocity,
        convert_gravity_vector,
        single_floating_body_h_g,
    ],
)
def test_public_functions_document_units_and_representation(
    public_function: object,
) -> None:
    doc = inspect.getdoc(public_function)
    assert doc is not None
    assert "[angular; linear]" in doc or "Gravity vector" in doc
    assert "m/s" in doc or "m/s^2" in doc
    assert "representation" in doc


# ---------------------------------------------------------------------------
# Finiteness guards (#6931)
# ---------------------------------------------------------------------------


def _identity_rotation() -> np.ndarray:
    return np.eye(3)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_convert_velocity_rejects_non_finite_velocity(bad: float) -> None:
    velocity = np.array([0.1, -0.2, 0.3, 1.0, 2.0, bad])
    with pytest.raises(ValueError, match="finite"):
        convert_floating_base_velocity(
            velocity,
            source=VelocityRepresentation.INERTIAL,
            target=VelocityRepresentation.BODY_FIXED,
            rotation_inertial_from_body=_identity_rotation(),
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_convert_gravity_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        convert_gravity_vector(
            np.array([0.0, 0.0, bad]),
            target=VelocityRepresentation.BODY_FIXED,
            rotation_inertial_from_body=_identity_rotation(),
        )


def test_single_floating_body_rejects_non_finite_inertia() -> None:
    inertia = np.eye(3)
    inertia[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        single_floating_body_h_g(
            mass_kg=1.0,
            inertia_body_kg_m2=inertia,
            angular_velocity=np.zeros(3),
            representation=VelocityRepresentation.INERTIAL,
            rotation_inertial_from_body=_identity_rotation(),
        )


@pytest.mark.parametrize("bad_mass", [np.nan, np.inf])
def test_single_floating_body_rejects_non_finite_mass(bad_mass: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        single_floating_body_h_g(
            mass_kg=bad_mass,
            inertia_body_kg_m2=np.eye(3),
            angular_velocity=np.zeros(3),
            representation=VelocityRepresentation.INERTIAL,
            rotation_inertial_from_body=_identity_rotation(),
        )
