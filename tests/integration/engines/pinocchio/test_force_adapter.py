"""Real native equation and coordinate checks for the shared force factory."""

import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest

if isinstance(sys.modules.get("pinocchio"), Mock):
    pytest.skip("Native Pinocchio required", allow_module_level=True)
pin = pytest.importorskip("pinocchio")

from src.engines.physics_engines.pinocchio.python.native_model import (  # noqa: E402
    FullBodyPinocchioModel,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (  # noqa: E402
    create_engine_force_adapter,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]


@pytest.fixture
def pair():
    path = (
        Path(__file__).resolve().parents[4]
        / "docs/development/full_body_models/full_body_spec_v1.json"
    )
    plant = FullBodyPinocchioModel(json.loads(path.read_bytes()))
    adapter = create_engine_force_adapter("pinocchio", path)
    q = np.linspace(-0.04, 0.06, plant.model.nq)
    v = np.linspace(-0.2, 0.3, plant.model.nv)
    a = np.linspace(-0.3, 0.5, plant.model.nv)
    return adapter, plant, q, v, a


def test_factory_raw_equation_and_fresh_forward_parity(pair):
    adapter, plant, q, v, a = pair
    mass = pin.crba(plant.model, plant.data, q).copy()
    bias = pin.nonLinearEffects(plant.model, plant.data, q, v).copy()
    effort = mass @ a + bias
    assert adapter.verify_acceleration_parity(q, v, effort, a) < 1e-7
    np.testing.assert_allclose(
        adapter.compute_inverse_dynamics(q, v, a), effort, atol=1e-9
    )


def test_native_coordinate_names_and_actuation_are_explicit(pair):
    adapter, plant, *_ = pair
    names = tuple(
        name
        for name, _ in sorted(plant._velocity_indices.items(), key=lambda item: item[1])
    )
    assert adapter.coordinate_order == names
    assert adapter.contact_names == tuple(s.name for s in plant.contact_spheres)
    assert adapter.nv == len(names)
    assert len(adapter.actuated_indices) == len(names) - 6


def test_contact_jacobian_matches_independent_native_position_difference(pair):
    adapter, plant, q, v, _ = pair
    jacobian = adapter.compute_contact_jacobian(q)
    positions = []
    for sign in (-1, 1):
        pin.framesForwardKinematics(
            plant.model, plant.data, pin.integrate(plant.model, q, sign * 1e-6 * v)
        )
        positions.append(
            np.concatenate(
                [
                    plant.data.oMf[plant._contact_frames[s.name]].translation.copy()
                    for s in plant.contact_spheres
                ]
            )
        )
    np.testing.assert_allclose(
        jacobian @ v, (positions[1] - positions[0]) / 2e-6, atol=1e-8
    )


def test_grip_jacobian_uses_native_velocity_order(pair):
    adapter, plant, q, v, _ = pair
    names = adapter.coordinate_order
    values = dict(zip(names, q, strict=True))
    rates = dict(zip(names, v, strict=True))
    expected = plant.closure_trajectory_residuals(
        values, rates, dict.fromkeys(names, 0.0)
    ).rate
    np.testing.assert_allclose(
        adapter.compute_grip_jacobian(q) @ v, expected, atol=1e-10
    )


def test_grip_jacobian_is_independent_of_prior_call_state(pair):
    adapter, plant, q, v, _ = pair
    names = adapter.coordinate_order
    first_j = adapter.compute_grip_jacobian(q)
    q_perturbed = q + 0.05
    adapter.compute_grip_jacobian(q_perturbed)
    third_j = adapter.compute_grip_jacobian(q)
    np.testing.assert_allclose(third_j, first_j, atol=1e-12)

    values_p = dict(zip(names, q_perturbed, strict=True))
    rates = dict(zip(names, v, strict=True))
    expected_p = plant.closure_trajectory_residuals(
        values_p, rates, dict.fromkeys(names, 0.0)
    ).rate
    np.testing.assert_allclose(
        adapter.compute_grip_jacobian(q_perturbed) @ v, expected_p, atol=1e-10
    )


@pytest.mark.parametrize("bad", [np.array([0.0]), np.full(41, np.nan)])
def test_invalid_configuration_fails_closed(pair, bad):
    adapter, _, _, v, a = pair
    with pytest.raises(ValueError):
        adapter.compute_inverse_dynamics(bad, v, a)
