"""Shared ground-contact law and cross-engine parity harness (FB-2)."""

import numpy as np
import pytest

from src.shared.python.motion_matching import contact_law as module

pytestmark = pytest.mark.unit


def _params() -> module.ContactParameters:
    return module.ContactParameters(
        stiffness_n_m=1e5,
        dissipation_s_m=0.5,
        static_friction=0.9,
        dynamic_friction=0.6,
        viscous_friction=0.1,
        transition_velocity_m_s=0.1,
    )


def _ground() -> module.GroundPlane:
    return module.GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)


def test_no_force_without_penetration() -> None:
    sample = module.sphere_ground_contact(
        np.array([0.0, 0.0, 0.05]),
        np.array([1.0, 0.0, -1.0]),
        0.03,
        _ground(),
        _params(),
    )
    assert sample.penetration_m == 0.0
    np.testing.assert_array_equal(sample.normal_force_n, 0.0)
    np.testing.assert_array_equal(sample.friction_force_n, 0.0)


def test_normal_force_matches_closed_form() -> None:
    # Centre at z=0.02 with radius 0.03: penetration 0.01 m, approaching at 0.2 m/s.
    sample = module.sphere_ground_contact(
        np.array([0.0, 0.0, 0.02]),
        np.array([0.0, 0.0, -0.2]),
        0.03,
        _ground(),
        _params(),
    )
    assert sample.penetration_m == pytest.approx(0.01)
    expected = 1e5 * 0.01 * (1.0 + 0.5 * 0.2)
    np.testing.assert_allclose(sample.normal_force_n, [0.0, 0.0, expected])
    np.testing.assert_allclose(sample.contact_point_m, [0.0, 0.0, -0.01])
    # Separating fast enough clips the normal force at zero, never negative.
    pulling = module.sphere_ground_contact(
        np.array([0.0, 0.0, 0.02]),
        np.array([0.0, 0.0, 5.0]),
        0.03,
        _ground(),
        _params(),
    )
    np.testing.assert_array_equal(pulling.normal_force_n, 0.0)


def test_friction_opposes_sliding_and_is_regularised() -> None:
    p = _params()
    slow = module.sphere_ground_contact(
        np.array([0.0, 0.0, 0.02]), np.array([0.01, 0.0, 0.0]), 0.03, _ground(), p
    )
    fast = module.sphere_ground_contact(
        np.array([0.0, 0.0, 0.02]), np.array([2.0, 0.0, 0.0]), 0.03, _ground(), p
    )
    normal = 1e5 * 0.01
    assert slow.friction_force_n[0] < 0 and fast.friction_force_n[0] < 0
    assert abs(slow.friction_force_n[0]) < abs(fast.friction_force_n[0])
    mu_fast = p.dynamic_friction + p.viscous_friction * 2.0
    np.testing.assert_allclose(
        abs(fast.friction_force_n[0]), mu_fast * normal * np.tanh(2.0 / 0.1), rtol=1e-6
    )
    assert slow.friction_force_n[2] == 0.0


def test_parameters_and_plane_are_validated() -> None:
    with pytest.raises(ValueError):
        module.ContactParameters(0.0, 0.5, 0.9, 0.6, 0.0, 0.1)
    with pytest.raises(ValueError):
        module.ContactParameters(1e5, 0.5, 0.5, 0.6, 0.0, 0.1)  # static below dynamic
    with pytest.raises(ValueError):
        module.GroundPlane(normal=(0.0, 0.0, 0.0), height_m=0.0)
    with pytest.raises(ValueError):
        module.sphere_ground_contact(
            np.zeros(2), np.zeros(3), 0.03, _ground(), _params()
        )


def test_parity_harness_reports_max_differences_between_adapters() -> None:
    def reference(position, velocity, radius):
        return module.sphere_ground_contact(
            position, velocity, radius, _ground(), _params()
        )

    def biased(position, velocity, radius):
        sample = reference(position, velocity, radius)
        return sample._replace(normal_force_n=sample.normal_force_n * 1.01)

    states = module.random_contact_states(seed=1, count=20, radius=0.03)
    report = module.contact_parity_report(
        {"ref": reference, "biased": biased}, states, radius=0.03
    )
    assert report["reference"] == "ref"
    assert report["max_normal_force_difference_n"]["biased"] > 0
    assert report["max_friction_force_difference_n"]["biased"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert report["states"] == 20 and report["penetrating_states"] > 0
    with pytest.raises(ValueError):
        module.contact_parity_report({"only": reference}, states, radius=0.03)
