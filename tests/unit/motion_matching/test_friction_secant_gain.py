"""Secant slip gain of the shared regularised friction law (#13306)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_kernel import (
    contact_law_kernel,
    friction_secant_gain,
)
from src.shared.python.motion_matching.contact_law import ContactParameters

pytestmark = pytest.mark.unit

LAW = ContactParameters(
    stiffness_n_m=2.0e5,
    dissipation_s_m=0.5,
    static_friction=0.9,
    dynamic_friction=0.6,
    viscous_friction=0.05,
    transition_velocity_m_s=1.0e-3,
)


def test_gain_times_speed_is_the_kernel_friction_magnitude() -> None:
    speed = np.array([1e-6, 3e-4, 1e-3, 4e-3, 0.2])
    penetration = np.full(speed.shape, 1e-4)
    normal_force = LAW.stiffness_n_m * penetration
    res = contact_law_kernel(penetration, np.zeros(speed.shape), speed, LAW)
    gain = friction_secant_gain(speed, LAW)
    np.testing.assert_allclose(
        gain * normal_force * speed, np.abs(res.tangential_force_n), rtol=1e-12
    )


def test_gain_at_rest_is_the_static_slope() -> None:
    gain = friction_secant_gain(np.array([0.0, 1e-15]), LAW)
    expected = LAW.static_friction / LAW.transition_velocity_m_s
    np.testing.assert_allclose(gain, expected, rtol=1e-9)


def test_gain_is_positive_and_falls_with_speed() -> None:
    gain = friction_secant_gain(np.geomspace(1e-7, 1.0, 50), LAW)
    assert np.all(gain > 0.0)
    assert np.all(np.diff(gain) < 0.0)


def test_scalar_speed_returns_a_float() -> None:
    assert isinstance(friction_secant_gain(1e-3, LAW), float)


@pytest.mark.parametrize("bad", [-1e-3, float("nan"), float("inf")])
def test_rejects_negative_or_nonfinite_speed(bad: float) -> None:
    with pytest.raises(ValueError, match="speed"):
        friction_secant_gain(np.array([1e-3, bad]), LAW)
