from __future__ import annotations

import numpy as np  # noqa: TID253
import pytest
from double_pendulum_model.physics.triple_pendulum import (
    TriplePendulumDynamics,
    TriplePendulumState,
)

pytestmark = pytest.mark.unit


def test_triple_pendulum_mass_matrix_positive_definite() -> None:
    dynamics = TriplePendulumDynamics()
    state = TriplePendulumState(
        theta1=0.1, theta2=-0.2, theta3=0.3, omega1=0.0, omega2=0.0, omega3=0.0
    )
    mass = dynamics.mass_matrix(state)
    eigenvalues = np.linalg.eigvals(mass)
    assert np.all(eigenvalues > 0)


def test_inverse_matches_forward() -> None:
    dynamics = TriplePendulumDynamics()
    state = TriplePendulumState(
        theta1=0.2, theta2=-0.3, theta3=0.4, omega1=0.1, omega2=-0.2, omega3=0.05
    )
    desired_acc = (0.5, -0.1, 0.3)
    torques = dynamics.inverse_dynamics(state, desired_acc)
    computed_acc = dynamics.forward_dynamics(state, torques)
    assert np.allclose(computed_acc, desired_acc, atol=1e-6)


def test_polynomial_profile_cached_eval_matches_poly1d() -> None:
    """Cached poly/deriv must equal a fresh np.poly1d evaluation (#7559)."""
    from double_pendulum_model.physics.triple_pendulum import PolynomialProfile

    coeffs = (2.0, -3.0, 0.5, 1.0)  # 2t^3 - 3t^2 + 0.5t + 1
    profile = PolynomialProfile(coeffs)
    poly = np.poly1d(coeffs)
    deriv = np.poly1d(np.polyder(coeffs))

    for t in (-2.0, 0.0, 0.25, 1.0, 3.5):
        assert profile.omega(t) == float(poly(t))
        assert profile.alpha(t) == float(deriv(t))
