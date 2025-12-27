from __future__ import annotations

import numpy as np

from double_pendulum_model.physics.triple_pendulum import (
    TriplePendulumDynamics,
    TriplePendulumState,
)


def test_mass_matrix_positive_definite():
    dynamics = TriplePendulumDynamics()
    state = TriplePendulumState(
        theta1=0.1, theta2=-0.2, theta3=0.3, omega1=0.0, omega2=0.0, omega3=0.0
    )
    mass = dynamics.mass_matrix(state)
    eigenvalues = np.linalg.eigvals(mass)
    assert np.all(eigenvalues > 0)


def test_inverse_matches_forward():
    dynamics = TriplePendulumDynamics()
    state = TriplePendulumState(
        theta1=0.2, theta2=-0.3, theta3=0.4, omega1=0.1, omega2=-0.2, omega3=0.05
    )
    desired_acc = (0.5, -0.1, 0.3)
    torques = dynamics.inverse_dynamics(state, desired_acc)
    computed_acc = dynamics.forward_dynamics(state, torques)
    assert np.allclose(computed_acc, desired_acc, atol=1e-6)
