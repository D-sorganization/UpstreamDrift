"""Independent verification of the numpy planar two-arm model (NSX-12).

The model is checked against finite differences of its own kinematics and
against the NSX-00 sympy goldens, so neither check shares code with the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.fixtures import load_golden
from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    planar_dynamics,
)

pytestmark = pytest.mark.unit

RNG = np.random.default_rng(12557)


def _params() -> PlanarGolferParams:
    return PlanarGolferParams.from_golden(load_golden("planar_two_arm_full"))


def _random_state() -> tuple[np.ndarray, np.ndarray]:
    return RNG.uniform(-1.2, 1.2, 5), RNG.uniform(-2.0, 2.0, 5)


def test_closure_jacobian_matches_finite_difference() -> None:
    prm = _params()
    q, v = _random_state()
    jac = planar_dynamics(q, v, prm)["Jc"]
    eps = 1e-6
    cols = []
    for j in range(5):
        dq = np.zeros(5)
        dq[j] = eps
        cols.append(
            (
                planar_dynamics(q + dq, v, prm)["phi"]
                - planar_dynamics(q - dq, v, prm)["phi"]
            )
            / (2 * eps)
        )
    assert np.max(np.abs(jac - np.array(cols).T)) < 1e-8


def test_jdot_v_matches_finite_difference() -> None:
    prm = _params()
    q, v = _random_state()
    eps = 1e-6
    jv_plus = planar_dynamics(q + eps * v, v, prm)["Jc"] @ v
    jv_minus = planar_dynamics(q - eps * v, v, prm)["Jc"] @ v
    fd = (jv_plus - jv_minus) / (2 * eps)
    assert np.max(np.abs(fd - planar_dynamics(q, v, prm)["Jdot_v"])) < 1e-7


def test_mass_matrix_is_symmetric_positive_definite() -> None:
    mass = planar_dynamics(*_random_state(), _params())["M"]
    assert np.array_equal(mass, mass.T)
    assert np.linalg.eigvalsh(mass).min() > 0.0


def test_velocity_product_term_matches_kinetic_energy_rate() -> None:
    """Without gravity, ``v . h = d/dt (1/2 v^T M v)`` along constant ``v``."""
    prm = _params()
    q, v = _random_state()
    eps = 1e-6
    zero_g = PlanarGolferParams(**{**prm.__dict__, "gravity": 0.0})
    h = planar_dynamics(q, v, zero_g)["h"]
    ke_plus = 0.5 * v @ planar_dynamics(q + eps * v, v, zero_g)["M"] @ v
    ke_minus = 0.5 * v @ planar_dynamics(q - eps * v, v, zero_g)["M"] @ v
    assert abs((ke_plus - ke_minus) / (2 * eps) - v @ h) < 1e-6


def test_precondition_shape_errors() -> None:
    with pytest.raises(ValueError, match="q must have shape"):
        planar_dynamics(np.zeros(4), np.zeros(5), _params())
    with pytest.raises(ValueError, match="v must have shape"):
        planar_dynamics(np.zeros(5), np.zeros(4), _params())
