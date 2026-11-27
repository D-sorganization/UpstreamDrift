"""Regression tests for Gemini SCI-10/12/13/14 (#15046)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_cantilever_tip_deflection_matches_euler_bernoulli():
    """Linear Euler–Bernoulli: δ = P L³ / (3 EI) for tip load P (no K_G or CG offset)."""
    from src.shared.python.physics._shaft_data import BeamElement
    from src.shared.python.physics._shaft_fem import FiniteElementShaftModel

    length = 1.0
    ei = 50.0
    p_load = 12.0
    element = BeamElement(
        node_i=0, node_j=1, length=length, EI=ei, mass_per_length=0.0, damping=0.0
    )
    fem = FiniteElementShaftModel(1)
    k_local = fem._element_stiffness_matrix(element)
    # Cantilever: fixed w1, θ1; loaded DOF w2 with tip force P.
    k_free = k_local[2:, 2:]
    delta_expected = p_load * length**3 / (3.0 * ei)
    delta = np.linalg.solve(k_free, np.array([p_load, 0.0]))[0]
    assert delta == pytest.approx(delta_expected, rel=1e-12)


def test_pendulum_scalar_uniform_rod_about_pivot():
    """Uniform rod: I_O = m L²/3 ⇒ tip-equivalent tangential term (I_O/L) q̈ = m L q̈ / 3."""
    from src.shared.python.biomechanics.ztcf import pendulum_tangential_scalar

    mass, length, qddot = 2.0, 0.9, 4.0
    i_o = mass * length**2 / 3.0
    point_mass = pendulum_tangential_scalar(qddot, mass, length)
    assert point_mass == pytest.approx(mass * length * qddot)
    distributed = pendulum_tangential_scalar(
        qddot, mass, length, inertia_about_pivot=i_o
    )
    assert distributed == pytest.approx(mass * length * qddot / 3.0, rel=1e-14)


def test_redundancy_frame_cost_matches_minimized_objective():
    from src.shared.python.myofullbody import redundancy

    active = np.array([800.0, 400.0])
    passive = np.zeros(2)
    moment = np.array([[0.05], [-0.03]])
    tau = np.array([15.0])
    weight = 25.0
    sol = redundancy.solve_frame(active, passive, moment, tau, reserve_weight=weight)
    gain = moment.T * active[None, :]
    offset = tau - moment.T @ passive
    reserve = offset - gain @ sol.activation
    objective = float(sol.activation @ sol.activation + weight * reserve @ reserve)
    assert sol.cost == pytest.approx(objective, rel=1e-14)
    assert sol.cost == pytest.approx(
        redundancy.frame_quadratic_cost(
            sol.activation, active, passive, moment, tau, reserve_weight=weight
        ),
        rel=1e-14,
    )


def test_spatial_jacobian_ordering_conversion_restores_mobility():
    from src.shared.python.loop_nullspace.decomposition import CharacteristicScales
    from src.shared.python.spatial_algebra.operational_space import mobility_matrix
    from src.shared.python.spatial_algebra.spatial_ordering import (
        SpatialOrdering,
        permute_task_jacobian,
    )

    m_diag = np.diag([2.0, 3.0, 1.5, 4.0, 2.5, 3.5])
    rng = np.random.default_rng(11)
    b = rng.normal(size=(6, 6))
    m_mat = b @ b.T + m_diag
    j_lin = rng.normal(size=(6, 6))
    j_ang = permute_task_jacobian(
        j_lin, SpatialOrdering.LINEAR_FIRST, SpatialOrdering.ANGULAR_FIRST
    )
    scales = CharacteristicScales()
    ref = mobility_matrix(j_lin, m_mat, scales=scales)
    converted = mobility_matrix(
        j_ang,
        m_mat,
        scales=scales,
        spatial_ordering=SpatialOrdering.ANGULAR_FIRST,
    )
    np.testing.assert_allclose(converted, ref, rtol=1e-12, atol=1e-12)


def test_force_ellipsoid_tikhonov_damping_softens_singular_axis():
    from src.shared.python.spatial_algebra.operational_space import (
        force_ellipsoid,
        mobility_matrix,
    )

    j = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    m = np.eye(2)
    mob = mobility_matrix(j, m)
    hard = force_ellipsoid(mob, tol=1e-9)
    damped = force_ellipsoid(mob, tol=1e-9, tikhonov=1e-6)
    assert damped.radii[1] < hard.radii[1]
