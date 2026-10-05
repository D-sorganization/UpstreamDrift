"""Tests for the vectorized planar-chain inertial regressor (MOSAIC fixture).

The regressor is checked against the closed-form double-pendulum equations of
motion, so a wrong sign or a missing coupling term fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.planar_chain import (
    GRAVITY_DOWN,
    PlanarChain,
    PlanarMarkerSet,
)

pytestmark = pytest.mark.unit

G0 = 9.81


def _double_pendulum() -> tuple[PlanarChain, np.ndarray, dict[str, float]]:
    m1, m2 = 1.3, 0.7
    l1, l2 = 0.9, 0.6
    lc1, lc2 = 0.45, 0.25
    i1c, i2c = 0.08, 0.02
    chain = PlanarChain(
        link_lengths=np.array([l1, l2]), actuated=np.array([True, True])
    )
    pi = np.array(
        [m1, m1 * lc1, 0.0, i1c + m1 * lc1**2, m2, m2 * lc2, 0.0, i2c + m2 * lc2**2]
    )
    consts = {
        "m1": m1,
        "m2": m2,
        "l1": l1,
        "lc1": lc1,
        "lc2": lc2,
        "i1o": pi[3],
        "i2o": pi[7],
    }
    return chain, pi, consts


def _closed_form(
    q: np.ndarray, v: np.ndarray, c: dict[str, float]
) -> tuple[np.ndarray, ...]:
    q1, q2 = q
    v1, v2 = v
    m2, l1, lc2 = c["m2"], c["l1"], c["lc2"]
    k = m2 * l1 * lc2
    mass = np.array(
        [
            [
                c["i1o"] + c["i2o"] + m2 * l1**2 + 2 * k * np.cos(q2),
                c["i2o"] + k * np.cos(q2),
            ],
            [c["i2o"] + k * np.cos(q2), c["i2o"]],
        ]
    )
    coriolis = np.array(
        [-k * np.sin(q2) * (2 * v1 * v2 + v2**2), k * np.sin(q2) * v1**2]
    )
    gravity = np.array(
        [
            (c["m1"] * c["lc1"] + m2 * l1) * G0 * np.cos(q1)
            + m2 * lc2 * G0 * np.cos(q1 + q2),
            m2 * lc2 * G0 * np.cos(q1 + q2),
        ]
    )
    return mass, coriolis, gravity


def test_regressor_matches_closed_form_double_pendulum() -> None:
    chain, pi, consts = _double_pendulum()
    rng = np.random.default_rng(1)
    q = rng.uniform(-2, 2, size=(7, 2))
    v = rng.normal(size=(7, 2))
    a = rng.normal(size=(7, 2))
    tau = chain.regressor(q, v, a) @ pi
    for n in range(7):
        mass, cor, grav = _closed_form(q[n], v[n], consts)
        np.testing.assert_allclose(
            tau[n], mass @ a[n] + cor + grav, rtol=1e-10, atol=1e-10
        )


def test_mass_matrix_and_bias_from_regressor() -> None:
    chain, pi, consts = _double_pendulum()
    q = np.array([[0.3, -1.1], [1.0, 0.4]])
    v = np.array([[0.5, -0.2], [-1.0, 2.0]])
    mass = chain.mass_matrix(q, pi)
    bias = chain.bias(q, v, pi)
    for n in range(2):
        m_ref, c_ref, g_ref = _closed_form(q[n], v[n], consts)
        np.testing.assert_allclose(mass[n], m_ref, atol=1e-12)
        np.testing.assert_allclose(bias[n], c_ref + g_ref, atol=1e-12)


def test_drift_decomposition_is_ztcf_and_zvcf() -> None:
    """ZTCF = -M^{-1} h(q,v); ZVCF = -M^{-1} g(q); both come from the regressor."""
    chain, pi, consts = _double_pendulum()
    q = np.array([[0.7, 0.9]])
    v = np.array([[1.5, -0.5]])
    drift = chain.drift_decomposition(q, v, pi)
    m_ref, c_ref, g_ref = _closed_form(q[0], v[0], consts)
    np.testing.assert_allclose(
        drift.ztcf[0], -np.linalg.solve(m_ref, c_ref + g_ref), atol=1e-10
    )
    np.testing.assert_allclose(
        drift.zvcf[0], -np.linalg.solve(m_ref, g_ref), atol=1e-10
    )
    np.testing.assert_allclose(
        drift.velocity_part[0], drift.ztcf[0] - drift.zvcf[0], atol=1e-12
    )


def test_forward_dynamics_round_trip_with_inverse_dynamics() -> None:
    chain, pi, _ = _double_pendulum()
    q = np.array([[0.2, 0.1], [-0.4, 1.3]])
    v = np.array([[0.0, 0.3], [2.0, -1.0]])
    u = np.array([[1.0, -0.5], [0.0, 0.25]])
    a = chain.forward_dynamics(q, v, u, pi)
    tau = chain.regressor(q, v, a) @ pi
    np.testing.assert_allclose(tau, u, atol=1e-10)


def test_unactuated_rows_are_selected_by_mask() -> None:
    chain = PlanarChain(
        link_lengths=np.array([1.0, 1.0, 0.5]), actuated=np.array([False, True, True])
    )
    assert chain.n_links == 3
    assert chain.n_parameters == 12
    np.testing.assert_array_equal(chain.unactuated_indices, [0])
    np.testing.assert_array_equal(chain.input_matrix.shape, (3, 2))


def test_markers_forward_kinematics_and_analytic_jacobians() -> None:
    chain = PlanarChain(
        link_lengths=np.array([0.8, 0.5]), actuated=np.array([True, True])
    )
    markers = PlanarMarkerSet(
        link_index=np.array([0, 1, 1]),
        offsets=np.array([[0.4, 0.05], [0.1, 0.0], [0.5, -0.02]]),
    )
    q = np.array([[0.3, -0.7], [1.2, 0.4]])
    pos = chain.marker_positions(q, markers)
    assert pos.shape == (2, 3, 2)
    # marker 1 sits 0.1 along link 2: p1 + 0.1 e_1
    phi1 = q[:, 0] + q[:, 1]
    p1 = 0.8 * np.stack([np.cos(q[:, 0]), np.sin(q[:, 0])], axis=-1)
    expected = p1 + 0.1 * np.stack([np.cos(phi1), np.sin(phi1)], axis=-1)
    np.testing.assert_allclose(pos[:, 1], expected, atol=1e-12)

    jq, jl = chain.marker_jacobians(q, markers)
    eps = 1e-7
    for k in range(2):
        dq = np.zeros_like(q)
        dq[:, k] = eps
        fd = (chain.marker_positions(q + dq, markers) - pos) / eps
        np.testing.assert_allclose(jq[..., k], fd, atol=1e-5)
        bumped = PlanarChain(
            link_lengths=chain.link_lengths + eps * np.eye(2)[k],
            actuated=chain.actuated,
        )
        fd_l = (bumped.marker_positions(q, markers) - pos) / eps
        np.testing.assert_allclose(jl[..., k], fd_l, atol=1e-5)


def test_rk4_rollout_conserves_energy_without_input() -> None:
    chain, pi, _ = _double_pendulum()
    x0 = np.array([[0.5, 0.2, 0.0, 0.0]])
    dt, steps = 1e-3, 400
    u = np.zeros((1, steps, 2))
    q, v = chain.rollout(x0, u, pi, dt)
    assert q.shape == (1, steps + 1, 2)
    e0 = chain.total_energy(q[:, 0], v[:, 0], pi)
    e1 = chain.total_energy(q[:, -1], v[:, -1], pi)
    np.testing.assert_allclose(e1, e0, rtol=1e-6)


def test_gravity_constant_points_down() -> None:
    np.testing.assert_allclose(GRAVITY_DOWN, [0.0, -9.81])


def test_rejects_inconsistent_shapes() -> None:
    chain, pi, _ = _double_pendulum()
    with pytest.raises(ContractViolationError):
        chain.regressor(np.zeros((3, 2)), np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ContractViolationError):
        PlanarChain(link_lengths=np.array([1.0, -1.0]), actuated=np.array([True, True]))
