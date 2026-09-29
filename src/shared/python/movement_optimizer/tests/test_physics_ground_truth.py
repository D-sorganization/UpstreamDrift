# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Ground-truth physics validation (issue #493).

Unlike the parity/smoke tests elsewhere (which only prove two code paths agree
or that arrays are finite), these tests assert that the dynamics reproduce
hand-computed reference values. A systematic error in the equations of motion
(e.g. a double-counted parallel-axis inertia term, #490) is invisible to parity
tests but fails here.

Markers:
    ``validates_physics`` — physical-correctness assertions against hand
    calculations, not internal consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.constants import (
    ARM_RADIUS_OF_GYRATION_FRAC,
    RADIUS_OF_GYRATION_FRAC,
)
from movement_optimizer.models.body_model import BodyModel
from movement_optimizer.models.lagrangian_dynamics import LagrangianDynamics

pytestmark = pytest.mark.validates_physics


def _squat_dynamics() -> tuple[LagrangianDynamics, BodyModel]:
    body = BodyModel(height=1.8, body_mass=80.0)
    dyn = LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), 0.0)
    return dyn, body


def test_static_gravity_torque_matches_hand_calculation() -> None:
    """At a held pose (qd=qdd=0) joint torques equal the gravitational moment.

    The gravitational moment about each proximal joint is
    ``g * (m_i * d_i + sum_{j>i} m_j * L_i) * sin(q_i)`` for the standing
    (non-supine) chain. This is independent of the inertia convention, so it
    pins the gravity model directly.
    """
    dyn, body = _squat_dynamics()
    m, length, d, g = body.m_squat, body.L, body.d, body.g
    q = np.array([0.30, 0.20, 0.10])
    zero = np.zeros(3)

    tau = dyn.inverse_dynamics(q, zero, zero)

    g0 = g * (m[0] * d[0] + (m[1] + m[2]) * length[0])
    g1 = g * (m[1] * d[1] + m[2] * length[1])
    g2 = g * (m[2] * d[2])
    expected = np.array([g0 * np.sin(q[0]), g1 * np.sin(q[1]), g2 * np.sin(q[2])])
    np.testing.assert_allclose(tau, expected, rtol=1e-12)


def test_diagonal_inertia_uses_centroidal_convention() -> None:
    """Pure proximal acceleration yields the centroidal-convention inertia (#490).

    Holding the pose at zero velocity and accelerating only the first joint, the
    first-joint torque must equal ``M00 * qdd0 + gravity0`` where ``M00`` is the
    physically correct diagonal inertia built from *centroidal* segment inertia
    (``I_com = m*(rho*L)^2``) plus a single parallel-axis term ``m0*d0^2``. A
    proximal-axis inertia passed into the dynamics would double-count
    ``m0*d0^2`` and inflate this torque.
    """
    dyn, body = _squat_dynamics()
    m, length, d, g = body.m_squat, body.L, body.d, body.g
    rho = np.array(
        [
            RADIUS_OF_GYRATION_FRAC["lower_leg"],
            RADIUS_OF_GYRATION_FRAC["upper_leg"],
            RADIUS_OF_GYRATION_FRAC["trunk"],
        ]
    )
    i_com = m * (rho * length) ** 2

    q = np.array([0.30, 0.20, 0.10])
    qdd = np.array([1.0, 0.0, 0.0])
    tau = dyn.inverse_dynamics(q, np.zeros(3), qdd)

    m00 = i_com[0] + m[0] * d[0] ** 2 + (m[1] + m[2]) * length[0] ** 2
    g0 = g * (m[0] * d[0] + (m[1] + m[2]) * length[0])
    expected_tau0 = m00 * 1.0 + g0 * np.sin(q[0])
    np.testing.assert_allclose(tau[0], expected_tau0, rtol=1e-12)

    # Guard against the double-count regression explicitly: a proximal-axis
    # build would add an extra m0*d0^2 to the inertia term.
    double_counted = expected_tau0 + m[0] * d[0] ** 2
    assert not np.isclose(tau[0], double_counted, rtol=1e-9)


def test_bodymodel_stores_centroidal_inertia() -> None:
    """BodyModel.I_squat/I_deadlift are centroidal (I_com), not proximal (#490)."""
    body = BodyModel(height=1.75, body_mass=75.0)
    rho = np.array(
        [
            RADIUS_OF_GYRATION_FRAC["lower_leg"],
            RADIUS_OF_GYRATION_FRAC["upper_leg"],
            RADIUS_OF_GYRATION_FRAC["trunk"],
        ]
    )
    expected_squat = body.m_squat * (rho * body.L) ** 2
    expected_deadlift = body.m_deadlift * (rho * body.L) ** 2
    np.testing.assert_allclose(body.I_squat, expected_squat, rtol=1e-12)
    np.testing.assert_allclose(body.I_deadlift, expected_deadlift, rtol=1e-12)


def test_bench_inertia_uses_winter_radius_of_gyration() -> None:
    """BenchPressModel inertia is centroidal with Winter arm rho values (#490)."""
    from movement_optimizer.models.bench_press_model import BenchPressModel

    body = BodyModel(height=1.8, body_mass=80.0)
    bench = BenchPressModel(body)
    rho_arm = np.array(
        [
            ARM_RADIUS_OF_GYRATION_FRAC["upper_arm"],
            ARM_RADIUS_OF_GYRATION_FRAC["forearm"],
            ARM_RADIUS_OF_GYRATION_FRAC["hand"],
        ]
    )
    expected = bench.m * (rho_arm * bench.L) ** 2
    np.testing.assert_allclose(bench.I, expected, rtol=1e-12)
    # The old uniform-rod COM-frame value (1/12)*m*L^2 used a larger generic
    # rho (~0.289) and is no longer used.
    old_rod = (1.0 / 12.0) * bench.m * bench.L**2
    assert not np.allclose(bench.I, old_rod)


def test_single_pendulum_torque_matches_analytic() -> None:
    """A degenerate chain (segments 2,3 massless) reduces to a single pendulum.

    With ``m1=m2=load=0`` the first link is an isolated physical pendulum about
    its proximal joint. Inverse dynamics at qd=0 must give
    ``tau0 = (I_com0 + m0*d0^2)*qdd0 + m0*g*d0*sin(q0)`` — the textbook physical
    pendulum equation.
    """
    body = BodyModel(height=1.8, body_mass=80.0)
    m = body.m_squat.copy()
    inertia = body.I_squat.copy()
    # Zero out distal segments to isolate link 0.
    m[1] = m[2] = 0.0
    inertia[1] = inertia[2] = 0.0
    dyn = LagrangianDynamics(body, m, inertia, 0.0)

    q = np.array([0.4, 0.0, 0.0])
    qdd = np.array([2.0, 0.0, 0.0])
    tau = dyn.inverse_dynamics(q, np.zeros(3), qdd)

    i_prox0 = inertia[0] + m[0] * body.d[0] ** 2
    expected = i_prox0 * 2.0 + m[0] * body.g * body.d[0] * np.sin(q[0])
    np.testing.assert_allclose(tau[0], expected, rtol=1e-12)
