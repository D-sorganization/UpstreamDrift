import math

import numpy as np
from bunkershot3d.kinematics.coupling import CoupledDoublePendulum, CoSimulator


def test_coupled_pendulum() -> None:
    pendulum = CoupledDoublePendulum()
    pos, quat, lvel, avel = pendulum.get_clubhead_pose()

    assert pos.shape == (3,)
    assert quat.shape == (4,)

    # Test step with zero wrench (gravity affects omega, so we just check it runs)
    zero_wrench = (np.zeros(3), np.zeros(3))
    pendulum.step(0.01, zero_wrench)

    assert np.isclose(pendulum.time, 0.01)


def test_cosimulator() -> None:
    pendulum = CoupledDoublePendulum()

    # Mock backend
    class MockBackend:
        pass

    cosim = CoSimulator(pendulum, MockBackend())

    # Step macro
    force, torque = cosim.step(0.01)

    assert force.shape == (3,)
    assert torque.shape == (3,)


def test_pure_force_torque_mapping() -> None:
    """Wrench->joint-torque mapping must equal the manipulator Jacobian transpose.

    Regression for #6987. For a planar 2R chain whose joints both rotate about
    the world y-axis, the manipulator Jacobian has angular row [1, 1]; therefore
    an external moment Ty projects onto BOTH joint torques. The position rows
    give the force contributions. This is the correct J^T wrench projection, NOT
    a double-count, so we lock the convention in here.
    """
    pendulum = CoupledDoublePendulum()
    # Use a non-trivial configuration so the Jacobian is fully populated.
    pendulum.state.theta1 = 0.3
    pendulum.state.theta2 = -0.4
    t1 = pendulum.state.theta1
    t2 = pendulum.state.theta2
    l1 = pendulum.params.upper_segment.length_m
    l2 = pendulum.params.lower_segment.length_m

    dx_dt1 = l1 * math.cos(t1) + l2 * math.cos(t1 + t2)
    dz_dt1 = l1 * math.sin(t1) + l2 * math.sin(t1 + t2)
    dx_dt2 = l2 * math.cos(t1 + t2)
    dz_dt2 = l2 * math.sin(t1 + t2)

    # Pure force (no moment): tau = J_v^T @ F.
    Fx, Fz = 7.0, -3.0
    pendulum.step(0.0, (np.array([Fx, 0.0, Fz]), np.zeros(3)))
    assert pendulum.external_tau1 == np.float64(Fx * dx_dt1 + Fz * dz_dt1)
    assert pendulum.external_tau2 == np.float64(Fx * dx_dt2 + Fz * dz_dt2)

    # Pure moment about y: angular Jacobian row is [1, 1] -> Ty hits both joints.
    Ty = 2.5
    pendulum.step(0.0, (np.zeros(3), np.array([0.0, Ty, 0.0])))
    assert pendulum.external_tau1 == np.float64(Ty)
    assert pendulum.external_tau2 == np.float64(Ty)


def test_zero_stiffness_regression() -> None:
    """
    Validate that, in the limit of zero sand stiffness (zero external wrench),
    the coupled system reproduces the free pendulum trajectory exactly.
    """
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        DoublePendulumDynamics,
        DoublePendulumState,
        DoublePendulumParameters,
    )

    coupled_pendulum = CoupledDoublePendulum()

    # Initialize native pendulum identically
    native_state = DoublePendulumState(theta1=0.0, theta2=0.0, omega1=10.0, omega2=15.0)
    native_params = DoublePendulumParameters.default()

    def zero_forcing(t: float, state: DoublePendulumState) -> float:
        return 0.0

    native_dynamics = DoublePendulumDynamics(
        parameters=native_params, forcing_functions=(zero_forcing, zero_forcing)
    )

    dt = 0.001
    zero_wrench = (np.zeros(3), np.zeros(3))
    time = 0.0

    for _ in range(50):
        # Step coupled
        coupled_pendulum.step(dt, zero_wrench)

        # Step native
        native_state = native_dynamics.step(time, native_state, dt)
        time += dt

        # Compare states
        assert np.isclose(coupled_pendulum.state.theta1, native_state.theta1)
        assert np.isclose(coupled_pendulum.state.theta2, native_state.theta2)
        assert np.isclose(coupled_pendulum.state.omega1, native_state.omega1)
        assert np.isclose(coupled_pendulum.state.omega2, native_state.omega2)
