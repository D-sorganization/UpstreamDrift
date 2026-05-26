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
