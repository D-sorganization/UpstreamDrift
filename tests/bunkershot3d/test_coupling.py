import pytest
import numpy as np
from bunkershot3d.kinematics.coupling import MockDoublePendulum, CoSimulator


def test_mock_pendulum() -> None:
    pendulum = MockDoublePendulum()
    pos, quat, lvel, avel = pendulum.get_clubhead_pose()

    assert pos.shape == (3,)
    assert quat.shape == (4,)

    # Test step with zero wrench
    zero_wrench = (np.zeros(3), np.zeros(3))
    pendulum.step(0.01, zero_wrench)

    assert pendulum.time == 0.01


def test_cosimulator() -> None:
    pendulum = MockDoublePendulum()

    # Mock backend
    class MockBackend:
        pass

    cosim = CoSimulator(pendulum, MockBackend())

    # Store initial omega
    initial_omega1 = pendulum.omega1

    # Step macro
    force, torque = cosim.step(0.01)

    # Omega should decrease because of the mock resistance
    assert pendulum.omega1 < initial_omega1
    assert force.shape == (3,)
    assert torque.shape == (3,)
