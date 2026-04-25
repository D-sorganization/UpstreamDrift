import sys
import types

import numpy as np

import src.engines.physics_engines.pendulum.python.golf_swing_physics_engine as golf_engine
from src.engines.physics_engines.pendulum.python.golf_swing_physics_engine import (

    GolfSwingPendulumEngine,
)
import pytest
pytestmark = pytest.mark.unit



def test_rk4_samples_time_varying_torque_profile_at_stage_times(monkeypatch):
    sampled: list[tuple[float, tuple[float, float]]] = []

    package = types.ModuleType("double_pendulum_golf")
    physics = types.ModuleType("double_pendulum_golf.physics")

    class PendulumParams:
        def __init__(self, **kwargs):
            self.params = kwargs

    def equations_of_motion(state, time, params, torque_func):
        del state, params
        tau = torque_func(time)
        sampled.append((time, tau))
        return np.array([0.0, 0.0, tau[0], tau[1]])

    physics.PendulumParams = PendulumParams
    physics.equations_of_motion = equations_of_motion
    package.physics = physics

    monkeypatch.setitem(sys.modules, "double_pendulum_golf", package)
    monkeypatch.setitem(sys.modules, "double_pendulum_golf.physics", physics)
    monkeypatch.setattr(golf_engine, "_TOOLS_PENDULUM_AVAILABLE", None)

    engine = GolfSwingPendulumEngine()
    engine.set_control_profile(lambda t: (t, 2.0 * t))

    engine.step(0.2)

    assert [round(time, 10) for time, _ in sampled] == [0.0, 0.1, 0.1, 0.2]
    assert sampled == [
        (0.0, (0.0, 0.0)),
        (0.1, (0.1, 0.2)),
        (0.1, (0.1, 0.2)),
        (0.2, (0.2, 0.4)),
    ]
    np.testing.assert_allclose(engine.get_state()[1], [0.02, 0.04])
