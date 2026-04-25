import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
pytestmark = pytest.mark.unit



def test_pinocchio_rk4_uses_stage_states(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.engines.physics_engines.pinocchio.python import pinocchio_physics_engine

    engine = pinocchio_physics_engine.PinocchioPhysicsEngine()
    engine.model = MagicMock(nq=1, nv=1)
    engine.data = MagicMock()
    engine.q = np.array([0.0])
    engine.v = np.array([2.0])
    engine.tau = np.array([0.0])

    mock_pin = MagicMock()
    mock_pin.aba.side_effect = [
        np.array([1.0]),
        np.array([2.0]),
        np.array([3.0]),
        np.array([4.0]),
    ]
    mock_pin.integrate.side_effect = lambda model, q, dq: q + dq
    monkeypatch.setattr(pinocchio_physics_engine, "pin", mock_pin, raising=False)

    engine.step(0.1)

    assert mock_pin.aba.call_count == 4
    np.testing.assert_allclose(engine.q, np.array([0.21]))
    np.testing.assert_allclose(engine.v, np.array([2.25]))
    np.testing.assert_allclose(engine.a, np.array([4.0]))


def test_pinocchio_semi_implicit_integrator_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.engines.physics_engines.pinocchio.python import pinocchio_physics_engine

    engine = pinocchio_physics_engine.PinocchioPhysicsEngine()
    engine.model = MagicMock(nq=1, nv=1)
    engine.data = MagicMock()
    engine.q = np.array([0.0])
    engine.v = np.array([0.0])
    engine.tau = np.array([0.0])

    mock_pin = MagicMock()
    mock_pin.aba.return_value = np.array([1.0])
    mock_pin.integrate.side_effect = lambda model, q, dq: q + dq
    monkeypatch.setattr(pinocchio_physics_engine, "pin", mock_pin, raising=False)

    engine.step(0.1, integrator="semi_implicit")

    assert mock_pin.aba.call_count == 1
    np.testing.assert_allclose(engine.q, np.array([0.01]))
    np.testing.assert_allclose(engine.v, np.array([0.1]))


@pytest.mark.parametrize("bad_integrator", ["RK4", "rkk4", ""])
def test_pinocchio_rejects_unsupported_integrators(
    monkeypatch: pytest.MonkeyPatch,
    bad_integrator: str,
) -> None:
    from src.engines.physics_engines.pinocchio.python import pinocchio_physics_engine

    engine = pinocchio_physics_engine.PinocchioPhysicsEngine()
    engine.model = MagicMock(nq=1, nv=1)
    engine.data = MagicMock()
    engine.q = np.array([0.0])
    engine.v = np.array([0.0])
    engine.tau = np.array([0.0])

    mock_pin = MagicMock()
    monkeypatch.setattr(pinocchio_physics_engine, "pin", mock_pin, raising=False)

    with pytest.raises(ValueError, match="Unsupported Pinocchio integrator"):
        engine.step(0.1, integrator=bad_integrator)  # type: ignore[arg-type]

    assert mock_pin.aba.call_count == 0
    assert mock_pin.integrate.call_count == 0


def test_golf_pendulum_rk4_samples_torque_profile_at_stage_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.engines.physics_engines.pendulum.python import (
        golf_swing_physics_engine,
    )

    sampled: list[tuple[float, tuple[float, float]]] = []

    def equations_of_motion(
        state: np.ndarray,
        time: float,
        params: Any,
        torque_func: Any,
    ) -> np.ndarray:
        del state, params
        sampled.append((time, torque_func(time)))
        return np.zeros(4)

    package = ModuleType("double_pendulum_golf")
    physics = ModuleType("double_pendulum_golf.physics")
    physics.equations_of_motion = equations_of_motion  # type: ignore[attr-defined]
    physics.PendulumParams = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "double_pendulum_golf", package)
    monkeypatch.setitem(sys.modules, "double_pendulum_golf.physics", physics)
    monkeypatch.setattr(golf_swing_physics_engine, "_TOOLS_PENDULUM_AVAILABLE", None)

    engine = golf_swing_physics_engine.GolfSwingPendulumEngine()
    engine._is_initialized = True
    engine._pendulum_params = object()
    engine.time = 1.0
    engine.set_control_profile(lambda t: (t, -t))

    engine.step(0.2)

    assert sampled == [
        (1.0, (1.0, -1.0)),
        (1.1, (1.1, -1.1)),
        (1.1, (1.1, -1.1)),
        (1.2, (1.2, -1.2)),
    ]


def test_golf_pendulum_reset_clears_torque_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.engines.physics_engines.pendulum.python import (
        golf_swing_physics_engine,
    )

    sampled: list[tuple[float, tuple[float, float]]] = []

    def equations_of_motion(
        state: np.ndarray,
        time: float,
        params: Any,
        torque_func: Any,
    ) -> np.ndarray:
        del state, params
        sampled.append((time, torque_func(time)))
        return np.zeros(4)

    package = ModuleType("double_pendulum_golf")
    physics = ModuleType("double_pendulum_golf.physics")
    physics.equations_of_motion = equations_of_motion  # type: ignore[attr-defined]
    physics.PendulumParams = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "double_pendulum_golf", package)
    monkeypatch.setitem(sys.modules, "double_pendulum_golf.physics", physics)
    monkeypatch.setattr(golf_swing_physics_engine, "_TOOLS_PENDULUM_AVAILABLE", None)

    engine = golf_swing_physics_engine.GolfSwingPendulumEngine()
    engine._is_initialized = True
    engine._pendulum_params = object()
    engine.time = 1.0
    engine.set_control_profile(lambda _t: (7.0, -7.0))
    engine.reset()
    engine.set_control(np.array([2.0, -3.0]))

    assert engine._torque_profile is None

    engine.step(0.2)

    assert sampled == [
        (0.0, (2.0, -3.0)),
        (0.1, (2.0, -3.0)),
        (0.1, (2.0, -3.0)),
        (0.2, (2.0, -3.0)),
    ]
