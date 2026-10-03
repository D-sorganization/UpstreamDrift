"""Engine-level tests for OpenSim set_state / set_control (#11344).

Real ``opensim`` (4.x) is required; the module skips cleanly without it. All
models are programmatic ``synthetic_`` fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

osim = pytest.importorskip("opensim", reason="OpenSim not installed")
if not hasattr(osim, "Model"):
    pytest.skip("real opensim is unavailable", allow_module_level=True)

from src.engines.physics_engines.opensim.python import (  # noqa: E402
    opensim_physics_engine as engine_module,
)
from src.engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E402
    OpenSimPhysicsEngine,
)
from src.shared.python.force_overlay import WrenchKind  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

CONTROL_NM = 3.0


@pytest.fixture(autouse=True)
def _real_opensim_in_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other unit modules rebind the engine's ``opensim`` to a mock; undo that."""
    monkeypatch.setattr(engine_module, "opensim", osim)


def _engine(tmp_path: Path, with_actuator: bool = True) -> OpenSimPhysicsEngine:
    model = osim.Model()
    model.setName("synthetic_pendulum")
    body = osim.Body("link", 2.0, osim.Vec3(0, -0.5, 0), osim.Inertia(0.01, 0.01, 0.01))
    model.addBody(body)
    joint = osim.PinJoint(
        "pin",
        model.getGround(),
        osim.Vec3(0, 1.0, 0),
        osim.Vec3(0, 0, 0),
        body,
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
    )
    model.addJoint(joint)
    if with_actuator:
        actuator = osim.CoordinateActuator("pin_coord")
        actuator.setName("pin_motor")
        model.addForce(actuator)
        actuator.setCoordinate(joint.updCoordinate())
        actuator.setOptimalForce(1.0)
    model.initSystem()
    path = tmp_path / "synthetic_pendulum.osim"
    model.printToXML(str(path))
    engine = OpenSimPhysicsEngine()
    engine.load_from_path(str(path))
    return engine


def _actuation(engine: OpenSimPhysicsEngine) -> float:
    model, state = engine._model, engine._state
    model.realizeDynamics(state)
    actuator = osim.CoordinateActuator.safeDownCast(
        model.getForceSet().get("pin_motor")
    )
    return float(actuator.getActuation(state))


def test_set_state_writes_q_and_speeds(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.set_state(np.array([0.4]), np.array([-1.5]))
    q, v = engine.get_state()
    np.testing.assert_allclose(q, [0.4])
    np.testing.assert_allclose(v, [-1.5])


def test_set_state_realizes_velocity_and_keeps_time(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._state.setTime(0.25)
    engine.set_state(np.array([0.4]), np.array([1.0]))
    assert engine.get_time() == pytest.approx(0.25)
    # getSpeedValue needs a Velocity-realized state: it throws if set_state
    # left the state unrealized.
    coordinate = engine._model.getCoordinateSet().get(0)
    assert coordinate.getSpeedValue(engine._state) == pytest.approx(1.0)


def test_set_state_changes_dynamics(tmp_path: Path) -> None:
    """Gravity acceleration about the pin follows the configuration that was set."""
    engine = _engine(tmp_path)

    def udot() -> float:
        engine._model.realizeAcceleration(engine._state)
        return float(engine._state.getUDot().get(0))

    engine.set_state(np.array([0.0]), np.array([0.0]))
    assert udot() == pytest.approx(0.0, abs=1e-9)
    engine.set_state(np.array([np.pi / 2]), np.array([0.0]))
    assert abs(udot()) > 1.0


def test_set_state_rejects_mismatched_lengths(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with pytest.raises(ValueError, match="q"):
        engine.set_state(np.zeros(3), np.zeros(1))
    with pytest.raises(ValueError, match="v"):
        engine.set_state(np.zeros(1), np.zeros(2))


def test_set_control_makes_nonzero_actuation(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    assert _actuation(engine) == pytest.approx(0.0)
    engine.set_control(np.array([CONTROL_NM]))
    assert _actuation(engine) == pytest.approx(CONTROL_NM)


def test_set_control_reaches_the_force_torque_frame(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.set_control(np.array([CONTROL_NM]))
    frame = engine.get_force_torque_frame()
    assert frame is not None
    (w,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert np.linalg.norm(w.torque_nm) == pytest.approx(CONTROL_NM)


def test_set_control_rejects_mismatched_length(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with pytest.raises(ValueError, match="u"):
        engine.set_control(np.zeros(4))


def test_counterfactuals_restore_controls_and_state(tmp_path: Path) -> None:
    """ZTCF zeroes controls temporarily; the caller's control is restored."""
    engine = _engine(tmp_path)
    engine.set_control(np.array([CONTROL_NM]))
    engine.compute_ztcf(np.array([0.3]), np.array([0.0]))
    q, v = engine.get_state()
    np.testing.assert_allclose(q, [0.0])
    np.testing.assert_allclose(v, [0.0])
    assert _actuation(engine) == pytest.approx(CONTROL_NM)


def test_controls_survive_a_later_set_state(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.set_control(np.array([CONTROL_NM]))
    assert _actuation(engine) == pytest.approx(CONTROL_NM)
    engine.set_state(np.array([0.2]), np.array([0.1]))
    assert _actuation(engine) == pytest.approx(CONTROL_NM)


def test_ztcf_zeroes_controls_during_the_call(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.set_control(np.array([CONTROL_NM]))
    zero_torque = engine.compute_ztcf(np.array([0.0]), np.array([0.0]))
    # Hanging at rest with zero control: no acceleration. With the 3 Nm
    # control left on it would be nonzero.
    np.testing.assert_allclose(zero_torque, [0.0], atol=1e-9)


def test_failed_counterfactual_still_restores_state_and_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    engine.set_state(np.array([0.2]), np.array([0.1]))
    engine.set_control(np.array([3.0]))

    def _boom(_state: object) -> None:
        raise RuntimeError("realize failed")

    monkeypatch.setattr(engine._model, "realizeDynamics", _boom, raising=False)
    for call in (
        lambda: engine.compute_ztcf(np.array([0.5]), np.array([0.4])),
        lambda: engine.compute_zvcf(np.array([0.5])),
    ):
        assert call().size == 0
        q, v = engine.get_state()
        assert q[0] == pytest.approx(0.2) and v[0] == pytest.approx(0.1)
        assert engine._controls is not None
        assert engine._controls[0] == pytest.approx(3.0)
