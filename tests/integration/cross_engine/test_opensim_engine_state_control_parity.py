"""Engine set_state/set_control match a model with state and control baked in.

Refs #11344. Until set_state/set_control worked on OpenSim 4.x the force-overlay
parity builders baked the initial coordinate and a ``PrescribedController`` into
the model. The engine API must now give the same force/torque frame.
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
from src.engines.physics_engines.opensim.python.opensim_force_torque import (  # noqa: E402
    OpenSimForceTorqueSource,
)
from src.engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E402
    OpenSimPhysicsEngine,
)
from src.shared.python.force_overlay import WrenchKind  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

Q_RAD = 0.4
CONTROL_NM = 3.0


@pytest.fixture(autouse=True)
def _real_opensim_in_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "opensim", osim)


def _pendulum(baked: bool):
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
    actuator = osim.CoordinateActuator("pin_coord")
    actuator.setName("pin_motor")
    model.addForce(actuator)
    actuator.setCoordinate(joint.updCoordinate())
    actuator.setOptimalForce(1.0)
    if baked:
        joint.upd_coordinates(0).set_default_value(Q_RAD)
        controller = osim.PrescribedController()
        controller.addActuator(actuator)
        controller.prescribeControlForActuator("pin_motor", osim.Constant(CONTROL_NM))
        model.addController(controller)
    model.finalizeConnections()
    return model


def _frame_of_baked_model():
    model = _pendulum(baked=True)
    state = model.initSystem()
    return OpenSimForceTorqueSource(model).sample(state)


def _frame_of_engine(tmp_path: Path):
    path = tmp_path / "synthetic_pendulum.osim"
    _pendulum(baked=False).printToXML(str(path))
    engine = OpenSimPhysicsEngine()
    engine.load_from_path(str(path))
    engine.set_state(np.array([Q_RAD]), np.array([0.0]))
    engine.set_control(np.array([CONTROL_NM]))
    return engine.get_force_torque_frame()


def _assert_same_optional(actual, expected) -> None:
    assert (actual is None) == (expected is None)
    if expected is not None:
        np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_engine_state_and_control_match_the_baked_model(tmp_path: Path) -> None:
    baked = _frame_of_baked_model()
    driven = _frame_of_engine(tmp_path)
    assert baked is not None and driven is not None
    for kind in (WrenchKind.JOINT_REACTION, WrenchKind.JOINT_ACTUATOR):
        (b,) = baked.by_kind(kind)
        (d,) = driven.by_kind(kind)
        for field in ("force_n", "point_m", "torque_nm"):
            _assert_same_optional(getattr(d, field), getattr(b, field))
    (actuator,) = driven.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert np.linalg.norm(actuator.torque_nm) == pytest.approx(CONTROL_NM)
