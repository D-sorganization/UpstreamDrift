"""Per-engine builders for the force/torque parity suite (FTO-21, #11306).

Each builder turns the shared ``synthetic_`` parameters into that engine's model and
returns the provider frame, so the test sees only ``ForceTorqueFrame``. Builders
import their engine lazily and raise ``pytest.skip`` when it is absent. Only the
engines whose provider is merged have a builder; ``PENDING_PROVIDERS`` names the
rest so the suite skips them with an explicit reason.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np
import pytest

from src.shared.python.force_overlay import ForceTorqueFrame, ForceTorqueSeries

from .force_overlay_fixtures import (
    BALL_MASS_KG,
    BALL_RADIUS_M,
    PENDULUM,
    PendulumParams,
    inverted_expected,
)

#: Engines parameterized in the suite whose provider has not merged yet, with the
#: child issue that delivers it. The skip reason names the issue.
PENDING_PROVIDERS: dict[str, str] = {
    "mujoco": "FTO-9 #11294",
    "drake": "FTO-11 #11296",
    "pinocchio": "FTO-13 #11298",
}

_HUNT_CROSSLEY_XML = """<HuntCrossleyForce name="hc">
<appliesForce>true</appliesForce>
<HuntCrossleyForce::ContactParametersSet name="contact_parameters"><objects>
<HuntCrossleyForce::ContactParameters>
<geometry>floor ball_cs</geometry>
<stiffness>1e6</stiffness><dissipation>1.0</dissipation>
<static_friction>0.8</static_friction><dynamic_friction>0.6</dynamic_friction>
<viscous_friction>0.01</viscous_friction>
</HuntCrossleyForce::ContactParameters></objects><groups/>
</HuntCrossleyForce::ContactParametersSet>
<transition_velocity>0.01</transition_velocity>
</HuntCrossleyForce>"""

# --- OpenSim (Y-up ground; the provider rotates to the Z-up world) -----------------


def _osim() -> object:
    osim = pytest.importorskip("opensim", reason="OpenSim not installed")
    if not hasattr(osim, "Model"):
        pytest.skip("real opensim is unavailable")
    return osim


def _opensim_frame(model: object, state: object) -> ForceTorqueFrame:
    from src.engines.physics_engines.opensim.python.opensim_force_torque import (
        OpenSimForceTorqueSource,
    )

    return OpenSimForceTorqueSource(model).sample(state)


def _opensim_pendulum(osim: object, p: PendulumParams, *, actuated: bool):
    """Link on a +z pin at height ``pivot_height_m``; COM ``length_m / 2`` down."""
    model = osim.Model()
    model.setName("synthetic_pendulum")
    body = osim.Body(
        "link",
        p.mass_kg,
        osim.Vec3(0, -p.com_offset_m, 0),
        osim.Inertia(0.01, 0.01, 0.01),
    )
    model.addBody(body)
    joint = osim.PinJoint(
        "pin",
        model.getGround(),
        osim.Vec3(0, p.pivot_height_m, 0),
        osim.Vec3(0, 0, 0),
        body,
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
    )
    model.addJoint(joint)
    if actuated:
        actuator = osim.CoordinateActuator("pin_coord")
        actuator.setName("pin_motor")
        model.addForce(actuator)
        actuator.setCoordinate(joint.updCoordinate())
        actuator.setOptimalForce(1.0)
    return model, model.initSystem()


def opensim_hanging(p: PendulumParams = PENDULUM) -> ForceTorqueFrame:
    osim = _osim()
    model, state = _opensim_pendulum(osim, p, actuated=False)
    return _opensim_frame(model, state)


def opensim_inverted(p: PendulumParams = PENDULUM) -> ForceTorqueFrame:
    """Link standing at ``hold_angle_rad`` from vertical, held by an exact actuation.

    Rotation about the +z pin tips the COM toward -x (OpenSim), which is the world
    ``rod_direction_world`` direction after the Y-up to Z-up rotation.
    """
    osim = _osim()
    model, state = _opensim_pendulum(osim, p, actuated=True)
    # COM starts hanging (-y); theta - pi puts it theta from straight up toward -x.
    model.getCoordinateSet().get(0).setValue(state, p.hold_angle_rad - np.pi)
    actuator = osim.CoordinateActuator.safeDownCast(model.getForceSet().get(0))
    model.realizeDynamics(state)
    actuator.overrideActuation(state, True)
    # The actuator acts about the world axis in the expected frame; recover the
    # signed magnitude about the pin (+z of OpenSim == world -y).
    world_axis = np.array([0.0, -1.0, 0.0])
    torque = np.asarray(inverted_expected(p)["actuator_torque_nm"])
    actuator.setOverrideActuation(state, float(torque @ world_axis))
    return _opensim_frame(model, state)


def opensim_resting_ball(directory: Path) -> ForceTorqueFrame:
    """Free ball with a Hunt-Crossley sphere resting on a half-space floor."""
    osim = _osim()
    model = osim.Model()
    model.setName("synthetic_ball")
    body = osim.Body(
        "ball", BALL_MASS_KG, osim.Vec3(0, 0, 0), osim.Inertia(0.01, 0.01, 0.01)
    )
    model.addBody(body)
    free = osim.FreeJoint(
        "free",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    model.addJoint(free)
    model.addContactGeometry(
        osim.ContactHalfSpace(
            osim.Vec3(0), osim.Vec3(0, 0, -np.pi / 2), model.getGround(), "floor"
        )
    )
    model.addContactGeometry(
        osim.ContactSphere(BALL_RADIUS_M, osim.Vec3(0, 0, 0), body, "ball_cs")
    )
    model.finalizeConnections()
    path = directory / "synthetic_ball.osim"
    model.printToXML(str(path))
    text = re.sub(
        r'(<ForceSet name="forceset">\s*<objects)\s*/>',
        lambda m: m.group(1) + ">" + _HUNT_CROSSLEY_XML + "</objects>",
        path.read_text(),
    )
    path.write_text(text)
    model = osim.Model(str(path))
    state = model.initSystem()
    # free_coord_4 is the vertical (Y-up) translation: start just touching.
    model.getJointSet().get(0).get_coordinates(4).setValue(state, BALL_RADIUS_M)
    manager = osim.Manager(model)
    state.setTime(0.0)
    manager.initialize(state)
    settled = osim.State(manager.integrate(1.0))
    return _opensim_frame(model, settled)


# --- Simscape (file based; encodes the same statics as a dataset CSV) --------------

_RX_MINUS_90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_RX_PLUS_90 = _RX_MINUS_90.T


def _simscape_csv(
    path: Path, columns: dict[str, float], rotation: np.ndarray, joint: str
) -> None:
    for i in range(3):
        for j in range(3):
            columns[f"{joint}Logs_Rotation_Transform_I{i + 1}{j + 1}"] = float(
                rotation[i, j]
            )
    names = ["time", *columns]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerow([repr(0.0)] + [repr(columns[n]) for n in names[1:]])


def _load_simscape(path: Path) -> ForceTorqueFrame:
    from src.engines.simscape.force_channels import load_simscape_force_series

    series: ForceTorqueSeries
    series, _missing = load_simscape_force_series(path)
    return series[0]


def simscape_hanging(directory: Path, p: PendulumParams = PENDULUM) -> ForceTorqueFrame:
    """Joint-local reaction (0, m*g, 0) rotated by Rx(+90) to world (0, 0, m*g)."""
    local = _RX_PLUS_90.T @ np.array([0.0, 0.0, p.weight_n])
    columns: dict[str, float] = {}
    for i, value in enumerate(local, start=1):
        columns[f"LFLogs_ConstraintForceLocal_{i}"] = float(value)
        columns[f"LFLogs_ConstraintTorqueLocal_{i}"] = 0.0
        columns[f"LFLogs_GlobalPosition_{i}"] = float(p.pivot_world[i - 1])
    path = directory / "synthetic_hanging.csv"
    _simscape_csv(path, columns, _RX_PLUS_90, "LF")
    return _load_simscape(path)


def simscape_inverted(
    directory: Path, p: PendulumParams = PENDULUM
) -> ForceTorqueFrame:
    """Actuator torque about the joint-local Z axis (LF is actuated about Z only)."""
    world = np.asarray(inverted_expected(p)["actuator_torque_nm"])
    local = _RX_MINUS_90.T @ world
    # DbC: LF is driven about local Z only; a fixture that needs other axes is wrong.
    assert abs(local[0]) < 1e-12 and abs(local[1]) < 1e-12, local
    columns = {
        "LFLogs_ActuatorTorqueZ": float(local[2]),
        **{
            f"LFLogs_GlobalPosition_{i}": float(p.pivot_world[i - 1]) for i in (1, 2, 3)
        },
        **{f"LFLogs_ConstraintForceLocal_{i}": 0.0 for i in (1, 2, 3)},
        **{f"LFLogs_ConstraintTorqueLocal_{i}": 0.0 for i in (1, 2, 3)},
    }
    path = directory / "synthetic_inverted.csv"
    _simscape_csv(path, columns, _RX_MINUS_90, "LF")
    return _load_simscape(path)


__all__ = [
    "PENDING_PROVIDERS",
    "opensim_hanging",
    "opensim_inverted",
    "opensim_resting_ball",
    "simscape_hanging",
    "simscape_inverted",
]
