"""Cross-engine physical parity of force/torque overlays (FTO-21, #11306).

Every engine is asked the same three analytic statics questions from the one
shared parameter set in ``force_overlay_fixtures.py``, and is judged only by
its provider frame (``get_force_torque_frame()``) through one assertion helper:

1. static hanging pendulum: reaction ``(0, 0, +m*g)`` at the pivot, axial load
   ``+m*g`` (tension);
2. inverted pendulum held by an actuator at ``theta``: actuator torque
   ``+m*g*(l/2)*sin(theta)`` about the hinge, axial load ``-m*g*cos(theta)``
   (compression);
3. body resting on the ground: contact forces sum to ``(0, 0, +m*g)`` with
   contact points on the ground plane.

Rows for engines that are not installed, or whose provider is not on this
branch, skip with an explicit reason and go live automatically. The Simscape
row is file-based: it checks the FTO-18 loader keeps the world /
applied-to-body convention. All models are ``synthetic_`` and are built here,
beside the test; no model generators live in ``src/``.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.simscape.force_channels import load_simscape_force_series
from src.shared.python.force_overlay import ForceTorqueFrame, WrenchKind
from tests.integration.cross_engine.force_overlay_fixtures import (
    BOX,
    HINGE,
    LINK,
    STANDARD,
    PendulumFixture,
)

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

# --- Tolerances, set once ----------------------------------------------------
# Analytic statics (hanging / held pendulum): the engines solve rigid-body
# equations in closed form (Pinocchio RNEA/ABA, Drake reaction port, Simbody
# acceleration-level reactions), so they agree with the formula to solver
# round-off. Observed worst case on the reference host is 4e-15 (N, N*m, m);
# 1e-6 relative / 1e-6 absolute leaves ~9 orders of margin for other BLAS and
# platform builds and is still ~6 orders below a sign or frame error (~1e1).
ANALYTIC_RTOL = 1e-6
ANALYTIC_ATOL = 1e-6
# Compliant contact (resting body): the penetration that carries the weight is
# set by the contact stiffness, so the settled sum equals m*g only to the
# compliance: 2 % relative / 0.05 N absolute is the bound the Drake provider's
# own resting-contact test uses (hydroelastic modulus 1e6 Pa, 1 ms step) and
# also bounds Hunt-Crossley (stiffness 1e6) and MuJoCo soft contact here.
CONTACT_RTOL = 0.02
CONTACT_ATOL_N = 0.05
# Contact points lie on the ground plane to within the penetration depth and
# the hydroelastic patch centroid offset: 2 cm.
CONTACT_POINT_ATOL_M = 0.02
# Seconds a resting body is simulated before sampling its contact forces.
SETTLE_S = 1.0

ENGINES = ("mujoco", "drake", "pinocchio", "opensim")
ENGINE_PARAMS = [
    pytest.param(name, marks=getattr(pytest.mark, f"requires_{name}"))
    for name in ENGINES
]
_PROVIDER_ISSUE = {"mujoco": "FTO-9 (#11294)"}
_NO_CONTACT_REASON = {
    "pinocchio": (
        "Pinocchio has no contact model: contacts are caller-supplied "
        "ContactSample records passed through, so a resting-contact row would "
        "only echo its own input"
    ),
    "simscape": (
        "Simscape is file-based (logged CSV channels): there is no resting "
        "contact configuration to simulate"
    ),
}


# --- Shared assertion helper -------------------------------------------------
def select_wrenches(
    frame: ForceTorqueFrame, label_prefix: str, kind: WrenchKind, body: str
) -> list[Any]:
    """Wrenches of ``kind`` on ``body`` whose name after ``category:`` starts
    with ``label_prefix`` (engines differ in the category word only)."""
    return [
        w
        for w in frame.by_kind(kind)
        if w.body == body and w.label.split(":", 1)[1].startswith(label_prefix)
    ]


def assert_wrench(
    frame: ForceTorqueFrame,
    label_prefix: str,
    kind: WrenchKind,
    *,
    body: str = LINK,
    force: tuple[float, float, float] | None = None,
    torque: tuple[float, float, float] | None = None,
    point: tuple[float, float, float] | None = None,
    rtol: float = ANALYTIC_RTOL,
    atol: float = ANALYTIC_ATOL,
) -> None:
    """Assert exactly one ``kind`` wrench matches the expected world values.

    Unavailable data is omitted by providers, so zero matches is a failure,
    never a pass. Only the expected quantities that are given are compared.
    """
    assert frame.world_frame == "world_Zup"
    found = select_wrenches(frame, label_prefix, kind, body)
    assert len(found) == 1, (
        f"expected exactly one {kind.value} wrench on {body!r} with name prefix "
        f"{label_prefix!r}, got {[w.label for w in frame.wrenches]}"
    )
    wrench = found[0]
    for name, expected, actual in (
        ("force_n", force, wrench.force_n),
        ("torque_nm", torque, wrench.torque_nm),
        ("point_m", point, wrench.point_m),
    ):
        if expected is None:
            continue
        assert actual is not None, f"{wrench.label}: {name} is unavailable"
        np.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=atol, err_msg=f"{wrench.label} {name}"
        )


def assert_axial_load(frame: ForceTorqueFrame, expected: float) -> None:
    """Assert the proximal axial load of the pendulum link (tension positive)."""
    assert frame.axial_loads is not None, "frame has no axial loads"
    actual = frame.axial_loads.values_n[LINK]
    assert actual is not None, "axial load of the link is unavailable"
    assert actual == pytest.approx(expected, rel=ANALYTIC_RTOL, abs=ANALYTIC_ATOL)


def assert_resting_contact(frame: ForceTorqueFrame, fx: PendulumFixture) -> None:
    """Contact forces on the body sum to ``m*g`` up, on the ground plane."""
    contacts = [w for w in frame.by_kind(WrenchKind.CONTACT) if w.body == BOX]
    assert contacts, (
        f"no contact wrench on {BOX!r}: {[w.label for w in frame.wrenches]}"
    )
    forces = [w.force_n for w in contacts]
    assert all(f is not None for f in forces)
    np.testing.assert_allclose(
        np.sum(forces, axis=0),
        (0.0, 0.0, fx.weight),
        rtol=CONTACT_RTOL,
        atol=CONTACT_ATOL_N,
    )
    assert all(abs(w.point_m[2]) <= CONTACT_POINT_ATOL_M for w in contacts)


# --- Engine availability -----------------------------------------------------
# Engines are probed once at import (collection) time, not inside tests:
# tests/conftest.py restores the protected engine modules after every test, so
# a library first imported inside a test body cannot be imported again.
_ENGINE_SOURCES = {
    "mujoco": (
        "mujoco",
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine",
        "MuJoCoPhysicsEngine",
    ),
    "drake": (
        "pydrake.multibody.plant",
        "src.engines.physics_engines.drake.python.drake_physics_engine",
        "DrakePhysicsEngine",
    ),
    "pinocchio": (
        "pinocchio",
        "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine",
        "PinocchioPhysicsEngine",
    ),
    "opensim": (
        "opensim",
        "src.engines.physics_engines.opensim.python.opensim_physics_engine",
        "OpenSimPhysicsEngine",
    ),
}


def _probe(name: str) -> tuple[type | None, str]:
    """``(engine class, "")`` if the row can run, else ``(None, skip reason)``."""
    library, module_name, class_name = _ENGINE_SOURCES[name]
    try:
        importlib.import_module(library)
        cls = getattr(importlib.import_module(module_name), class_name)
    except ImportError as exc:
        return None, f"{name} is not installed ({exc})"
    if not hasattr(cls, "get_force_torque_frame"):
        return None, (
            f"{name} force/torque provider {_PROVIDER_ISSUE.get(name, '')} is not "
            f"on this branch: {class_name} has no get_force_torque_frame"
        )
    return cls, ""


_PROBED = {name: _probe(name) for name in ENGINES}


def engine_class(name: str) -> type:
    """Engine class for ``name``; skips with an explicit reason if absent."""
    cls, reason = _PROBED[name]
    if cls is None:
        pytest.skip(reason)
    return cls


# --- Per-engine model builders (synthetic_, beside the test) -----------------
Builder = Callable[[type, PendulumFixture, str, Path], Any]
_INERTIA = 0.01  # kg*m^2; irrelevant to statics, only needs to be physical.


def _hold(engine: Any, fx: PendulumFixture, scenario: str) -> Any:
    """Put a loaded pendulum engine in its static state for ``scenario``."""
    inverted = scenario == "inverted"
    engine.set_state(np.array([fx.inverted_angle if inverted else 0.0]), np.zeros(1))
    if inverted:
        engine.set_control(np.array([fx.hold_torque]))
    return engine


def _pendulum_urdf(fx: PendulumFixture, base: str) -> str:
    """Hanging-pendulum URDF with a massless tip welded at the link end.

    The tip gives Drake a distal joint, which its segment axes require (Pinocchio
    and OpenSim fall back to the centre of mass). The transmission gives Drake
    its actuator.
    """
    return f"""<robot name="synthetic_pendulum">
  <link name="{base}"/>
  <link name="{LINK}"><inertial><origin xyz="0 0 {-fx.com_distance}"/>
    <mass value="{fx.mass}"/>
    <inertia ixx="{_INERTIA}" iyy="{_INERTIA}" izz="{_INERTIA}" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <joint name="{HINGE}" type="revolute">
    <parent link="{base}"/><child link="{LINK}"/>
    <origin xyz="0 0 {fx.pivot_height}"/><axis xyz="0 1 0"/>
    <limit lower="-4" upper="4" effort="100" velocity="10"/>
  </joint>
  <link name="tip"/>
  <joint name="tip_weld" type="fixed">
    <parent link="{LINK}"/><child link="tip"/><origin xyz="0 0 {-fx.length}"/>
  </joint>
  <transmission name="t"><type>transmission_interface/SimpleTransmission</type>
    <joint name="{HINGE}"><hardwareInterface>EffortJointInterface</hardwareInterface></joint>
    <actuator name="motor"><mechanicalReduction>1</mechanicalReduction></actuator>
  </transmission>
</robot>"""


def _build_pinocchio(cls: type, fx: PendulumFixture, scenario: str, tmp: Path) -> Any:
    engine = cls()
    engine.load_from_string(_pendulum_urdf(fx, "base"), "urdf")
    return _hold(engine, fx, scenario)


_DRAKE_COMPLIANT = (
    '<drake:proximity_properties xmlns:drake="http://drake.mit.edu">'
    "<drake:compliant_hydroelastic/>"
    '<drake:hydroelastic_modulus value="1e6"/>'
    '<drake:mesh_resolution_hint value="0.05"/></drake:proximity_properties>'
)
_DRAKE_RIGID = (
    '<drake:proximity_properties xmlns:drake="http://drake.mit.edu">'
    "<drake:rigid_hydroelastic/></drake:proximity_properties>"
)


def _drake_resting_urdf(fx: PendulumFixture) -> str:
    """Compliant box on a rigid ground slab (the ``world`` link is welded)."""
    side = fx.box_side
    return f"""<robot name="synthetic_box">
  <link name="world"><collision><origin xyz="0 0 -0.5"/>
    <geometry><box size="5 5 1"/></geometry>{_DRAKE_RIGID}</collision></link>
  <link name="{BOX}"><inertial><mass value="{fx.mass}"/>
    <inertia ixx="{_INERTIA}" iyy="{_INERTIA}" izz="{_INERTIA}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision><geometry><box size="{side} {side} {side}"/></geometry>
    {_DRAKE_COMPLIANT}</collision></link>
</robot>"""


def _build_drake(cls: type, fx: PendulumFixture, scenario: str, tmp: Path) -> Any:
    if scenario == "resting":
        engine = cls()  # discrete plant (default step) for hydroelastic contact
        engine.load_from_string(_drake_resting_urdf(fx), "urdf")
        pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, fx.box_rest_height])
        engine.set_state(pose, np.zeros(6))
        engine.step(SETTLE_S)
        return engine
    engine = cls(time_step=0.0)  # continuous plant: reactions are valid at t=0
    engine.load_from_string(_pendulum_urdf(fx, "world"), "urdf")
    return _hold(engine, fx, scenario)


def _opensim_model(fx: PendulumFixture) -> Any:
    """Empty OpenSim model with Y-up gravity (the OpenSim convention)."""
    import opensim as osim

    model = osim.Model()
    model.setName("synthetic_model")
    model.setGravity(osim.Vec3(0, -fx.gravity, 0))
    return model


def _load_opensim(cls: type, model: Any, tmp_path: Path) -> Any:
    model.finalizeConnections()
    path = tmp_path / "synthetic_model.osim"
    model.printToXML(str(path))
    engine = cls()
    engine.load_from_path(str(path))
    return engine


def _build_opensim_pendulum(
    cls: type, fx: PendulumFixture, scenario: str, tmp_path: Path
) -> Any:
    import opensim as osim

    model = _opensim_model(fx)
    com = osim.Vec3(0, -fx.com_distance, 0)
    body = osim.Body(LINK, fx.mass, com, osim.Inertia(_INERTIA))
    model.addBody(body)
    # Both joint frames are turned half a revolution about x so the pin axis
    # (OpenSim z) is world +y, making the angle convention match the others.
    flip = osim.Vec3(math.pi, 0, 0)
    pivot = osim.Vec3(0, fx.pivot_height, 0)
    joint = osim.PinJoint(
        HINGE, model.getGround(), pivot, flip, body, osim.Vec3(0), flip
    )
    model.addJoint(joint)
    motor = osim.CoordinateActuator(joint.get_coordinates(0).getName())
    motor.setName("motor")
    motor.setOptimalForce(1.0)
    model.addForce(motor)
    if scenario == "inverted":
        # Initial angle and hold torque are baked into the model instead of
        # using engine.set_state / set_control: on OpenSim 4.x the former raises
        # TypeError (opensim.Vector(n) has no such constructor) and the latter
        # never marks the controls valid, so they are overwritten with zeros.
        joint.get_coordinates(0).setDefaultValue(fx.inverted_angle)
        holder = osim.PrescribedController()
        holder.addActuator(motor)
        holder.prescribeControlForActuator("motor", osim.Constant(fx.hold_torque))
        model.addController(holder)
    return _load_opensim(cls, model, tmp_path)


def _build_opensim_resting(
    cls: type, fx: PendulumFixture, scenario: str, tmp_path: Path
) -> Any:
    """Sphere on a half-space: OpenSim's supported contact pair (no box)."""
    import opensim as osim

    model = _opensim_model(fx)
    radius = fx.box_rest_height
    zero = osim.Vec3(0)
    body = osim.Body(BOX, fx.mass, zero, osim.Inertia(_INERTIA))
    model.addBody(body)
    free = osim.FreeJoint("free", model.getGround(), zero, zero, body, zero, zero)
    model.addJoint(free)
    free.get_coordinates(4).setDefaultValue(radius)  # vertical (Y-up) translation
    # Half-space turned so its outward normal is OpenSim +y (world up).
    tilt = osim.Vec3(0, 0, -math.pi / 2)
    floor = osim.ContactHalfSpace(zero, tilt, model.getGround(), "floor")
    ball = osim.ContactSphere(radius, zero, body, "ball_cs")
    model.addContactGeometry(floor)
    model.addContactGeometry(ball)
    contact = osim.SmoothSphereHalfSpaceForce("contact", ball, floor)
    contact.set_stiffness(1e6)
    contact.set_dissipation(1.0)
    model.addForce(contact)
    engine = _load_opensim(cls, model, tmp_path)
    engine.step(SETTLE_S)
    return engine


def _mjcf(fx: PendulumFixture, worldbody: str, actuator: str = "") -> str:
    return (
        f'<mujoco model="synthetic"><option gravity="0 0 {-fx.gravity}" '
        f'timestep="0.002"/><worldbody>{worldbody}</worldbody>{actuator}</mujoco>'
    )


def mujoco_pendulum_mjcf(fx: PendulumFixture) -> str:
    """Hanging pendulum MJCF with a unit-gear motor on the hinge."""
    body = (
        f'<body name="{LINK}" pos="0 0 {fx.pivot_height}">'
        f'<joint name="{HINGE}" type="hinge" axis="0 1 0"/>'
        f'<inertial pos="0 0 {-fx.com_distance}" mass="{fx.mass}" '
        f'diaginertia="{_INERTIA} {_INERTIA} {_INERTIA}"/>'
        # Non-colliding rod from the pivot to the COM: MuJoCo's native axial
        # source only reports rods. The explicit <inertial> above keeps the
        # body mass and inertia unchanged.
        f'<geom type="capsule" fromto="0 0 0 0 0 {-fx.com_distance}" '
        'size="0.01" contype="0" conaffinity="0" group="3"/></body>'
    )
    motor = f'<actuator><motor name="motor" joint="{HINGE}" gear="1"/></actuator>'
    return _mjcf(fx, body, motor)


def mujoco_resting_mjcf(fx: PendulumFixture) -> str:
    """Box resting on a ground plane (default soft contact)."""
    half = fx.box_rest_height
    body = (
        '<geom type="plane" size="5 5 0.1"/>'
        f'<body name="{BOX}" pos="0 0 {half}"><freejoint/>'
        f'<geom type="box" size="{half} {half} {half}" mass="{fx.mass}"/></body>'
    )
    return _mjcf(fx, body)


def _build_mujoco(cls: type, fx: PendulumFixture, scenario: str, tmp: Path) -> Any:
    engine = cls()
    if scenario != "resting":
        engine.load_from_string(mujoco_pendulum_mjcf(fx), "xml")
        return _hold(engine, fx, scenario)
    engine.load_from_string(mujoco_resting_mjcf(fx), "xml")
    pose = np.array([0.0, 0.0, fx.box_rest_height, 1.0, 0.0, 0.0, 0.0])
    engine.set_state(pose, np.zeros(6))
    for _ in range(round(SETTLE_S / 0.002)):
        engine.step()
    return engine


_BUILDERS: dict[str, dict[str, Builder]] = {
    "mujoco": dict.fromkeys(("hanging", "inverted", "resting"), _build_mujoco),
    "drake": dict.fromkeys(("hanging", "inverted", "resting"), _build_drake),
    "pinocchio": dict.fromkeys(("hanging", "inverted"), _build_pinocchio),
    "opensim": {
        "hanging": _build_opensim_pendulum,
        "inverted": _build_opensim_pendulum,
        "resting": _build_opensim_resting,
    },
}


def build(name: str, scenario: str, tmp_path: Path, fx: PendulumFixture) -> Any:
    """Build the engine for ``scenario`` ('hanging', 'inverted', 'resting')."""
    cls = engine_class(name)
    if scenario == "resting" and name in _NO_CONTACT_REASON:
        pytest.skip(_NO_CONTACT_REASON[name])
    return _BUILDERS[name][scenario](cls, fx, scenario, tmp_path)


def provider_frame(
    name: str, scenario: str, tmp_path: Path, fx: PendulumFixture = STANDARD
) -> ForceTorqueFrame:
    """The provider frame of ``scenario`` on engine ``name``."""
    frame = build(name, scenario, tmp_path, fx).get_force_torque_frame()
    assert frame is not None, f"{name} returned no force/torque frame"
    assert frame.engine == name
    return frame


# --- The suite -----------------------------------------------------------------
@pytest.mark.parametrize("engine", ENGINE_PARAMS)
def test_hanging_pendulum_reaction_is_weight_up_and_tension(
    engine: str, tmp_path: Path
) -> None:
    fx = STANDARD
    frame = provider_frame(engine, "hanging", tmp_path)
    assert_wrench(
        frame,
        HINGE,
        WrenchKind.JOINT_REACTION,
        force=(0.0, 0.0, fx.weight),
        point=fx.pivot,
    )
    assert_axial_load(frame, fx.weight)


@pytest.mark.parametrize("engine", ENGINE_PARAMS)
def test_inverted_pendulum_holds_with_actuator_torque_and_compression(
    engine: str, tmp_path: Path
) -> None:
    fx = STANDARD
    frame = provider_frame(engine, "inverted", tmp_path)
    assert_wrench(
        frame,
        HINGE,
        WrenchKind.JOINT_ACTUATOR,
        torque=(0.0, fx.hold_torque, 0.0),
        point=fx.pivot,
    )
    assert_wrench(
        frame,
        HINGE,
        WrenchKind.JOINT_REACTION,
        force=(0.0, 0.0, fx.weight),
        point=fx.pivot,
    )
    assert_axial_load(frame, fx.inverted_axial_load)
    assert fx.inverted_axial_load < 0.0 < fx.weight


@pytest.mark.parametrize("engine", ENGINE_PARAMS + ["simscape"])
def test_resting_body_contact_sums_to_weight_on_ground_plane(
    engine: str, tmp_path: Path
) -> None:
    if engine in _NO_CONTACT_REASON:
        pytest.skip(_NO_CONTACT_REASON[engine])
    assert_resting_contact(provider_frame(engine, "resting", tmp_path), STANDARD)


@pytest.mark.parametrize("engine", ENGINE_PARAMS)
def test_sign_convention_guard_flipped_expectations_fail(
    engine: str, tmp_path: Path
) -> None:
    """The suite bites: flipping any expected sign makes the engine row fail."""
    fx = STANDARD
    hanging = provider_frame(engine, "hanging", tmp_path)
    inverted = provider_frame(engine, "inverted", tmp_path)
    reaction = WrenchKind.JOINT_REACTION
    with pytest.raises(AssertionError):
        assert_wrench(hanging, HINGE, reaction, force=(0.0, 0.0, -fx.weight))
    with pytest.raises(AssertionError):
        assert_axial_load(hanging, -fx.weight)
    with pytest.raises(AssertionError):
        assert_wrench(
            inverted,
            HINGE,
            WrenchKind.JOINT_ACTUATOR,
            torque=(0.0, -fx.hold_torque, 0.0),
        )
    with pytest.raises(AssertionError):
        assert_axial_load(inverted, -fx.inverted_axial_load)
    # Omitted data is a failure, never a pass.
    with pytest.raises(AssertionError, match="exactly one"):
        assert_wrench(hanging, "no_such_joint", reaction, force=(0.0, 0.0, fx.weight))


# --- Simscape row (file-based, FTO-18 loader) ----------------------------------
# Logged joint-local columns of the LS joint; R = [[1,0,0],[0,0,-1],[0,1,0]] maps
# the local +y axis to world +z, so a weight-up reaction is logged as (0, m*g, 0).
_SIMSCAPE_JOINT = "LS"
_RX90 = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))


def _write_simscape_hanging_csv(path: Path, fx: PendulumFixture) -> Path:
    columns: dict[str, float] = {"time": 0.0}
    local_force = (0.0, fx.weight, 0.0)
    for i in range(3):
        columns[f"LSLogs_ConstraintForceLocal_{i + 1}"] = local_force[i]
        columns[f"LSLogs_ConstraintTorqueLocal_{i + 1}"] = 0.0
        columns[f"LSLogs_GlobalPosition_{i + 1}"] = fx.pivot[i]
        for j in range(3):
            columns[f"LSLogs_Rotation_Transform_I{i + 1}{j + 1}"] = _RX90[i][j]
    path.write_text(
        ",".join(columns) + "\n" + ",".join(repr(v) for v in columns.values()) + "\n"
    )
    return path


def _simscape_frame(tmp_path: Path, fx: PendulumFixture) -> ForceTorqueFrame:
    csv_path = _write_simscape_hanging_csv(tmp_path / "synthetic_hanging.csv", fx)
    series, _missing = load_simscape_force_series(csv_path)
    return series.frames[0]


def test_simscape_loader_keeps_world_applied_to_body_convention(
    tmp_path: Path,
) -> None:
    fx = STANDARD
    frame = _simscape_frame(tmp_path, fx)
    assert frame.engine == "simscape"
    assert_wrench(
        frame,
        _SIMSCAPE_JOINT,
        WrenchKind.JOINT_REACTION,
        body=_SIMSCAPE_JOINT,
        force=(0.0, 0.0, fx.weight),
        torque=(0.0, 0.0, 0.0),
        point=fx.pivot,
    )
    (wrench,) = select_wrenches(
        frame, _SIMSCAPE_JOINT, WrenchKind.JOINT_REACTION, _SIMSCAPE_JOINT
    )
    assert wrench.APPLICATION_FRAME == "world"
    assert wrench.DIRECTION_CONVENTION == "applied_to_body"
    # The logged local vector is not the world vector: the rotation is applied.
    assert wrench.force_n != (0.0, fx.weight, 0.0)


def test_simscape_sign_convention_guard_flipped_expectation_fails(
    tmp_path: Path,
) -> None:
    frame = _simscape_frame(tmp_path, STANDARD)
    with pytest.raises(AssertionError):
        assert_wrench(
            frame,
            _SIMSCAPE_JOINT,
            WrenchKind.JOINT_REACTION,
            body=_SIMSCAPE_JOINT,
            force=(0.0, 0.0, -STANDARD.weight),
        )


# --- Fixture and ground-truth self-checks --------------------------------------
@pytest.mark.parametrize(
    "field, value",
    [
        ("mass", 0.0),
        ("mass", -1.0),
        ("length", float("nan")),
        ("gravity", 0.0),
        ("pivot_height", float("inf")),
        ("box_side", -0.2),
        ("theta", 0.0),
        ("theta", math.pi / 2),
    ],
)
def test_fixture_rejects_non_physical_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        PendulumFixture(**{field: value})


def test_fixture_analytic_values_match_the_issue() -> None:
    fx = PendulumFixture(mass=3.0, length=2.0, gravity=10.0, theta=math.pi / 6)
    assert fx.weight == pytest.approx(30.0)
    assert fx.hold_torque == pytest.approx(30.0 * 1.0 * 0.5)
    assert fx.inverted_axial_load == pytest.approx(-30.0 * math.cos(math.pi / 6))


@pytest.mark.requires_mujoco
def test_mujoco_models_are_statically_consistent() -> None:
    """The MJCF builders encode the analytic statics (independent of any provider).

    Guards the MuJoCo row, which goes live when its provider lands (FTO-9).
    """
    mujoco = pytest.importorskip("mujoco")
    fx = STANDARD
    model = mujoco.MjModel.from_xml_string(mujoco_pendulum_mjcf(fx))
    data = mujoco.MjData(model)
    data.qpos[0] = fx.inverted_angle
    data.ctrl[0] = fx.hold_torque
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qacc, 0.0, atol=ANALYTIC_ATOL)
    data.ctrl[0] = 0.0
    data.qpos[0] = 0.0
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qacc, 0.0, atol=ANALYTIC_ATOL)

    rest = mujoco.MjModel.from_xml_string(mujoco_resting_mjcf(fx))
    rest_data = mujoco.MjData(rest)
    for _ in range(round(SETTLE_S / rest.opt.timestep)):
        mujoco.mj_step(rest, rest_data)
    wrench = np.zeros(6)
    total = np.zeros(3)
    for i in range(rest_data.ncon):
        mujoco.mj_contactForce(rest, rest_data, i, wrench)
        total += rest_data.contact[i].frame.reshape(3, 3).T @ wrench[:3]
    np.testing.assert_allclose(
        total, (0.0, 0.0, fx.weight), rtol=CONTACT_RTOL, atol=CONTACT_ATOL_N
    )
