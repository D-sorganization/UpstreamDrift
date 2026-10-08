"""OpenSim ground-reaction breakdown on the force frame (GCV-2, #11708).

A free pelvis carries two welded foot bodies (``calcn_l`` / ``calcn_r``), each
with a Hunt-Crossley contact sphere against a half-space floor.  After settling
for at least 0.2 s the frame carries per-foot and net GRF, CoP, free moment and
moment about the centre of mass with net vertical force equal to the weight.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

osim = pytest.importorskip("opensim", reason="OpenSim not installed")
if not hasattr(osim, "Model"):
    pytest.skip("real opensim is unavailable", allow_module_level=True)

from src.engines.physics_engines.opensim.python.opensim_force_torque import (  # noqa: E402
    OpenSimForceTorqueSource,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

G = 9.80665
PELVIS_KG = 70.0
FOOT_KG = 1.0
FOOT_Y = 0.15  # world +y is left
RADIUS = 0.05
LEG_M = 0.9
SETTLE_S = 1.0
WEIGHT_RTOL = 0.02
WEIGHT_N = (PELVIS_KG + 2 * FOOT_KG) * G

_HC = """<HuntCrossleyForce name="hc_{side}">
<appliesForce>true</appliesForce>
<HuntCrossleyForce::ContactParametersSet name="contact_parameters"><objects>
<HuntCrossleyForce::ContactParameters>
<geometry>floor cs_{side}</geometry>
<stiffness>1e6</stiffness><dissipation>2.0</dissipation>
<static_friction>0.8</static_friction><dynamic_friction>0.6</dynamic_friction>
<viscous_friction>0.0</viscous_friction>
</HuntCrossleyForce::ContactParameters></objects><groups/>
</HuntCrossleyForce::ContactParametersSet>
<transition_velocity>0.2</transition_velocity>
</HuntCrossleyForce>"""


def _stance_model_path(directory: Path) -> Path:
    model = osim.Model()
    model.setName("synthetic_stance")
    pelvis = osim.Body("pelvis", PELVIS_KG, osim.Vec3(0), osim.Inertia(1, 1, 1))
    model.addBody(pelvis)
    model.addJoint(
        osim.FreeJoint(
            "root",
            model.getGround(),
            osim.Vec3(0),
            osim.Vec3(0),
            pelvis,
            osim.Vec3(0),
            osim.Vec3(0),
        )
    )
    floor = osim.ContactHalfSpace(
        osim.Vec3(0), osim.Vec3(0, 0, -np.pi / 2), model.getGround(), "floor"
    )
    model.addContactGeometry(floor)
    # OpenSim (x, y, z) -> world (x, -z, y): world +y (left) is OpenSim -z.
    for side, z in (("l", -FOOT_Y), ("r", FOOT_Y)):
        foot = osim.Body(f"calcn_{side}", FOOT_KG, osim.Vec3(0), osim.Inertia(0.01))
        model.addBody(foot)
        model.addJoint(
            osim.WeldJoint(
                f"weld_{side}",
                pelvis,
                osim.Vec3(0, -LEG_M, z),
                osim.Vec3(0),
                foot,
                osim.Vec3(0),
                osim.Vec3(0),
            )
        )
        # heel and forefoot spheres: a four-point support is statically stable
        for tag, x in (("h", -0.1), ("t", 0.1)):
            model.addContactGeometry(
                osim.ContactSphere(RADIUS, osim.Vec3(x, 0, 0), foot, f"cs_{side}_{tag}")
            )
    model.finalizeConnections()
    path = directory / "synthetic_stance.osim"
    model.printToXML(str(path))
    text = path.read_text()
    text = re.sub(
        r'(<ForceSet name="forceset">\s*<objects)\s*/>',
        lambda m: m.group(1)
        + ">"
        + "".join(_HC.format(side=f"{a}_{b}") for a in "lr" for b in "ht")
        + "</objects>",
        text,
    )
    path.write_text(text)
    return path


def build_stance(directory):
    path = _stance_model_path(directory)
    model = osim.Model(str(path))
    state = model.initSystem()
    joint = model.getJointSet().get("root")
    joint.get_coordinates(4).setValue(state, LEG_M + RADIUS)  # just touching
    manager = osim.Manager(model)
    state.setTime(0.0)
    manager.initialize(state)
    return model, osim.State(manager.integrate(SETTLE_S))


@pytest.fixture(scope="module")
def stance(tmp_path_factory):
    return build_stance(tmp_path_factory.mktemp("osim_stance"))


def _labelled(stance):
    model, state = stance
    return {w.label: w for w in OpenSimForceTorqueSource(model).sample(state).wrenches}


def test_static_stance_net_grf_equals_body_weight(stance) -> None:
    w = _labelled(stance)
    net = np.asarray(w["contact:grf_net"].force_n)
    assert abs(net[2] - WEIGHT_N) / WEIGHT_N < WEIGHT_RTOL
    np.testing.assert_allclose(net[:2], 0.0, atol=0.01 * WEIGHT_N)  # friction ringing
    left = np.asarray(w["contact:grf_left"].force_n)
    right = np.asarray(w["contact:grf_right"].force_n)
    np.testing.assert_allclose(left + right, net, atol=1e-6)
    assert abs(left[2] - right[2]) / WEIGHT_N < WEIGHT_RTOL


def test_left_foot_is_on_world_positive_y(stance) -> None:
    w = _labelled(stance)
    assert w["contact:grf_left"].point_m[1] > 0.0 > w["contact:grf_right"].point_m[1]
    assert abs(w["contact:grf_net"].point_m[1]) < FOOT_Y
    assert abs(w["contact:grf_net"].point_m[2]) < 0.01


def test_com_moment_uses_the_model_centre_of_mass(stance) -> None:
    model, state = stance
    w = _labelled(stance)
    com = np.asarray(model.calcMassCenterPosition(state).to_numpy())
    com_world = np.array([com[0], -com[2], com[1]])
    np.testing.assert_allclose(w["contact:moment_com_net"].point_m, com_world, atol=1e-6)
    for label in ("contact:free_moment_net", "contact:free_moment_left"):
        assert label in w
    m_com = np.asarray(w["contact:moment_com_net"].torque_nm)
    com_height = w["contact:moment_com_net"].point_m[2]
    # residual shear of at most 1 % of weight acting at the CoM height
    assert np.linalg.norm(m_com[:2]) < 0.01 * WEIGHT_N * com_height
