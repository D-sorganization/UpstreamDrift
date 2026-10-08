"""MuJoCo and MyoSuite ground-reaction breakdown on the force frame (GCV-2, #11708).

Static double stance: a pelvis with a foot body (``calcn_l`` / ``calcn_r``) on
each side stands on a plane.  After the contact settles the frame must carry the
per-foot and net GRF, CoP, free moment and moment about the CoM, with the net
vertical force equal to the weight.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (  # noqa: E402
    MujocoForceTorqueSource,
)
from src.engines.physics_engines.myosuite.python.myosuite_force_torque import (  # noqa: E402
    MyoSuiteForceTorqueSource,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

G = 9.81
PELVIS_KG = 70.0
FOOT_KG = 1.0
HALF_X, HALF_Y, HALF_Z = 0.12, 0.05, 0.02
FOOT_Y = 0.15
SETTLE_S = 0.5  # the issue asks for a hold of at least 0.2 s
#: Net vertical force tolerance of the issue (2 %).
WEIGHT_RTOL = 0.02

STANCE = f"""
<mujoco model="double_stance">
  <option gravity="0 0 -{G}" timestep="0.001"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="pelvis" pos="0 0 0.9">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{PELVIS_KG}" diaginertia="5 5 5"/>
      <body name="calcn_l" pos="0 {FOOT_Y} -0.88">
        <geom type="box" size="{HALF_X} {HALF_Y} {HALF_Z}" mass="{FOOT_KG}"/>
      </body>
      <body name="calcn_r" pos="0 -{FOOT_Y} -0.88">
        <geom type="box" size="{HALF_X} {HALF_Y} {HALF_Z}" mass="{FOOT_KG}"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

WEIGHT_N = (PELVIS_KG + 2 * FOOT_KG) * G


@pytest.fixture(scope="module")
def stance():
    model = mujoco.MjModel.from_xml_string(STANCE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    while data.time < SETTLE_S:
        mujoco.mj_step(model, data)
    return model, data


def _by_label(frame):
    return {w.label: w for w in frame.wrenches}


@pytest.mark.parametrize(
    ("source_cls", "engine"),
    [(MujocoForceTorqueSource, "mujoco"), (MyoSuiteForceTorqueSource, "myosuite")],
)
def test_static_stance_net_grf_equals_body_weight(stance, source_cls, engine) -> None:
    model, data = stance
    frame = source_cls(model).sample(data)
    assert frame.engine == engine
    w = _by_label(frame)
    for label in ("contact:grf_left", "contact:grf_right", "contact:grf_net"):
        assert label in w, label
    net = np.asarray(w["contact:grf_net"].force_n)
    assert abs(net[2] - WEIGHT_N) / WEIGHT_N < WEIGHT_RTOL
    np.testing.assert_allclose(net[:2], 0.0, atol=0.01 * WEIGHT_N)
    left = np.asarray(w["contact:grf_left"].force_n)
    right = np.asarray(w["contact:grf_right"].force_n)
    np.testing.assert_allclose(left + right, net, atol=1e-9)
    assert abs(left[2] - right[2]) / WEIGHT_N < WEIGHT_RTOL  # symmetric stance


def test_static_stance_cop_lies_inside_the_support_polygon(stance) -> None:
    model, data = stance
    w = _by_label(MujocoForceTorqueSource(model).sample(data))
    cop = np.asarray(w["contact:grf_net"].point_m)
    assert abs(cop[0]) <= HALF_X and abs(cop[1]) <= FOOT_Y + HALF_Y
    assert cop[2] == pytest.approx(0.0)
    left = np.asarray(w["contact:grf_left"].point_m)
    right = np.asarray(w["contact:grf_right"].point_m)
    assert left[1] > 0.0 > right[1]
    assert abs(left[1] - FOOT_Y) <= HALF_Y and abs(right[1] + FOOT_Y) <= HALF_Y


def test_static_stance_free_moment_and_com_moment_are_present(stance) -> None:
    model, data = stance
    w = _by_label(MujocoForceTorqueSource(model).sample(data))
    for label in (
        "contact:free_moment_left",
        "contact:free_moment_right",
        "contact:free_moment_net",
        "contact:moment_com_net",
        "contact:moment_com_left",
    ):
        assert label in w, label
    # the free moment is vertical
    np.testing.assert_allclose(w["contact:free_moment_net"].torque_nm[:2], 0.0)
    com = np.asarray(w["contact:moment_com_net"].point_m)
    np.testing.assert_allclose(com, data.subtree_com[0], atol=1e-9)


def test_static_stance_com_moment_balance_residual_is_small(stance) -> None:
    """Quasi-static balance: the ground wrench has no horizontal moment about the CoM.

    The bound is a residual shear of at most 1 % of weight acting at the CoM
    height (the lever of a horizontal ground force about the CoM); the measured
    residuals are recorded in the pull request.
    """
    model, data = stance
    w = _by_label(MujocoForceTorqueSource(model).sample(data))
    m_com = np.asarray(w["contact:moment_com_net"].torque_nm)
    com_height = w["contact:moment_com_net"].point_m[2]
    assert np.linalg.norm(m_com[:2]) < 0.01 * WEIGHT_N * com_height


def test_newton_euler_identity_diagnostic(stance) -> None:
    """F_net + m g = d(momentum)/dt = 0 at rest (reported as a diagnostic)."""
    model, data = stance
    w = _by_label(MujocoForceTorqueSource(model).sample(data))
    net = np.asarray(w["contact:grf_net"].force_n)
    weight = np.array([0.0, 0.0, -WEIGHT_N])
    residual = np.linalg.norm(net + weight) / WEIGHT_N
    assert residual < WEIGHT_RTOL


def test_airborne_has_no_ground_reaction_not_zero_arrows() -> None:
    model = mujoco.MjModel.from_xml_string(STANCE)
    data = mujoco.MjData(model)
    data.qpos[2] = 3.0
    mujoco.mj_forward(model, data)
    labels = {w.label for w in MujocoForceTorqueSource(model).sample(data).wrenches}
    assert not any(label.startswith("contact:grf") for label in labels)


def test_club_contact_is_not_ground_reaction() -> None:
    """An airborne foot pressed by the club carries a contact but no GRF."""
    xml = STANCE.replace(
        "</worldbody>",
        '<body name="golf_club" pos="0 0.15 1.15"><freejoint/>'
        '<geom type="sphere" size="0.03" mass="0.3"/></body></worldbody>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[2] = 2.0  # lift the pelvis; the club sphere overlaps the left foot
    mujoco.mj_forward(model, data)
    labels = {w.label for w in MujocoForceTorqueSource(model).sample(data).wrenches}
    assert any(label.startswith("contact:calcn_l") for label in labels)
    assert not any(label.startswith("contact:grf") for label in labels)


def test_rejects_a_foreign_model() -> None:
    with pytest.raises(TypeError):
        MyoSuiteForceTorqueSource(object())  # type: ignore[arg-type]
