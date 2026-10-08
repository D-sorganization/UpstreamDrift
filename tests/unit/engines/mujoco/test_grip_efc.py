"""MuJoCo weld efc_force grip extraction (GCV-8, #11714).

Minimal MJCF fixture: a free club held by two site welds to mocap hands.
Sign convention: wrench exerted by the hand ON THE CLUB, world frame.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.grip_efc import (  # noqa: E402
    grip_analysis_from_efc,
    weld_wrench_on_club,
)

pytestmark = pytest.mark.unit

MASS = 0.5
G = 9.81
XML = f"""
<mujoco><option gravity="0 0 -{G}"/>
<worldbody>
 <body name="hand_r" mocap="true" pos="0.1 0 1"><site name="hr" size="0.01"/></body>
 <body name="hand_l" mocap="true" pos="-0.1 0 1"><site name="hl" size="0.01"/></body>
 <body name="club" pos="0 0 1"><freejoint name="club_free"/>
  <geom type="box" size="0.2 0.02 0.02" mass="{MASS}"/>
  <site name="cr" pos="0.1 0 0" size="0.01"/>
  <site name="cl" pos="-0.1 0 0" size="0.01"/></body>
</worldbody>
<equality>
 <weld name="grip_weld_r" site1="hr" site2="cr"/>
 <weld name="grip_weld_l" site1="hl" site2="cl"/>
</equality>
</mujoco>"""


def _settled(xml: str = XML, steps: int = 3000):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture(scope="module")
def static():
    return _settled()


def test_static_two_weld_hold_balances_weight_and_moment(static) -> None:
    model, data = static
    g = grip_analysis_from_efc(model, data)
    assert g.split_method == "efc_force"
    assert g.left is not None and g.right is not None
    for h in (g.left, g.right):
        assert np.isfinite(h.force_on_club_n).all()
        assert h.torque_on_club_nm is not None
        assert np.isfinite(h.torque_on_club_nm).all()
    weight = np.array([0.0, 0.0, -MASS * G])
    total = np.array(g.left.force_on_club_n) + np.array(g.right.force_on_club_n)
    np.testing.assert_allclose(total + weight, 0.0, atol=1e-3)
    # Moment balance about the club centre of mass.
    com = np.array(data.xipos[model.body("club").id])
    moment = np.zeros(3)
    for h in (g.left, g.right):
        moment += np.cross(np.array(h.point_m) - com, h.force_on_club_n)
        moment += np.array(h.torque_on_club_nm)
    np.testing.assert_allclose(moment, 0.0, atol=1e-3)


def test_force_on_club_points_up_when_holding_against_gravity(static) -> None:
    model, data = static
    g = grip_analysis_from_efc(model, data)
    assert g.net_force_n is not None
    assert g.net_force_n[2] > 0.0


def test_lift_test_upward_acceleration_increases_upward_force() -> None:
    """Sign test: lifting the club upward gives m(a - g) upward on the club."""
    model = mujoco.MjModel.from_xml_string(XML)
    rest = mujoco.MjData(model)
    lifted = mujoco.MjData(model)
    mocap = [model.body(n).mocapid[0] for n in ("hand_r", "hand_l")]
    lift_acc = 5.0

    def set_hands(t: float) -> None:
        for m in mocap:
            lifted.mocap_pos[m][2] = 1.0 + 0.5 * lift_acc * t * t

    steps = 300
    for _ in range(steps):
        mujoco.mj_step(model, rest)
    for i in range(steps):
        set_hands(i * model.opt.timestep)
        mujoco.mj_step(model, lifted)
    set_hands(steps * model.opt.timestep)  # hands consistent with the state
    mujoco.mj_forward(model, rest)
    mujoco.mj_forward(model, lifted)
    f_rest = grip_analysis_from_efc(model, rest).net_force_n
    f_lift = grip_analysis_from_efc(model, lifted).net_force_n
    assert f_rest is not None and f_lift is not None
    assert f_lift[2] > f_rest[2] > 0.0
    # Newton: net hand force on the club equals m (a - g) with a the club
    # acceleration MuJoCo solved for in the same forward pass.
    np.testing.assert_allclose(
        f_lift[2], MASS * (lifted.qacc[2] + G), rtol=1e-6, atol=1e-6
    )


def test_single_weld_wrench_has_expected_shape(static) -> None:
    model, data = static
    club = model.body("club").id
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_weld_r")
    wrench, point = weld_wrench_on_club(model, data, eq, club)
    assert wrench.shape == (6,) and point.shape == (3,)
    np.testing.assert_allclose(point, data.site_xpos[model.site("cr").id])


def test_rejects_non_weld_and_unknown_club(static) -> None:
    model, data = static
    with pytest.raises(ValueError, match="equality"):
        weld_wrench_on_club(model, data, 99, model.body("club").id)
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_weld_r")
    with pytest.raises(ValueError, match="club"):
        weld_wrench_on_club(model, data, eq, model.body("hand_l").id)


def test_inactive_weld_is_unavailable_not_zero() -> None:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.eq_active[:] = 0
    mujoco.mj_forward(model, data)
    g = grip_analysis_from_efc(model, data)
    assert g.net_force_n is None
    assert "inactive" in g.unavailable_reason


def test_model_without_grip_welds_is_unavailable() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    data = mujoco.MjData(model)
    g = grip_analysis_from_efc(model, data)
    assert g.net_force_n is None and g.split_method == "unavailable"
