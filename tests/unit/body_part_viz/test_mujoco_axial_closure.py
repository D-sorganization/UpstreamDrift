"""Grip-weld closure wrench in the prescribed-motion axial-load evaluation (#13179).

Synthetic closed loop: a free body A welded (6-D) to a ball-jointed rod B. The
closure multiplier comes from an explicit KKT solve (the production solver); the
inverse-dynamics residual against the applied efforts then has to vanish when the
closure wrench is injected, and must not when it is left to MuJoCo's soft weld.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.native_model import (
    _evaluate_weld_closure,
    _solve_kkt_with_multipliers,
)
from src.shared.python.body_part_viz.mujoco_axial_loads import (
    GripClosure,
    MujocoAxialLoadSource,
)

pytestmark = pytest.mark.unit

LOOP = """<mujoco><option gravity="0 0 -9.81" timestep="0.002"/><worldbody>
<body name="hand" pos="0 0 1"><freejoint/>
<geom type="sphere" size="0.05" mass="1.5"/><site name="a" pos="0 0 0"/></body>
<body name="club" pos="0.5 0 2"><joint type="ball"/>
<geom type="capsule" fromto="0 0 0 -0.5 0 -1" size="0.02" mass="0.4"/>
<site name="b" pos="-0.5 0 -1"/></body>
</worldbody><equality><weld name="native_grip" site1="a" site2="b"/></equality>
</mujoco>"""

NO_FIX_MIN_RESIDUAL = 1e-2


def _state(model):
    """Closed-loop state with a velocity that satisfies the weld (J v = 0)."""
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    sites = [model.site(n).id for n in ("a", "b")]
    jac, _ = _evaluate_weld_closure(mujoco, model, data, sites)
    raw = np.linspace(-0.4, 0.5, model.nv)
    data.qvel[:] = raw - np.linalg.pinv(jac) @ (jac @ raw)
    return data


def _forward(model, data):
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)


def _kkt(model, data, efforts):
    """Explicit KKT solve: (qacc, lambda) exactly like the full-body adapter."""
    _forward(model, data)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    sites = [model.site(n).id for n in ("a", "b")]
    jac, drift = _evaluate_weld_closure(mujoco, model, data, sites)
    return _solve_kkt_with_multipliers(
        mass, efforts - data.qfrc_bias, jac, drift, regularization=0.0
    )


def _closure(model, data, lam):
    return GripClosure.from_multiplier(
        body_a="hand",
        body_b="club",
        point_a_m=data.site_xpos[model.site("a").id],
        point_b_m=data.site_xpos[model.site("b").id],
        multiplier=lam,
    )


@pytest.fixture
def loop():
    model = mujoco.MjModel.from_xml_string(LOOP)
    data = _state(model)
    rng = np.random.default_rng(13179)
    efforts = rng.normal(0.0, 3.0, model.nv)
    accel, lam = _kkt(model, data, efforts)
    return model, data, efforts, accel, lam


def test_closure_pair_obeys_newtons_third_law(loop):
    model, data, _, _, lam = loop
    closure = _closure(model, data, lam)
    on_hand, on_club = closure.wrenches
    assert (on_hand.body, on_club.body) == ("hand", "club")
    force = np.add(on_hand.force_n, on_club.force_n)
    assert np.abs(force).max() <= 1e-9
    origin = np.array([0.3, -0.2, 0.7])
    moment = sum(
        np.cross(np.asarray(w.point_m) - origin, w.force_n) + np.asarray(w.torque_nm)
        for w in closure.wrenches
    )
    assert np.abs(moment).max() <= 1e-9
    assert np.linalg.norm(lam) > 1.0, "fixture must load the weld"


def test_unavailable_closure_is_never_zero():
    closure = GripClosure.unavailable("no efforts supplied")
    assert not closure.available
    assert closure.wrenches == ()
    assert closure.multiplier_world is None
    assert closure.reason == "no efforts supplied"
    with pytest.raises(ValueError, match="reason"):
        GripClosure.unavailable("")


def test_closure_rejects_bad_multiplier(loop):
    model, data, *_ = loop
    with pytest.raises(ValueError, match="shape"):
        _closure(model, data, np.zeros(5))
    with pytest.raises(ValueError, match="finite"):
        _closure(model, data, np.full(6, np.nan))


def test_inverse_recovers_efforts_with_closure_injected(loop):
    model, data, efforts, accel, lam = loop
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    closure = _closure(model, data, lam)
    required = source.required_generalized_force(
        data.qpos, data.qvel, accel, external_wrenches=[], closure=closure
    )
    assert np.abs(required - efforts).max() <= 1e-6


def test_fix_bites_against_soft_weld_and_open_loop(loop):
    model, data, efforts, accel, lam = loop
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    legacy = mujoco.MjData(model)
    legacy.qpos[:], legacy.qvel[:], legacy.qacc[:] = data.qpos, data.qvel, accel
    mujoco.mj_inverse(model, legacy)  # weld allocated by solref/solimp impedance
    assert np.abs(legacy.qfrc_inverse - efforts).max() > NO_FIX_MIN_RESIDUAL
    open_loop = source.required_generalized_force(
        data.qpos,
        data.qvel,
        accel,
        external_wrenches=[],
        closure=GripClosure.unavailable("test"),
    )
    assert np.abs(open_loop - efforts).max() > NO_FIX_MIN_RESIDUAL


def test_live_model_equality_is_untouched(loop):
    model, data, _, accel, lam = loop
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    active = model.eq_active0.copy(), model.opt.disableflags
    source.sample_prescribed(
        data.qpos, data.qvel, accel, 0.0, [], closure=_closure(model, data, lam)
    )
    np.testing.assert_array_equal(model.eq_active0, active[0])
    assert model.opt.disableflags == active[1]


def test_sample_with_closure_reports_loads_and_status(loop):
    model, data, _, accel, lam = loop
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    frame = source.sample_prescribed(
        data.qpos, data.qvel, accel, 0.0, [], closure=_closure(model, data, lam)
    )
    assert np.isfinite(frame.values_n["club"])
    assert "grip closure wrench applied" in frame.source


@pytest.mark.parametrize("closure", ["omitted", "unavailable"])
def test_missing_closure_marks_loop_bodies_unavailable(loop, closure):
    model, data, _, accel, _ = loop
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    kwargs = {}
    if closure == "unavailable":
        kwargs["closure"] = GripClosure.unavailable("efforts not provided")
    frame = source.sample_prescribed(data.qpos, data.qvel, accel, 0.0, [], **kwargs)
    assert frame.values_n["club"] is None
    assert "grip closure unavailable" in frame.source
