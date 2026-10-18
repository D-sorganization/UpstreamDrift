"""Native MuJoCo equilibrium sign and simulation-state isolation probes."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.shared.python.body_part_viz.mujoco_axial_loads import MujocoAxialLoadSource

pytestmark = pytest.mark.unit


ROD = """<mujoco><option gravity="0 0 -9.81"/><worldbody>
<body name="arbitrary_rod" pos="0 0 2"><joint type="hinge" axis="0 1 0"/>
<geom type="capsule" fromto="0 0 0 0 0 -1" size="0.03" mass="2"/>
</body></worldbody></mujoco>"""


@pytest.mark.parametrize("angle,sign", [(0.0, 1.0), (np.pi, -1.0)])
def test_native_static_tension_compression_and_no_state_mutation(angle, sign):
    model = mujoco.MjModel.from_xml_string(ROD)
    data = mujoco.MjData(model)
    data.qpos[0] = angle
    data.time = 1.25
    before = data.qpos.copy(), data.qvel.copy(), data.qacc.copy(), data.cfrc_int.copy()
    source = MujocoAxialLoadSource(model)
    frame = source.sample(data)
    assert frame.values_n["arbitrary_rod"] == pytest.approx(sign * 19.62)
    assert frame.time_s == 1.25
    for actual, expected in zip(
        (data.qpos, data.qvel, data.qacc, data.cfrc_int), before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_missing_section_axis_is_unavailable():
    model = mujoco.MjModel.from_xml_string(
        ROD.replace(
            'type="capsule" fromto="0 0 0 0 0 -1"', 'type="sphere" pos="0 0 -1"'
        )
    )
    assert MujocoAxialLoadSource(model).sample(mujoco.MjData(model)) is None


def test_native_dynamic_centripetal_tension():
    model = mujoco.MjModel.from_xml_string(ROD.replace("0 0 -9.81", "0 0 0"))
    data = mujoco.MjData(model)
    data.qvel[0] = 2.0
    frame = MujocoAxialLoadSource(model).sample(data)
    assert frame.values_n["arbitrary_rod"] == pytest.approx(4.0)


def test_engine_provider_and_scene_colors_preserve_native_model():
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )
    from src.shared.python.body_part_viz import ForceColorScale
    from src.shared.python.body_part_viz.mujoco_force_colors import (
        apply_mujoco_scene_colors,
    )

    engine = MuJoCoPhysicsEngine()
    engine.load_from_string(ROD)
    frame = engine.get_segment_axial_loads()
    model, data = engine.model, engine.data
    scene = mujoco.MjvScene(model, maxgeom=20)
    option, camera = mujoco.MjvOption(), mujoco.MjvCamera()
    mujoco.mj_forward(model, data)
    mujoco.mjv_updateScene(
        model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    originals = model.geom_rgba.copy()
    positions = [scene.geoms[i].pos.copy() for i in range(scene.ngeom)]
    apply_mujoco_scene_colors(
        model, scene, frame, ForceColorScale(enabled=True, tension_limit_n=1)
    )
    np.testing.assert_allclose(scene.geoms[0].rgba[:3], [0, 0, 1])
    np.testing.assert_array_equal(model.geom_rgba, originals)
    for i, position in enumerate(positions):
        np.testing.assert_array_equal(scene.geoms[i].pos, position)
    mujoco.mjv_updateScene(
        model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    apply_mujoco_scene_colors(model, scene, frame, ForceColorScale())
    np.testing.assert_allclose(scene.geoms[0].rgba, originals[0])


def test_native_widget_exposes_shared_force_controls(monkeypatch):
    from PyQt6.QtWidgets import QApplication, QToolButton
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
        sim_widget,
    )
    from src.shared.python.body_part_viz import ForceColorScale

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(sim_widget, "MuJoCoMeshcatAdapter", lambda: None)
    widget = sim_widget.MuJoCoSimWidget()
    widget.timer.stop()
    assert widget.findChild(QToolButton, "segment_force_colors") is not None
    widget.set_axial_color_scale(ForceColorScale(enabled=True))
    assert widget.axial_color_scale.enabled
    widget.close()
    app.processEvents()


def _discovered_bodies(xml: str) -> set[int]:
    model = mujoco.MjModel.from_xml_string(xml)
    return set(MujocoAxialLoadSource(model)._axes)


@pytest.mark.parametrize("geom_type", ["capsule", "cylinder"])
def test_axis_discovery_accepts_capsule_and_cylinder_rods(geom_type):
    """Geom/joint type ints from MjModel must match enums on every MuJoCo version."""
    assert _discovered_bodies(ROD.replace("capsule", geom_type)) == {1}


def test_axis_discovery_skips_free_bodies_and_non_rod_geoms():
    free = ROD.replace('<joint type="hinge" axis="0 1 0"/>', "<freejoint/>")
    assert _discovered_bodies(free) == set()
    box = ROD.replace(
        'type="capsule" fromto="0 0 0 0 0 -1" size="0.03"',
        'type="box" size="0.1 0.1 0.1"',
    )
    assert _discovered_bodies(box) == set()


# --- kinematic axis policy and prescribed-motion (inverse dynamics) sampling ---

G = 9.81
PENDULUM = f"""<mujoco><option gravity="0 0 -{G}"/><worldbody>
<body name="upper" pos="0 0 3"><joint name="j1" type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.05" pos="0 0 -0.5" mass="3"/>
<body name="lower" pos="0 0 -1"><joint name="j2" type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.05" pos="0 0 -0.5" mass="2"/>
</body></body></worldbody></mujoco>"""
COLUMN = f"""<mujoco><option gravity="0 0 -{G}"/><worldbody>
<body name="base" pos="0 0 0"><joint type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.05" pos="0 0 0.5" mass="3"/>
<body name="top" pos="0 0 1"><joint type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.05" pos="0 0 0.5" mass="2"/>
</body></body></worldbody></mujoco>"""


def _source(xml: str, policy: str = "kinematic") -> MujocoAxialLoadSource:
    return MujocoAxialLoadSource(
        mujoco.MjModel.from_xml_string(xml), axis_policy=policy
    )


def test_default_policy_is_rod_and_unchanged():
    model = mujoco.MjModel.from_xml_string(PENDULUM)
    source = MujocoAxialLoadSource(model)
    assert source.axis_policy == "rod"
    assert dict(source.axis_sources) == {}
    assert source.sample(mujoco.MjData(model)) is None


def test_invalid_axis_policy_rejected():
    model = mujoco.MjModel.from_xml_string(PENDULUM)
    with pytest.raises(ValueError, match="axis_policy"):
        MujocoAxialLoadSource(model, axis_policy="guess")


def test_kinematic_axis_sources_record_rule():
    source = _source(PENDULUM)
    assert dict(source.axis_sources) == {"upper": "kinematic", "lower": "kinematic"}
    rod = _source(ROD)
    assert dict(rod.axis_sources) == {"arbitrary_rod": "rod"}


def test_kinematic_policy_keeps_rod_rule_first():
    source = _source(ROD.replace("capsule", "cylinder"))
    assert dict(source.axis_sources) == {"arbitrary_rod": "rod"}


def test_kinematic_skips_free_and_degenerate_bodies():
    free = PENDULUM.replace(
        '<joint name="j1" type="hinge" axis="0 1 0"/>', "<freejoint/>"
    )
    assert "upper" not in _source(free).axis_sources
    degenerate = """<mujoco><worldbody><body name="a"><joint type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>"""
    assert dict(_source(degenerate).axis_sources) == {}


def test_kinematic_uses_first_joint_anchor_when_anchors_differ():
    xml = PENDULUM.replace(
        '<joint name="j2" type="hinge" axis="0 1 0"/>',
        '<joint type="hinge" axis="0 1 0"/>'
        '<joint type="hinge" axis="1 0 0" pos="0.2 0 0"/>',
    )
    assert _source(xml).axis_sources["lower"] == "kinematic"
    # The strict rod rule still demands a common anchor.
    assert "lower" not in _source(xml, "rod").axis_sources


TEE = f"""<mujoco><option gravity="0 0 -{G}"/><worldbody>
<body name="stem" pos="0 0 3"><joint type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.05" pos="0 0 -0.5" mass="1"/>
<body name="left" pos="-0.3 0 -1"><joint type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.05" mass="2"/></body>
<body name="right" pos="0.3 0 -1"><joint type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.05" mass="2"/></body>
</body></worldbody></mujoco>"""


def test_centroid_rule_axis_runs_along_stem_for_symmetric_tee():
    source = _source(TEE)
    assert source.axis_sources["stem"] == "kinematic_centroid"
    assert "left" not in source.axis_sources  # COM at anchor: degenerate
    (_, proximal, distal) = next(v for v in source._axes.values() if v[0] == "stem")
    np.testing.assert_allclose(proximal, [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(distal, [0, 0, -1], atol=1e-12)


def test_hanging_tee_reads_tension_of_mass_below():
    model = mujoco.MjModel.from_xml_string(TEE)
    frame = MujocoAxialLoadSource(model, axis_policy="kinematic").sample(
        mujoco.MjData(model)
    )
    assert frame.values_n["stem"] == pytest.approx(5 * G, rel=1e-2)


def test_centroid_degenerate_axis_is_skipped():
    xml = TEE.replace('pos="-0.3 0 -1"', 'pos="-0.3 0 0"').replace(
        'pos="0.3 0 -1"', 'pos="0.3 0 0"'
    )
    assert "stem" not in _source(xml).axis_sources


def test_translational_root_is_skipped():
    root = """<mujoco><worldbody><body name="pelvis" pos="0 0 1">
<joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
<joint type="slide" axis="0 0 1"/><geom type="sphere" size="0.1" mass="1"/>
<body name="a" pos="0 0 0.5"><joint type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.1" mass="1"/></body>
<body name="b" pos="0 0.2 0.5"><joint type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.1" mass="1"/></body></body></worldbody></mujoco>"""
    assert "pelvis" not in _source(root).axis_sources


def test_kinematic_hanging_pendulum_reads_tension_static():
    model = mujoco.MjModel.from_xml_string(PENDULUM)
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    frame = source.sample(mujoco.MjData(model))
    assert frame.values_n["lower"] == pytest.approx(2 * G, rel=1e-2)
    assert frame.values_n["upper"] == pytest.approx(5 * G, rel=1e-2)
    assert "kinematic" in frame.source


def test_kinematic_upright_column_reads_compression_static():
    model = mujoco.MjModel.from_xml_string(COLUMN)
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    frame = source.sample(mujoco.MjData(model))
    assert frame.values_n["top"] == pytest.approx(-2 * G, rel=1e-2)
    assert frame.values_n["base"] == pytest.approx(-5 * G, rel=1e-2)


@pytest.mark.parametrize(
    "xml,lower,upper",
    [(PENDULUM, 2 * G, 5 * G), (COLUMN, -2 * G, -5 * G)],
)
def test_sample_prescribed_matches_static_physics(xml, lower, upper):
    model = mujoco.MjModel.from_xml_string(xml)
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    zeros = np.zeros(model.nv)
    frame = source.sample_prescribed(np.zeros(model.nq), zeros, zeros, 0.5)
    key = ("lower", "upper") if xml is PENDULUM else ("top", "base")
    assert frame.time_s == 0.5
    assert frame.values_n[key[0]] == pytest.approx(lower, rel=1e-2)
    assert frame.values_n[key[1]] == pytest.approx(upper, rel=1e-2)


def test_sample_prescribed_does_not_mutate_inputs():
    model = mujoco.MjModel.from_xml_string(PENDULUM)
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    qpos, qvel, qacc = np.array([0.3, -0.2]), np.array([1.0, 2.0]), np.array([0.5, 0.1])
    copies = qpos.copy(), qvel.copy(), qacc.copy()
    source.sample_prescribed(qpos, qvel, qacc, 0.0)
    for actual, expected in zip((qpos, qvel, qacc), copies, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_sample_prescribed_uses_prescribed_acceleration():
    """Free-space (no gravity) pendulum: tension = centripetal + tangential parts."""
    model = mujoco.MjModel.from_xml_string(PENDULUM.replace(f"-{G}", "0"))
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    qvel = np.array([0.0, 2.0])
    zero = np.zeros(2)
    moving = source.sample_prescribed(zero, qvel, zero, 0.0)
    at_rest = source.sample_prescribed(zero, zero, zero, 0.0)
    assert at_rest.values_n["lower"] == pytest.approx(0.0, abs=1e-9)
    # lower COM is 0.5 m from j2; centripetal force m w^2 r = 2*4*0.5 = 4 N.
    assert moving.values_n["lower"] == pytest.approx(4.0, rel=1e-2)


def test_sample_prescribed_validates_inputs():
    model = mujoco.MjModel.from_xml_string(PENDULUM)
    source = MujocoAxialLoadSource(model, axis_policy="kinematic")
    ok = np.zeros(2)
    with pytest.raises(ValueError, match="qpos"):
        source.sample_prescribed(np.zeros(3), ok, ok, 0.0)
    with pytest.raises(ValueError, match="qvel"):
        source.sample_prescribed(ok, np.zeros(1), ok, 0.0)
    with pytest.raises(ValueError, match="qacc"):
        source.sample_prescribed(ok, ok, np.array([0.0, np.nan]), 0.0)
    with pytest.raises(ValueError, match="time_s"):
        source.sample_prescribed(ok, ok, ok, float("inf"))


def test_sample_prescribed_unavailable_without_axes():
    model = mujoco.MjModel.from_xml_string(PENDULUM)
    source = MujocoAxialLoadSource(model)  # rod policy: no qualified bodies
    assert source.sample_prescribed(np.zeros(2), np.zeros(2), np.zeros(2), 0.0) is None
