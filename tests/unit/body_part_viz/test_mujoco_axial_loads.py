"""Native MuJoCo equilibrium sign and simulation-state isolation probes."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.shared.python.body_part_viz.mujoco_axial_loads import MujocoAxialLoadSource


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
