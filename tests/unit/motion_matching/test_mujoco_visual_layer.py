"""MuJoCo visual layer: capsule skeleton, floor and lights without physics change."""

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FULL_BODY = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def test_visual_export_adds_geoms_but_keeps_physics_identical() -> None:
    mujoco = pytest.importorskip("mujoco")
    raw = FULL_BODY.read_bytes()
    plain_xml, plain_meta = exporter.export_full_body_mjcf(raw)
    visual_xml, visual_meta = exporter.export_full_body_mjcf(raw, visual=True)
    plain = mujoco.MjModel.from_xml_string(plain_xml)
    visual = mujoco.MjModel.from_xml_string(visual_xml)
    assert visual.nq == plain.nq == 41 and visual.nv == plain.nv == 41
    assert visual.nbody == plain.nbody
    assert visual.ngeom > plain.ngeom + 20  # one capsule per body plus floor
    assert visual.nlight >= 1
    assert visual_meta["visual_layer"]["capsules"] >= plain.nbody - 1
    assert visual_meta["visual_layer"]["floor"] is True
    assert "visual_layer" not in plain_meta
    # Visual geoms never collide and never contribute inertia.
    for i in range(visual.ngeom):
        if visual.geom_group[i] == 1:
            assert visual.geom_contype[i] == 0 and visual.geom_conaffinity[i] == 0
    np.testing.assert_allclose(visual.body_mass, plain.body_mass)
    np.testing.assert_allclose(visual.body_inertia, plain.body_inertia)
    d_plain, d_visual = mujoco.MjData(plain), mujoco.MjData(visual)
    rng = np.random.default_rng(0)
    q = rng.normal(scale=0.2, size=plain.nq)
    d_plain.qpos[:] = q
    d_visual.qpos[:] = q
    mujoco.mj_forward(plain, d_plain)
    mujoco.mj_forward(visual, d_visual)
    np.testing.assert_allclose(d_visual.xpos, d_plain.xpos, atol=1e-12)
    np.testing.assert_allclose(d_visual.qacc, d_plain.qacc, atol=1e-9)


def test_visual_export_renders_offscreen(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml, _ = exporter.export_full_body_mjcf(FULL_BODY.read_bytes(), visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, 120, 160)
    except (RuntimeError, ValueError, OSError) as error:  # headless CI without GL
        pytest.skip(f"no offscreen GL: {error}")
    renderer.update_scene(data)
    image = renderer.render()
    assert image.shape == (120, 160, 3)
    assert image.mean() > 5.0  # not a black frame: floor and lit capsules visible


def test_whole_body_com_and_scene_markers(tmp_path: Path) -> None:
    import mujoco

    from src.engines.physics_engines.mujoco.python import visual_layer

    xml, _ = exporter.export_full_body_mjcf(FULL_BODY.read_bytes(), visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    com = visual_layer.whole_body_com(model, data)
    masses = model.body_mass[1:]
    expected = (masses[:, None] * data.xipos[1:]).sum(axis=0) / masses.sum()
    np.testing.assert_allclose(com, expected, atol=1e-9)
    scene = mujoco.MjvScene(model, maxgeom=4000)
    option = mujoco.MjvOption()
    mujoco.mjv_updateScene(
        model, data, option, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    before = scene.ngeom
    visual_layer.add_com_markers(scene, model, data, 0.0)
    assert scene.ngeom == before + 2
    with pytest.raises(ValueError):
        visual_layer.add_scene_marker(scene, com, -0.01, (1, 0, 0, 1))
