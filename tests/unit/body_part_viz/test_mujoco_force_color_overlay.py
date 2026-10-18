"""Translucent overlay applied to a native MuJoCo scene keeps the base visible."""

import pytest

mujoco = pytest.importorskip("mujoco")

from src.shared.python.body_part_viz import AxialLoadFrame, ForceColorScale  # noqa: E402
from src.shared.python.body_part_viz.mujoco_force_colors import (  # noqa: E402
    apply_mujoco_scene_colors,
)

pytestmark = pytest.mark.unit

XML = """<mujoco><worldbody><body name="rod">
<geom type="capsule" size="0.05 0.2" rgba="1 1 0 0.4"/>
</body></worldbody></mujoco>"""


def _scene():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    scene = mujoco.MjvScene(model, maxgeom=10)
    mujoco.mj_forward(model, data)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return model, scene


def _rgba(scale, force):
    model, scene = _scene()
    apply_mujoco_scene_colors(
        model, scene, AxialLoadFrame(0, {"rod": force}, "s"), scale
    )
    return [float(v) for v in scene.geoms[0].rgba]


def test_replace_opacity_blends_with_scene_base_and_keeps_alpha():
    scale = ForceColorScale(
        enabled=True, tension_limit_n=1, compression_limit_n=1, opacity=0.5
    )
    assert _rgba(scale, 10) == pytest.approx([0.5, 0.5, 0.5, 0.4])


def test_opaque_replace_is_unchanged_legacy_behaviour():
    scale = ForceColorScale(enabled=True, tension_limit_n=1, compression_limit_n=1)
    assert _rgba(scale, 10) == pytest.approx([0.0, 0.0, 1.0, 0.4])


def test_tint_keeps_original_color_at_zero_load():
    scale = ForceColorScale(
        enabled=True, opacity=0.5, blend_mode="tint", deadband_n=1.0
    )
    assert _rgba(scale, 0.0) == pytest.approx([1.0, 1.0, 0.0, 0.4])
    assert _rgba(scale, -1e6) == pytest.approx([1.0, 0.5, 0.0, 0.4])
