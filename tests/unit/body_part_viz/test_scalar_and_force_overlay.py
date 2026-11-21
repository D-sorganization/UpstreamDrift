"""Force shading wins when stored-KE overlay is also enabled (#15396)."""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from src.shared.python.body_part_viz import AxialLoadFrame, ForceColorScale  # noqa: E402
from src.shared.python.body_part_viz.meshcat_force_colors import MeshcatForceColors  # noqa: E402
from src.shared.python.body_part_viz.meshcat_scalar_colors import MeshcatScalarColors  # noqa: E402
from src.shared.python.body_part_viz.mujoco_force_colors import (  # noqa: E402
    apply_mujoco_scene_colors,
)
from src.shared.python.body_part_viz.mujoco_scalar_colors import (  # noqa: E402
    apply_mujoco_scalar_scene_colors,
)
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale  # noqa: E402

pytestmark = pytest.mark.unit

XML = """<mujoco><worldbody><body name="rod">
<geom type="capsule" size="0.05 0.2" rgba="0.8 0.2 0.2 1"/>
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


def _bytes_rgb(rgb: tuple[float, ...]) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in rgb)


def test_mujoco_force_and_ke_overlay_force_wins() -> None:
    model, scene = _scene()
    force_scale = ForceColorScale(
        enabled=True, tension_limit_n=1, compression_limit_n=1
    )
    frame = AxialLoadFrame(0.0, {"rod": 10.0}, "t")
    apply_mujoco_scene_colors(model, scene, frame, force_scale)
    force_only = _bytes_rgb(tuple(scene.geoms[0].rgba[:3]))

    model2, scene2 = _scene()
    ke_scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=100.0,
    )
    apply_mujoco_scalar_scene_colors(model2, scene2, {"rod": 50.0}, ke_scale)
    apply_mujoco_scene_colors(model2, scene2, frame, force_scale)
    both = _bytes_rgb(tuple(scene2.geoms[0].rgba[:3]))
    assert both == force_only


def test_meshcat_force_and_ke_overlay_force_wins() -> None:
    base = [0.8, 0.2, 0.2, 1.0]
    bindings = {"rod": {"visuals/rod/<object>": base}}
    force = ForceColorScale(enabled=True, tension_limit_n=1, compression_limit_n=1)
    ke = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=100.0,
    )
    frame = AxialLoadFrame(0.0, {"rod": 10.0}, "t")
    colors: list[list[float]] = []

    def setter(path: str, prop: str, value: list[float]) -> None:
        colors.append(list(value))

    force_adapter = MeshcatForceColors(setter, bindings)
    force_adapter.apply(frame, force)
    force_only = _bytes_rgb(tuple(colors[-1][:3]))

    colors.clear()
    scalar_adapter = MeshcatScalarColors(setter, bindings)
    scalar_adapter.apply({"rod": 50.0}, ke)
    force_adapter.apply(frame, force)
    both = _bytes_rgb(tuple(colors[-1][:3]))
    assert both == force_only


def test_mujoco_ke_disabled_restores_scene_base_after_refresh() -> None:
    model, scene = _scene()
    base = [float(v) for v in scene.geoms[0].rgba[:3]]
    ke_scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=100.0,
    )
    apply_mujoco_scalar_scene_colors(model, scene, {"rod": 50.0}, ke_scale)
    assert list(scene.geoms[0].rgba[:3]) != pytest.approx(base)

    _, scene2 = _scene()
    assert list(scene2.geoms[0].rgba[:3]) == pytest.approx(base)
