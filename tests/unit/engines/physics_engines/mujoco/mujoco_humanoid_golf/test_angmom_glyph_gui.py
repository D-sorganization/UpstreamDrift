"""MuJoCo angular-momentum overlay smoke (#15398)."""

from __future__ import annotations

from typing import Any

import pytest

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.angmom_glyph_overlay import (
    build_angmom_overlay_glyphs,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
    add_glyphs_to_scene,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_angmom_mixin import (
    SimAngMomMixin,
)
from src.shared.python.body_part_viz.angular_momentum_glyphs import AngularMomentumArrowStyle
from src.shared.python.body_part_viz.segment_angular_momentum import SegmentGroupCatalog
from src.shared.python.force_overlay.renderers.meshcat_glyphs import MeshcatGlyphRenderer
from tests.unit.force_overlay.test_meshcat_glyphs import RecordingFakeSink

pytestmark = pytest.mark.unit

_XML = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="8"/>
      <body name="torso" pos="0 0 0.15">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.07" fromto="0 0 0 0 0 0.3" mass="10"/>
        <body name="club" pos="0.3 0 0.2">
          <geom type="capsule" size="0.02" fromto="0 0 0 0.4 0 0" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _catalog() -> SegmentGroupCatalog:
    return SegmentGroupCatalog(
        legs=frozenset(),
        pelvis=frozenset({"pelvis"}),
        torso=frozenset({"torso"}),
        arms=frozenset(),
        club=frozenset({"club"}),
    )


def test_native_scene_receives_angmom_arrows() -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    data.qvel[:] = 2.0
    mujoco.mj_forward(model, data)
    glyphs = build_angmom_overlay_glyphs(
        model,
        data,
        catalog=_catalog(),
        enabled=True,
        style=AngularMomentumArrowStyle(),
    )
    assert glyphs is not None
    scene = mujoco.MjvScene(model, maxgeom=200)
    before = scene.ngeom
    receipt = add_glyphs_to_scene(scene, glyphs)
    assert receipt.added > 0
    assert scene.ngeom > before


def test_meshcat_renderer_accepts_angmom_glyphs() -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    data.qvel[:] = 2.0
    mujoco.mj_forward(model, data)
    glyphs = build_angmom_overlay_glyphs(
        model,
        data,
        catalog=_catalog(),
        enabled=True,
        style=AngularMomentumArrowStyle(),
    )
    assert glyphs is not None
    sink = RecordingFakeSink()
    MeshcatGlyphRenderer(sink, root="/angmom").update(glyphs)
    assert sink.cylinders or sink.transforms


def test_widget_mixin_builds_glyphs_when_enabled() -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    data.qvel[:] = 2.0
    mujoco.mj_forward(model, data)

    class _Engine:
        def segment_group_catalog(self) -> SegmentGroupCatalog:
            return _catalog()

    class _Widget(SimAngMomMixin):
        def __init__(self) -> None:
            self._init_angmom_overlay_state()
            self.show_angular_momentum_arrows = True
            self.model = model
            self.data = data
            self.engine = _Engine()

    widget = _Widget()
    glyphs = widget._build_angular_momentum_glyphs()
    assert glyphs is not None
    assert len(glyphs.arrows) >= 1