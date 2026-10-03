"""Unit tests for MuJoCo MjvScene glyph renderer (FTO-6, #11291)."""

from __future__ import annotations

import math
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
    SceneGlyphReceipt,
    add_glyphs_to_scene,
    segment_geom_count,
)
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
)


def _make_glyph_set(
    arrows: tuple[ArrowGlyph, ...] = (),
    torque_arcs: tuple[TorqueArcGlyph, ...] = (),
) -> GlyphSet:
    return GlyphSet(
        time_s=0.0,
        arrows=arrows,
        torque_arcs=torque_arcs,
        legend=LegendSpec(
            force_reference_n=None,
            force_reference_length_m=None,
            torque_reference_nm=None,
            torque_reference_radius_m=None,
            kinds_present=(),
            unavailable_labels=(),
            engine="mujoco",
            source_labels=(),
        ),
    )


def _make_dummy_arrow(
    label: str = "test_arrow",
    tail_m: tuple[float, float, float] = (1.0, 2.0, 3.0),
    tip_m: tuple[float, float, float] = (4.0, 6.0, 8.0),
    rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    shaft_radius_m: float = 0.02,
) -> ArrowGlyph:
    tail_arr = np.asarray(tail_m, dtype=np.float64)
    tip_arr = np.asarray(tip_m, dtype=np.float64)
    head_base_arr = tip_arr - 0.2 * (tip_arr - tail_arr)
    head_base_m = (
        float(head_base_arr[0]),
        float(head_base_arr[1]),
        float(head_base_arr[2]),
    )
    return ArrowGlyph(
        label=label,
        kind="applied",
        tail_m=tail_m,
        tip_m=tip_m,
        head_base_m=head_base_m,
        shaft_radius_m=shaft_radius_m,
        head_radius_m=2.4 * shaft_radius_m,
        rgba=rgba,
        magnitude=100.0,
        units="N",
        clamped=False,
    )


def _make_dummy_arc(
    label: str = "test_arc",
    n_segments: int = 32,
    radius_m: float = 0.2,
    rgba: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> TorqueArcGlyph:
    thetas = np.linspace(0.0, 1.5 * math.pi, n_segments + 1)
    pts = tuple(
        (float(radius_m * math.cos(th)), float(radius_m * math.sin(th)), 0.0)
        for th in thetas
    )
    head_tip = pts[-1]
    head_base = (float(pts[-1][0]), float(pts[-1][1] - 0.05), float(pts[-1][2]))
    return TorqueArcGlyph(
        label=label,
        kind="joint_torque",
        center_m=(0.0, 0.0, 0.0),
        axis_unit=(0.0, 0.0, 1.0),
        radius_m=radius_m,
        polyline_m=pts,
        head_tip_m=head_tip,
        head_base_m=head_base,
        rgba=rgba,
        magnitude=50.0,
        units="N*m",
        clamped=False,
    )


@pytest.mark.unit
def test_one_arrow_geom_and_endpoints_match() -> None:
    """One arrow adds exactly one mjGEOM_ARROW geom whose endpoints match within 1e-9."""
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    scene = mujoco.MjvScene(model, maxgeom=10)
    assert scene.ngeom == 0

    arrow = _make_dummy_arrow(
        tail_m=(1.0, 2.0, 3.0),
        tip_m=(4.0, 6.0, 8.0),
        rgba=(0.8, 0.2, 0.1, 1.0),
        shaft_radius_m=0.03,
    )
    glyph_set = _make_glyph_set(arrows=(arrow,))

    receipt = add_glyphs_to_scene(scene, glyph_set)
    assert receipt.added == 1
    assert receipt.dropped == 0
    assert scene.ngeom == 1

    geom = scene.geoms[0]
    assert geom.type == int(mujoco.mjtGeom.mjGEOM_ARROW)
    np.testing.assert_allclose(geom.rgba, [0.8, 0.2, 0.1, 1.0], atol=1e-6)

    # In MuJoCo mjv_connector for mjGEOM_ARROW:
    # geom.pos is the tail of the arrow
    # geom.mat[:, 2] is the normalized direction vector along the shaft
    # geom.size[2] is the total length
    tail_reconstructed = geom.pos
    np.testing.assert_allclose(tail_reconstructed, arrow.tail_m, atol=1e-9)

    z_dir = geom.mat.reshape(3, 3)[:, 2]
    tip_reconstructed = geom.pos + geom.size[2] * z_dir
    np.testing.assert_allclose(tip_reconstructed, arrow.tip_m, atol=1e-9)


@pytest.mark.unit
def test_32_segment_arc_adds_32_capsules_plus_1_arrow() -> None:
    """A 32-segment arc adds 32 capsules plus 1 arrow."""
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    scene = mujoco.MjvScene(model, maxgeom=100)
    assert scene.ngeom == 0

    arc = _make_dummy_arc(n_segments=32, rgba=(0.1, 0.5, 0.9, 1.0))
    glyph_set = _make_glyph_set(torque_arcs=(arc,))

    receipt = add_glyphs_to_scene(scene, glyph_set)
    assert receipt.added == 33  # 32 capsules + 1 arrow head
    assert receipt.dropped == 0
    assert scene.ngeom == 33

    capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW)

    # First 32 geoms must be capsules
    for i in range(32):
        g = scene.geoms[i]
        assert g.type == capsule_type, f"geom[{i}] should be capsule"
        np.testing.assert_allclose(g.rgba, [0.1, 0.5, 0.9, 1.0], atol=1e-6)

    # 33rd geom (index 32) must be the arrow head
    head_geom = scene.geoms[32]
    assert head_geom.type == arrow_type
    np.testing.assert_allclose(head_geom.rgba, [0.1, 0.5, 0.9, 1.0], atol=1e-6)


@pytest.mark.unit
def test_overflow_records_dropped_without_exception() -> None:
    """A scene with maxgeom=ngeom+2 and 5 glyphs gives added=2, dropped=3."""
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    scene = mujoco.MjvScene(model, maxgeom=2)
    assert scene.ngeom == 0
    assert scene.maxgeom == 2

    arrows = tuple(
        _make_dummy_arrow(
            label=f"arrow_{i}",
            tail_m=(float(i), 0.0, 0.0),
            tip_m=(float(i), 1.0, 0.0),
        )
        for i in range(5)
    )
    glyph_set = _make_glyph_set(arrows=arrows)

    receipt = add_glyphs_to_scene(scene, glyph_set)
    assert receipt == SceneGlyphReceipt(added=2, dropped=3)
    assert scene.ngeom == 2


@pytest.mark.unit
def test_segment_geom_count() -> None:
    """segment_geom_count correctly returns expected geom capacity."""
    arrow1 = _make_dummy_arrow("a1")
    arrow2 = _make_dummy_arrow("a2")
    arc = _make_dummy_arc(n_segments=32)

    assert segment_geom_count(_make_glyph_set()) == 0
    assert segment_geom_count(_make_glyph_set(arrows=(arrow1, arrow2))) == 2
    assert (
        segment_geom_count(_make_glyph_set(arrows=(arrow1,), torque_arcs=(arc,)))
        == 1 + 33
    )


@pytest.mark.unit
@pytest.mark.requires_gl
def test_offscreen_pixel_test_renders_arrow_colour() -> None:
    """Render a tiny MJCF with one large arrow and assert arrow-coloured pixels exist."""
    xml = """
    <mujoco>
      <visual>
        <global offwidth="64" offheight="64"/>
      </visual>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="1 1 0.1" rgba="0.2 0.2 0.2 1"/>
        <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.5 1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, 64, 64)

    # 1. Baseline clean render without glyphs
    renderer.update_scene(data, "cam")
    clean_pixels = renderer.render().copy()

    # Red color filter: bright red, low green, low blue
    def count_red_pixels(img: np.ndarray) -> int:
        mask = (img[:, :, 0] > 180) & (img[:, :, 1] < 40) & (img[:, :, 2] < 40)
        return int(np.sum(mask))

    assert count_red_pixels(clean_pixels) == 0

    # 2. Render with a bright red arrow
    arrow = _make_dummy_arrow(
        tail_m=(-0.5, 0.0, 0.2),
        tip_m=(0.5, 0.0, 0.2),
        rgba=(1.0, 0.0, 0.0, 1.0),
        shaft_radius_m=0.04,
    )
    glyphs = _make_glyph_set(arrows=(arrow,))

    renderer.update_scene(data, "cam")
    receipt = add_glyphs_to_scene(renderer.scene, glyphs)
    assert receipt.added == 1
    assert receipt.dropped == 0

    arrow_pixels = renderer.render().copy()
    red_count = count_red_pixels(arrow_pixels)
    assert red_count > 0, f"Expected red arrow pixels, found {red_count}"
