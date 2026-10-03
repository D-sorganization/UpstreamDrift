"""Unit tests for ForceGlyphStyle, build_glyphs, and FORCE_KIND_PALETTE (#11288)."""

from __future__ import annotations

import math
from typing import Mapping
import pytest

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.plot_style import FORCE_KIND_PALETTE

try:
    from src.shared.python.force_overlay.glyphs import (
        ArrowGlyph,
        ForceGlyphStyle,
        GlyphSet,
        LegendSpec,
        TorqueArcGlyph,
        build_glyphs,
        scale_for_view,
    )
except ImportError:
    # Expected during TDD red phase before implementation exists
    pass

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_force_kind_palette_registration_and_reservation() -> None:
    """FORCE_KIND_PALETTE must contain all WrenchKind keys and never use reserved red/blue."""
    assert isinstance(FORCE_KIND_PALETTE, (dict, Mapping))
    for kind in WrenchKind:
        assert kind.value in FORCE_KIND_PALETTE or kind in FORCE_KIND_PALETTE

    # Reserved colors for tension/compression fills: #0000ff and #ff0000
    banned_colors = {"#0000ff", "#ff0000", "#00f", "#f00"}
    for k, hex_val in FORCE_KIND_PALETTE.items():
        assert hex_val.lower() not in banned_colors, (
            f"Banned color {hex_val} in palette for {k}"
        )
        assert hex_val.startswith("#")
        assert len(hex_val) in (4, 7, 9)


def test_force_glyph_style_defaults_and_validation() -> None:
    """ForceGlyphStyle enforces valid ranges and provides documented defaults."""
    style = ForceGlyphStyle()
    assert style.force_scale_m_per_n == pytest.approx(1.0 / 1000.0)
    assert style.torque_scale_m_per_nm == pytest.approx(1.0 / 200.0)
    assert style.min_length_m == pytest.approx(0.02)
    assert style.max_length_m == pytest.approx(0.6)
    assert style.shaft_radius_m == pytest.approx(0.006)
    assert style.head_length_ratio == pytest.approx(0.22)
    assert style.head_radius_ratio == pytest.approx(2.4)
    assert style.torque_style == "arc"
    assert style.arc_sweep_rad == pytest.approx(1.5 * math.pi)
    assert style.arc_segments == 32
    assert style.magnitude_floor_n == pytest.approx(1.0)
    assert style.magnitude_floor_nm == pytest.approx(0.1)
    assert not style.show_labels

    # Validation errors
    with pytest.raises((ValueError, TypeError)):
        ForceGlyphStyle(force_scale_m_per_n=-1.0)
    with pytest.raises((ValueError, TypeError)):
        ForceGlyphStyle(min_length_m=1.0, max_length_m=0.5)
    with pytest.raises((ValueError, TypeError)):
        ForceGlyphStyle(arc_segments=2)
    with pytest.raises((ValueError, TypeError)):
        ForceGlyphStyle(torque_style="invalid_style")  # type: ignore[arg-type]


def test_force_glyph_style_serialization() -> None:
    """ForceGlyphStyle round-trips to dict and rejects unknown keys."""
    style = ForceGlyphStyle()
    d = style.to_dict()
    assert isinstance(d, dict)
    restored = ForceGlyphStyle.from_dict(d)
    assert restored == style

    with pytest.raises((ValueError, TypeError)):
        bad = dict(d)
        bad["unknown_field"] = 123
        ForceGlyphStyle.from_dict(bad)


def test_build_glyphs_force_arrow_defaults() -> None:
    """500 N along +x at origin produces tip at (0.5, 0, 0) and head base at 0.39 m."""
    wrench = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:test",
        body="link1",
        point_m=(0.0, 0.0, 0.0),
        force_n=(500.0, 0.0, 0.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=1.0, engine="engine_a", wrenches=(wrench,))
    style = ForceGlyphStyle()

    glyph_set = build_glyphs(frame, style)
    assert len(glyph_set.arrows) == 1
    assert len(glyph_set.torque_arcs) == 0

    arrow = glyph_set.arrows[0]
    assert arrow.label == "joint:test"
    assert arrow.kind == WrenchKind.JOINT_ACTUATOR
    assert arrow.tail_m == (0.0, 0.0, 0.0)
    assert arrow.tip_m[0] == pytest.approx(0.5)
    assert arrow.tip_m[1] == pytest.approx(0.0)
    assert arrow.tip_m[2] == pytest.approx(0.0)
    assert arrow.head_base_m[0] == pytest.approx(0.39)
    assert arrow.head_base_m[1] == pytest.approx(0.0)
    assert arrow.head_base_m[2] == pytest.approx(0.0)
    assert not arrow.clamped
    assert arrow.magnitude == pytest.approx(500.0)
    assert arrow.units == "N"


def test_build_glyphs_clamping() -> None:
    """10 kN is clamped to max_length_m (0.6 m); 5 N is raised to min_length_m (0.02 m)."""
    w_large = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:large",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 10000.0),
        source="test",
    )
    w_small = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:small",
        body="hand",
        point_m=(0.0, 0.0, 0.0),
        force_n=(5.0, 0.0, 0.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w_large, w_small))
    style = ForceGlyphStyle()

    glyph_set = build_glyphs(frame, style)
    arrows_by_label = {a.label: a for a in glyph_set.arrows}

    large_arrow = arrows_by_label["contact:large"]
    assert large_arrow.clamped
    assert large_arrow.tip_m[2] == pytest.approx(0.6)

    small_arrow = arrows_by_label["contact:small"]
    assert small_arrow.clamped
    assert small_arrow.tip_m[0] == pytest.approx(0.02)


def test_build_glyphs_torque_arc_geometry() -> None:
    """Torque about +z has all polyline points in z=const with counter-clockwise winding."""
    wrench = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:yaw",
        body="torso",
        point_m=(1.0, 2.0, 3.0),
        torque_nm=(0.0, 0.0, 50.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(wrench,))
    style = ForceGlyphStyle()

    glyph_set = build_glyphs(frame, style)
    assert len(glyph_set.torque_arcs) == 1
    arc = glyph_set.torque_arcs[0]

    assert arc.center_m == (1.0, 2.0, 3.0)
    assert arc.axis_unit == (0.0, 0.0, 1.0)
    # Expected radius: clamp(50 * 0.005, 0.02, 0.6) / 2 = 0.25 / 2 = 0.125
    assert arc.radius_m == pytest.approx(0.125)

    pts = arc.polyline_m
    assert len(pts) == style.arc_segments + 1
    # All points lie in plane z = 3.0
    for pt in pts:
        assert pt[2] == pytest.approx(3.0)
        dist = math.sqrt((pt[0] - 1.0) ** 2 + (pt[1] - 2.0) ** 2)
        assert dist == pytest.approx(arc.radius_m, abs=1e-6)

    # 2D signed area using Shoelace formula on x, y relative to center
    signed_area_2x = sum(
        (pts[i][0] - 1.0) * (pts[i + 1][1] - 2.0)
        - (pts[i + 1][0] - 1.0) * (pts[i][1] - 2.0)
        for i in range(len(pts) - 1)
    )
    assert signed_area_2x > 0.0, "Positive torque about +z must wind counter-clockwise"


def test_build_glyphs_unavailable_halves() -> None:
    """Missing halves are recorded in legend.unavailable_labels."""
    w_torque_only = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="actuator:torque_only",
        body="shaft",
        point_m=(0.0, 0.0, 0.0),
        torque_nm=(0.0, 10.0, 0.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="engine_x", wrenches=(w_torque_only,))
    glyph_set = build_glyphs(frame, ForceGlyphStyle())

    assert len(glyph_set.arrows) == 0
    assert len(glyph_set.torque_arcs) == 1
    assert "actuator:torque_only" in glyph_set.legend.unavailable_labels


def test_build_glyphs_kind_filtering_and_order_invariance() -> None:
    """Kind filtering selects subsets; wrench ordering does not alter sorted glyph output."""
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        source="test",
    )
    w2 = OverlayWrench(
        kind=WrenchKind.MUSCLE,
        label="muscle:biceps",
        body="arm",
        point_m=(0.0, 1.0, 0.0),
        force_n=(0.0, 50.0, 0.0),
        source="test",
    )

    frame1 = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w1, w2))
    frame2 = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w2, w1))

    style_all = ForceGlyphStyle()
    res1 = build_glyphs(frame1, style_all)
    res2 = build_glyphs(frame2, style_all)

    assert res1.to_dict() == res2.to_dict()

    style_contact_only = ForceGlyphStyle(kinds=frozenset({WrenchKind.CONTACT}))
    res_filtered = build_glyphs(frame1, style_contact_only)
    assert len(res_filtered.arrows) == 1
    assert res_filtered.arrows[0].label == "contact:ground"


def test_scale_for_view() -> None:
    """scale_for_view scales all coordinates and lengths while preserving magnitudes."""
    wrench = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(1.0, 2.0, 3.0),
        force_n=(500.0, 0.0, 0.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(wrench,))
    glyphs = build_glyphs(frame, ForceGlyphStyle())

    scaled = scale_for_view(glyphs, scale_factor=2.0)
    assert scaled.arrows[0].magnitude == pytest.approx(500.0)
    assert scaled.arrows[0].tail_m == (2.0, 4.0, 6.0)
    assert scaled.arrows[0].tip_m[0] == pytest.approx((1.0 + 0.5) * 2.0)
