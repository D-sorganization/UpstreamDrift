"""Unit tests for glyph builder ForceGlyphStyle and build_glyphs (FTO-3, #11288)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    ForceGlyphStyle,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
    build_glyphs,
    scale_for_view,
)
from src.shared.python.force_overlay.palette import (
    FORCE_KIND_PALETTE,
    get_kind_rgba,
    hex_to_rgba,
)

pytestmark = pytest.mark.unit


def test_force_glyph_style_defaults_and_validation() -> None:
    style = ForceGlyphStyle()
    assert style.force_scale_m_per_n == 0.001
    assert style.torque_scale_m_per_nm == 0.005
    assert style.min_length_m == 0.02
    assert style.max_length_m == 0.6
    assert style.shaft_radius_m == 0.006
    assert style.head_length_ratio == 0.22
    assert style.head_radius_ratio == 2.4
    assert style.torque_style == "arc"
    assert style.arc_sweep_rad == pytest.approx(1.5 * math.pi)
    assert style.arc_segments == 32
    assert style.magnitude_floor_n == 1.0
    assert style.magnitude_floor_nm == 0.1
    assert not style.show_labels

    # Validation errors
    with pytest.raises(
        ValueError, match="min_length_m must be strictly less than max_length_m"
    ):
        ForceGlyphStyle(min_length_m=1.0, max_length_m=0.5)

    with pytest.raises(ValueError, match="must be positive"):
        ForceGlyphStyle(force_scale_m_per_n=-0.01)

    with pytest.raises(
        ValueError, match="torque_style must be 'arc' or 'axis_double_head'"
    ):
        ForceGlyphStyle(torque_style="invalid")  # type: ignore[arg-type]


def test_force_arrow_500n_along_x_origin() -> None:
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(500.0, 0.0, 0.0),
        torque_nm=None,
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.1, engine="test", wrenches=(w,))
    glyphs = build_glyphs(frame)

    assert len(glyphs.arrows) == 1
    assert len(glyphs.torque_arcs) == 0

    arrow = glyphs.arrows[0]
    assert arrow.label == "contact:ground"
    assert arrow.kind == "contact"
    assert arrow.tail_m == (0.0, 0.0, 0.0)
    assert arrow.tip_m == pytest.approx((0.5, 0.0, 0.0), abs=1e-6)
    assert arrow.head_base_m == pytest.approx((0.39, 0.0, 0.0), abs=1e-6)
    assert not arrow.clamped
    assert arrow.magnitude == pytest.approx(500.0)
    assert arrow.units == "N"


def test_force_arrow_clamping_max_and_min() -> None:
    # 10 kN -> clamped to 0.6 m
    w_large = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:large",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 10000.0, 0.0),
        torque_nm=None,
        source="test",
    )
    # 5 N -> clamped to 0.02 m
    w_small = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:small",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 5.0, 0.0),
        torque_nm=None,
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w_large, w_small))
    glyphs = build_glyphs(frame)

    arrows_by_label = {a.label: a for a in glyphs.arrows}

    a_large = arrows_by_label["contact:large"]
    assert a_large.clamped
    assert a_large.tip_m == pytest.approx((0.0, 0.6, 0.0), abs=1e-6)

    a_small = arrows_by_label["contact:small"]
    assert a_small.clamped
    assert a_small.tip_m == pytest.approx((0.0, 0.02, 0.0), abs=1e-6)


def test_torque_arc_positive_and_negative_z() -> None:
    # +z torque: counter-clockwise
    w_pos = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="actuator:pos_z",
        body="shaft",
        point_m=(1.0, 2.0, 3.0),
        force_n=None,
        torque_nm=(0.0, 0.0, 40.0),  # 40 * 0.005 = 0.2 m effective length, r = 0.1 m
        source="test",
    )
    # -z torque: clockwise
    w_neg = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="actuator:neg_z",
        body="shaft",
        point_m=(1.0, 2.0, 3.0),
        force_n=None,
        torque_nm=(0.0, 0.0, -40.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w_pos, w_neg))
    glyphs = build_glyphs(frame)

    arcs = {a.label: a for a in glyphs.torque_arcs}
    pos_arc = arcs["actuator:pos_z"]
    neg_arc = arcs["actuator:neg_z"]

    assert pos_arc.radius_m == pytest.approx(0.1, abs=1e-6)
    assert neg_arc.radius_m == pytest.approx(0.1, abs=1e-6)

    # All points lie in z = 3 plane
    for pt in pos_arc.polyline_m:
        assert pt[2] == pytest.approx(3.0, abs=1e-6)
        dx = pt[0] - 1.0
        dy = pt[1] - 2.0
        assert math.hypot(dx, dy) == pytest.approx(0.1, abs=1e-6)

    # Signed area in xy-plane for counter-clockwise vs clockwise
    pts_pos = np.array(pos_arc.polyline_m)[:, :2]
    v0_pos = pts_pos[:-1] - np.array([1.0, 2.0])
    v1_pos = pts_pos[1:] - np.array([1.0, 2.0])
    cross_pos = v0_pos[:, 0] * v1_pos[:, 1] - v0_pos[:, 1] * v1_pos[:, 0]
    assert np.all(cross_pos > 0)  # Counter-clockwise

    pts_neg = np.array(neg_arc.polyline_m)[:, :2]
    v0_neg = pts_neg[:-1] - np.array([1.0, 2.0])
    v1_neg = pts_neg[1:] - np.array([1.0, 2.0])
    cross_neg = v0_neg[:, 0] * v1_neg[:, 1] - v0_neg[:, 1] * v1_neg[:, 0]
    assert np.all(cross_neg < 0)  # Clockwise


def test_torque_arc_basis_deterministic_across_axes() -> None:
    # Test x, y, z and diagonal axes
    axes = [
        (10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (0.0, 0.0, 10.0),
        (10.0, 10.0, 10.0),
    ]
    wrenches = tuple(
        OverlayWrench(
            kind=WrenchKind.JOINT_ACTUATOR,
            label=f"actuator:axis_{i}",
            body="torso",
            point_m=(0.0, 0.0, 0.0),
            force_n=None,
            torque_nm=ax,
            source="test",
        )
        for i, ax in enumerate(axes)
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=wrenches)
    g1 = build_glyphs(frame)
    g2 = build_glyphs(frame)

    assert len(g1.torque_arcs) == 4
    for a1, a2 in zip(g1.torque_arcs, g2.torque_arcs, strict=True):
        assert a1.polyline_m == a2.polyline_m
        assert a1.axis_unit == pytest.approx(a2.axis_unit, abs=1e-9)


def test_torque_only_and_force_only_unavailable_labels() -> None:
    w_t_only = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="actuator:wrist",
        body="hand",
        point_m=(0.0, 0.0, 0.0),
        force_n=None,
        torque_nm=(0.0, 5.0, 0.0),
        source="test",
    )
    w_f_only = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:toe",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 50.0),
        torque_nm=None,
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w_t_only, w_f_only))
    glyphs = build_glyphs(frame)

    assert len(glyphs.arrows) == 1
    assert glyphs.arrows[0].label == "contact:toe"
    assert len(glyphs.torque_arcs) == 1
    assert glyphs.torque_arcs[0].label == "actuator:wrist"

    # Both wrenches had an unavailable half
    assert "actuator:wrist" in glyphs.legend.unavailable_labels
    assert "contact:toe" in glyphs.legend.unavailable_labels


def test_filtering_and_order_invariance() -> None:
    w1 = OverlayWrench(
        kind=WrenchKind.GRIP,
        label="grip:lead",
        body="club",
        point_m=(0.0, 0.0, 1.0),
        force_n=(10.0, 0.0, 0.0),
        torque_nm=None,
        source="test",
    )
    w2 = OverlayWrench(
        kind=WrenchKind.GRAVITY,
        label="gravity:pelvis",
        body="pelvis",
        point_m=(0.0, 0.0, 0.8),
        force_n=(0.0, 0.0, -700.0),
        torque_nm=None,
        source="test",
    )
    # Kind filter: only GRIP
    style = ForceGlyphStyle(kinds=(WrenchKind.GRIP,))
    frame1 = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w1, w2))
    frame2 = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w2, w1))

    g1 = build_glyphs(frame1, style)
    g2 = build_glyphs(frame2, style)

    assert len(g1.arrows) == 1
    assert g1.arrows[0].label == "grip:lead"
    assert g1.arrows == g2.arrows


def test_scale_for_view() -> None:
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(1.0, 2.0, 0.0),
        force_n=(500.0, 0.0, 0.0),
        torque_nm=(0.0, 0.0, 20.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w,))
    glyphs = build_glyphs(frame)

    scaled = scale_for_view(glyphs, 2.0)

    # Arrow checks
    assert scaled.arrows[0].tail_m == (1.0, 2.0, 0.0)  # Anchor unchanged
    # Tip vector was (1.5, 2.0, 0.0) relative to (1.0, 2.0, 0.0) -> scaled by 2 -> (2.0, 2.0, 0.0)
    assert scaled.arrows[0].tip_m == pytest.approx((2.0, 2.0, 0.0), abs=1e-6)
    assert (
        scaled.arrows[0].magnitude == glyphs.arrows[0].magnitude
    )  # Magnitude unchanged

    # Arc checks
    assert scaled.torque_arcs[0].center_m == (1.0, 2.0, 0.0)
    assert scaled.torque_arcs[0].radius_m == pytest.approx(
        glyphs.torque_arcs[0].radius_m * 2.0
    )
    assert scaled.torque_arcs[0].magnitude == glyphs.torque_arcs[0].magnitude


def test_scale_for_view_errors() -> None:
    glyphs = GlyphSet(
        time_s=0.0,
        arrows=(),
        torque_arcs=(),
        legend=LegendSpec(None, None, None, None, (), (), "test", ()),
    )
    with pytest.raises(ValueError, match="scale_factor must be positive and finite"):
        scale_for_view(glyphs, 0.0)
    with pytest.raises(ValueError, match="scale_factor must be positive and finite"):
        scale_for_view(glyphs, -1.0)
    with pytest.raises(ValueError, match="scale_factor must be positive and finite"):
        scale_for_view(glyphs, float("nan"))


def test_palette_and_rgba_coverage() -> None:
    # 3-digit hex
    assert hex_to_rgba("#123") == pytest.approx(
        (0x11 / 255.0, 0x22 / 255.0, 0x33 / 255.0, 1.0)
    )
    # 6-digit hex
    assert hex_to_rgba("#123456") == pytest.approx(
        (0x12 / 255.0, 0x34 / 255.0, 0x56 / 255.0, 1.0)
    )
    # 8-digit hex
    assert hex_to_rgba("#12345678") == pytest.approx(
        (0x12 / 255.0, 0x34 / 255.0, 0x56 / 255.0, 0x78 / 255.0)
    )
    # Non-string
    with pytest.raises(TypeError, match="hex_str must be str"):
        hex_to_rgba(123)  # type: ignore[arg-type]
    # Invalid hex
    with pytest.raises(ValueError, match="Invalid hex color"):
        hex_to_rgba("#xyz")

    # get_kind_rgba
    assert get_kind_rgba(WrenchKind.CONTACT) == get_kind_rgba("contact")
    assert get_kind_rgba("unknown_kind") == (0.0, 0.0, 0.0, 1.0)
