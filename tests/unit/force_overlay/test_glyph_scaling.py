"""Scale modes, clamping, per-kind scale and group toggles for glyphs (GCV-4, #11710).

Tests assert glyph geometry (lengths, flags, labels), never pixels.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import (
    ForceGlyphStyle,
    GlyphSet,
    build_glyphs,
    label_group,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

BW = 80.0 * 9.80665


def _wrench(
    label: str, fz: float, kind: WrenchKind = WrenchKind.CONTACT
) -> OverlayWrench:
    return OverlayWrench(
        kind=kind,
        label=label,
        body="b",
        source="test",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, fz),
        torque_nm=None,
    )


def _frame(*wrenches: OverlayWrench) -> ForceTorqueFrame:
    return ForceTorqueFrame(time_s=0.0, engine="e", wrenches=tuple(wrenches))


def _length(glyphs: GlyphSet, index: int = 0) -> float:
    a = glyphs.arrows[index]
    return math.dist(a.tail_m, a.tip_m)


def test_body_weight_mode_gives_reference_length_per_bw() -> None:
    style = ForceGlyphStyle(
        scale_mode="body_weight", reference_force_n=BW, max_length_m=3.0
    )
    out = build_glyphs(_frame(_wrench("c:a", BW), _wrench("c:b", 2.0 * BW)), style)
    assert _length(out, 0) == pytest.approx(0.5)
    assert _length(out, 1) == pytest.approx(1.0)
    assert not any(a.clamped for a in out.arrows)
    assert out.legend.force_reference_n == pytest.approx(BW)
    assert out.legend.force_reference_length_m == pytest.approx(0.5)
    assert out.legend.scale_mode == "body_weight"


def test_reference_length_is_configurable() -> None:
    style = ForceGlyphStyle(
        scale_mode="body_weight",
        reference_force_n=BW,
        reference_length_m=0.25,
        max_length_m=3.0,
    )
    out = build_glyphs(_frame(_wrench("c:a", BW)), style)
    assert _length(out) == pytest.approx(0.25)


def test_peak_mode_maps_series_peak_to_reference_length() -> None:
    peak = 2100.0
    style = ForceGlyphStyle(
        scale_mode="peak",
        reference_force_n=peak,
        reference_length_m=0.8,
        max_length_m=2.0,
    )
    out = build_glyphs(_frame(_wrench("c:a", peak), _wrench("c:b", peak / 2)), style)
    assert _length(out, 0) == pytest.approx(0.8)
    assert _length(out, 1) == pytest.approx(0.4)


def test_fixed_mode_uses_force_scale() -> None:
    style = ForceGlyphStyle(force_scale_m_per_n=0.0005)
    out = build_glyphs(_frame(_wrench("c:a", 400.0)), style)
    assert _length(out) == pytest.approx(0.2)
    assert style.scale_mode == "fixed"


def test_clamped_iff_raw_length_exceeds_max() -> None:
    style = ForceGlyphStyle(
        scale_mode="body_weight",
        reference_force_n=BW,
        max_length_m=0.9,
        min_length_m=0.02,
    )
    out = build_glyphs(
        _frame(_wrench("c:ok", 1.5 * BW), _wrench("c:big", 3.0 * BW)), style
    )
    by_label = {a.label: a for a in out.arrows}
    assert not by_label["c:ok"].clamped
    assert by_label["c:big"].clamped
    assert math.dist(
        by_label["c:big"].tail_m, by_label["c:big"].tip_m
    ) == pytest.approx(0.9)
    assert out.legend.clamped_labels == ("c:big",)


def test_raising_to_min_length_is_not_flagged_clamped() -> None:
    style = ForceGlyphStyle(force_scale_m_per_n=1e-5, min_length_m=0.02)
    out = build_glyphs(_frame(_wrench("c:tiny", 50.0)), style)
    assert not out.arrows[0].clamped
    assert _length(out) == pytest.approx(0.02)


def test_kind_scale_multiplies_only_that_kind() -> None:
    style = ForceGlyphStyle(
        scale_mode="body_weight",
        reference_force_n=BW,
        max_length_m=3.0,
        kind_scale={WrenchKind.GRIP: 2.0},
    )
    out = build_glyphs(
        _frame(_wrench("c:a", BW), _wrench("grip:a", BW, WrenchKind.GRIP)), style
    )
    by_label = {a.label: a for a in out.arrows}
    assert math.dist(by_label["c:a"].tail_m, by_label["c:a"].tip_m) == pytest.approx(
        0.5
    )
    assert math.dist(
        by_label["grip:a"].tail_m, by_label["grip:a"].tip_m
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale_mode": "bogus"},
        {"scale_mode": "body_weight"},
        {"scale_mode": "peak", "reference_force_n": -1.0},
        {"scale_mode": "peak", "reference_force_n": float("nan")},
        {"reference_length_m": 0.0},
        {"kind_scale": {WrenchKind.GRIP: -1.0}},
        {"groups": frozenset({"nope"})},
    ],
)
def test_invalid_scale_configuration_raises_value_error(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ForceGlyphStyle(**kwargs)


def test_style_round_trip_and_unknown_key_rejected() -> None:
    style = ForceGlyphStyle(
        scale_mode="peak",
        reference_force_n=1800.0,
        reference_length_m=0.7,
        kind_scale={WrenchKind.CONTACT: 1.5},
        groups=frozenset({"per_foot", "net"}),
    )
    data = style.to_dict()
    assert data["scale_mode"] == "peak"
    assert data["kind_scale"] == {"contact": 1.5}
    assert data["groups"] == ["net", "per_foot"]
    again = ForceGlyphStyle.from_dict(data)
    assert again.to_dict() == data
    assert again.kind_scale[WrenchKind.CONTACT] == 1.5
    with pytest.raises(ValueError):
        ForceGlyphStyle.from_dict({**data, "surprise": 1})


def test_default_shaft_is_thicker_for_video() -> None:
    assert ForceGlyphStyle().shaft_radius_m >= 0.01


def test_label_groups() -> None:
    assert label_group("contact:grf_left_foot") == "per_foot"
    assert label_group("contact:grf_net") == "net"
    assert label_group("contact:free_moment_left_foot") == "free_moment"
    assert label_group("contact:moment_com_net") == "moment_about_com"
    assert label_group("contact:ball:3") == "contact_points"
    assert label_group("grip:hand_lead") == "grip_per_hand"
    assert label_group("grip:net_midpoint") == "grip_net"
    assert label_group("grip:couple_midpoint") == "grip_couple"
    assert label_group("grip:mof_lead") == "grip_mof"
    assert label_group("joint:elbow") is None


def test_group_toggles_filter_by_label_prefix() -> None:
    frame = _frame(
        _wrench("contact:grf_left_foot", 700.0),
        _wrench("contact:grf_net", 1400.0),
        _wrench("contact:ball:0", 50.0),
        _wrench("grip:hand_lead", 100.0, WrenchKind.GRIP),
        _wrench("joint:elbow", 100.0, WrenchKind.JOINT_REACTION),
    )
    labels = lambda s: {a.label for a in build_glyphs(frame, s).arrows}  # noqa: E731
    default = labels(ForceGlyphStyle())
    assert default == {
        "contact:grf_left_foot",
        "contact:grf_net",
        "grip:hand_lead",
        "joint:elbow",
    }
    only_net = labels(ForceGlyphStyle(groups=frozenset({"net"})))
    assert only_net == {"contact:grf_net", "joint:elbow"}
    with_points = labels(
        ForceGlyphStyle(groups=frozenset({"per_foot", "net", "contact_points"}))
    )
    assert "contact:ball:0" in with_points
    assert "grip:hand_lead" not in with_points


def test_raw_contacts_shown_when_no_aggregated_grf_present() -> None:
    frame = _frame(_wrench("contact:ball:0", 50.0), _wrench("contact:foot:1", 90.0))
    out = build_glyphs(frame, ForceGlyphStyle())
    assert {a.label for a in out.arrows} == {"contact:ball:0", "contact:foot:1"}


def test_glyphset_round_trip_keeps_legend_fields() -> None:
    style = ForceGlyphStyle(
        scale_mode="body_weight", reference_force_n=BW, max_length_m=0.6
    )
    out = build_glyphs(_frame(_wrench("c:big", 3 * BW)), style)
    again = GlyphSet.from_dict(out.to_dict())
    assert again.legend.clamped_labels == ("c:big",)
    assert again.legend.scale_mode == "body_weight"
