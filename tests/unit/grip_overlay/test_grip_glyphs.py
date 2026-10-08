"""Grip glyph builder: per-hand, net, couple, MOF (GCV-10, #11716, ADR-0052)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.biomechanics.grip_extraction import net_only_analysis
from src.shared.python.biomechanics.grip_wrench import (
    HandWrench,
    analyze_grip,
    to_overlay_wrenches,
)
from src.shared.python.force_overlay import (
    ForceGlyphStyle,
    ForceTorqueFrame,
    LegendSpec,
    WrenchKind,
    build_glyphs,
)
from src.shared.python.force_overlay.grip_frame import grip_frame

pytestmark = pytest.mark.unit

RL = (0.0, 0.0, 1.0)
RR = (0.0, 0.0, 0.8)


def _analysis(method="constraint_multiplier"):
    return analyze_grip(
        HandWrench("L", RL, (60.0, 0, 0), (0, 0, 2.0)),
        HandWrench("R", RR, (-20.0, 10.0, 0), (0, 0, 3.0)),
        split_method=method,
    )


STYLE = ForceGlyphStyle(magnitude_floor_n=1.0, magnitude_floor_nm=0.1)


def _glyphs(analysis=None, style=STYLE):
    frame = grip_frame(0.5, analysis or _analysis(), engine="mujoco", source="t:grip")
    return build_glyphs(frame, style), frame


def test_known_analysis_emits_labelled_glyphs():
    g = _analysis()
    glyphs, _ = _glyphs(g)
    arrows = {a.label: a for a in glyphs.arrows}
    arcs = {a.label: a for a in glyphs.torque_arcs}
    assert {"grip:hand_left", "grip:hand_right", "grip:net_midpoint"} <= set(arrows)
    assert "grip:couple_midpoint" in arcs
    # per-hand arrows start at each hand's grip point, net at the midpoint
    assert arrows["grip:hand_left"].tail_m == pytest.approx(RL)
    assert arrows["grip:hand_right"].tail_m == pytest.approx(RR)
    assert arrows["grip:net_midpoint"].tail_m == pytest.approx(g.midpoint_m)
    assert arrows["grip:net_midpoint"].magnitude == pytest.approx(
        np.linalg.norm(g.net_force_n)
    )


def test_couple_arc_axis_equals_couple_direction():
    g = _analysis()
    glyphs, _ = _glyphs(g)
    arc = next(a for a in glyphs.torque_arcs if a.label == "grip:couple_midpoint")
    m = np.array(g.couple_at_midpoint_nm)
    np.testing.assert_allclose(arc.axis_unit, m / np.linalg.norm(m), atol=1e-12)
    assert arc.center_m == pytest.approx(g.midpoint_m)
    assert arc.magnitude == pytest.approx(np.linalg.norm(m))


def test_hand_arrows_have_distinct_colours_from_net():
    glyphs, _ = _glyphs()
    rgba = {a.label: a.rgba for a in glyphs.arrows}
    assert len({rgba["grip:hand_left"], rgba["grip:hand_right"], rgba["grip:net_midpoint"]}) == 3
    # GRIP #56B4E9 stays the net colour
    assert rgba["grip:net_midpoint"][:3] == pytest.approx(
        (0x56 / 255, 0xB4 / 255, 0xE9 / 255)
    )
    # left lighter, right darker than the net
    assert sum(rgba["grip:hand_left"][:3]) > sum(rgba["grip:net_midpoint"][:3])
    assert sum(rgba["grip:hand_right"][:3]) < sum(rgba["grip:net_midpoint"][:3])


def test_group_toggles_hide_each_glyph_class():
    base = dict(magnitude_floor_n=1.0, magnitude_floor_nm=0.1)
    only_net = ForceGlyphStyle(groups=frozenset({"grip_net"}), **base)
    glyphs, _ = _glyphs(style=only_net)
    assert [a.label for a in glyphs.arrows] == ["grip:net_midpoint"]
    assert glyphs.torque_arcs == ()
    only_couple = ForceGlyphStyle(groups=frozenset({"grip_couple"}), **base)
    glyphs, _ = _glyphs(style=only_couple)
    assert glyphs.arrows == ()
    assert [a.label for a in glyphs.torque_arcs] == ["grip:couple_midpoint"]
    mof = ForceGlyphStyle(groups=frozenset({"grip_mof"}), **base)
    glyphs, _ = _glyphs(style=mof)
    assert {a.label for a in glyphs.torque_arcs} == {"grip:mof_left", "grip:mof_right"}


def test_free_torques_are_separate_from_the_couple():
    g = _analysis()
    wrenches = {w.label: w for w in to_overlay_wrenches(g, source="t")}
    assert wrenches["grip:hand_left"].torque_nm == (0, 0, 2.0)
    glyphs, _ = _glyphs(g)
    labels = {a.label for a in glyphs.torque_arcs}
    assert {"grip:hand_left", "grip:hand_right", "grip:couple_midpoint"} <= labels
    left = next(a for a in glyphs.torque_arcs if a.label == "grip:hand_left")
    assert left.center_m == pytest.approx(RL)


def test_split_method_label_is_on_the_legend_and_hud_text():
    glyphs, frame = _glyphs(_analysis("efc_force"))
    assert frame.metadata["grip_split_method"] == "efc_force"
    assert glyphs.legend.grip_split_method == "efc_force"
    from src.shared.python.force_overlay.renderers.meshcat_glyphs import legend_text

    assert "Grip split: efc_force" in legend_text(glyphs)
    again = LegendSpec.from_dict(glyphs.legend.to_dict())
    assert again.grip_split_method == "efc_force"


def test_unavailable_split_is_listed_not_drawn_as_zero():
    net_only = net_only_analysis(
        point_m=(0, 0, 0.9),
        force_on_club_n=(5.0, 0, 0),
        torque_on_club_nm=(0, 2.0, 0),
        split_method="allocation",
        reason="allocation yields one net 6-D wrench; left/right split unavailable",
    )
    glyphs, frame = _glyphs(net_only)
    labels = {a.label for a in glyphs.arrows}
    assert labels == {"grip:net_midpoint"}
    assert {"grip:hand_left", "grip:hand_right"} <= set(glyphs.legend.unavailable_labels)
    assert glyphs.legend.grip_split_method == "allocation"
    from src.shared.python.force_overlay.renderers.meshcat_glyphs import legend_text

    text = legend_text(glyphs)
    assert "Grip split: allocation" in text and "unavailable" in text.lower()


def test_fully_unavailable_analysis_gives_no_glyphs_and_a_reason():
    from src.shared.python.biomechanics.grip_extraction import unavailable_analysis

    frame = grip_frame(
        0.0, unavailable_analysis("no grip welds"), engine="x", source="t"
    )
    glyphs = build_glyphs(frame, STYLE)
    assert glyphs.arrows == () and glyphs.torque_arcs == ()
    assert glyphs.legend.grip_split_method == "unavailable"
    assert "grip:net_midpoint" in glyphs.legend.unavailable_labels
    assert frame.metadata["grip_unavailable_reason"] == "no grip welds"


def test_grip_frame_rejects_bad_inputs():
    with pytest.raises(TypeError):
        grip_frame(0.0, object(), engine="x", source="t")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        grip_frame(math.nan, _analysis(), engine="x", source="t")
    with pytest.raises(ValueError):
        grip_frame(0.0, _analysis(), engine="", source="t")
    assert WrenchKind.GRIP in {w.kind for w in grip_frame(0.0, _analysis(), engine="x", source="t").wrenches}
