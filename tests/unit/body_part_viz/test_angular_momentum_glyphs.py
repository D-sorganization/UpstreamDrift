"""Angular-momentum arrow overlay contracts (#15398)."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.body_part_viz.angular_momentum_glyphs import (
    DEFAULT_SCALE_M_PER_NMS,
    AngularMomentumArrowStyle,
    angular_momentum_legend_line,
    build_angular_momentum_glyphs,
    merge_glyph_sets,
)
from src.shared.python.body_part_viz.segment_angular_momentum import (
    REFERENCE_POINT,
    UNITS_ANGMOM,
    SegmentAngularMomentumFrame,
    SegmentGroupCatalog,
)
from src.shared.python.force_overlay.glyphs import GlyphSet, LegendSpec

pytestmark = pytest.mark.unit


def _frame() -> SegmentAngularMomentumFrame:
    catalog = SegmentGroupCatalog(
        legs=frozenset({"thigh"}),
        pelvis=frozenset({"pelvis"}),
        torso=frozenset({"torso"}),
        arms=frozenset({"arm"}),
        club=frozenset({"club"}),
    )
    h = np.array([4.0, 0.0, 0.0])
    return SegmentAngularMomentumFrame(
        time_s=0.0,
        reference_point=REFERENCE_POINT,
        frame="world",
        units=UNITS_ANGMOM,
        source="fixture",
        contribution_nm_s={
            "pelvis": tuple(h),
            "torso": (0.0, 2.0, 0.0),
            "thigh": (0.0, 0.0, 1.0),
            "arm": (1.0, 1.0, 0.0),
            "club": (0.5, 0.0, 0.0),
        },
        spin_only_nm_s=dict.fromkeys(("pelvis", "torso", "thigh", "arm", "club"), (0.0, 0.0, 0.0)),
        groups_nm_s={
            "legs": (0.0, 0.0, 1.0),
            "pelvis": (4.0, 0.0, 0.0),
            "torso": (0.0, 2.0, 0.0),
            "arms": (1.0, 1.0, 0.0),
            "club": (0.5, 0.0, 0.0),
        },
        whole_body_nm_s=(5.5, 3.0, 1.0),
    )


def test_groups_mode_draws_six_arrows() -> None:
    anchors = {
        "whole_body": (0.0, 0.0, 1.0),
        "legs": (0.1, 0.0, 0.5),
        "pelvis": (0.0, 0.0, 0.9),
        "torso": (0.0, 0.0, 1.2),
        "arms": (0.2, 0.0, 1.4),
        "club": (0.5, 0.0, 1.0),
    }
    glyphs = build_angular_momentum_glyphs(
        _frame(),
        style=AngularMomentumArrowStyle(display_mode="groups"),
        anchors_m=anchors,
        enabled=True,
    )
    assert glyphs is not None
    assert len(glyphs.arrows) == 6
    assert len(glyphs.torque_arcs) == 0


def test_arrow_length_respects_fixed_scale_and_cap() -> None:
    style = AngularMomentumArrowStyle(
        scale_m_per_nms=0.1,
        max_length_m=0.5,
        min_length_m=0.02,
    )
    anchors = {"whole_body": (0.0, 0.0, 0.0)}
    frame = SegmentAngularMomentumFrame(
        time_s=0.0,
        reference_point=REFERENCE_POINT,
        frame="world",
        units=UNITS_ANGMOM,
        source="fixture",
        contribution_nm_s={},
        spin_only_nm_s={},
        groups_nm_s={
            "legs": None,
            "pelvis": None,
            "torso": None,
            "arms": None,
            "club": None,
        },
        whole_body_nm_s=(10.0, 0.0, 0.0),
    )
    glyphs = build_angular_momentum_glyphs(
        frame, style=style, anchors_m=anchors, enabled=True
    )
    assert glyphs is not None
    arrow = glyphs.arrows[0]
    length = float(np.linalg.norm(np.subtract(arrow.tip_m, arrow.tail_m)))
    assert length == pytest.approx(0.5)
    assert arrow.clamped is True


def test_spin_only_mode_adds_arc_not_in_groups_mode() -> None:
    anchors = {"pelvis": (0.0, 0.0, 1.0)}
    frame = _frame()
    spin_map = dict(frame.spin_only_nm_s)
    spin_map["pelvis"] = (0.0, 0.0, 3.0)
    frame = SegmentAngularMomentumFrame(
        time_s=frame.time_s,
        reference_point=frame.reference_point,
        frame=frame.frame,
        units=frame.units,
        source=frame.source,
        contribution_nm_s=frame.contribution_nm_s,
        spin_only_nm_s=spin_map,
        groups_nm_s=frame.groups_nm_s,
        whole_body_nm_s=frame.whole_body_nm_s,
    )
    groups = build_angular_momentum_glyphs(
        frame,
        style=AngularMomentumArrowStyle(display_mode="groups"),
        anchors_m=anchors,
        enabled=True,
    )
    spin = build_angular_momentum_glyphs(
        frame,
        style=AngularMomentumArrowStyle(display_mode="spin_only"),
        anchors_m=anchors,
        enabled=True,
    )
    assert groups is not None and spin is not None
    assert not groups.torque_arcs
    assert spin.torque_arcs


def test_legend_contains_reference_frame_and_units() -> None:
    text = angular_momentum_legend_line(AngularMomentumArrowStyle())
    assert "whole body com" in text.lower()
    assert "world" in text
    assert UNITS_ANGMOM in text
    ref = 1.0 / DEFAULT_SCALE_M_PER_NMS
    assert str(int(ref)) in text


def test_settings_round_trip() -> None:
    from src.shared.python.body_part_viz.angular_momentum_settings import (
        dump_angular_momentum_state,
        parse_angular_momentum_state,
    )

    style = AngularMomentumArrowStyle(display_mode="per_segment", show_spin_component=True)
    blob = dump_angular_momentum_state(True, style)
    enabled, parsed = parse_angular_momentum_state(blob)
    assert enabled is True
    assert parsed.display_mode == "per_segment"
    assert parsed.show_spin_component is True


def test_merge_glyph_sets_concatenates_arrows() -> None:
    empty = GlyphSet(time_s=0.0, arrows=(), torque_arcs=(), legend=LegendSpec())
    one = build_angular_momentum_glyphs(
        _frame(),
        style=AngularMomentumArrowStyle(display_mode="groups"),
        anchors_m={"whole_body": (0.0, 0.0, 1.0), **dict.fromkeys(("legs", "pelvis", "torso", "arms", "club"), (0.0, 0.0, 1.0))},
        enabled=True,
    )
    merged = merge_glyph_sets(empty, one)
    assert merged is not None
    assert len(merged.arrows) == len(one.arrows)


def test_headless_glyphs_module_has_no_matplotlib() -> None:
    mod = importlib.import_module("src.shared.python.body_part_viz.angular_momentum_glyphs")
    source_path = mod.__file__
    assert source_path is not None
    text = Path(source_path).read_text(encoding="utf-8")
    assert "matplotlib" not in text