"""Angular-momentum arrow overlay contracts (#15398)."""

from __future__ import annotations

import subprocess
import sys
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
        spin_only_nm_s=dict.fromkeys(
            ("pelvis", "torso", "thigh", "arm", "club"), (0.0, 0.0, 0.0)
        ),
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


def test_arrow_length_equals_h_times_scale_when_in_range() -> None:
    style = AngularMomentumArrowStyle(
        scale_m_per_nms=0.05,
        max_length_m=2.0,
        min_draw_length_m=0.001,
    )
    magnitude_nms = 3.0
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
        whole_body_nm_s=(magnitude_nms, 0.0, 0.0),
    )
    glyphs = build_angular_momentum_glyphs(
        frame, style=style, anchors_m=anchors, enabled=True
    )
    assert glyphs is not None
    arrow = glyphs.arrows[0]
    length = float(np.linalg.norm(np.subtract(arrow.tip_m, arrow.tail_m)))
    assert length == pytest.approx(magnitude_nms * style.scale_m_per_nms)
    assert arrow.clamped is False


def test_arrow_omitted_below_min_draw_length() -> None:
    style = AngularMomentumArrowStyle(
        scale_m_per_nms=1e-4,
        min_draw_length_m=0.05,
        max_length_m=1.0,
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
    assert glyphs is None


def test_arrow_length_respects_max_cap() -> None:
    style = AngularMomentumArrowStyle(
        scale_m_per_nms=0.1,
        max_length_m=0.5,
        min_draw_length_m=0.001,
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


def test_per_segment_threshold_top_n_and_count_cap() -> None:
    contributions = {f"seg{i}": (float(i + 1), 0.0, 0.0) for i in range(12)}
    frame = SegmentAngularMomentumFrame(
        time_s=0.0,
        reference_point=REFERENCE_POINT,
        frame="world",
        units=UNITS_ANGMOM,
        source="fixture",
        contribution_nm_s=contributions,
        spin_only_nm_s=contributions,
        groups_nm_s={
            "legs": None,
            "pelvis": None,
            "torso": None,
            "arms": None,
            "club": None,
        },
        whole_body_nm_s=(78.0, 0.0, 0.0),
    )
    style = AngularMomentumArrowStyle(
        display_mode="per_segment",
        per_segment_threshold_nms=5.0,
        per_segment_top_n=3,
        per_segment_max_count=2,
        scale_m_per_nms=0.01,
        min_draw_length_m=0.001,
    )
    anchors = {name: (0.0, 0.0, float(i) * 0.1) for i, name in enumerate(contributions)}
    glyphs = build_angular_momentum_glyphs(
        frame, style=style, anchors_m=anchors, enabled=True
    )
    assert glyphs is not None
    labels = {a.label for a in glyphs.arrows}
    assert labels == {"angmom:seg11", "angmom:seg10"}


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


def test_legend_contains_reference_frame_units_and_calibration() -> None:
    text = angular_momentum_legend_line(AngularMomentumArrowStyle())
    assert "whole body com" in text.lower()
    assert "world" in text
    assert UNITS_ANGMOM in text
    assert "FULL_BODY" in text
    ref = 1.0 / DEFAULT_SCALE_M_PER_NMS
    assert f"{ref:.1f}" in text


def test_spin_only_legend_notes_schematic_arc_radii() -> None:
    text = angular_momentum_legend_line(
        AngularMomentumArrowStyle(display_mode="spin_only")
    )
    assert "schematic" in text.lower()


def test_settings_round_trip() -> None:
    from src.shared.python.body_part_viz.angular_momentum_settings import (
        dump_angular_momentum_state,
        parse_angular_momentum_state,
    )

    style = AngularMomentumArrowStyle(
        display_mode="per_segment", show_spin_component=True
    )
    blob = dump_angular_momentum_state(True, style)
    enabled, parsed = parse_angular_momentum_state(blob)
    assert enabled is True
    assert parsed.display_mode == "per_segment"
    assert parsed.show_spin_component is True


def test_merge_glyph_sets_concatenates_arrows_and_keeps_primary_time() -> None:
    primary = GlyphSet(
        time_s=1.25,
        arrows=(),
        torque_arcs=(),
        legend=LegendSpec(engine="force", source_labels=(), scale_mode="fixed"),
    )
    one = build_angular_momentum_glyphs(
        _frame(),
        style=AngularMomentumArrowStyle(display_mode="groups"),
        anchors_m={
            "whole_body": (0.0, 0.0, 1.0),
            **dict.fromkeys(
                ("legs", "pelvis", "torso", "arms", "club"), (0.0, 0.0, 1.0)
            ),
        },
        enabled=True,
    )
    merged = merge_glyph_sets(primary, one)
    assert merged is not None
    assert merged.time_s == pytest.approx(1.25)
    assert len(merged.arrows) == len(one.arrows)


def test_angular_momentum_glyphs_module_has_no_matplotlib() -> None:
    script = """
import sys
class NoRendering:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('matplotlib', 'PyQt6', 'pyqtgraph'):
            raise ImportError('Rendering package imported: ' + fullname)
sys.meta_path.insert(0, NoRendering())
from src.shared.python.body_part_viz.angular_momentum_glyphs import (
    AngularMomentumArrowStyle,
    build_angular_momentum_glyphs,
)
from src.shared.python.body_part_viz.segment_angular_momentum import (
    REFERENCE_POINT,
    UNITS_ANGMOM,
    SegmentAngularMomentumFrame,
)
frame = SegmentAngularMomentumFrame(
    time_s=0.0,
    reference_point=REFERENCE_POINT,
    frame='world',
    units=UNITS_ANGMOM,
    source='headless',
    contribution_nm_s={},
    spin_only_nm_s={},
    groups_nm_s={'legs': None, 'pelvis': None, 'torso': None, 'arms': None, 'club': None},
    whole_body_nm_s=(1.0, 0.0, 0.0),
)
glyphs = build_angular_momentum_glyphs(
    frame,
    style=AngularMomentumArrowStyle(),
    anchors_m={'whole_body': (0.0, 0.0, 0.0)},
    enabled=True,
)
assert glyphs is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
