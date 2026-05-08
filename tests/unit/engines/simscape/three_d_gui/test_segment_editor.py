"""Tests for the user-defined segment editor + 3D viewer plumbing."""

from __future__ import annotations

import sys

import pytest

from ._viewer_test_helpers import ANATOMICAL_28, make_synthetic_model

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


def test_reset_to_default_populates_canonical_set(qt_app, tmp_path) -> None:
    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore

    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=20)
    tab = SegmentsTab()
    tab.update_from_model(model)

    # 26 canonical segments for the 28-marker subset.
    assert len(tab.segments) == 26


def test_add_segment_emits_signal_and_renders_line(qt_app) -> None:
    from src.apps.services.segment_set_io import SegmentSpec  # type: ignore
    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=20)
    seg_tab = SegmentsTab()
    viewer = Viewer3DTab()
    seg_tab.segments_changed.connect(viewer.set_user_segments)

    seg_tab.update_from_model(model)
    viewer.update_from_model(model)
    viewer.select_body_markers()

    spec = SegmentSpec(a="WaistLeft", b="LKneeOut", geometry="line", group="left_leg")
    seg_tab.add_segment(spec)
    assert any(s.a == "WaistLeft" and s.b == "LKneeOut" for s in seg_tab.segments)
    # Line collection allocated; some line segments rendered.
    assert viewer.user_line_segment_count >= 1


def test_geometry_switch_swaps_artist_kind(qt_app) -> None:
    from src.apps.services.segment_set_io import SegmentSpec  # type: ignore
    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=20)
    seg_tab = SegmentsTab()
    viewer = Viewer3DTab()
    seg_tab.segments_changed.connect(viewer.set_user_segments)
    seg_tab.update_from_model(model)
    viewer.update_from_model(model)
    viewer.select_body_markers()

    seg_tab.add_segment(
        SegmentSpec(a="WaistLeft", b="LKneeOut", geometry="line", group="left_leg")
    )
    last = len(seg_tab.segments) - 1
    seg_tab.set_segment_geometry(last, "cylinder")
    assert seg_tab.segments[last].geometry == "cylinder"
    assert viewer.user_cylinder_count >= 1


def test_visibility_toggle(qt_app) -> None:
    from src.apps.services.segment_set_io import SegmentSpec  # type: ignore
    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore

    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=20)
    tab = SegmentsTab()
    tab.update_from_model(model)
    tab.add_segment(
        SegmentSpec(a="WaistLeft", b="LKneeOut", geometry="line", group="left_leg")
    )
    last = len(tab.segments) - 1
    tab.set_segment_visibility(last, False)
    assert tab.segments[last].visible is False


def test_save_load_round_trip(qt_app, tmp_path) -> None:
    from src.apps.services.segment_set_io import (  # type: ignore
        SegmentSet,
        SegmentSpec,
        load_segment_set,
        save_segment_set,
    )

    seg_set = SegmentSet(
        segments=(
            SegmentSpec(
                a="WaistLeft",
                b="WaistRight",
                geometry="line",
                group="pelvis",
                visible=True,
                radius=0.02,
            ),
            SegmentSpec(
                a="WaistLeft",
                b="LKneeOut",
                geometry="cylinder",
                group="left_leg",
                visible=False,
                radius=0.018,
            ),
        )
    )
    out = tmp_path / "segments.json"
    save_segment_set(out, seg_set)
    loaded = load_segment_set(out)
    assert loaded.segments == seg_set.segments


def test_invalid_segment_spec_raises() -> None:
    from src.apps.services.segment_set_io import SegmentSpec  # type: ignore

    with pytest.raises(ValueError):
        SegmentSpec(a="A", b="A")
    with pytest.raises(ValueError):
        SegmentSpec(a="A", b="B", geometry="bezier")
    with pytest.raises(ValueError):
        SegmentSpec(a="A", b="B", radius=-1.0)
