"""Skeleton-overlay tests for the C3D viewer's 3D tab."""

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


def test_skeleton_segment_count_matches_canonical(qt_app) -> None:
    """The 28-marker anatomical subset yields 26 canonical segments."""
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore
    from src.shared.python.motion_matching.body_skeleton import (  # type: ignore
        default_body_segments,
    )

    expected_segments = default_body_segments(list(ANATOMICAL_28))
    assert len(expected_segments) == 26

    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=50)
    tab = Viewer3DTab()
    tab.update_from_model(model)
    tab.select_body_markers()

    assert tab.skeleton_segment_count == len(expected_segments)


def test_skeleton_visibility_toggle(qt_app) -> None:
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=20)
    tab = Viewer3DTab()
    tab.update_from_model(model)
    tab.select_body_markers()

    coll = tab._skeleton_collection
    assert coll is not None
    assert coll.get_visible() is True

    tab.check_skeleton.setChecked(False)
    assert coll.get_visible() is False

    tab.check_skeleton.setChecked(True)
    assert coll.get_visible() is True


def test_skeleton_partial_marker_set(qt_app) -> None:
    """Missing endpoints simply drop their segments — no exception."""
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    partial = list(ANATOMICAL_28[:10])  # head + pelvis + a few torso
    model = make_synthetic_model(partial, n_frames=20)
    tab = Viewer3DTab()
    tab.update_from_model(model)
    tab.select_all_markers()

    # Some segments should exist (pelvis + head); count must be < full 26.
    assert 0 < tab.skeleton_segment_count < 26


def test_view_preset_validation(qt_app) -> None:
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    tab = Viewer3DTab()
    with pytest.raises(ValueError):
        tab.set_view_preset("Bogus")
