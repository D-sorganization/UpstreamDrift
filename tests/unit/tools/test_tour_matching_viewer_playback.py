"""Tests for TourMatchingViewerWidget transport playback controls (MV-04 #10480)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt6 import QtWidgets

pytestmark = pytest.mark.unit

from src.tools.tour_matching_viewer.core import ReplayData
from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget


ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


@pytest.fixture
def viewer_widget(qtbot) -> TourMatchingViewerWidget:  # type: ignore[no-untyped-def]
    widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
    qtbot.addWidget(widget)
    return widget


def test_viewer_has_playback_transport_controls(
    viewer_widget: TourMatchingViewerWidget,
) -> None:
    assert hasattr(viewer_widget, "_transport")
    transport = viewer_widget._transport
    assert transport.play_button is not None
    assert transport.scrubber is not None
    assert transport.speed_combo is not None
    assert len(transport.event_buttons) >= 2


def test_viewer_load_replay_configures_transport(
    viewer_widget: TourMatchingViewerWidget,
) -> None:
    import json

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    coords_order = tuple(spec["coordinate_order"])
    n_coords = len(coords_order)
    n_frames = 100
    times_s = np.linspace(0.0, 1.814, n_frames)
    q = np.zeros((n_frames, n_coords))
    replay = ReplayData(
        time_s=times_s,
        coordinates=q,
        model_markers_m=np.zeros((n_frames, 5, 3)),
        target_markers_m=np.zeros((n_frames, 5, 3)),
        valid_mask=np.ones((n_frames, 5), dtype=bool),
        coordinate_names=coords_order,
    )

    viewer_widget.load_replay_data(replay)
    transport = viewer_widget._transport
    assert transport.duration_s() == pytest.approx(1.814)

    # Jump to events
    transport.jump_to_event(0)  # Address
    assert transport.current_time_s() == pytest.approx(0.0)

    # Jump to Finish
    transport.jump_to_event(len(transport.event_buttons) - 1)
    assert transport.current_time_s() == pytest.approx(1.814)


def test_camera_orbit_usable_while_paused(
    viewer_widget: TourMatchingViewerWidget,
) -> None:
    transport = viewer_widget._transport
    transport.pause()
    assert not transport.timer().isActive()

    # Orbit camera while paused
    viewer_widget._ax.view_init(elev=35, azim=60)
    assert viewer_widget._ax.elev == 35
    assert viewer_widget._ax.azim == 60
