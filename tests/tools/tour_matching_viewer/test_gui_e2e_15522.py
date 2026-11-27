"""Native headless journeys through the real model replay viewer (Refs #15522)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtTest, QtWidgets  # noqa: E402

from src.tools.tour_matching_viewer.gui import (  # noqa: E402
    DEFAULT_SPEC_PATH,
    TourMatchingViewerWidget,
    TourMatchingViewerWindow,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """Write a synthetic model trajectory; no private captures or engine mocks."""
    spec = json.loads(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))
    coordinates = np.zeros((9, len(spec["coordinate_order"])))
    coordinates[:, 0] = np.linspace(0.0, 0.1, 9)
    targets = np.full((9, 2, 3), 0.5)
    path = tmp_path / "mujoco_demo.npz"
    np.savez(
        path,
        time_s=np.linspace(0.0, 2.0, 9),
        coordinates=coordinates,
        model_markers_m=targets + 0.01,
        target_markers_m=targets,
        valid_mask=np.ones((9, 2), dtype=bool),
    )
    return path


@pytest.fixture
def window(qtbot):
    """Launch the same shell used by the standalone module entry point."""
    shell = TourMatchingViewerWindow()
    qtbot.addWidget(shell)
    shell.show()
    qtbot.waitUntil(shell.isVisible)
    yield shell
    shell.close()


def _open(viewer, path: Path, monkeypatch) -> None:
    # Replace only the OS file chooser; loading, FK, rendering and signals are real.
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), "")
    )
    QtTest.QTest.mouseClick(viewer._open_btn, QtCore.Qt.MouseButton.LeftButton)


def test_launch_open_scrub_camera_and_play(window, archive, monkeypatch, qtbot) -> None:
    viewer = window.widget
    assert "No replay loaded" in viewer.title_caption
    _open(viewer, archive, monkeypatch)
    assert "mujoco_demo" in viewer.title_caption
    assert viewer.current_rms_error == pytest.approx(np.sqrt(3) * 0.01)
    assert len(viewer._ax.collections) > 0
    transport = viewer._transport
    QtTest.QTest.keyClick(transport.scrubber, QtCore.Qt.Key.Key_End)
    assert viewer.current_frame == 8
    assert viewer.current_frame_time_s == pytest.approx(2.0)
    QtTest.QTest.keyClick(viewer._camera_combo, QtCore.Qt.Key.Key_Down)
    assert viewer._ax.elev == 0
    assert viewer._ax.azim == -90
    QtTest.QTest.mouseClick(
        transport.event_buttons[0], QtCore.Qt.MouseButton.LeftButton
    )
    assert viewer.current_frame == 0
    QtTest.QTest.mouseClick(transport.play_button, QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: viewer.current_frame > 0, timeout=5000)
    QtTest.QTest.mouseClick(transport.play_button, QtCore.Qt.MouseButton.LeftButton)
    assert not transport.timer().isActive()
    frame = viewer.current_frame
    QtTest.QTest.qWait(80)
    assert viewer.current_frame == frame


def test_invalid_archive_preserves_replay_then_recovers(
    window, archive, tmp_path, monkeypatch
) -> None:
    viewer = window.widget
    _open(viewer, archive, monkeypatch)
    caption = viewer.title_caption
    bad = tmp_path / "invalid.npz"
    np.savez(bad, coordinates=np.zeros((2, 1)))
    errors = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *args: errors.append(args[2])
    )
    _open(viewer, bad, monkeypatch)
    assert len(errors) == 1
    assert "Missing 'time_s'" in errors[0]
    assert viewer.title_caption == caption
    _open(viewer, archive, monkeypatch)
    QtTest.QTest.mouseClick(
        viewer._transport.event_buttons[-1], QtCore.Qt.MouseButton.LeftButton
    )
    assert viewer.current_frame == 8


@pytest.mark.parametrize("standalone", [True, False], ids=["window", "embedded"])
def test_close_stops_playback(qtbot, archive, monkeypatch, standalone) -> None:
    shell = TourMatchingViewerWindow() if standalone else TourMatchingViewerWidget()
    viewer = shell.widget if standalone else shell
    qtbot.addWidget(shell)
    shell.show()
    try:
        _open(viewer, archive, monkeypatch)
        transport = viewer._transport
        QtTest.QTest.mouseClick(transport.play_button, QtCore.Qt.MouseButton.LeftButton)
        assert transport.timer().isActive()
        shell.close()
        assert not transport.timer().isActive()
        if not standalone:
            shell.show()
            QtTest.QTest.mouseClick(
                transport.event_buttons[-1], QtCore.Qt.MouseButton.LeftButton
            )
            assert viewer.current_frame == 8
            assert len(viewer._ax.collections) > 0
    finally:
        viewer.cleanup()
