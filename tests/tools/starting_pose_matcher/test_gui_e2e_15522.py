"""Native offscreen Starting-Pose Matcher interactions (Refs #15522).

Only capture discovery and OS file pickers are replaced. FK skeletons,
transforms, CSV loading, rendering, JSON export and Qt timers remain real.
"""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def matcher(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    try:
        from PyQt6.QtWidgets import QApplication
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib not loadable: {exc}")

    from src.motion_capture import capture_registry
    from src.tools.starting_pose_matcher.gui import StartingPoseMatcher

    def no_capture(*args: Any, **kwargs: Any) -> None:
        raise capture_registry.CaptureRegistryError("No private capture in E2E")

    monkeypatch.setattr(capture_registry, "resolve_capture", no_capture)
    app = QApplication.instance() or QApplication([])
    window = StartingPoseMatcher()
    window.resize(1280, 900)
    window.show()
    app.processEvents()
    try:
        assert window.isVisible()
        assert "Starting-Pose Matcher" in window.windowTitle()
        yield window
    finally:
        window.cleanup()
        window.close()
        window.deleteLater()
        app.processEvents()


def _edit(spin: Any, text: str) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, text)
    QTest.keyClick(spin, Qt.Key.Key_Return)


def test_transform_export_and_reset(
    matcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QFileDialog

    slot = matcher.poses["TopofBackswing"]
    points = np.asarray(list(slot.skeleton.joints.values()))
    assert len(points) > 5
    _edit(matcher.s_tx.spin, "0.250")
    _edit(matcher.s_rz.spin, "90.0")
    _edit(matcher.s_scale.spin, "1.15")
    assert matcher.ax.lines or matcher.ax.collections
    pivot = np.asarray(matcher.transform.pivot)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    expected = (points - pivot) @ rotation.T * 1.15 + pivot + [0.25, 0, 0]
    np.testing.assert_allclose(matcher.transform.apply(points), expected, atol=1e-10)

    path = tmp_path / "offsets.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
    QTest.mouseClick(matcher.btn_save, Qt.MouseButton.LeftButton)
    exported = json.loads(path.read_text(encoding="utf-8"))["transform"]
    assert exported["tx"] == pytest.approx(0.25)
    assert exported["rz"] == pytest.approx(90)
    assert exported["scale"] == pytest.approx(1.15)
    assert exported["units"]["translation"] == "metres"
    QTest.mouseClick(matcher.btn_reset_all, Qt.MouseButton.LeftButton)
    np.testing.assert_allclose(matcher.transform.apply(points), points, atol=1e-10)


def test_rotation_lock_discards_unlocked_tilt(matcher: Any) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    assert not matcher.s_rx.spin.isEnabled()
    # Space activates the focused checkbox even in a scrolled control panel.
    matcher.cb_lock_xy.setFocus()
    QTest.keyClick(matcher.cb_lock_xy, Qt.Key.Key_Space)
    assert matcher.s_rx.spin.isEnabled() and matcher.s_ry.spin.isEnabled()
    _edit(matcher.s_rx.spin, "30.0")
    _edit(matcher.s_ry.spin, "-15.0")
    assert matcher.transform.rx == pytest.approx(30)
    assert matcher.transform.ry == pytest.approx(-15)
    QTest.keyClick(matcher.cb_lock_xy, Qt.Key.Key_Space)
    assert matcher.lock_xy_rotation
    assert not matcher.s_rx.spin.isEnabled() and not matcher.s_ry.spin.isEnabled()
    assert matcher.transform.rx == matcher.transform.ry == 0


def test_csv_playback_stops_when_standalone_window_closes(
    matcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QFileDialog

    path = tmp_path / "synthetic-trajectory.csv"
    path.write_text(
        "time,club_head_X,club_head_Y,club_head_Z\n"
        + "".join(f"{i * 0.01},{i * 0.01},0,1\n" for i in range(100)),
        encoding="utf-8",
    )
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    QTest.mouseClick(
        matcher._pose_trajectory_buttons["TopofBackswing"], Qt.MouseButton.LeftButton
    )
    slot = matcher.poses["TopofBackswing"]
    assert len(slot.trajectory) == 100
    matcher.combo_playback_target.setFocus()
    QTest.keyClick(matcher.combo_playback_target, Qt.Key.Key_Home)
    QTest.keyClick(matcher.combo_playback_target, Qt.Key.Key_Down)
    assert matcher.playback_target == "Skeleton"
    QTest.mouseClick(matcher.btn_play, Qt.MouseButton.LeftButton)
    QTest.qWait(150)
    assert matcher._timer.isActive()
    assert slot.trajectory_frame_index > 0
    QTest.mouseClick(matcher.btn_play, Qt.MouseButton.LeftButton)
    paused = slot.trajectory_frame_index
    QTest.qWait(100)
    assert slot.trajectory_frame_index == paused
    QTest.mouseClick(matcher.btn_play, Qt.MouseButton.LeftButton)
    assert matcher._timer.isActive()
    matcher.close()
    assert not matcher.isVisible()
    assert not matcher._timer.isActive(), (
        "Closing the standalone GUI must stop playback"
    )
    closed = slot.trajectory_frame_index
    QTest.qWait(100)
    assert slot.trajectory_frame_index == closed
    assert not matcher.is_playing
    assert "Play" in matcher.btn_play.text()
    matcher.show()
    QTest.mouseClick(matcher.btn_play, Qt.MouseButton.LeftButton)
    assert matcher.is_playing and matcher._timer.isActive()
