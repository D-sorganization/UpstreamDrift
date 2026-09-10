"""Original-video scrubbing is responsive and writes evidence only after selection."""

from pathlib import Path
import threading

import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets")
cv2 = pytest.importorskip("cv2")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def video(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    assert writer.isOpened()
    for index in range(8):
        writer.write(
            np.full((size[1], size[0], 3), (index * 20, 80, 160), dtype=np.uint8)
        )
    writer.release()


@pytest.mark.parametrize("size", [(64, 48), (2560, 1440)])
def test_preview_reads_original_frame_and_releases_decoder(qtbot, tmp_path, size):
    from src.tools.capture_rig.reference_calibration.video_preview import VideoPreview

    path = tmp_path / "source.avi"
    video(path, size)
    host = QWidget()
    qtbot.addWidget(host)
    loader = VideoPreview(path, host)
    frames = []
    loader.ready.connect(frames.append)
    loader.request(4)
    qtbot.waitUntil(lambda: bool(frames), timeout=10000)
    assert frames[-1].index == 4 and frames[-1].frame_count == 8
    assert frames[-1].fps == pytest.approx(30)
    assert frames[-1].image.dtype == np.uint8
    scale = min(1.0, 1280 / max(size))
    assert frames[-1].image.shape == (round(size[1] * scale), round(size[0] * scale), 3)
    assert frames[-1].image[10, 10, 0] == pytest.approx(80, abs=6)
    loader.close()
    with pytest.raises(ValueError, match="closed"):
        loader.request(0)
    assert not (tmp_path / "reference_calibration").exists()


def test_scrubbing_coalesces_requests_and_closes_after_active_read(
    qtbot, tmp_path, monkeypatch
):
    from src.tools.capture_rig.reference_calibration import video_preview

    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    reads = []

    class Reader:
        frame_count, fps, width, height = 10, 30, 64, 48

        def __init__(self, path):
            pass

        def read(self, index):
            reads.append(index)
            if index == 0:
                started.set()
                assert release.wait(5)
            return np.zeros((48, 64, 3), dtype=np.uint8)

        def close(self):
            closed.set()

    monkeypatch.setattr(video_preview, "VideoReader", Reader)
    host = QWidget()
    qtbot.addWidget(host)
    loader = video_preview.VideoPreview(tmp_path / "slow.avi", host)
    frames = []
    loader.ready.connect(frames.append)
    try:
        loader.request(0)
        qtbot.waitUntil(started.is_set)
        loader.request(1)
        loader.request(8)
        release.set()
        qtbot.waitUntil(lambda: bool(frames) and frames[-1].index == 8, timeout=10000)
        assert reads == [0, 8]
    finally:
        release.set()
        loader.close()
    qtbot.waitUntil(closed.is_set, timeout=5000)


def test_visual_selector_returns_displayed_original_frame_and_fullscreen_restores(
    qtbot, tmp_path
):
    from src.tools.capture_rig.reference_calibration.frame_selector import (
        ReferenceFrameSelector,
    )

    path = tmp_path / "source.avi"
    video(path)
    dialog = ReferenceFrameSelector(path, context="Practice · Front", initial_index=0)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.use.isEnabled(), timeout=10000)
    dialog.slider.setValue(5)
    qtbot.waitUntil(
        lambda: dialog.displayed_index == 5 and dialog.use.isEnabled(), timeout=10000
    )
    assert "Frame 5" in dialog.clock.text()
    qtbot.mouseDClick(dialog.canvas, Qt.MouseButton.LeftButton)
    assert dialog.isFullScreen()
    dialog.leave_fullscreen()
    assert not dialog.isFullScreen() and dialog.isVisible()
    dialog.use.click()
    assert dialog.selected_frame == 5
    assert not (tmp_path / "reference_calibration").exists()


def test_missing_video_is_actionable_and_cannot_be_selected(qtbot, tmp_path):
    from src.tools.capture_rig.reference_calibration.frame_selector import (
        ReferenceFrameSelector,
    )

    dialog = ReferenceFrameSelector(
        tmp_path / "missing.avi", context="Front", initial_index=0
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: "Cannot preview" in dialog.status.text(), timeout=10000)
    assert not dialog.use.isEnabled()
    dialog.reject()
    assert dialog.selected_frame is None


def test_closing_during_decode_returns_immediately_and_drops_late_result(
    qtbot, tmp_path, monkeypatch
):
    from src.tools.capture_rig.reference_calibration import video_preview

    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Reader:
        frame_count, fps, width, height = 10, 30, 64, 48

        def __init__(self, path):
            pass

        def read(self, index):
            started.set()
            assert release.wait(5)
            return np.zeros((48, 64, 3), dtype=np.uint8)

        def close(self):
            closed.set()

    monkeypatch.setattr(video_preview, "VideoReader", Reader)
    host = QWidget()
    qtbot.addWidget(host)
    loader = video_preview.VideoPreview(tmp_path / "slow.avi", host)
    frames = []
    loader.ready.connect(frames.append)
    try:
        loader.request(0)
        qtbot.waitUntil(started.is_set)
        loader.close()
        assert not closed.is_set()
    finally:
        release.set()
        loader.close()
    qtbot.waitUntil(closed.is_set)
    assert frames == []


def test_small_window_and_pause_never_accept_an_undisplayed_seek(qtbot, tmp_path):
    from src.tools.capture_rig.reference_calibration.frame_selector import (
        ReferenceFrameSelector,
    )

    path = tmp_path / "source.avi"
    video(path)
    dialog = ReferenceFrameSelector(path, context="Practice · Front")
    qtbot.addWidget(dialog)
    dialog.resize(640, 560)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.use.isEnabled(), timeout=10000)
    assert dialog.width() <= 640 and dialog.height() <= 560
    dialog.play.click()
    assert not dialog.use.isEnabled()
    dialog.play.click()
    assert not dialog.use.isEnabled()
    qtbot.waitUntil(lambda: dialog.use.isEnabled())
    dialog.reject()
    assert dialog.selected_frame is None
