"""Duplicate raw/overlay requests decode once and cannot mutate later frames."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from src.tools.capture_rig.player import VideoReader

pytestmark = pytest.mark.unit


class Capture:
    def __init__(self) -> None:
        self.reads = 0
        self.seeks: list[int] = []
        self.index = 0
        self.closed = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API
        return True

    def get(self, _prop: int) -> int:
        return 3

    def set(self, _prop: int, index: int) -> None:
        self.seeks.append(index)
        self.index = index

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        if self.closed or self.index >= 3:
            return False, None
        frame = np.full((3, 4, 3), self.index + 20, dtype=np.uint8)
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.closed = True


def reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[VideoReader, Capture]:
    path = tmp_path / "fixture.avi"
    path.touch()
    cap = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: cap)
    return VideoReader(path), cap


def test_raw_and_overlay_share_decode_but_not_mutable_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, cap = reader(tmp_path, monkeypatch)
    raw = video.read(1)
    assert raw is not None
    raw[:] = 255
    overlay = video.read(1)
    assert overlay is not None and np.all(overlay == 21)
    assert cap.reads == 1
    assert cap.seeks == [1]
    overlay[:] = 0
    assert np.all(video.read(1) == 21)


def test_new_frames_evict_cache_and_random_seek_remains_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, cap = reader(tmp_path, monkeypatch)
    assert np.all(video.read(0) == 20)
    assert np.all(video.read(1) == 21)
    assert np.all(video.read(0) == 20)
    assert cap.reads == 3
    assert cap.seeks == [0]
    assert video.read(3) is None
    assert np.all(video.read(2) == 22)


def test_close_releases_cached_pixels_and_does_not_return_stale_frame(
    tmp_path, monkeypatch
):
    video, cap = reader(tmp_path, monkeypatch)
    video.read(0)
    video.close()
    assert cap.closed
    assert video.read(0) is None


def test_benchmark_reports_bounded_composition_and_decode_samples() -> None:
    from scripts.benchmark_capture_responsiveness import run_benchmark

    with pytest.raises(ValueError):
        run_benchmark(samples=0)
    result = run_benchmark(samples=3)
    assert [item["views"] for item in result["composition"]] == [1, 3, 6]
    assert result["decode"]["duplicate"]["samples"] == 3
    assert (
        result["decode"]["duplicate"]["p95_ms"]
        >= result["decode"]["duplicate"]["median_ms"]
    )
