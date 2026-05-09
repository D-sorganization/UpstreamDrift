"""Production-readiness regressions for async video background processing."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from src.api.routes import video
from src.api.task_manager import TaskManager


class _FakeConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeResult:
    total_frames = 1
    valid_frames = 1
    average_confidence = 0.9
    quality_metrics = {"ok": True}


class _BlockingPipeline:
    def __init__(self, config: _FakeConfig) -> None:
        self.config = config

    def process_video(self, video_path: Path) -> _FakeResult:
        time.sleep(0.15)
        return _FakeResult()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_video_background_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blocking pipeline work runs off the event loop (#3942)."""
    monkeypatch.setattr(
        video,
        "_load_video_pipeline_classes",
        lambda: (_BlockingPipeline, _FakeConfig),
    )
    task_manager = TaskManager()
    video_path = tmp_path / "swing.mp4"
    video_path.write_bytes(b"video")
    task_manager.set("task-1", {"status": "started"})

    started = time.perf_counter()
    task = asyncio.create_task(
        video._process_video_background(
            "task-1",
            video_path,
            "swing.mp4",
            "mediapipe",
            0.5,
            "fake-hash",
            task_manager,
        )
    )
    await asyncio.sleep(0.02)
    elapsed = time.perf_counter() - started

    await task
    await task_manager.shutdown()
    assert elapsed < 0.1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_video_background_logs_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Temporary video cleanup failures are logged instead of escaping (#3942)."""
    monkeypatch.setattr(
        video,
        "_load_video_pipeline_classes",
        lambda: (_BlockingPipeline, _FakeConfig),
    )
    task_manager = TaskManager()
    video_path = tmp_path / "swing.mp4"
    video_path.write_bytes(b"video")
    task_manager.set("task-1", {"status": "started"})

    original_unlink = Path.unlink

    def raising_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == video_path:
            raise OSError("locked")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", raising_unlink)

    with caplog.at_level("WARNING", logger=video.logger.name):
        await video._process_video_background(
            "task-1",
            video_path,
            "swing.mp4",
            "mediapipe",
            0.5,
            "fake-hash",
            task_manager,
        )

    await task_manager.shutdown()
    assert any(
        "Failed to clean up temp video" in record.message for record in caplog.records
    )
