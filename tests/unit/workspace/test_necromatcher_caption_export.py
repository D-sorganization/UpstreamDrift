"""Tiny synthetic codec exports preserve source-sized and legacy annotations."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.unit.workspace.test_necromatcher_video import video_case as _video_case

video_case = _video_case

pytestmark = pytest.mark.unit


def test_compact_two_frame_codec_and_default_pixel_identity(
    video_case: Any, tmp_path: Path
) -> None:
    import cv2
    from src.shared.python.workspace import CaptionOverlayOptions, export_fit_video
    from src.shared.python.workspace.necromatcher_caption import (
        validate_caption_manifest,
    )

    library, capture, _ = video_case
    original = Path(capture.path).read_bytes()
    legacy = export_fit_video(
        library, "video-fit", tmp_path / "legacy", selected_frames=(0, 1)
    )
    disabled = export_fit_video(
        library,
        "video-fit",
        tmp_path / "disabled",
        selected_frames=(0, 1),
        caption_overlay=None,
    )
    assert legacy == disabled
    for name in ("frame-000000.png", "frame-000001.png"):
        assert (tmp_path / "legacy" / name).read_bytes() == (
            tmp_path / "disabled" / name
        ).read_bytes()
    compact = export_fit_video(
        library,
        "video-fit",
        tmp_path / "compact",
        selected_frames=(0, 1),
        caption_overlay=CaptionOverlayOptions(),
    )
    validate_caption_manifest(compact, CaptionOverlayOptions())
    assert compact["image_size"] == [320, 240]
    assert len(compact["frames"]) == 2
    for row in compact["frames"]:
        assert row["caption_overlay"]["rectangle"][3] <= 48
        assert row["frame"] == legacy["frames"][row["frame_index"]]["frame"]
    assert Path(capture.path).read_bytes() == original
    decoded = cv2.imread(str(tmp_path / "compact/frame-000000.png"))
    old = cv2.imread(str(tmp_path / "legacy/frame-000000.png"))
    np.testing.assert_array_equal(decoded[:150], old[:150])
    record = json.loads(
        (tmp_path / "compact/manifest.json").read_text(encoding="utf-8")
    )
    record["frames"][0]["caption_overlay"]["lines"][0]["text"] = "Accepted"
    with pytest.raises(ValueError, match="Caption"):
        validate_caption_manifest(record, CaptionOverlayOptions())


def test_worker_caption_keyword_absent_preserves_legacy_call(
    video_case: Any, monkeypatch: Any
) -> None:
    from src.shared.python.workspace import necromatcher_video_jobs as jobs
    from src.shared.python.workspace import necromatcher_video_worker as worker
    from src.shared.python.workspace import CaptionOverlayOptions
    from tests.unit.workspace.test_necromatcher_caption_jobs import _caption_worker
    from tests.unit.workspace.test_necromatcher_video_jobs import _wait

    library, _, _ = video_case
    stamp = jobs.fit_execution_stamp()
    monkeypatch.setattr(jobs, "fit_execution_stamp", lambda: stamp)
    monkeypatch.setattr(worker, "fit_execution_stamp", lambda: stamp)
    monkeypatch.setattr(jobs, "_execute_worker", _caption_worker)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("video-fit", caption_overlay=CaptionOverlayOptions())[
            "run_id"
        ]
        assert _wait(session, run)["status"] == "succeeded"
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            worker, "export_fit_video", lambda *args, **kwargs: calls.append(kwargs)
        )
        request_path = library.root / "video-runs" / run / "request.json"
        worker.execute(request_path)
        assert calls == [
            {"selected_frames": (0, 1), "caption_overlay": CaptionOverlayOptions()}
        ]
        request = json.loads(request_path.read_text(encoding="utf-8"))
        del request["caption_overlay"]
        request_path.write_text(json.dumps(request), encoding="utf-8")
        worker.execute(request_path)
        assert calls[-1] == {"selected_frames": (0, 1)}
    finally:
        session.close()
