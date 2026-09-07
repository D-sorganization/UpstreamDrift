"""Clip export and take comparison on a synthetic bundle (#9680, #9681)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.capture_rig.clips import (
    ClipRange,
    clip_from_session,
    compare_from_sessions,
    events_for,
    metric_deltas,
    resolve_frame,
)
from src.tools.capture_rig.session import load_session
from tests.tools.capture_rig.test_core import SIZE, _bundle, _observations

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.unit


def _with_analysis(root: Path, events: dict[str, int], extra: float = 1.0) -> None:
    (root / "analysis_2d").mkdir(exist_ok=True)
    payload = {
        "view": "cam_a",
        "peak_hand_speed_bh_per_s": 5.0 * extra,
        "events": {
            "address_frame": events["address"],
            "top_frame": events["top"],
            "peak_speed_frame": events["peak"],
            "finish_frame": events["finish"],
            "backswing_s": 0.3,
            "downswing_s": 0.1,
            "tempo_ratio": 3.0,
        },
    }
    (root / "analysis_2d" / "cam_a.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_resolve_frame_by_event_offset_and_number() -> None:
    events = {"address": 10, "top": 40, "peak": 50, "finish": 70}
    assert resolve_frame("peak", events) == 50
    assert resolve_frame("top-5", events) == 35
    assert resolve_frame("address+3", events) == 13
    assert resolve_frame("7", events) == 7
    assert resolve_frame("address-30", events) == 0  # clamped
    with pytest.raises(Exception, match="unknown event"):
        resolve_frame("impact", events)
    with pytest.raises(Exception, match="before first"):
        ClipRange(5, 4)


def test_clip_export_writes_slow_motion_with_overlay(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    _observations(root)
    _with_analysis(root, {"address": 1, "top": 3, "peak": 5, "finish": 8})
    media = load_session(root)
    assert events_for(media, "cam_a") == {
        "address": 1,
        "top": 3,
        "peak": 5,
        "finish": 8,
    }
    out = tmp_path / "clip.avi"
    result = clip_from_session(
        root, "cam_a", start="address", end="finish+1", out=out, speed=0.5
    )
    assert result["first"] == 1 and result["last"] == 9 and result["frames"] == 9
    assert result["fps"] == pytest.approx(5.0)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 9
    assert (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    ) == SIZE
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.any()
    with pytest.raises(Exception, match="speed"):
        clip_from_session(root, "cam_a", start="0", end="2", out=out, speed=2.0)


def test_compare_takes_side_by_side_and_deltas(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    a = _bundle(tmp_path / "a")
    _observations(a)
    _with_analysis(a, {"address": 1, "top": 4, "peak": 6, "finish": 9}, extra=1.0)
    (tmp_path / "b").mkdir()
    b = _bundle(tmp_path / "b")
    _observations(b)
    _with_analysis(b, {"address": 0, "top": 2, "peak": 5, "finish": 8}, extra=1.2)
    out = tmp_path / "compare.avi"
    result = compare_from_sessions(
        a, "cam_a", b, "cam_a", out=out, align="top", speed=1.0
    )
    assert result["left"]["event_frame"] == 4 and result["right"]["event_frame"] == 2
    assert result["frames"] == 20  # (1 s + 1 s) at 10 fps
    assert result["deltas"]["peak_hand_speed_bh_per_s"]["delta"] == pytest.approx(1.0)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 2 * SIZE[0]
    cap.release()
    assert out.with_suffix(".json").is_file()
    assert metric_deltas({"x": 1.0, "s": "text"}, {"x": 3.0, "s": "other"}) == {
        "x": {"a": 1.0, "b": 3.0, "delta": 2.0}
    }
