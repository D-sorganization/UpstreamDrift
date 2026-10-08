"""Hands close-up camera tracking and grip plot payload in the native export."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.shared.python.biomechanics.grip_extraction import unavailable_analysis
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.core import (
    ExportSettings,
    OverlayFeed,
    view_lookats,
)
from src.tools.native_viewer_export.runner import write_grip_json

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _feed(points: list) -> OverlayFeed:
    return OverlayFeed(
        lambda k: SimpleNamespace(
            metadata={} if points[k] is None else {"grip_midpoint_m": points[k]}
        )
    )


def test_focus_at_reads_grip_midpoint_and_caches_frames() -> None:
    calls: list[int] = []

    def frame_at(k: int) -> SimpleNamespace:
        calls.append(k)
        return SimpleNamespace(metadata={"grip_midpoint_m": (k, 0.0, 1.0)})

    feed = OverlayFeed(frame_at)
    assert feed.focus_at(3) == (3.0, 0.0, 1.0)
    assert feed.focus_at(3) == (3.0, 0.0, 1.0)
    assert calls == [3]


def test_focus_at_none_without_grip_metadata() -> None:
    assert _feed([None]).focus_at(0) is None


def test_view_lookats_target_is_grip_midpoint_over_time() -> None:
    points = [(0.1, 0.2, 1.0), None, (0.3, 0.2, 0.9)]
    settings = replace(
        ExportSettings(),
        views=("hands_closeup", "face_on"),
        multiview=False,
        lookat_m=(0.0, 0.0, 0.9),
    )
    looks = view_lookats(settings, [0, 1, 2], _feed(points))
    assert looks["hands_closeup"][0] == pytest.approx((0.1, 0.2, 1.0))
    # missing sample holds the last finite point, never the origin
    assert looks["hands_closeup"][1] == pytest.approx((0.1, 0.2, 1.0))
    assert looks["hands_closeup"][2] == pytest.approx((0.3, 0.2, 0.9))
    assert all(p == pytest.approx((0.0, 0.0, 0.9)) for p in looks["face_on"])


def test_view_lookats_without_overlay_keeps_static_lookat() -> None:
    settings = replace(
        ExportSettings(),
        views=("hands_closeup",),
        multiview=False,
        lookat_m=(0.0, 0.0, 0.9),
    )
    looks = view_lookats(settings, [0, 1], None)
    assert looks["hands_closeup"] == [pytest.approx((0.0, 0.0, 0.9))] * 2


def test_worker_job_lookat_for_roundtrip(tmp_path: Path) -> None:
    job = WorkerJob(
        "b",
        "q",
        [0, 1],
        ["hands_closeup", "face_on"],
        64,
        64,
        [0.0, 0.0, 0.9],
        None,
        "o",
        lookats={"hands_closeup": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
    )
    path = tmp_path / "job.json"
    job.dump(path)
    loaded = WorkerJob.load(path)
    assert loaded.lookat_for("hands_closeup", 1) == [4.0, 5.0, 6.0]
    assert loaded.lookat_for("face_on", 1) == [0.0, 0.0, 0.9]


def test_write_grip_json_marks_unavailable_not_zero(tmp_path: Path) -> None:
    bundle = SimpleNamespace(steps=8, dt_s=0.001)
    swing = SimpleNamespace(bundle=bundle, swing="s")
    feed = OverlayFeed(
        lambda k: SimpleNamespace(metadata={}),
        grip_analyses=lambda idx: [unavailable_analysis("no grip") for _ in idx],
    )
    path = write_grip_json(swing, "mujoco", feed, tmp_path, impact_time_s=0.004)
    payload = json.loads(path.read_text())
    assert path.name == "s_mujoco_grip_wrench.json"
    assert payload["split_method"] == "unavailable"
    assert payload["events"] == {"impact": 0.004}
    assert payload["traces"]["net_force_n"]["magnitude"] == [None] * 3
