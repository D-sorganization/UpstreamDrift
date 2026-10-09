"""Hands close-up camera tracking and grip plot payload in the native export."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
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


def test_impact_only_clip_set_and_clean_hud_flags() -> None:
    from src.tools.native_viewer_export import cli

    args = cli.build_parser().parse_args(
        [
            "--bundle",
            "b.npz",
            "--out",
            "o",
            "--swing",
            "s",
            "--speeds",
            "",
            "--impact-window",
            "0.5",
            "--impact-speed",
            "0.25",
            "--no-hud",
        ]
    )
    settings = cli.build_settings(args)
    assert settings.speeds == () and settings.hud is False
    assert settings.impact_speed == 0.25
    with pytest.raises(ValueError):
        replace(settings, impact_window_s=None)


def _head_path(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic swing: address at the ball, back, then a fast return through it."""
    t = np.arange(n) * 0.005
    x = np.zeros(n)
    z = np.zeros(n)
    half = n // 2
    x[:half] = -0.8 * np.sin(np.linspace(0, np.pi / 2, half))
    z[:half] = 1.0 * np.sin(np.linspace(0, np.pi / 2, half))
    ret = np.linspace(0, 1, n - half)
    x[half:] = -0.8 * (1 - ret**2) + 0.4 * ret**3
    z[half:] = 1.0 * (1 - ret**1.5)
    return t, np.stack([x, np.zeros(n), z], axis=1)


def test_hud_impact_time_matches_impact_frame() -> None:
    """The exporter's impact time is the shared ``impact_frame`` rule, one detector."""
    from src.shared.python.model_appearance.club_face import impact_frame
    from src.tools.native_viewer_export import overlay

    t, head = _head_path()
    k = impact_frame(t, head)
    dt = float(t[1] - t[0])
    swing = SimpleNamespace(
        bundle=SimpleNamespace(coordinate_order=("a",), spec_bytes=b"{}"),
        q=np.zeros((len(t), 1)),
        source_times_s=tuple(t),
    )

    class _Source:
        def __init__(self, _spec: bytes) -> None:
            self.i = -1

        def frame_poses(self, _coords: dict) -> dict:
            self.i += 1
            pose = np.eye(4)
            pose[:3, 3] = head[self.i]
            return {"Clubhead": pose}

    import sys
    import types

    mod = types.ModuleType("overlay_source_stub")
    mod.MujocoOverlaySource = _Source  # type: ignore[attr-defined]
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(
            sys.modules,
            "src.engines.physics_engines.mujoco.python.overlay_source",
            mod,
        )
        got = overlay.detect_impact_time_s(swing)
    assert abs(got - t[k]) <= dt


def test_hud_impact_time_for_capture_a_fixture_is_within_one_frame() -> None:
    """Capture-A driver fixture: exporter impact time vs ``impact_frame`` (OSV-10)."""
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.overlay_source import (
        MujocoOverlaySource,
    )
    from src.shared.python.model_appearance.club_face import impact_frame
    from src.tools.native_viewer_export import overlay

    root = Path(__file__).resolve().parents[4]
    fixtures = root / "tests/fixtures/club_face"
    spec = root / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    if not spec.exists():
        pytest.skip("capture-A specification not present")
    q = np.load(fixtures / "swing_q_driver.npz")["q"].astype(float)
    order = json.loads(spec.read_text())["coordinate_order"]
    dt = 0.002  # the fixture holds every second sample of the 1 kHz reference
    times = np.arange(len(q)) * dt
    source = MujocoOverlaySource(spec.read_bytes())
    head = np.array(
        [
            source.frame_poses(dict(zip(order, row, strict=True)))["Clubhead"][:3, 3]
            for row in q
        ]
    )
    swing = SimpleNamespace(
        bundle=SimpleNamespace(
            coordinate_order=tuple(order), spec_bytes=spec.read_bytes()
        ),
        q=q,
        source_times_s=tuple(times),
    )
    got = overlay.detect_impact_time_s(swing)
    assert abs(got - times[impact_frame(times, head)]) <= dt
    assert 1.2 < got < 1.45  # downswing of the 1.8 s capture, not its end
