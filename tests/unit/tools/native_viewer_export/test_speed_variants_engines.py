"""Full-, half- and quarter-speed variants for every engine (GCV-14, #11720).

Frame counts follow from the simulation time span; the HUD reads 0 ms from
impact at the frame nearest the impact detected from clubhead kinematics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance.club_face import ball_passage
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.video_timing.frame_schedule import (
    SPEED_VARIANTS,
    FrameSchedule,
    speed_suffix,
)
from src.tools.native_viewer_export import core
from src.tools.native_viewer_export.core import ENGINES, ExportSettings
from src.tools.native_viewer_export.runner import ExportJob, run_export

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COORDS = ("A", "B")
DT = 0.002
STEPS = 800  # 1.6 s at 500 Hz


def _overshoot_head() -> tuple[np.ndarray, np.ndarray]:
    """Head path whose peak-speed sample is not at the ball (GCV-14 bug shape)."""
    pts = [
        (0.00, 0.0, 0.0),
        (0.60, -0.8, 0.3),
        (0.90, -0.05, 0.02),
        (0.94, 0.03, -0.01),
        (1.00, 0.5, 0.14),
        (1.30, 0.6, 0.30),
        (1.60, 0.6, 0.30),
    ]
    k = np.array(pts)
    t = np.arange(STEPS + 1) * DT
    head = np.stack(
        [
            np.interp(t, k[:, 0], k[:, 1]),
            np.zeros_like(t),
            np.interp(t, k[:, 0], k[:, 2]),
        ],
        axis=1,
    )
    return t, head


def _bundle() -> InputBundle:
    t = np.arange(STEPS + 1) * DT
    q = np.stack([t, t * 2.0], axis=1)
    return InputBundle(
        spec_bytes=json.dumps({"coordinate_order": list(COORDS)}).encode(),
        coordinate_order=COORDS,
        dt_s=DT,
        q0=q[0],
        v0=np.zeros(2),
        efforts=np.zeros((STEPS, 2)),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
    )


class _Backend:
    def __init__(self, engine: str) -> None:
        self.engine = engine

    def unavailable_reason(self) -> None:
        return None

    def render(self, swing, settings, indices, overlay):
        for _ in indices:
            yield {
                v: np.zeros((settings.height, settings.width, 3), np.uint8)
                for v in settings.views
            }


class _Writer:
    def __init__(self, path: Path, fps: int) -> None:
        self.path, self.fps, self.n = path, fps, 0

    def append_data(self, frame) -> None:
        self.n += 1

    def close(self) -> None:
        self.path.write_bytes(b"x")


@pytest.mark.parametrize("speed", SPEED_VARIANTS)
def test_schedule_frame_count_follows_time_span(speed: float) -> None:
    times = np.arange(STEPS + 1) * DT
    schedule = FrameSchedule(times, 60, speed)
    span = times[-1] - times[0]
    assert schedule.n_frames == math.floor(span * 60 / speed + 1e-9) + 1
    np.testing.assert_allclose(np.diff(schedule.sample_times_s), speed / 60)


def test_variants_are_full_half_and_quarter() -> None:
    assert SPEED_VARIANTS == (1.0, 0.5, 0.25)
    assert [speed_suffix(v) for v in SPEED_VARIANTS] == ["_1x", "_0p5x", "_0p25x"]
    assert ExportSettings().speeds == SPEED_VARIANTS


def test_every_engine_exports_all_variants_with_zero_ms_at_impact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t, head = _overshoot_head()
    t_impact, _, _ = ball_passage(t, head)  # kinematic impact, not a fixed time
    huds: list[tuple[str, core.HudInfo]] = []

    def spy(frame, hud):
        huds.append((hud.speed, hud))
        return frame

    monkeypatch.setattr(core, "draw_hud", spy)
    bundle = _bundle()
    bpath = tmp_path / "b.npz"
    bundle.save(bpath)
    backends = {e: _Backend(e) for e in ENGINES}
    writers: dict[Path, _Writer] = {}

    def make_writer(path: Path, fps: int) -> _Writer:
        writers[path] = _Writer(path, fps)
        return writers[path]

    job = ExportJob(bpath, tmp_path / "out", "driver", "Driver", ENGINES)
    results = run_export(
        job,
        ExportSettings(width=64, height=48, views=("face_on",), multiview=False),
        backends.__getitem__,
        lambda swing, engine: (None, (0.0, 0.0, 0.9)),
        make_writer,
        impact_detector=lambda swing: ball_passage(
            np.asarray(swing.source_times_s), head
        )[0],
    )
    span = t[-1] - t[0]
    assert [r.engine for r in results] == list(ENGINES)
    for r in results:
        assert sorted(r.frames_by_suffix) == ["_0p25x", "_0p5x", "_1x"]
        for speed in SPEED_VARIANTS:
            expected = math.floor(span * 60 / speed + 1e-9) + 1
            assert r.frames_by_suffix[speed_suffix(speed)] == expected
            name = f"driver_{r.engine}_face_on{speed_suffix(speed)}.mp4"
            clip = tmp_path / "out" / name
            assert writers[clip].fps == 60 and writers[clip].n == expected
    for speed in SPEED_VARIANTS:
        rows = [h for s, h in huds if s == speed]
        nearest = min(rows, key=lambda h: abs(h.time_s - t_impact))
        assert abs(nearest.ms_from_impact) <= 0.5 * speed / 60 * 1000.0 + 1e-6
        assert nearest.ms_from_impact == pytest.approx(
            (nearest.time_s - t_impact) * 1000.0
        )
