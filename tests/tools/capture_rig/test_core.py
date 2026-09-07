"""Capture Rig tool: command vectors, session view, overlay and reader (no Qt)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.rig.bundle import build_index, write_bundle
from src.motion_capture.rig.plan import (
    CameraBinding,
    CameraControls,
    CaptureMode,
    RigPlan,
)
from src.motion_capture.rig.probe import RecordingProbe
from src.motion_capture.rig.recorder import RecordingResult
from src.tools.capture_rig import commands
from src.tools.capture_rig.overlay import PoseTrack, draw_pose, skeleton_edges
from src.tools.capture_rig.player import VideoReader, clamp_index
from src.tools.capture_rig.session import flatten_numbers, load_session

pytestmark = pytest.mark.unit

SIZE = (320, 200)


def _selection(tmp_path: Path) -> commands.PlanSelection:
    return commands.PlanSelection(
        plan=tmp_path / "plan.json",
        mode=CaptureMode(width=1280, height=720, fps=120),
        views=("cam_b",),
        controls=CameraControls(exposure=-6, auto_exposure=False),
    )


def test_plan_selection_args_mirror_the_cli_flags(tmp_path: Path) -> None:
    args = _selection(tmp_path).args()
    assert args[:2] == ["--plan", str(tmp_path / "plan.json")]
    assert args[2:4] == ["--mode", "1280x720@120:MJPG"]
    assert args[4:6] == ["--views", "cam_b"]
    assert args[6:] == ["--exposure", "-6", "--auto-exposure", "off"]
    assert commands.PlanSelection(plan=tmp_path / "p.json").args() == [
        "--plan",
        str(tmp_path / "p.json"),
    ]


def test_every_command_runs_the_rig_module(tmp_path: Path) -> None:
    sel = commands.PlanSelection(plan=tmp_path / "plan.json")
    session = tmp_path / "s"
    for argv in (
        commands.plan_check_command(sel),
        commands.record_command(sel, session, duration_s=30, warmup_s=4, dry_run=True),
        commands.proxy_command(session, encoder="libx264"),
        commands.ingest_command(session, estimator="openpose_dnn", max_frames=5),
        commands.reconstruct_command(
            session, anchor_segment="neck", anchor_m=0.53, intrinsics=tmp_path / "i"
        ),
        commands.calibrate_command(session, square_m=0.025),
    ):
        assert argv[:3] == [sys.executable, "-m", commands.RIG_MODULE]
    rec = commands.record_command(sel, session, duration_s=30, warmup_s=4, dry_run=True)
    assert rec[3] == "record" and "--dry-run" in rec and "--warmup" in rec
    ing = commands.ingest_command(session, estimator="openpose_dnn", max_frames=5)
    assert ing[-4:] == ["--estimator", "openpose_dnn", "--max-frames", "5"]
    rc = commands.reconstruct_command(
        session, anchor_segment="neck", anchor_m=0.53, cameras=tmp_path / "c.json"
    )
    assert "--anchor" in rc and rc[rc.index("--anchor") + 1] == "neck=0.53"
    assert commands.repo_root().joinpath("src", "motion_capture", "rig").is_dir()


def test_command_preconditions(tmp_path: Path) -> None:
    sel = commands.PlanSelection(plan=tmp_path / "plan.json")
    with pytest.raises(Exception, match="positive"):
        commands.record_command(sel, tmp_path, duration_s=0)
    with pytest.raises(Exception, match="not both"):
        commands.reconstruct_command(
            tmp_path,
            anchor_segment="neck",
            anchor_m=0.5,
            cameras=tmp_path,
            intrinsics=tmp_path,
        )
    with pytest.raises(Exception, match="not both"):
        commands.reconstruct_command(tmp_path, anchor_segment="neck", anchor_m=0.5)


def test_estimator_choices_come_from_the_registry() -> None:
    names = {c.name for c in commands.estimator_choices()}
    assert {"mediapipe", "openpose", "openpose_dnn"} <= names
    for choice in commands.estimator_choices():
        assert choice.available or choice.hint


def test_plan_override_controls_are_recorded_in_the_name() -> None:
    plan = RigPlan(name="p", cameras=(CameraBinding(view="a", serial="1"),))
    derived = plan.with_overrides(controls=CameraControls(gain=4, auto_exposure=True))
    assert derived.cameras[0].controls.gain == 4
    assert derived.name == "p+auto_exposure=1,gain=4"
    assert plan.with_overrides(controls=CameraControls()) is plan


def _write_video(path: Path, frames: int) -> None:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, SIZE)
    for i in range(frames):
        img = np.full((SIZE[1], SIZE[0], 3), 40, dtype=np.uint8)
        img[:, : 10 * (i + 1)] = 200  # a bar that grows with the frame index
        writer.write(img)
    writer.release()


def _bundle(tmp_path: Path) -> Path:
    plan = RigPlan(
        name="rig",
        cameras=(
            CameraBinding(view="cam_a", serial="1"),
            CameraBinding(view="cam_b", serial="2"),
        ),
    )
    video = tmp_path / "cam_a_1.avi"
    _write_video(video, 12)
    results = [
        RecordingResult("1", video, 0, video.stat().st_size),
        RecordingResult("2", tmp_path / "missing.avi", 1, 0),
    ]
    index = build_index(
        plan,
        results,
        1.2,
        tmp_path,
        prober=lambda p: RecordingProbe(12, 1.2, SIZE[0], SIZE[1], 10.0),
    )
    write_bundle(tmp_path, plan, index, started_utc="2026-09-07T00:00:00+00:00")
    return tmp_path


def _observations(tmp_path: Path) -> Path:
    names = ["nose", "left_shoulder", "right_shoulder", "left_elbow"]
    rows = []
    for i in (0, 1, 3):
        rows.append(
            {
                "camera_id": "1",
                "time_s": i / 10.0,
                "keypoints_px": [[10 + i, 20], [30, 40], [50, 40], [30, 80]],
                "confidence": [0.9, 0.9, 0.2, 0.9],
            }
        )
    payload = {
        "view": "cam_a",
        "identity": "1",
        "camera_id": "1",
        "fps": 10.0,
        "width": SIZE[0],
        "height": SIZE[1],
        "frames_total": 12,
        "frames_with_pose": 3,
        "detector_layout": {"name": "test", "keypoint_names": names},
        "frames": rows,
        "provenance": {"estimator": "mediapipe"},
    }
    obs_dir = tmp_path / "observations"
    obs_dir.mkdir()
    (obs_dir / "cam_a.json").write_text(json.dumps(payload), encoding="utf-8")
    (obs_dir / "observations.json").write_text(
        json.dumps(
            {
                "plan_name": "rig",
                "views": [
                    {
                        "view": "cam_a",
                        "identity": "1",
                        "status": "available",
                        "file": "cam_a.json",
                    },
                    {"view": "cam_b", "identity": "2", "status": "unavailable"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return obs_dir / "cam_a.json"


def test_session_media_reports_what_exists(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    obs = _observations(root)
    recon = root / "reconstruct"
    recon.mkdir()
    (recon / "swing_summary.json").write_text(
        json.dumps({"tempo": {"ratio": 3.0}, "x_factor_deg": 42.5, "ok": True}),
        encoding="utf-8",
    )
    media = load_session(root)
    assert media.plan_name == "rig" and media.ingested
    a, b = media.view("cam_a"), media.view("cam_b")
    assert a.recording == root / "cam_a_1.avi" and a.playable == a.recording
    assert a.observations == obs and a.fps == pytest.approx(10.0)
    assert b.recording is None and b.playable is None and b.observations is None
    assert media.problems and media.problems[0].startswith("cam_b")
    rows = dict(flatten_numbers(media.swing_summary or {}))
    assert rows == {"tempo.ratio": "3", "x_factor_deg": "42.5", "ok": "True"}
    (tmp_path / "nope").mkdir()
    with pytest.raises(ValueError):
        load_session(tmp_path / "nope")


def test_pose_track_indexes_frames_and_draws(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    track = PoseTrack.load(_observations(root))
    assert track.coverage == 3 and track.at(2) is None and track.at(3) is not None
    assert track.estimator == "mediapipe"
    # MediaPipe's registry skeleton parents left_elbow to left_shoulder and the
    # shoulders to the nose; only pairs present in the layout become edges.
    assert set(track.edges) == {(1, 0), (2, 0), (3, 1)}
    px, conf = track.at(0) or (None, None)
    assert px is not None and conf is not None
    frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    drawn = draw_pose(frame, px, conf, track.edges)
    assert drawn.shape == frame.shape and drawn.any() and not frame.any()
    assert tuple(drawn[40, 50]) == (0, 0, 255)  # low-confidence joint drawn red
    assert skeleton_edges(["a"], [{"name": "a", "parent": "zzz"}]) == ()
    with pytest.raises(Exception, match="one confidence"):
        draw_pose(frame, px, conf[:2])


def test_video_reader_seeks_to_exact_frames(tmp_path: Path) -> None:
    video = tmp_path / "v.avi"
    _write_video(video, 12)
    with VideoReader(video) as reader:
        assert reader.frame_count == 12 and reader.fps == pytest.approx(10.0)
        assert (reader.width, reader.height) == SIZE
        f5 = reader.read(5)
        f1 = reader.read(1)
        assert f5 is not None and f1 is not None
        assert int(f5[100, 55, 0]) > 150 > int(f1[100, 55, 0])  # bar width differs
        assert reader.read(12) is None
    assert clamp_index(-3, 12) == 0 and clamp_index(40, 12) == 11
    assert clamp_index(4, 0) == 0


def test_child_environment_puts_src_first_on_pythonpath() -> None:
    import os

    env = commands.child_environment({"PYTHONPATH": "x", "HOME": "h"})
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(commands.repo_root() / "src") and parts[1] == "x"
    assert env["HOME"] == "h"
    again = commands.child_environment(env)
    assert again["PYTHONPATH"] == env["PYTHONPATH"]  # idempotent
