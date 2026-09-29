"""Composite (multipicture) video export on a synthetic two-view session (#9815)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.motion_capture.rig import __main__ as rig_cli
from src.motion_capture.rig.bundle import build_index, write_bundle
from src.motion_capture.rig.plan import CameraBinding, RigPlan
from src.motion_capture.rig.probe import RecordingProbe
from src.motion_capture.rig.recorder import RecordingResult
from src.tools.capture_rig import commands, workflow
from src.tools.capture_rig.commands import MultipictureArgs
from src.tools.capture_rig.layout_model import (
    PRESET_NAMES,
    LayoutSpec,
    SourceRef,
    Tile,
    preset,
)
from src.tools.capture_rig.layout_presets import SESSION, USER, LayoutStore
from src.tools.capture_rig.mosaic import (
    MOSAIC_SCHEMA,
    MosaicOptions,
    MosaicResult,
    default_sources,
    export_from_session,
    export_multipicture,
    frame_offsets,
    parse_size,
    resolve_layout,
)
from src.tools.capture_rig.overlay import POINT_COLOUR
from src.tools.capture_rig.overlay_render import ClipRange
from src.tools.capture_rig.session import load_session
from tests.tools.capture_rig.test_core import SIZE, _write_video

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.unit

FRAMES = 12
FPS = 10.0


def _two_view_bundle(root: Path, *, frames_b: int = FRAMES) -> Path:
    """A bundle whose two views both have a recording (``cam_b`` may be shorter)."""
    plan = RigPlan(
        name="rig",
        cameras=(
            CameraBinding(view="cam_a", serial="1"),
            CameraBinding(view="cam_b", serial="2"),
        ),
    )
    video_a, video_b = root / "cam_a_1.avi", root / "cam_b_2.avi"
    _write_video(video_a, FRAMES)
    _write_video(video_b, frames_b)
    results = [
        RecordingResult("1", video_a, 0, video_a.stat().st_size),
        RecordingResult("2", video_b, 0, video_b.stat().st_size),
    ]
    counts = {str(video_a): FRAMES, str(video_b): frames_b}
    index = build_index(
        plan,
        results,
        1.2,
        root,
        prober=lambda p: RecordingProbe(counts[str(p)], 1.2, SIZE[0], SIZE[1], FPS),
    )
    write_bundle(root, plan, index, started_utc="2026-09-07T00:00:00+00:00")
    return root


def _pose_at(px: tuple[int, int]) -> list[list[int]]:
    return [
        list(px),
        [px[0] + 40, px[1]],
        [px[0] + 40, px[1] + 40],
        [px[0], px[1] + 40],
    ]


def _observations(root: Path, view: str, px: tuple[int, int]) -> None:
    """An observation set with a confident 4-joint square pose on every frame."""
    names = ["nose", "left_shoulder", "right_shoulder", "left_hip"]
    rows = [
        {
            "camera_id": "1",
            "time_s": i / FPS,
            "keypoints_px": _pose_at(px),
            "confidence": [0.9, 0.9, 0.9, 0.9],
        }
        for i in range(FRAMES)
    ]
    payload = {
        "view": view,
        "identity": "1",
        "camera_id": "1",
        "fps": FPS,
        "width": SIZE[0],
        "height": SIZE[1],
        "frames_total": FRAMES,
        "frames_with_pose": FRAMES,
        "detector_layout": {"name": "test", "keypoint_names": names},
        "frames": rows,
        "provenance": {"estimator": "mediapipe"},
    }
    obs_dir = root / "observations"
    obs_dir.mkdir(exist_ok=True)
    (obs_dir / f"{view}.json").write_text(json.dumps(payload), encoding="utf-8")
    (obs_dir / "observations.json").write_text(
        json.dumps(
            {
                "plan_name": "rig",
                "views": [
                    {
                        "view": view,
                        "identity": "1",
                        "status": "available",
                        "file": f"{view}.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _video_props(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    props = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        float(cap.get(cv2.CAP_PROP_FPS)),
    )
    cap.release()
    return props


def _first_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame


# ------------------------------------------------------------------ export


def test_side_by_side_export_writes_video_and_provenance_sidecar(
    tmp_path: Path,
) -> None:
    root = _two_view_bundle(tmp_path)
    media = load_session(root)
    spec = preset("side_by_side", default_sources(media))
    assert spec.source_keys() == ("recorded:cam_a", "recorded:cam_b")
    out = tmp_path / "mosaic.mp4"
    result = export_multipicture(
        root, spec, out, MosaicOptions(size=(2 * SIZE[0], SIZE[1]))
    )
    assert isinstance(result, MosaicResult)
    assert result.frames == FRAMES and result.first == 0 and result.last == 11
    assert result.fps == pytest.approx(FPS) and result.size == (2 * SIZE[0], SIZE[1])
    assert _video_props(out)[:3] == (FRAMES, 2 * SIZE[0], SIZE[1])
    frame = _first_frame(out)
    # cam_a fills the left cell 1:1, cam_b the right one: the frame-0 bar is
    # 10 px wide in each, so both cells show footage rather than placeholder.
    assert frame[100, 5].mean() > 150 and frame[100, SIZE[0] + 5].mean() > 150
    assert 30 < frame[100, 100].mean() < 60
    sidecar = json.loads(result.sidecar.read_text(encoding="utf-8"))
    assert result.sidecar == out.with_suffix(".json")
    assert sidecar["schema_version"] == MOSAIC_SCHEMA
    assert sidecar["layout"]["name"] == "side_by_side"
    assert sidecar["layout"]["tiles"][1]["source"] == {
        "kind": "recorded",
        "view": "cam_b",
    }
    assert [s["key"] for s in sidecar["sources"]] == [
        "recorded:cam_a",
        "recorded:cam_b",
    ]
    assert sidecar["sources"][0]["file"] == "cam_a_1.avi"
    assert sidecar["first"] == 0 and sidecar["last"] == 11 and sidecar["speed"] == 1.0
    provenance = sidecar["provenance"]
    assert provenance["generated_by"]["module"] == "src.tools.capture_rig.mosaic"
    assert "version" in provenance["generated_by"]
    assert "git_sha" in provenance["generated_by"]
    assert {i["path"] for i in provenance["inputs"]} == {"cam_a_1.avi", "cam_b_2.avi"}
    assert provenance["parameters"]["layout"] == "side_by_side"


def test_overlay_tile_draws_the_observed_pose_in_its_cell(tmp_path: Path) -> None:
    root = _two_view_bundle(tmp_path)
    _observations(root, "cam_b", (100, 60))
    spec = preset(
        "three_across",
        [
            SourceRef("recorded", "cam_a"),
            SourceRef("overlay", "cam_b"),
            SourceRef("recorded", "cam_b"),
        ],
    )
    spec = LayoutSpec(
        name=spec.name,
        rows=1,
        cols=3,
        tiles=tuple(
            Tile(source=t.source, cell=t.cell, show_label=False) for t in spec.tiles
        ),
    )
    out = tmp_path / "overlay.mp4"
    result = export_multipicture(
        root, spec, out, MosaicOptions(size=(3 * SIZE[0], SIZE[1]))
    )
    assert result.frames == FRAMES
    frame = _first_frame(out)
    b, g, r = (int(c) for c in frame[60, SIZE[0] + 100])
    # POINT_COLOUR (yellow, BGR) survives the codec as low blue, high green/red.
    assert b < 120 < g and r > 120, (b, g, r, POINT_COLOUR)
    # The plain recording of the same view on the right has no pose on it.
    assert frame[60, 2 * SIZE[0] + 100].max() < 90
    assert result.sources[1] == "overlay:cam_b"


def test_speed_keeps_every_frame_and_scales_the_fps(tmp_path: Path) -> None:
    """Slow motion is a playback-rate change (as ``clip``/``overlay``): 0.5 halves fps."""
    root = _two_view_bundle(tmp_path)
    media = load_session(root)
    spec = preset("side_by_side", default_sources(media))
    out = tmp_path / "slow.mp4"
    result = export_multipicture(
        root, spec, out, MosaicOptions(clip=ClipRange(2, 7), speed=0.5, size=(320, 100))
    )
    assert result.frames == 6 and (result.first, result.last) == (2, 7)
    assert result.fps == pytest.approx(FPS * 0.5)
    count, _, _, fps = _video_props(out)
    assert count == 6 and fps == pytest.approx(5.0)
    sidecar = json.loads(result.sidecar.read_text(encoding="utf-8"))
    assert sidecar["speed"] == 0.5 and sidecar["fps"] == pytest.approx(5.0)


def test_short_source_holds_its_last_frame_and_range_is_checked(
    tmp_path: Path,
) -> None:
    root = _two_view_bundle(tmp_path, frames_b=5)
    media = load_session(root)
    spec = preset("side_by_side", default_sources(media))
    out = tmp_path / "hold.mp4"
    result = export_multipicture(
        root, spec, out, MosaicOptions(size=(2 * SIZE[0], SIZE[1]))
    )
    assert result.frames == FRAMES, "the longest source sets the length"
    with pytest.raises(Exception, match="start"):
        export_multipicture(root, spec, out, MosaicOptions(clip=ClipRange(40, 50)))
    with pytest.raises(Exception, match="canvas"):
        export_multipicture(root, spec, out, MosaicOptions(size=(0, 10)))
    with pytest.raises(Exception, match="source"):
        export_multipicture(root, preset("single"), out)


def test_frame_offsets_follow_the_manifest_timing_block(tmp_path: Path) -> None:
    root = _two_view_bundle(tmp_path)
    assert frame_offsets(root, {"cam_a": FPS, "cam_b": FPS}) == {}
    manifest = root / "session_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["timing"] = {
        "method": "flash_event",
        "clock_domain": "host_monotonic",
        "reference_view": "cam_a",
        "status": "available",
        "views": [
            {"view": "cam_a", "status": "available", "offset_ns": 0, "frames": 12},
            {
                "view": "cam_b",
                "status": "available",
                "offset_ns": 250_000_000,
                "frames": 12,
            },
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    # cam_b stamps 0.25 s later than the reference, so at 10 fps its frame
    # for output index k is k + 2 (rounded).
    assert frame_offsets(root, {"cam_a": FPS, "cam_b": FPS}) == {"cam_b": 2}
    media = load_session(root)
    spec = preset("side_by_side", default_sources(media))
    out = tmp_path / "aligned.mp4"
    result = export_multipicture(
        root, spec, out, MosaicOptions(size=(2 * SIZE[0], SIZE[1]))
    )
    assert result.offsets == {"cam_b": 2}
    frame = _first_frame(out)
    # frame 0 of cam_a has a 10 px bar; cam_b shows its frame 2 (30 px bar).
    assert frame[100, 25].mean() < 60 and frame[100, SIZE[0] + 25].mean() > 150


# ------------------------------------------------------------- resolution


def test_resolve_layout_from_builtin_saved_and_path(tmp_path: Path) -> None:
    root = _two_view_bundle(tmp_path)
    sources = default_sources(load_session(root))
    builtin = resolve_layout("two_by_two", root, sources)
    assert builtin.name == "two_by_two" and builtin.source_keys()[0] == "recorded:cam_a"
    store = LayoutStore(user_root=tmp_path / "user", session=root)
    store.save("mine", preset("single", sources[1:]), SESSION)
    store.save("mine", preset("three_across", sources), USER)
    store.save("theirs", preset("two_by_two", sources), USER)
    session_first = resolve_layout("mine", root, sources, user_root=tmp_path / "user")
    assert session_first.rows == 1 and session_first.cols == 1
    from_user = resolve_layout("theirs", root, sources, user_root=tmp_path / "user")
    assert from_user.rows == 2
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps(preset("side_by_side", sources).renamed("custom").to_dict()),
        encoding="utf-8",
    )
    assert resolve_layout(str(path), root, sources).name == "custom"
    with pytest.raises(ValueError, match="layout"):
        resolve_layout("no_such_layout", root, sources, user_root=tmp_path / "user")
    assert all(n in PRESET_NAMES for n in ("single", "side_by_side"))


def test_parse_size_and_default_sources_skip_unplayable_views(tmp_path: Path) -> None:
    assert parse_size("1280x720") == (1280, 720)
    assert parse_size(" 640X360 ") == (640, 360)
    for bad in ("1280", "0x10", "ax b", "10x-5"):
        with pytest.raises(ValueError, match="WxH"):
            parse_size(bad)
    root = _two_view_bundle(tmp_path)
    media = load_session(root)
    assert [s.view for s in default_sources(media)] == ["cam_a", "cam_b"]


# --------------------------------------------------------------- CLI + tile


def test_cli_and_command_builder(tmp_path: Path) -> None:
    root = _two_view_bundle(tmp_path)
    out = tmp_path / "cli.mp4"
    argv = commands.multipicture_command(
        root,
        "side_by_side",
        out,
        MultipictureArgs(
            start=1,
            stop=6,
            variants=("", "pair"),
            observation_set="observations",
            speed=0.5,
            size=(640, 200),
        ),
    )
    assert argv[:3] == commands.python_module_command([])[:3]
    assert argv[3:] == [
        "multipicture",
        "--session",
        str(root),
        "--layout",
        "side_by_side",
        "--out",
        str(out),
        "--variants",
        "",
        "pair",
        "--set",
        "observations",
        "--from",
        "1",
        "--to",
        "6",
        "--speed",
        "0.5",
        "--size",
        "640x200",
    ]
    minimal = commands.multipicture_command(root, "single", out)
    assert "--variants" not in minimal and "--size" not in minimal
    with pytest.raises(Exception, match="layout"):
        commands.multipicture_command(root, " ", out)
    with pytest.raises(Exception, match="speed"):
        commands.multipicture_command(root, "single", out, MultipictureArgs(speed=0))
    argv = commands.multipicture_command(
        root,
        "side_by_side",
        out,
        MultipictureArgs(start=1, stop=6, speed=0.5, size=(640, 200)),
    )
    assert rig_cli.main(argv[3:]) == 0
    assert _video_props(out) == (6, 640, 200, pytest.approx(5.0))
    assert out.with_suffix(".json").is_file()


def test_export_from_session_resolves_the_layout_text(tmp_path: Path) -> None:
    root = _two_view_bundle(tmp_path)
    out = tmp_path / "from_session.avi"
    result = export_from_session(
        root, "side_by_side", out, MosaicOptions(size=(320, 100))
    )
    assert result.layout == "side_by_side" and result.frames == FRAMES


def test_multipicture_is_a_workflow_action_with_help() -> None:
    assert "multipicture" in workflow.EXPORT.actions
    assert "multipicture" in workflow.ANALYZE_2D.actions
    for step in workflow.STEPS:
        for action in step.actions:
            assert action in workflow.ACTION_HELP, action
    assert "multiview" in workflow.ACTION_HELP["multipicture"].lower()
