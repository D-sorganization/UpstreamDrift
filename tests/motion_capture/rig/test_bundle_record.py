"""Session bundles and the record / session-check commands, without cameras."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.motion_capture.rig import __main__ as cli
from src.motion_capture.rig.bundle import (
    BUNDLE_SCHEMA_VERSION,
    MANIFEST_FILE,
    PLAN_FILE,
    RECORDINGS_FILE,
    RecordingsIndex,
    build_index,
    check_bundle,
    recording_stats,
    write_bundle,
)
from src.motion_capture.rig.plan import CameraBinding, CaptureMode, RigPlan
from src.motion_capture.rig.probe import RecordingProbe, parse_ffmpeg_decode_log
from src.motion_capture.rig.recorder import RecordingResult
from src.motion_capture.rig.session import CaptureOutcome, SessionManifest

pytestmark = pytest.mark.unit


def _plan() -> RigPlan:
    return RigPlan(
        name="bundle-test",
        cameras=(
            CameraBinding(view="face_on", serial="2605160001"),
            CameraBinding(
                view="down_line", serial="2601240001", mode=CaptureMode(fps=30)
            ),
        ),
    )


def _full(path: Path) -> RecordingProbe:
    return RecordingProbe(
        frames=600, duration_s=10.0, width=1920, height=1200, nominal_fps=60.0
    )


def _short(path: Path) -> RecordingProbe:
    return RecordingProbe(
        frames=472, duration_s=7.91, width=1920, height=1200, nominal_fps=60.0
    )


def _results(bundle: Path, *, second_bytes: int = 4096, second_rc: int | None = 0):
    a = bundle / "face_on_2605160001.mkv"
    b = bundle / "down_line_2601240001.mkv"
    bundle.mkdir(parents=True, exist_ok=True)
    a.write_bytes(b"x" * 8192)
    b.write_bytes(b"y" * second_bytes)
    return [
        RecordingResult("2605160001", a, 0, 8192),
        RecordingResult("2601240001", b, second_rc, second_bytes),
    ]


def test_recording_stats_maps_success_and_failures() -> None:
    from src.motion_capture.rig.bundle import RecordingEntry

    mode = CaptureMode()
    ok = RecordingEntry(
        view="a",
        identity="1",
        file="a.mkv",
        bytes=10,
        returncode=0,
        requested_mode=mode,
        requested_duration_s=10.0,
    )
    empty = RecordingEntry(
        view="b",
        identity="2",
        file="b.mkv",
        bytes=0,
        returncode=0,
        requested_mode=mode,
        requested_duration_s=10.0,
    )
    failed = RecordingEntry(
        view="c",
        identity="3",
        file="c.mkv",
        bytes=5,
        returncode=1,
        requested_mode=mode,
        requested_duration_s=10.0,
    )
    assert (
        recording_stats(ok).state == "ok" and recording_stats(ok).effective_mode == mode
    )
    assert recording_stats(empty).state == "no_stream"
    assert recording_stats(empty).reason == "recording is empty"
    assert recording_stats(failed).reason == "recorder exited 1"


def test_build_index_requires_results_for_every_camera(tmp_path: Path) -> None:
    plan = _plan()
    results = _results(tmp_path)
    index = build_index(plan, results, 10.0, tmp_path, prober=_full)
    assert index.schema_version == BUNDLE_SCHEMA_VERSION
    assert [e.view for e in index.recordings] == ["face_on", "down_line"]
    assert index.recordings[0].file == "face_on_2605160001.mkv"  # relative to bundle
    assert index.recordings[1].requested_mode.fps == 30
    with pytest.raises(Exception, match="exactly the plan cameras"):
        build_index(plan, results[:1], 10.0, tmp_path, prober=_full)
    with pytest.raises(Exception, match="duration_s"):
        build_index(plan, results, 0, tmp_path, prober=_full)


def test_write_and_check_bundle_supported(tmp_path: Path) -> None:
    plan = _plan()
    index = build_index(plan, _results(tmp_path), 10.0, tmp_path, prober=_full)
    manifest = write_bundle(
        tmp_path, plan, index, started_utc="2026-09-06T20:00:00+00:00"
    )
    assert manifest.outcome is CaptureOutcome.SUPPORTED
    for name in (PLAN_FILE, RECORDINGS_FILE, MANIFEST_FILE):
        assert (tmp_path / name).is_file()
    assert RigPlan.load(tmp_path / PLAN_FILE) == plan
    reloaded = RecordingsIndex.model_validate_json(
        (tmp_path / RECORDINGS_FILE).read_text()
    )
    assert reloaded == index
    assert (
        SessionManifest.model_validate_json((tmp_path / MANIFEST_FILE).read_text())
        == manifest
    )
    assert check_bundle(tmp_path).ok


def test_empty_recording_blocks_the_session_and_is_reported(tmp_path: Path) -> None:
    plan = _plan()
    index = build_index(
        plan, _results(tmp_path, second_bytes=0), 10.0, tmp_path, prober=_full
    )
    manifest = write_bundle(
        tmp_path, plan, index, started_utc="2026-09-06T20:00:00+00:00"
    )
    assert manifest.outcome is CaptureOutcome.BLOCKED
    assert any("down_line" in r and "empty" in r for r in manifest.reasons)
    assert check_bundle(tmp_path).ok  # consistent, even though blocked


def test_check_bundle_finds_missing_files_and_inconsistent_outcome(
    tmp_path: Path,
) -> None:
    plan = _plan()
    index = build_index(plan, _results(tmp_path), 10.0, tmp_path, prober=_full)
    write_bundle(tmp_path, plan, index, started_utc="2026-09-06T20:00:00+00:00")
    (tmp_path / "down_line_2601240001.mkv").unlink()
    (tmp_path / "face_on_2605160001.mkv").write_bytes(b"z")  # size no longer matches
    problems = check_bundle(tmp_path).problems
    assert any("down_line" in p and "missing" in p for p in problems)
    assert any("face_on" in p and "size differs" in p for p in problems)

    manifest = SessionManifest.model_validate_json(
        (tmp_path / MANIFEST_FILE).read_text()
    )
    tampered = manifest.model_copy(update={"outcome": CaptureOutcome.UNAVAILABLE})
    tampered.save(tmp_path / MANIFEST_FILE)
    assert any(
        "manifest outcome unavailable != supported" in p
        for p in check_bundle(tmp_path).problems
    )


def test_check_bundle_reports_missing_and_malformed_files(tmp_path: Path) -> None:
    problems = check_bundle(tmp_path).problems
    assert set(problems) == {
        f"missing {PLAN_FILE}",
        f"missing {RECORDINGS_FILE}",
        f"missing {MANIFEST_FILE}",
    }
    (tmp_path / PLAN_FILE).write_text("{not json", encoding="utf-8")
    assert any(p.startswith(f"{PLAN_FILE}:") for p in check_bundle(tmp_path).problems)


def test_cli_record_dry_run_then_session_check(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _plan().save(plan_path)
    out = tmp_path / "session"
    code = cli.main(
        [
            "record",
            "--plan",
            str(plan_path),
            "--duration",
            "1",
            "--out",
            str(out),
            "--dry-run",
        ]
    )
    # NullRecorder writes no bytes, so every view is "empty" -> unavailable (2). The bundle itself is sound.
    assert code == 2
    assert (out / RECORDINGS_FILE).is_file() and (out / MANIFEST_FILE).is_file()
    manifest = SessionManifest.model_validate_json((out / MANIFEST_FILE).read_text())
    assert manifest.outcome is CaptureOutcome.UNAVAILABLE
    assert manifest.tools_schema["status"] == "unavailable"
    assert cli.main(["session-check", "--session", str(out)]) == 0
    (out / RECORDINGS_FILE).unlink()
    assert cli.main(["session-check", "--session", str(out)]) == 1


def test_short_recording_is_degraded_not_supported(tmp_path: Path) -> None:
    """The rig's first recordings: exit 0, bytes on disk, 20 % of the frames missing."""
    plan = _plan()
    index = build_index(plan, _results(tmp_path), 10.0, tmp_path, prober=_short)
    entry = index.recordings[0]
    assert entry.frames == 472 and entry.duration_s == pytest.approx(7.91)
    assert entry.achieved_fps == pytest.approx(472 / 7.91)
    assert (
        entry.coverage_reason() is not None and "recorded <" in entry.coverage_reason()
    )
    manifest = write_bundle(
        tmp_path, plan, index, started_utc="2026-09-06T20:00:00+00:00"
    )
    assert manifest.outcome is CaptureOutcome.DEGRADED
    assert all(c.state == "degraded" for c in manifest.cameras)
    assert check_bundle(tmp_path).ok


def test_unprobed_entries_do_not_invent_coverage() -> None:
    from src.motion_capture.rig.bundle import RecordingEntry

    entry = RecordingEntry(
        view="a",
        identity="1",
        file="a.mkv",
        bytes=10,
        returncode=0,
        requested_mode=CaptureMode(),
        requested_duration_s=10.0,
    )
    assert entry.coverage_reason() is None and entry.achieved_fps is None
    assert recording_stats(entry).state == "ok"


def test_parse_ffmpeg_decode_log() -> None:
    text = "\n".join(
        [
            "Input #0, matroska,webm, from 'x.mkv':",
            "  Stream #0:0: Video: mjpeg (Baseline), yuvj422p, 1920x1200, 60 fps, 60 tbr",
            "frame=  200 fps=0.0 q=-0.0 size=N/A time=00:00:03.33 bitrate=N/A",
            "frame=  472 fps=135 q=-0.0 Lsize=N/A time=00:00:07.91 bitrate=N/A speed=2.2x",
        ]
    )
    probe = parse_ffmpeg_decode_log(text)
    assert (probe.frames, probe.width, probe.height, probe.nominal_fps) == (
        472,
        1920,
        1200,
        60.0,
    )
    assert probe.duration_s == pytest.approx(7.91)
    assert probe.achieved_fps == pytest.approx(472 / 7.91)
    empty = parse_ffmpeg_decode_log("")
    assert empty.frames == 0 and empty.achieved_fps is None


def test_record_all_warms_up_then_signals_every_recorder_before_reaping(
    tmp_path: Path,
) -> None:
    from src.motion_capture.rig.recorder import NullRecorder, record_all

    plan = _plan()
    made: list[NullRecorder] = []

    def factory() -> NullRecorder:
        rec = NullRecorder()
        made.append(rec)
        return rec

    slept: list[float] = []
    results = record_all(
        plan,
        {"face_on": "r1", "down_line": "r2"},
        10.0,
        tmp_path,
        factory,
        warmup_s=2.0,
        sleep=slept.append,
    )
    assert slept == [12.0]
    assert all(rec.signalled for rec in made)
    assert [r.identity for r in results] == ["2605160001", "2601240001"]
