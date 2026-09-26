"""Tests for ``record --repeat N [--pause S]`` soak mode (#9613)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.core.contracts import PreconditionError
from src.motion_capture.rig import __main__ as cli
from src.motion_capture.rig.bundle import (
    MANIFEST_FILE,
    RECORDINGS_FILE,
    build_index,
    check_bundle,
    write_bundle,
)
from src.motion_capture.rig.plan import CameraBinding, CaptureMode, RigPlan
from src.motion_capture.rig.probe import RecordingProbe
from src.motion_capture.rig.recorder import RecordingResult
from src.motion_capture.rig.session import CaptureOutcome, SessionManifest
from src.motion_capture.rig.soak import (
    SOAK_SUMMARY_FILE,
    SOAK_SUMMARY_SCHEMA,
    SOAK_SUMMARY_VERSION,
    TakeSummary,
    build_soak_summary,
    soak_exit_code,
)

pytestmark = pytest.mark.unit


def _plan() -> RigPlan:
    """Two-camera test plan reusing the bundle test layout."""
    return RigPlan(
        name="repeat-test",
        cameras=(
            CameraBinding(view="face_on", serial="2605160001"),
            CameraBinding(
                view="down_line", serial="2601240001", mode=CaptureMode(fps=30)
            ),
        ),
    )


def test_cli_record_repeat_dry_run(tmp_path: Path) -> None:
    """dry-run --repeat 3 --pause 0 writes take_01..take_03 bundles + soak_summary.json."""
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
            "--repeat",
            "3",
            "--pause",
            "0",
        ]
    )

    # In dry-run, NullRecorder produces 0 bytes -> unavailable (2).
    # The soak exit code is the worst take's exit code (2).
    assert code == 2

    # Check each take bundle
    for k in (1, 2, 3):
        take_dir = out / f"take_{k:02d}"
        assert take_dir.is_dir()
        assert (take_dir / RECORDINGS_FILE).is_file()
        assert (take_dir / MANIFEST_FILE).is_file()
        assert check_bundle(take_dir).ok

    # Check soak_summary.json
    summary_path = out / SOAK_SUMMARY_FILE
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["schema"] == SOAK_SUMMARY_SCHEMA
    assert summary["version"] == SOAK_SUMMARY_VERSION
    assert summary["repeat"] == 3
    assert summary["ok"] is False  # dry-run outcome is unavailable
    assert len(summary["takes"]) == 3

    for k, take in enumerate(summary["takes"], start=1):
        assert take["take"] == k
        assert take["dir"] == f"take_{k:02d}"
        assert take["outcome"] == CaptureOutcome.UNAVAILABLE.value
        recordings = take["recordings"]
        assert len(recordings) == 2
        for rec in recordings:
            assert set(rec.keys()) == {
                "view",
                "identity",
                "returncode",
                "frames",
                "duration_s",
                "bytes",
            }
            assert rec["returncode"] == 0
            assert rec["frames"] is None
            assert rec["bytes"] == 0


def test_cli_record_default_no_repeat_writes_bundle_directly_in_out(
    tmp_path: Path,
) -> None:
    """Default (no --repeat) writes the bundle directly in --out and no soak summary."""
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
    assert code == 2
    assert (out / RECORDINGS_FILE).is_file()
    assert (out / MANIFEST_FILE).is_file()
    assert not (out / "take_01").exists()
    assert not (out / SOAK_SUMMARY_FILE).exists()


def test_cli_record_repeat_one_matches_default(tmp_path: Path) -> None:
    """Explicit --repeat 1 must behave identically to default (no take_01, no summary)."""
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
            "--repeat",
            "1",
        ]
    )
    assert code == 2
    assert (out / RECORDINGS_FILE).is_file()
    assert (out / MANIFEST_FILE).is_file()
    assert not (out / "take_01").exists()
    assert not (out / SOAK_SUMMARY_FILE).exists()


def test_cli_record_invalid_repeat_exits(tmp_path: Path) -> None:
    """--repeat 0 or negative exits with SystemExit."""
    plan_path = tmp_path / "plan.json"
    _plan().save(plan_path)

    with pytest.raises(SystemExit) as exc_info_0:
        cli.main(
            [
                "record",
                "--plan",
                str(plan_path),
                "--repeat",
                "0",
            ]
        )
    assert "--repeat must be >= 1" in str(exc_info_0.value)

    with pytest.raises(SystemExit) as exc_info_neg:
        cli.main(
            [
                "record",
                "--plan",
                str(plan_path),
                "--repeat",
                "-2",
            ]
        )
    assert "--repeat must be >= 1" in str(exc_info_neg.value)


def test_cli_record_invalid_pause_exits(tmp_path: Path) -> None:
    """--pause -1 exits with SystemExit."""
    plan_path = tmp_path / "plan.json"
    _plan().save(plan_path)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "record",
                "--plan",
                str(plan_path),
                "--pause",
                "-1",
            ]
        )
    assert "--pause must be >= 0" in str(exc_info.value)


def test_build_soak_summary_and_exit_code_rule() -> None:
    """build_soak_summary unit test: one non-ok take -> ok is False; exit-code rule picks worst."""
    rec_ok = {
        "view": "face_on",
        "identity": "2605160001",
        "returncode": 0,
        "frames": 600,
        "duration_s": 10.0,
        "bytes": 8192,
    }
    rec_deg = {
        "view": "face_on",
        "identity": "2605160001",
        "returncode": 0,
        "frames": 472,
        "duration_s": 7.91,
        "bytes": 6000,
    }
    rec_zero = {
        "view": "face_on",
        "identity": "2605160001",
        "returncode": 0,
        "frames": 0,
        "duration_s": 0.0,
        "bytes": 512,
    }

    t_sup1 = TakeSummary(
        take=1, dir="take_01", outcome="supported", recordings=(rec_ok,)
    )
    t_sup2 = TakeSummary(
        take=2, dir="take_02", outcome="supported", recordings=(rec_ok,)
    )
    t_deg = TakeSummary(
        take=2, dir="take_02", outcome="degraded", recordings=(rec_deg,)
    )
    t_blk = TakeSummary(
        take=2, dir="take_02", outcome="blocked", recordings=(rec_zero,)
    )
    t_unav = TakeSummary(
        take=3, dir="take_03", outcome="unavailable", recordings=(rec_zero,)
    )

    # 1. All supported takes: ok is True, exit code is 0
    s_clean = build_soak_summary([t_sup1, t_sup2], repeat=2)
    assert s_clean["ok"] is True
    assert s_clean["repeat"] == 2
    assert soak_exit_code([t_sup1, t_sup2]) == 0

    # 2. One degraded take: ok is False, exit code is 1
    s_deg = build_soak_summary([t_sup1, t_deg], repeat=2)
    assert s_deg["ok"] is False
    assert soak_exit_code([t_sup1, t_deg]) == 1

    # 3. One blocked take: ok is False, exit code is 1
    s_blk = build_soak_summary([t_sup1, t_blk], repeat=2)
    assert s_blk["ok"] is False
    assert soak_exit_code([t_sup1, t_blk]) == 1

    # 4. One unavailable take (with supported and degraded): exit code picks worst (2)
    s_worst = build_soak_summary([t_sup1, t_deg, t_unav], repeat=3)
    assert s_worst["ok"] is False
    assert soak_exit_code([t_sup1, t_deg, t_unav]) == 2

    # Exit code mapping verification across pairs
    with pytest.raises(PreconditionError):
        soak_exit_code([])  # an empty soak is not a success
    assert soak_exit_code([t_sup1]) == 0
    assert soak_exit_code([t_deg]) == 1
    assert soak_exit_code([t_blk]) == 1
    assert soak_exit_code([t_unav]) == 2
    assert soak_exit_code([t_sup1, t_deg]) == 1
    assert soak_exit_code([t_deg, t_blk]) == 1
    assert soak_exit_code([t_blk, t_unav]) == 2


def test_zero_decoded_frames_marked_non_ok(tmp_path: Path) -> None:
    """Prove that existing outcome logic marks a 0-frame recording non-ok (outcome blocked)."""
    plan = _plan()
    a = tmp_path / "face_on_2605160001.mkv"
    b = tmp_path / "down_line_2601240001.mkv"
    a.write_bytes(b"x" * 8192)
    b.write_bytes(b"y" * 512)
    results = [
        RecordingResult("2605160001", a, 0, 8192),
        RecordingResult("2601240001", b, 0, 512, "I/O error"),
    ]

    def probe(path: Path) -> RecordingProbe:
        if path.name.startswith("down_line"):
            return RecordingProbe(
                frames=0, duration_s=0.0, width=None, height=None, nominal_fps=None
            )
        return RecordingProbe(
            frames=600, duration_s=10.0, width=1920, height=1200, nominal_fps=60.0
        )

    index = build_index(plan, results, 10.0, tmp_path, prober=probe)
    manifest = write_bundle(
        tmp_path, plan, index, started_utc="2026-09-06T20:00:00+00:00"
    )

    # Existing outcome logic classifies 0 decoded frames as BLOCKED
    assert manifest.outcome is CaptureOutcome.BLOCKED

    # Building a TakeSummary from this bundle reflects the blocked outcome
    take_summary = TakeSummary.from_bundle(1, tmp_path, manifest)
    assert take_summary.outcome == "blocked"

    # Soak summary must be non-ok
    soak = build_soak_summary([take_summary], repeat=1)
    assert soak["ok"] is False

    # Even if an outcome was claimed "supported", 0 frames forces ok=False
    falsely_supported = TakeSummary(
        take=1,
        dir="take_01",
        outcome="supported",
        recordings=take_summary.recordings,
    )
    soak_forced = build_soak_summary([falsely_supported], repeat=1)
    assert soak_forced["ok"] is False


def test_cmd_record_sleep_injection(tmp_path: Path) -> None:
    """Sleep injection: pause is called exactly N-1 times (not after the last)."""
    plan_path = tmp_path / "plan.json"
    _plan().save(plan_path)
    out = tmp_path / "session"

    slept: list[float] = []

    parser = cli._parser()
    args = parser.parse_args(
        [
            "record",
            "--plan",
            str(plan_path),
            "--duration",
            "1",
            "--out",
            str(out),
            "--dry-run",
            "--repeat",
            "3",
            "--pause",
            "1.5",
        ]
    )

    code = cli.cmd_record(args, sleep=slept.append)
    assert code == 2
    # Pause was 1.5, called exactly 3 - 1 = 2 times
    assert slept == [1.5, 1.5]

    # For repeat 1, pause must be called 0 times
    slept_single: list[float] = []
    args_single = parser.parse_args(
        [
            "record",
            "--plan",
            str(plan_path),
            "--duration",
            "1",
            "--out",
            str(out / "single"),
            "--dry-run",
            "--repeat",
            "1",
            "--pause",
            "1.5",
        ]
    )
    code_single = cli.cmd_record(args_single, sleep=slept_single.append)
    assert code_single == 2
    assert slept_single == []
