"""Contract tests for isolated Moco ladder orchestration (no native solve)."""

from __future__ import annotations

import sys
import os
import time
import hashlib
import json
from pathlib import Path

import pytest

from tests.opensim.native_moco_ladder_harness import (
    DEFAULT_RUNGS,
    LadderRung,
    _read_blocked_failure,
    run_ladder,
)

pytestmark = pytest.mark.unit


def test_default_ladder_adds_one_feature_per_rung() -> None:
    assert tuple(rung.name for rung in DEFAULT_RUNGS) == (
        "plain",
        "nonlinear_joint",
        "moving_path",
        "conditional_path",
        "wrap",
        "coupler",
    )
    assert all(
        len(current.features) - len(previous.features) == 1
        and set(previous.features) < set(current.features)
        for previous, current in zip(DEFAULT_RUNGS, DEFAULT_RUNGS[1:], strict=False)
    )


def test_ladder_stops_after_first_worker_failure(tmp_path: Path) -> None:
    invocations: list[str] = []

    def command(rung: LadderRung, _: Path) -> tuple[str, ...]:
        invocations.append(rung.name)
        return (sys.executable, "-c", "raise SystemExit(7)")

    result = run_ladder(DEFAULT_RUNGS[:2], command, tmp_path / "run")
    assert invocations == ["plain"]
    assert len(result) == 1
    assert result[0].status == "worker_failed"
    assert result[0].receipt_sha256 is None


def test_ladder_bounds_worker_time_and_stops(tmp_path: Path) -> None:
    invocations: list[str] = []

    def command(rung: LadderRung, _: Path) -> tuple[str, ...]:
        invocations.append(rung.name)
        return (sys.executable, "-c", "import time; time.sleep(10)")

    result = run_ladder(
        DEFAULT_RUNGS[:2], command, tmp_path / "run", timeout_seconds=0.05
    )
    assert invocations == ["plain"]
    assert len(result) == 1
    assert result[0].status == "timeout"
    assert result[0].elapsed_seconds < 3


def test_ladder_rejects_success_exit_without_native_evidence(tmp_path: Path) -> None:
    def command(_: LadderRung, __: Path) -> tuple[str, ...]:
        return (sys.executable, "-c", "pass")

    result = run_ladder(DEFAULT_RUNGS[:1], command, tmp_path / "run")
    assert len(result) == 1
    assert result[0].status == "invalid_evidence"
    assert result[0].receipt_sha256 is None


def test_blocked_failure_requires_exact_preserved_source_binding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.osim"
    source.write_bytes(b"synthetic source")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    failure_path = tmp_path / "failure.json"
    payload = {
        "status": "blocked",
        "source_sha256": source_sha,
        "code": "conditional_path_point_replay_policy_unavailable",
        "detail": "fails closed",
    }
    failure_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _read_blocked_failure(tmp_path) == {
        "code": payload["code"],
        "detail": payload["detail"],
    }
    payload["source_sha256"] = "0" * 64
    failure_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _read_blocked_failure(tmp_path) is None


def test_ladder_timeout_is_hard_capped_at_thirty_seconds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="30"):
        run_ladder(
            DEFAULT_RUNGS[:1],
            lambda *_: (sys.executable,),
            tmp_path,
            timeout_seconds=31,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object containment")
def test_windows_timeout_kills_launcher_descendant_tree(tmp_path: Path) -> None:
    marker = tmp_path / "orphan-was-not-killed.txt"
    child = (
        "import time; from pathlib import Path; time.sleep(0.6); "
        f"Path({str(marker)!r}).write_text('orphan')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(5)"
    )
    result = run_ladder(
        DEFAULT_RUNGS[:1],
        lambda *_: (sys.executable, "-c", parent),
        tmp_path / "run",
        timeout_seconds=0.2,
    )
    assert result[0].status == "timeout"
    assert result[0].peak_tree_rss_bytes is not None
    time.sleep(0.8)
    assert not marker.exists()
