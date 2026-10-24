"""Hermetic tests for scripts/check_fleet_merge_settings.py (UD-OPS-033)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_fleet_merge_settings.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "check_fleet_merge_settings", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
check_fleet_merge_settings = importlib.util.module_from_spec(_SPEC)
sys.modules["check_fleet_merge_settings"] = check_fleet_merge_settings
_SPEC.loader.exec_module(check_fleet_merge_settings)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_SNAPSHOT = (
    REPO_ROOT
    / "docs/operations/settings-snapshots/UpstreamDrift-merge-queue-target.json"
)


def _completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)


class FakeGh:
    def __init__(self, ruleset_payload: dict) -> None:
        self.ruleset_payload = ruleset_payload
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = list(cmd)
        self.calls.append(argv)
        if any("rulesets" in part for part in argv):
            return _completed(json.dumps(self.ruleset_payload))
        raise AssertionError(f"unexpected gh invocation: {argv}")


def test_target_snapshot_batch_size_within_ud_ops_033_range() -> None:
    """Precondition: the committed target snapshot must sit in the 2–3 batch window."""
    assert TARGET_SNAPSHOT.is_file(), f"missing target snapshot: {TARGET_SNAPSHOT}"
    snapshot = check_fleet_merge_settings.load_snapshot(TARGET_SNAPSHOT)
    params = check_fleet_merge_settings.merge_queue_parameters(snapshot)
    assert 2 <= params["max_entries_to_build"] <= 3
    assert 2 <= params["max_entries_to_merge"] <= 3
    assert params["max_entries_to_build"] == params["max_entries_to_merge"]


def test_compare_live_reports_no_diff_when_parameters_match() -> None:
    snapshot = {
        "ruleset_id": 24451991,
        "merge_queue": {
            "max_entries_to_build": 3,
            "max_entries_to_merge": 3,
            "merge_method": "SQUASH",
        },
    }
    live = {
        "id": 24451991,
        "rules": [
            {
                "type": "merge_queue",
                "parameters": {
                    "max_entries_to_build": 3,
                    "max_entries_to_merge": 3,
                    "merge_method": "SQUASH",
                },
            }
        ],
    }
    assert check_fleet_merge_settings.diff_merge_queue(snapshot, live) == []


def test_compare_live_reports_diff_when_batch_size_drifts() -> None:
    snapshot = {
        "ruleset_id": 24451991,
        "merge_queue": {"max_entries_to_build": 3, "max_entries_to_merge": 3},
    }
    live = {
        "id": 24451991,
        "rules": [
            {
                "type": "merge_queue",
                "parameters": {
                    "max_entries_to_build": 5,
                    "max_entries_to_merge": 5,
                },
            }
        ],
    }
    diffs = check_fleet_merge_settings.diff_merge_queue(snapshot, live)
    assert any("max_entries_to_build" in line for line in diffs)


def test_check_live_uses_ruleset_id_from_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = TARGET_SNAPSHOT
    live_payload = {
        "id": 24451991,
        "rules": [
            {
                "type": "merge_queue",
                "parameters": check_fleet_merge_settings.merge_queue_parameters(
                    check_fleet_merge_settings.load_snapshot(snapshot_path)
                ),
            }
        ],
    }
    fake = FakeGh(live_payload)

    def _runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return fake(cmd)

    ok, detail = check_fleet_merge_settings.check_against_live(
        snapshot_path, repo="D-sorganization/UpstreamDrift", run=_runner
    )
    assert ok is True
    assert detail == "live settings match snapshot"
    assert any("24451991" in " ".join(call) for call in fake.calls)
