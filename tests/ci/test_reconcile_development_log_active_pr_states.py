"""Regression tests for development-log PR-state reconciliation (fleet policy gate)."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from shared_scripts.development_log_schema import (
    MAX_ACTIVE_ENTRIES,
    MAX_BYTES,
    parse_entries,
)
from shared_scripts.development_log_validator import validate_devlog_content

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/ci/reconcile_development_log_active_pr_states.py"
# UpstreamDrift main before devlog fleet-policy repair (2026-10-22).
PRE_REPAIR_MAIN_SHA = "19ce0cc32d861c1bb27ce1abe5604e7c22d63007"
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "ci"
    / "devlog_fleet_policy_wip_ceiling_fixture.md"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "reconcile_development_log_active_pr_states_under_test", SCRIPT
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reconcile_module():
    return _load_module()


def _oversized_active_fixture() -> str:
    fixture = FIXTURE.read_text(encoding="utf-8")
    header = fixture.split("### DL-#90001")[0]
    first_entry = fixture.split("### DL-#90001", 1)[1].split("### DL-#90002", 1)[0]
    template = "### DL-#90001" + first_entry
    merged_blocks = []
    for issue in range(91000, 91000 + MAX_ACTIVE_ENTRIES):
        merged_blocks.append(
            template.replace("DL-#90001", f"DL-#{issue}")
            .replace("#90001", f"#{issue}")
            .replace("- **PR:** #101", f"- **PR:** #{issue}")
        )
    open_entry = fixture.split("### DL-#90003", 1)[1]
    return header + "".join(merged_blocks) + "### DL-#90003" + open_entry


def test_fixture_trips_wip_ceiling_before_reconciliation() -> None:
    content = _oversized_active_fixture()
    findings = validate_devlog_content(content, FIXTURE, check_portfolio_wip=True)
    kinds = {finding.kind for finding in findings}
    assert "wip_ceiling" in kinds
    assert len([e for e in parse_entries(content) if e.is_active]) > MAX_ACTIVE_ENTRIES


def test_reconcile_fixture_clears_wip_ceiling(reconcile_module) -> None:
    content = _oversized_active_fixture()
    records = {
        **{
            number: reconcile_module.PrRecord(
                state="MERGED", merged_at="2026-10-01T12:00:00Z"
            )
            for number in range(91000, 91000 + MAX_ACTIVE_ENTRIES)
        },
        201: reconcile_module.PrRecord(state="OPEN", merged_at=None),
    }
    updated, counts = reconcile_module.reconcile_active_pr_states(
        content, records=records
    )
    assert counts["merged"] == MAX_ACTIVE_ENTRIES
    assert counts["skipped"] == 1
    active = [entry for entry in parse_entries(updated) if entry.is_active]
    assert len(active) <= MAX_ACTIVE_ENTRIES
    findings = validate_devlog_content(updated, FIXTURE, check_portfolio_wip=True)
    assert not any(finding.kind == "wip_ceiling" for finding in findings)


def test_classify_entry_ignores_issue_numbers_in_pr_field(reconcile_module) -> None:
    records = {
        10227: reconcile_module.PrRecord(
            state="MERGED", merged_at="2026-09-16T03:14:50Z"
        ),
        10199: None,
    }
    assert reconcile_module.classify_entry([10227, 10199], records) == "merged"


def test_gate_metrics_match_fleet_policy_failure_snapshot() -> None:
    """Mirror the 2026-10-22 fleet-policy-gate JSON: 202 active / 31 WIP / 590917 bytes."""
    proc = subprocess.run(
        [
            "git",
            "show",
            f"{PRE_REPAIR_MAIN_SHA}:docs/development/DEVELOPMENT_LOG.md",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"pre-repair devlog unavailable: {proc.stderr.strip()}")
    content = proc.stdout
    entries = parse_entries(content)
    active = [entry for entry in entries if entry.is_active]
    in_progress = [entry for entry in entries if entry.state == "in_progress"]
    metrics = {
        "active": len(active),
        "in_progress": len(in_progress),
        "bytes": len(content.encode("utf-8")),
    }
    assert metrics["active"] == 202
    assert metrics["in_progress"] == 31
    assert metrics["bytes"] > MAX_BYTES


def test_repaired_devlog_passes_fleet_policy_ceilings() -> None:
    devlog = REPO_ROOT / "docs/development/DEVELOPMENT_LOG.md"
    content = devlog.read_text(encoding="utf-8")
    findings = validate_devlog_content(content, devlog, check_portfolio_wip=True)
    assert not findings
