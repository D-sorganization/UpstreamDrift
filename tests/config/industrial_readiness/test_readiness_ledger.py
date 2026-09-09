"""CI gate for the industrial-readiness ledger (epic #9539).

These tests ARE the readiness gate. They fail when:
    1. A ``merged`` entry claims completion without a merge SHA, a test, or
       user-visible acceptance evidence
    2. An ``open`` entry carries a merge SHA, or lacks an owner, a plan, or a
       dependency when no owner is named
    3. A referenced implementation or test path does not exist
    4. An open issue is absent from every acceptance blocker list
    5. ``release_status`` reads ``ready`` while work is outstanding

Plus loader contract (DbC) unit tests mirroring the feature-parity registry
test conventions.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from src.config.industrial_readiness_loader import (
    LEDGER_PATH,
    REPO_ROOT,
    UNASSIGNED_OWNER,
    AcceptanceCriterion,
    IndustrialReadinessLedger,
    ReadinessItem,
    ReadinessLedgerError,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def ledger() -> IndustrialReadinessLedger:
    """The committed ledger, loaded once per module."""
    return IndustrialReadinessLedger.load()


@pytest.fixture(scope="module")
def raw_ledger() -> dict[str, Any]:
    """The committed ledger as raw JSON, for mutation-based contract tests."""
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """Write a ledger payload to a temp file and return its path."""
    path = tmp_path / "industrial_readiness.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The committed ledger
# ---------------------------------------------------------------------------


def test_ledger_file_exists() -> None:
    assert LEDGER_PATH.exists(), f"Missing readiness ledger {LEDGER_PATH}"


def test_ledger_loads(ledger: IndustrialReadinessLedger) -> None:
    assert ledger.epic == 9539
    assert ledger.queue
    assert ledger.acceptance


def test_every_referenced_path_exists(ledger: IndustrialReadinessLedger) -> None:
    """A ledger that cites a path the tree does not have is not evidence."""
    missing = [
        path for path in ledger.referenced_paths() if not (REPO_ROOT / path).exists()
    ]
    assert not missing, (
        f"industrial_readiness.json references paths that do not exist: {missing}"
    )


def test_merged_entries_carry_full_evidence(
    ledger: IndustrialReadinessLedger,
) -> None:
    """Merge SHA, executed tests and acceptance evidence, for every claim."""
    for item in ledger.merged_items:
        assert item.merge_shas, f"{item.key} claims merged with no SHA"
        assert item.tests, f"{item.key} claims merged with no test"
        assert item.acceptance_evidence, f"{item.key} claims merged with no evidence"


def test_open_entries_carry_owner_and_plan(
    ledger: IndustrialReadinessLedger,
) -> None:
    """Outstanding blockers name an owner or the dependency they wait on."""
    for item in ledger.open_items:
        assert item.owner, f"{item.key} is open with no owner"
        assert item.plan, f"{item.key} is open with no narrow PR plan"
        if item.owner == UNASSIGNED_OWNER:
            assert item.depends_on, f"{item.key} has no named owner and no dependency"


def test_open_work_is_not_hidden(ledger: IndustrialReadinessLedger) -> None:
    """Every open issue surfaces in an acceptance blocker list."""
    blocked = {issue for c in ledger.acceptance for issue in c.blockers}
    for item in ledger.open_items:
        assert item.issue in blocked, (
            f"{item.key} (#{item.issue}) is open but blocks no acceptance criterion"
        )


def test_release_status_is_not_falsely_green(
    ledger: IndustrialReadinessLedger,
) -> None:
    """Open work must not masquerade as a green release status."""
    if ledger.open_items or not all(c.is_met for c in ledger.acceptance):
        assert ledger.release_status == "blocked"


# ---------------------------------------------------------------------------
# Loader contract (DbC)
# ---------------------------------------------------------------------------


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        IndustrialReadinessLedger.load(tmp_path / "absent.json")


def test_merged_entry_without_sha_rejected() -> None:
    with pytest.raises(ReadinessLedgerError, match="must record its merge SHA"):
        ReadinessItem.from_dict(
            {
                "key": "U1",
                "issue": 1,
                "priority": "P1",
                "title": "t",
                "status": "merged",
                "merge_shas": [],
                "tests": ["tests/x.py"],
                "acceptance_evidence": "e",
            }
        )


def test_merged_entry_with_short_sha_rejected() -> None:
    with pytest.raises(ReadinessLedgerError, match="40 lowercase hex chars"):
        ReadinessItem.from_dict(
            {
                "key": "U1",
                "issue": 1,
                "priority": "P1",
                "title": "t",
                "status": "merged",
                "merge_shas": ["ee7792c58"],
                "tests": ["tests/x.py"],
                "acceptance_evidence": "e",
            }
        )


def test_merged_entry_without_tests_rejected() -> None:
    with pytest.raises(ReadinessLedgerError, match="name the tests"):
        ReadinessItem.from_dict(
            {
                "key": "U1",
                "issue": 1,
                "priority": "P1",
                "title": "t",
                "status": "merged",
                "merge_shas": ["0" * 40],
                "tests": [],
                "acceptance_evidence": "e",
            }
        )


def test_open_entry_with_sha_rejected() -> None:
    with pytest.raises(ReadinessLedgerError, match="must not record a merge SHA"):
        ReadinessItem.from_dict(
            {
                "key": "U3",
                "issue": 3,
                "priority": "P1",
                "title": "t",
                "status": "open",
                "merge_shas": ["0" * 40],
                "owner": "someone",
                "plan": "p",
            }
        )


def test_unassigned_open_entry_without_dependency_rejected() -> None:
    with pytest.raises(ReadinessLedgerError, match="dependency it is waiting on"):
        ReadinessItem.from_dict(
            {
                "key": "U3",
                "issue": 3,
                "priority": "P1",
                "title": "t",
                "status": "open",
                "owner": UNASSIGNED_OWNER,
                "depends_on": [],
                "plan": "p",
            }
        )


def test_unknown_queue_status_rejected() -> None:
    with pytest.raises(ReadinessLedgerError, match="status must be one of"):
        ReadinessItem.from_dict(
            {
                "key": "U1",
                "issue": 1,
                "priority": "P1",
                "title": "t",
                "status": "in_progress",
            }
        )


def test_malformed_key_rejected() -> None:
    with pytest.raises(ReadinessLedgerError, match="key must match"):
        ReadinessItem.from_dict(
            {
                "key": "first",
                "issue": 1,
                "priority": "P1",
                "title": "t",
                "status": "open",
            }
        )


def test_boolean_issue_number_rejected() -> None:
    """``bool`` subclasses ``int``; ``True`` must not pass as issue 1."""
    with pytest.raises(ReadinessLedgerError, match="positive int"):
        ReadinessItem.from_dict(
            {
                "key": "U1",
                "issue": True,
                "priority": "P1",
                "title": "t",
                "status": "open",
            }
        )


def test_unmet_criterion_without_blockers_rejected() -> None:
    with pytest.raises(ReadinessLedgerError, match="must name its blocking issues"):
        AcceptanceCriterion.from_dict(
            {"id": "x", "criterion": "c", "status": "unmet", "evidence": "e"}
        )


def test_partial_criterion_without_blockers_rejected() -> None:
    with pytest.raises(ReadinessLedgerError, match="must name its blocking issues"):
        AcceptanceCriterion.from_dict(
            {"id": "x", "criterion": "c", "status": "partial", "evidence": "e"}
        )


def test_ready_status_with_open_item_rejected(
    tmp_path: Path, raw_ledger: dict[str, Any]
) -> None:
    payload = copy.deepcopy(raw_ledger)
    payload["release_status"] = "ready"
    with pytest.raises(ReadinessLedgerError, match="may not be 'ready'"):
        IndustrialReadinessLedger.load(_write(tmp_path, payload))


def test_hidden_open_issue_rejected(tmp_path: Path, raw_ledger: dict[str, Any]) -> None:
    """Dropping an open issue from every blocker list must fail the gate."""
    payload = copy.deepcopy(raw_ledger)
    open_issues = {e["issue"] for e in payload["queue"] if e["status"] == "open"}
    for criterion in payload["acceptance"]:
        criterion["blockers"] = [
            issue for issue in criterion.get("blockers", []) if issue not in open_issues
        ]
        if criterion["status"] != "met" and not criterion["blockers"]:
            criterion["status"] = "met"
    with pytest.raises(ReadinessLedgerError, match="absent from every acceptance"):
        IndustrialReadinessLedger.load(_write(tmp_path, payload))


def test_duplicate_queue_key_rejected(
    tmp_path: Path, raw_ledger: dict[str, Any]
) -> None:
    payload = copy.deepcopy(raw_ledger)
    payload["queue"].append(copy.deepcopy(payload["queue"][0]))
    with pytest.raises(ReadinessLedgerError, match="duplicate queue keys"):
        IndustrialReadinessLedger.load(_write(tmp_path, payload))


def test_ledger_entries_are_immutable(ledger: IndustrialReadinessLedger) -> None:
    with pytest.raises((AttributeError, TypeError)):
        ledger.queue[0].status = "merged"  # type: ignore[misc]
