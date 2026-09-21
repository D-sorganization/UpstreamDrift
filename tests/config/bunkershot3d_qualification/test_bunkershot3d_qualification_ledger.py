"""CI gate for the BunkerShot3D product acceptance matrix (epic #9541).

Epic #9541 closes only when "the product acceptance matrix and independent
review pass". ``src/config/bunkershot3d_qualification.json`` is that matrix
in machine-readable form. It reuses the #9539 readiness-ledger contract
(``src.config.industrial_readiness_loader``) unchanged, so a merged entry
needs a merge SHA, a test and acceptance evidence, an open entry needs an
owner and a plan, every cited path must exist and open work cannot be
dropped from the acceptance blockers.

These tests add the epic-specific rules:
    1. Every child on the epic's delivery checklist is tracked
    2. Every dependency resolves to a tracked child or the epic itself
    3. The exact vendored Tools pin is recorded alongside the audit SHA
    4. ``release_status`` cannot read ``ready`` while the live V&V register
       is empty or any assessed credibility factor sits below its threshold,
       so the matrix cannot be greened by editing JSON
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from src.config.industrial_readiness_loader import (
    CONFIG_DIR,
    REPO_ROOT,
    IndustrialReadinessLedger,
    ReadinessLedgerError,
)

pytestmark = pytest.mark.unit

LEDGER_PATH = CONFIG_DIR / "bunkershot3d_qualification.json"

EPIC = 9541

#: The delivery checklist from the epic body: the four B1-B4 children, the
#: reused issues and the eight bounded children from the 2026-09-07 review.
EPIC_CHILDREN = frozenset(
    {
        9542,
        9543,
        9544,
        9545,
        9239,
        9286,
        8733,
        8880,
        9688,
        9689,
        9690,
        9691,
        9692,
        9693,
        9694,
        9695,
    }
)

_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture(scope="module")
def ledger() -> IndustrialReadinessLedger:
    """The committed matrix, loaded once per module."""
    return IndustrialReadinessLedger.load(LEDGER_PATH)


@pytest.fixture(scope="module")
def raw_ledger() -> dict[str, Any]:
    """The committed matrix as raw JSON, for mutation-based contract tests."""
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """Write a matrix payload to a temp file and return its path."""
    path = tmp_path / "bunkershot3d_qualification.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The committed matrix
# ---------------------------------------------------------------------------


def test_matrix_file_exists() -> None:
    assert LEDGER_PATH.exists(), f"Missing qualification matrix {LEDGER_PATH}"


def test_matrix_is_for_the_epic(ledger: IndustrialReadinessLedger) -> None:
    assert ledger.epic == EPIC
    assert ledger.queue
    assert ledger.acceptance


def test_every_referenced_path_exists(ledger: IndustrialReadinessLedger) -> None:
    """A matrix that cites a path the tree does not have is not evidence."""
    missing = [
        path for path in ledger.referenced_paths() if not (REPO_ROOT / path).exists()
    ]
    assert not missing, (
        f"bunkershot3d_qualification.json references missing paths: {missing}"
    )


def test_every_epic_child_is_tracked(ledger: IndustrialReadinessLedger) -> None:
    """Dropping a child from the queue must fail, not quietly shrink the epic."""
    tracked = {item.issue for item in ledger.queue}
    assert tracked == EPIC_CHILDREN, (
        f"untracked children: {sorted(EPIC_CHILDREN - tracked)}; "
        f"unexpected entries: {sorted(tracked - EPIC_CHILDREN)}"
    )


def test_dependencies_resolve_to_tracked_work(
    ledger: IndustrialReadinessLedger,
) -> None:
    """Every dependency names a tracked child or the epic; never itself."""
    tracked = {item.issue for item in ledger.queue} | {EPIC}
    for item in ledger.queue:
        assert item.issue not in item.depends_on, f"{item.key} depends on itself"
        dangling = [dep for dep in item.depends_on if dep not in tracked]
        assert not dangling, f"{item.key} depends on untracked issues {dangling}"


def test_tools_pin_is_recorded(raw_ledger: dict[str, Any]) -> None:
    """The execution contract requires both exact revisions, not a floating main."""
    assert _SHA_PATTERN.match(raw_ledger.get("tools_pin", "")), (
        "bunkershot3d_qualification.json must record the exact vendored Tools "
        "commit as a 40-character 'tools_pin'"
    )


def test_open_work_is_not_hidden(ledger: IndustrialReadinessLedger) -> None:
    """Every open child surfaces in an acceptance blocker list."""
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


def test_release_status_tracks_live_credibility(
    ledger: IndustrialReadinessLedger,
) -> None:
    """The matrix cannot outrun the V&V register it claims to summarise.

    ``credibility_assessment()`` is derived from the shipped measurement
    register, which holds zero measurements. While that is so, or while any
    assessed NASA-STD-7009B factor is below its threshold, the product
    cannot be ``ready`` no matter what the JSON says.
    """
    from bunkershot3d.vandv.credibility import credibility_assessment
    from bunkershot3d.vandv.measurement_intake import shipped_register

    register_is_empty = len(shipped_register().records) == 0
    short = [
        assessment.factor.value
        for assessment in credibility_assessment()
        if assessment.is_assessed and not assessment.meets_threshold
    ]
    if register_is_empty or short:
        assert ledger.release_status == "blocked", (
            f"release_status is {ledger.release_status!r} while the shipped "
            f"register is {'empty' if register_is_empty else 'populated'} and "
            f"factors below threshold are {short}"
        )


# ---------------------------------------------------------------------------
# Contract (DbC): the shared loader refuses a greened-by-edit matrix
# ---------------------------------------------------------------------------


def test_ready_status_with_open_child_rejected(
    tmp_path: Path, raw_ledger: dict[str, Any]
) -> None:
    payload = copy.deepcopy(raw_ledger)
    payload["release_status"] = "ready"
    with pytest.raises(ReadinessLedgerError, match="may not be 'ready'"):
        IndustrialReadinessLedger.load(_write(tmp_path, payload))


def test_hidden_open_child_rejected(tmp_path: Path, raw_ledger: dict[str, Any]) -> None:
    """Dropping an open child from every blocker list must fail the gate."""
    payload = copy.deepcopy(raw_ledger)
    hidden = payload["queue"][0]["issue"]
    for criterion in payload["acceptance"]:
        criterion["blockers"] = [
            issue for issue in criterion.get("blockers", []) if issue != hidden
        ]
        if criterion["status"] != "met" and not criterion["blockers"]:
            criterion["status"] = "met"
    with pytest.raises(ReadinessLedgerError, match="absent from every acceptance"):
        IndustrialReadinessLedger.load(_write(tmp_path, payload))


def test_completion_claim_without_evidence_rejected(
    tmp_path: Path, raw_ledger: dict[str, Any]
) -> None:
    """Flipping a child to merged without a SHA is an issue closure, not proof."""
    payload = copy.deepcopy(raw_ledger)
    payload["queue"][0]["status"] = "merged"
    with pytest.raises(ReadinessLedgerError, match="must record its merge SHA"):
        IndustrialReadinessLedger.load(_write(tmp_path, payload))
