"""Industrial-readiness ledger loader — the execution index for epic #9539.

Epic #9539 is a repository-level execution index for the 2026-09-04 industrial
readiness review. Its own acceptance is reconciliation, not implementation: the
priority children hold the code changes, and the epic must record, truthfully
and against current ``main``, which of them landed, what proves it, and which
remain open with an owner or a dependency.

This module loads ``industrial_readiness.json``, the machine-readable form of
that record, and enforces the contract that keeps it honest. The rules exist so
the ledger cannot drift into a green summary that the tree does not support.

Design by Contract:
    Preconditions:
        - Ledger file must exist and be valid JSON conforming to the schema
        - Every ``merged`` entry must carry at least one 40-character merge
          SHA, at least one test path, and acceptance evidence
        - Every ``open`` entry must carry no merge SHA, a non-empty owner and
          a non-empty plan, plus a dependency when the owner is unassigned
        - Every referenced implementation and test path must exist in the tree
        - Every unmet or partial acceptance criterion must name its blockers
    Postconditions:
        - Queue keys and acceptance ids are unique
        - ``release_status`` is ``ready`` only when every queue entry is
          merged and every acceptance criterion is met
        - Every open queue issue appears in at least one acceptance blocker
          list, so open work cannot be dropped from the summary
    Invariants:
        - Ledger is immutable after loading (frozen dataclasses)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).parent
LEDGER_PATH = CONFIG_DIR / "industrial_readiness.json"
REPO_ROOT = CONFIG_DIR.parent.parent

#: A queue entry is either landed on protected ``main`` or still outstanding.
#: There is deliberately no "in progress" value: an unmerged slice is open.
VALID_QUEUE_STATUSES = frozenset({"merged", "open"})

#: ``partial`` exists because two of the epic's criteria are genuinely half
#: satisfied; it is treated as not-met everywhere the contract asks for green.
VALID_ACCEPTANCE_STATUSES = frozenset({"met", "partial", "unmet"})

VALID_RELEASE_STATUSES = frozenset({"blocked", "ready"})

#: Owner value recorded when a GitHub issue has no assignee. Fabricating a
#: name would defeat the point of the owner field, so the contract instead
#: demands a dependency for entries carrying this value.
UNASSIGNED_OWNER = "unassigned"

_KEY_PATTERN: re.Pattern[str] = re.compile(r"^U[1-9][0-9]*$")
_SHA_PATTERN: re.Pattern[str] = re.compile(r"^[0-9a-f]{40}$")


class ReadinessLedgerError(ValueError):
    """Raised when the ledger violates its contract."""


def _require_str(value: Any, field_name: str, context: str) -> str:
    """Return ``value`` as a non-empty string or raise."""
    if not isinstance(value, str) or not value.strip():
        raise ReadinessLedgerError(
            f"{context}: '{field_name}' must be a non-empty string, got {value!r}"
        )
    return value


def _require_issue(value: Any, context: str) -> int:
    """Return ``value`` as a positive GitHub issue number or raise.

    ``bool`` is a subclass of ``int``, so reject it explicitly rather than let
    ``True`` pass as issue 1.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ReadinessLedgerError(
            f"{context}: issue number must be a positive int, got {value!r}"
        )
    return value


def _require_str_list(value: Any, field_name: str, context: str) -> tuple[str, ...]:
    """Return ``value`` as a tuple of non-empty strings or raise."""
    if not isinstance(value, list):
        raise ReadinessLedgerError(
            f"{context}: '{field_name}' must be a list, got {type(value).__name__}"
        )
    return tuple(_require_str(item, field_name, context) for item in value)


@dataclass(frozen=True)
class ReadinessItem:
    """One entry in the epic's priority implementation queue.

    Attributes:
        key: Queue label from the epic body (``U1`` … ``U4``)
        issue: GitHub issue number of the child
        priority: Review priority band (``P0``, ``P1``, ``P3``)
        title: Human-readable summary of the defect or program
        status: ``merged`` or ``open``
        merge_shas: Full 40-char merge SHAs (merged entries only)
        implementation: Repo-relative implementation paths
        tests: Repo-relative test paths that exercise the change
        acceptance_evidence: User-visible acceptance statement (merged only)
        owner: Responsible party for an open entry
        depends_on: Issue numbers this entry waits on
        plan: Ordered narrow-PR plan for an open entry
    """

    key: str
    issue: int
    priority: str
    title: str
    status: str
    merge_shas: tuple[str, ...]
    implementation: tuple[str, ...]
    tests: tuple[str, ...]
    acceptance_evidence: str | None
    owner: str | None
    depends_on: tuple[int, ...]
    plan: str | None

    @property
    def is_open(self) -> bool:
        """True when this slice has not landed on protected ``main``."""
        return self.status == "open"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReadinessItem:
        """Build an item from a ledger dict, validating the contract.

        Args:
            data: One element of the ledger's ``queue`` list

        Returns:
            Validated, frozen ``ReadinessItem``

        Raises:
            ReadinessLedgerError: If any contract rule is violated
        """
        if not isinstance(data, dict):
            raise ReadinessLedgerError(
                f"queue entry must be an object, got {type(data).__name__}"
            )
        key = _require_str(data.get("key"), "key", "queue entry")
        context = f"queue entry {key}"
        if not _KEY_PATTERN.match(key):
            raise ReadinessLedgerError(
                f"{context}: key must match U<n> (e.g. 'U1'), got {key!r}"
            )

        status = _require_str(data.get("status"), "status", context)
        if status not in VALID_QUEUE_STATUSES:
            raise ReadinessLedgerError(
                f"{context}: status must be one of "
                f"{sorted(VALID_QUEUE_STATUSES)}, got {status!r}"
            )

        item = cls(
            key=key,
            issue=_require_issue(data.get("issue"), context),
            priority=_require_str(data.get("priority"), "priority", context),
            title=_require_str(data.get("title"), "title", context),
            status=status,
            merge_shas=_require_str_list(
                data.get("merge_shas", []), "merge_shas", context
            ),
            implementation=_require_str_list(
                data.get("implementation", []), "implementation", context
            ),
            tests=_require_str_list(data.get("tests", []), "tests", context),
            acceptance_evidence=data.get("acceptance_evidence"),
            owner=data.get("owner"),
            depends_on=tuple(
                _require_issue(dep, context) for dep in data.get("depends_on", [])
            ),
            plan=data.get("plan"),
        )
        item._validate_status_contract(context)
        return item

    def _validate_status_contract(self, context: str) -> None:
        """Enforce the merged/open field requirements.

        Raises:
            ReadinessLedgerError: If the entry claims completion without
                evidence, or leaves an open slice without an owner or plan.
        """
        if self.status == "merged":
            if not self.merge_shas:
                raise ReadinessLedgerError(
                    f"{context}: a merged entry must record its merge SHA; "
                    "an issue closure is not evidence of correctness"
                )
            for sha in self.merge_shas:
                if not _SHA_PATTERN.match(sha):
                    raise ReadinessLedgerError(
                        f"{context}: merge SHA must be 40 lowercase hex chars, "
                        f"got {sha!r}"
                    )
            if not self.tests:
                raise ReadinessLedgerError(
                    f"{context}: a merged entry must name the tests that prove it"
                )
            _require_str(self.acceptance_evidence, "acceptance_evidence", context)
            return

        if self.merge_shas:
            raise ReadinessLedgerError(
                f"{context}: an open entry must not record a merge SHA"
            )
        owner = _require_str(self.owner, "owner", context)
        _require_str(self.plan, "plan", context)
        if owner == UNASSIGNED_OWNER and not self.depends_on:
            raise ReadinessLedgerError(
                f"{context}: an open entry with no named owner must name the "
                "dependency it is waiting on"
            )


@dataclass(frozen=True)
class AcceptanceCriterion:
    """One acceptance box from the epic body, with its current standing.

    Attributes:
        id: Stable slug for the criterion
        criterion: The epic's wording, quoted
        status: ``met``, ``partial`` or ``unmet``
        evidence: What supports the recorded status
        blockers: Issue numbers preventing ``met``
    """

    id: str
    criterion: str
    status: str
    evidence: str
    blockers: tuple[int, ...]

    @property
    def is_met(self) -> bool:
        """True only for a fully satisfied criterion."""
        return self.status == "met"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcceptanceCriterion:
        """Build a criterion from a ledger dict, validating the contract.

        Args:
            data: One element of the ledger's ``acceptance`` list

        Returns:
            Validated, frozen ``AcceptanceCriterion``

        Raises:
            ReadinessLedgerError: If any contract rule is violated
        """
        if not isinstance(data, dict):
            raise ReadinessLedgerError(
                f"acceptance entry must be an object, got {type(data).__name__}"
            )
        criterion_id = _require_str(data.get("id"), "id", "acceptance entry")
        context = f"acceptance entry {criterion_id}"

        status = _require_str(data.get("status"), "status", context)
        if status not in VALID_ACCEPTANCE_STATUSES:
            raise ReadinessLedgerError(
                f"{context}: status must be one of "
                f"{sorted(VALID_ACCEPTANCE_STATUSES)}, got {status!r}"
            )

        blockers = tuple(
            _require_issue(issue, context) for issue in data.get("blockers", [])
        )
        if status != "met" and not blockers:
            raise ReadinessLedgerError(
                f"{context}: a {status} criterion must name its blocking issues"
            )

        return cls(
            id=criterion_id,
            criterion=_require_str(data.get("criterion"), "criterion", context),
            status=status,
            evidence=_require_str(data.get("evidence"), "evidence", context),
            blockers=blockers,
        )


@dataclass(frozen=True)
class IndustrialReadinessLedger:
    """The full reconciliation record for epic #9539.

    Attributes:
        version: Ledger schema version
        epic: Epic issue number (9539)
        audit_snapshot: SHA the original review was written against
        reconciled_against: SHA this record was reconciled against
        reconciled_on: ISO date of the reconciliation
        release_status: ``blocked`` or ``ready``
        queue: Priority implementation queue, in epic order
        acceptance: Acceptance criteria, in epic order
    """

    version: str
    epic: int
    audit_snapshot: str
    reconciled_against: str
    reconciled_on: str
    release_status: str
    queue: tuple[ReadinessItem, ...]
    acceptance: tuple[AcceptanceCriterion, ...]

    @property
    def open_items(self) -> tuple[ReadinessItem, ...]:
        """Queue entries that have not landed."""
        return tuple(item for item in self.queue if item.is_open)

    @property
    def merged_items(self) -> tuple[ReadinessItem, ...]:
        """Queue entries with recorded merge evidence."""
        return tuple(item for item in self.queue if not item.is_open)

    def referenced_paths(self) -> tuple[str, ...]:
        """Every repo-relative path the ledger cites, deduplicated in order."""
        seen: dict[str, None] = {}
        for item in self.queue:
            for path in (*item.implementation, *item.tests):
                seen.setdefault(path, None)
        return tuple(seen)

    @classmethod
    def load(cls, path: Path | None = None) -> IndustrialReadinessLedger:
        """Load and validate the ledger.

        Args:
            path: Ledger location; defaults to :data:`LEDGER_PATH`

        Returns:
            Validated, frozen ledger

        Raises:
            FileNotFoundError: If the ledger file is absent
            ReadinessLedgerError: If any contract rule is violated
        """
        ledger_path = path if path is not None else LEDGER_PATH
        if not ledger_path.exists():
            raise FileNotFoundError(f"Readiness ledger not found: {ledger_path}")

        raw = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ReadinessLedgerError("Ledger root must be a JSON object")

        queue = tuple(ReadinessItem.from_dict(entry) for entry in raw.get("queue", []))
        acceptance = tuple(
            AcceptanceCriterion.from_dict(entry) for entry in raw.get("acceptance", [])
        )

        release_status = _require_str(
            raw.get("release_status"), "release_status", "ledger"
        )
        if release_status not in VALID_RELEASE_STATUSES:
            raise ReadinessLedgerError(
                f"ledger: release_status must be one of "
                f"{sorted(VALID_RELEASE_STATUSES)}, got {release_status!r}"
            )

        ledger = cls(
            version=_require_str(raw.get("version"), "version", "ledger"),
            epic=_require_issue(raw.get("epic"), "ledger"),
            audit_snapshot=_require_str(
                raw.get("audit_snapshot"), "audit_snapshot", "ledger"
            ),
            reconciled_against=_require_str(
                raw.get("reconciled_against"), "reconciled_against", "ledger"
            ),
            reconciled_on=_require_str(
                raw.get("reconciled_on"), "reconciled_on", "ledger"
            ),
            release_status=release_status,
            queue=queue,
            acceptance=acceptance,
        )
        ledger._validate_ledger_contract()
        logger.debug(
            "Loaded readiness ledger: %d merged, %d open, release_status=%s",
            len(ledger.merged_items),
            len(ledger.open_items),
            ledger.release_status,
        )
        return ledger

    def _validate_ledger_contract(self) -> None:
        """Enforce the cross-entry rules.

        Raises:
            ReadinessLedgerError: On duplicate keys or ids, a green status
                that the queue does not support, or an open issue missing
                from every acceptance blocker list.
        """
        if not self.queue:
            raise ReadinessLedgerError("ledger: queue must not be empty")
        if not self.acceptance:
            raise ReadinessLedgerError("ledger: acceptance must not be empty")

        for sha in (self.audit_snapshot, self.reconciled_against):
            if not _SHA_PATTERN.match(sha):
                raise ReadinessLedgerError(
                    f"ledger: SHA must be 40 lowercase hex chars, got {sha!r}"
                )

        keys = [item.key for item in self.queue]
        if len(set(keys)) != len(keys):
            raise ReadinessLedgerError(f"ledger: duplicate queue keys in {keys}")
        issues = [item.issue for item in self.queue]
        if len(set(issues)) != len(issues):
            raise ReadinessLedgerError(f"ledger: duplicate queue issues in {issues}")
        ids = [criterion.id for criterion in self.acceptance]
        if len(set(ids)) != len(ids):
            raise ReadinessLedgerError(f"ledger: duplicate acceptance ids in {ids}")

        all_met = all(criterion.is_met for criterion in self.acceptance)
        if self.release_status == "ready" and (self.open_items or not all_met):
            raise ReadinessLedgerError(
                "ledger: release_status may not be 'ready' while any queue "
                "entry is open or any acceptance criterion is unmet"
            )

        blocked_issues = {
            issue for criterion in self.acceptance for issue in criterion.blockers
        }
        unlisted = sorted(
            item.issue for item in self.open_items if item.issue not in blocked_issues
        )
        if unlisted:
            raise ReadinessLedgerError(
                f"ledger: open issues {unlisted} are absent from every acceptance "
                "blocker list, which would hide them from the release summary"
            )


__all__ = [
    "LEDGER_PATH",
    "REPO_ROOT",
    "UNASSIGNED_OWNER",
    "VALID_ACCEPTANCE_STATUSES",
    "VALID_QUEUE_STATUSES",
    "VALID_RELEASE_STATUSES",
    "AcceptanceCriterion",
    "IndustrialReadinessLedger",
    "ReadinessItem",
    "ReadinessLedgerError",
]
