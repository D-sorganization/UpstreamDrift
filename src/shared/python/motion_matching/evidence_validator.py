"""Committed evidence provenance and acceptance consistency checks (#14935)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import json
import re
from pathlib import Path
from typing import Any

from src.shared.python.motion_matching.ledger import classify_receipt

__all__ = [
    "EvidenceIssue",
    "ValidationReport",
    "load_waivers",
    "validate_candidate_package",
    "validate_receipt_paths",
    "validate_receipt_taxonomy",
    "validate_committed_evidence",
]

_HOST_ABSOLUTE_RE = re.compile(
    r"(^/home/|^/Users/|^[A-Za-z]:\\|^candidate:/home/|^candidate:/Users/)"
)


@dataclass(frozen=True)
class EvidenceIssue:
    """One validation finding."""

    rule: str
    path: str
    message: str
    waiver_id: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate validation outcome."""

    issues: tuple[EvidenceIssue, ...]
    waived: tuple[EvidenceIssue, ...]

    @property
    def failed(self) -> bool:
        return bool(self.issues)


def _parse_gates_passed(text: str) -> tuple[int, int] | None:
    if "/" not in text:
        return None
    left, right = text.split("/", 1)
    try:
        return int(left.strip()), int(right.strip())
    except ValueError:
        return None


def validate_candidate_package(
    rel_path: str, data: Mapping[str, Any]
) -> list[EvidenceIssue]:
    """``accepted`` must agree with gate booleans and ``gates_passed``."""
    issues: list[EvidenceIssue] = []
    if not data.get("accepted"):
        return issues
    gates = data.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        issues.append(
            EvidenceIssue(
                rule="accepted_gates_consistency",
                path=rel_path,
                message="accepted=true but gates block is missing or empty",
            )
        )
        return issues
    failed = [name for name, passed in gates.items() if passed is not True]
    if failed:
        issues.append(
            EvidenceIssue(
                rule="accepted_gates_consistency",
                path=rel_path,
                message=(
                    f"accepted=true but gates failed: {', '.join(sorted(failed))}"
                ),
            )
        )
    passed_text = data.get("gates_passed")
    if isinstance(passed_text, str):
        parsed = _parse_gates_passed(passed_text)
        if parsed is not None:
            passed_count, total = parsed
            true_count = sum(1 for value in gates.values() if value is True)
            if passed_count != true_count or total != len(gates):
                issues.append(
                    EvidenceIssue(
                        rule="accepted_gates_consistency",
                        path=rel_path,
                        message=(
                            f"gates_passed={passed_text!r} disagrees with "
                            f"{true_count}/{len(gates)} gate booleans"
                        ),
                    )
                )
    return issues


def _walk_strings(
    value: Any, *, field_path: str, issues: list[EvidenceIssue], rel_path: str
) -> None:
    if isinstance(value, str):
        if _HOST_ABSOLUTE_RE.search(value):
            issues.append(
                EvidenceIssue(
                    rule="host_absolute_provenance",
                    path=rel_path,
                    message=f"{field_path}: host-absolute path {value!r}",
                )
            )
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            _walk_strings(
                nested,
                field_path=f"{field_path}.{key}",
                issues=issues,
                rel_path=rel_path,
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _walk_strings(
                nested,
                field_path=f"{field_path}[{index}]",
                issues=issues,
                rel_path=rel_path,
            )


def validate_receipt_paths(
    rel_path: str, data: Mapping[str, Any]
) -> list[EvidenceIssue]:
    """Committed receipts must not cite host-absolute provenance strings."""
    issues: list[EvidenceIssue] = []
    _walk_strings(data, field_path="receipt", issues=issues, rel_path=rel_path)
    return issues


def validate_receipt_taxonomy(
    rel_path: str, data: Mapping[str, Any]
) -> list[EvidenceIssue]:
    """Explicit engine/lane/capture metadata must not disagree with path heuristics."""
    issues: list[EvidenceIssue] = []
    engine, lane, capture, _reason = classify_receipt(rel_path, data)
    explicit_lane = data.get("lane")
    explicit_engine = data.get("engine") or data.get("backend")
    explicit_capture = data.get("capture")
    if isinstance(explicit_lane, str) and explicit_lane.strip():
        if lane != "unclassified" and explicit_lane.strip().lower() != lane.lower():
            issues.append(
                EvidenceIssue(
                    rule="explicit_taxonomy",
                    path=rel_path,
                    message=(
                        f"explicit lane={explicit_lane!r} disagrees with "
                        f"path heuristic lane={lane!r}"
                    ),
                )
            )
    if isinstance(explicit_engine, str) and explicit_engine.strip():
        normalized = explicit_engine.strip().lower()
        if engine != "unknown" and normalized != engine.lower():
            issues.append(
                EvidenceIssue(
                    rule="explicit_taxonomy",
                    path=rel_path,
                    message=(
                        f"explicit engine={explicit_engine!r} disagrees with "
                        f"path heuristic engine={engine!r}"
                    ),
                )
            )
    if isinstance(explicit_capture, str) and explicit_capture.strip():
        normalized = explicit_capture.strip().lower()
        if capture is not None and normalized != capture.lower():
            issues.append(
                EvidenceIssue(
                    rule="explicit_taxonomy",
                    path=rel_path,
                    message=(
                        f"explicit capture={explicit_capture!r} disagrees with "
                        f"path heuristic capture={capture!r}"
                    ),
                )
            )
    if lane == "unclassified" and any(
        isinstance(data.get(key), str) and str(data[key]).strip()
        for key in ("lane", "engine", "capture", "backend")
    ):
        issues.append(
            EvidenceIssue(
                rule="unclassified_with_metadata",
                path=rel_path,
                message=(
                    "receipt carries explicit taxonomy fields but remains "
                    "unclassified by path heuristics"
                ),
            )
        )
    return issues


def validate_ledger_unclassified_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[EvidenceIssue]:
    """Ledger rows must not remain ``unclassified`` without an explicit waiver."""
    issues: list[EvidenceIssue] = []
    for row in rows:
        lane = str(row.get("lane", "")).lower()
        if lane != "unclassified":
            continue
        receipt_path = str(row.get("receipt_path", "unknown"))
        issues.append(
            EvidenceIssue(
                rule="ledger_unclassified",
                path=receipt_path,
                message="ledger row classified as unclassified",
            )
        )
    return issues


def load_waivers(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    waivers = payload.get("waivers")
    if not isinstance(waivers, list):
        return []
    return [entry for entry in waivers if isinstance(entry, dict)]


def _waiver_matches(
    issue: EvidenceIssue, waiver: Mapping[str, Any], today: date
) -> bool:
    if waiver.get("rule") != issue.rule:
        return False
    waiver_path = waiver.get("path")
    if isinstance(waiver_path, str) and waiver_path != issue.path:
        return False
    expires = waiver.get("expires")
    if isinstance(expires, str):
        try:
            if date.fromisoformat(expires) < today:
                return False
        except ValueError:
            return False
    return True


def validate_committed_evidence(
    repo_root: Path,
    *,
    waivers_path: Path | None = None,
    today: date | None = None,
) -> ValidationReport:
    """Scan committed candidate packages and evidence receipts under repo_root."""
    waivers_file = waivers_path or (
        repo_root / "scripts" / "config" / "evidence_validation_waivers.json"
    )
    waivers = load_waivers(waivers_file)
    check_day = today or date.today()

    raw_issues: list[EvidenceIssue] = []

    candidates_dir = repo_root / "candidates"
    if candidates_dir.is_dir():
        for package in sorted(candidates_dir.glob("*.json")):
            rel = package.relative_to(repo_root).as_posix()
            data = json.loads(package.read_text(encoding="utf-8"))
            if isinstance(data, Mapping):
                raw_issues.extend(validate_candidate_package(rel, data))

    evidence_root = repo_root / "evidence"
    if evidence_root.is_dir():
        for receipt in sorted(evidence_root.rglob("receipt.json")):
            rel = receipt.relative_to(repo_root).as_posix()
            data = json.loads(receipt.read_text(encoding="utf-8"))
            if isinstance(data, Mapping):
                raw_issues.extend(validate_receipt_paths(rel, data))

    ledger_path = repo_root / "reports" / "matched_swing_ledger.json"
    if ledger_path.is_file():
        ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        rows = ledger_payload.get("rows")
        if isinstance(rows, list):
            row_maps = [row for row in rows if isinstance(row, Mapping)]
            raw_issues.extend(validate_ledger_unclassified_rows(row_maps))

    active: list[EvidenceIssue] = []
    waived: list[EvidenceIssue] = []
    for issue in raw_issues:
        matched = next(
            (w["id"] for w in waivers if _waiver_matches(issue, w, check_day)),
            None,
        )
        if matched:
            waived.append(
                EvidenceIssue(
                    rule=issue.rule,
                    path=issue.path,
                    message=issue.message,
                    waiver_id=str(matched),
                )
            )
        else:
            active.append(issue)

    return ValidationReport(issues=tuple(active), waived=tuple(waived))


def ledger_unclassified_count(ledger_rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1 for row in ledger_rows if str(row.get("lane", "")).lower() == "unclassified"
    )
