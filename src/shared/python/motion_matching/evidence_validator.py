"""Committed evidence provenance and acceptance consistency checks (#14935)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import json
import math
import re
from pathlib import Path
from typing import Any, TypeGuard

from src.shared.python.motion_matching.acceptance import AcceptanceGates
from src.shared.python.motion_matching.ledger import classify_receipt

__all__ = [
    "DEFAULT_MAX_LEDGER_UNCLASSIFIED",
    "DEFAULT_MAX_LEDGER_UNVERIFIED",
    "EvidenceIssue",
    "ValidationReport",
    "ledger_unclassified_count",
    "ledger_unverified_count",
    "load_waivers",
    "validate_candidate_gate_metrics",
    "validate_candidate_package",
    "validate_ledger_ratchets",
    "validate_receipt_paths",
    "validate_receipt_taxonomy",
    "validate_committed_evidence",
]

DEFAULT_MAX_LEDGER_UNCLASSIFIED = 37
DEFAULT_MAX_LEDGER_UNVERIFIED = 122

_HTTP_URL_RE = re.compile(r"https?://\S+")

_HOST_ABSOLUTE_RE = re.compile(
    r"(?:file://"
    r"|(?<![A-Za-z0-9_])~[\\/]"
    r"|/home/"
    r"|/Users/"
    r"|/mnt/"
    r"|(?<![A-Za-z0-9_])[A-Za-z]:[\\/])"
)

_CORE_CANDIDATE_GATES = (
    "whole_window_pass",
    "early_retention_pass",
    "terminal_rmse_pass",
    "clubhead_terminal_pass",
)
_PELVIS_YAW_GATES = (
    "pelvis_yaw_pass",
    "pelvis_yaw_080_pass",
    "pelvis_yaw_105_pass",
    "pelvis_yaw_115_pass",
    "pelvis_yaw_1233_pass",
)


def _is_valid_real_number(value: Any) -> TypeGuard[int | float]:
    """Return True iff ``value`` is a finite int/float (excluding bool)."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
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


_DEFAULT_GATES = AcceptanceGates()


def _thresholds_for_duration(duration_s: Any) -> dict[str, tuple[str, float]]:
    dur = float(duration_s) if isinstance(duration_s, (int, float)) else 0.8
    if dur > 1.35:
        whole, early = (
            _DEFAULT_GATES.g3_whole_driver_rmse_m,
            _DEFAULT_GATES.g3_early_rmse_m,
        )
        term, club = _DEFAULT_GATES.g3_terminal_rmse_m, _DEFAULT_GATES.g3_club_rmse_m
    elif dur > 0.90:
        whole, early = _DEFAULT_GATES.g2_whole_rmse_m, _DEFAULT_GATES.g2_early_rmse_m
        term, club = _DEFAULT_GATES.g2_terminal_rmse_m, _DEFAULT_GATES.g2_club_rmse_m
    else:
        whole, early = _DEFAULT_GATES.g1_whole_rmse_m, _DEFAULT_GATES.g1_early_rmse_m
        term, club = _DEFAULT_GATES.g1_terminal_rmse_m, _DEFAULT_GATES.g1_club_rmse_m
    yaw_pct = _DEFAULT_GATES.g1_pelvis_yaw_error_pct
    return {
        "whole_window_pass": ("rmse_m", whole),
        "early_retention_pass": ("early_rmse_m", early),
        "terminal_rmse_pass": ("terminal_rmse_m", term),
        "clubhead_terminal_pass": ("clubhead_terminal_rmse_m", club),
        "pelvis_yaw_pass": ("pelvis_yaw_error_pct", yaw_pct),
        "pelvis_yaw_080_pass": ("pelvis_yaw_080_error_pct", yaw_pct),
        "pelvis_yaw_105_pass": ("pelvis_yaw_105_error_pct", yaw_pct),
        "pelvis_yaw_115_pass": ("pelvis_yaw_115_error_pct", yaw_pct),
        "pelvis_yaw_1233_pass": ("pelvis_yaw_1233_error_pct", yaw_pct),
    }


def validate_candidate_gate_metrics(
    rel_path: str, data: Mapping[str, Any]
) -> list[EvidenceIssue]:
    """Recompute verifiable candidate gate booleans against AcceptanceGates limits."""
    gates = data.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        return []
    issues: list[EvidenceIssue] = []
    duration_s = data.get("duration_s")
    valid_duration = _is_valid_real_number(duration_s) and float(duration_s) > 0.0
    if not valid_duration:
        issues.append(
            EvidenceIssue(
                rule="gate_metric_mismatch",
                path=rel_path,
                message=f"duration_s={duration_s!r} must be a positive finite number",
            )
        )
    missing_core = [gate for gate in _CORE_CANDIDATE_GATES if gate not in gates]
    if missing_core:
        issues.append(
            EvidenceIssue(
                rule="gate_metric_mismatch",
                path=rel_path,
                message=f"missing required core gates: {', '.join(missing_core)}",
            )
        )
    if not any(gate in gates for gate in _PELVIS_YAW_GATES):
        issues.append(
            EvidenceIssue(
                rule="gate_metric_mismatch",
                path=rel_path,
                message="missing required pelvis_yaw gate",
            )
        )

    thresholds = _thresholds_for_duration(duration_s if valid_duration else 0.8)
    for gate_name, stored in gates.items():
        if not isinstance(stored, bool):
            issues.append(
                EvidenceIssue(
                    rule="gate_metric_mismatch",
                    path=rel_path,
                    message=f"{gate_name}={stored!r} must be a boolean",
                )
            )
            continue
        if gate_name not in thresholds:
            if stored is True:
                issues.append(
                    EvidenceIssue(
                        rule="gate_metric_mismatch",
                        path=rel_path,
                        message=f"{gate_name}=True has no verifiable metric threshold",
                    )
                )
            continue
        metric_key, limit = thresholds[gate_name]
        measured = data.get(metric_key)
        if (
            metric_key not in data
            or not _is_valid_real_number(measured)
            or float(measured) < 0.0
        ):
            issues.append(
                EvidenceIssue(
                    rule="gate_metric_mismatch",
                    path=rel_path,
                    message=(
                        f"{gate_name} requires finite non-negative numeric {metric_key}, "
                        f"got {measured!r}"
                    ),
                )
            )
            continue
        recomputed = float(measured) <= limit
        if stored != recomputed:
            issues.append(
                EvidenceIssue(
                    rule="gate_metric_mismatch",
                    path=rel_path,
                    message=(
                        f"{gate_name}={stored} disagrees with {metric_key}={measured} "
                        f"vs threshold {limit}"
                    ),
                )
            )
    return issues


def validate_candidate_package(
    rel_path: str, data: Mapping[str, Any]
) -> list[EvidenceIssue]:
    """``accepted`` must agree with gate booleans, metrics, and ``gates_passed``."""
    issues: list[EvidenceIssue] = []
    gates = data.get("gates")
    accepted = data.get("accepted")

    if isinstance(gates, Mapping) and gates:
        all_passed = all(passed is True for passed in gates.values())
        true_count = sum(1 for value in gates.values() if value is True)
        passed_text = data.get("gates_passed")
        if isinstance(passed_text, str) and passed_text.strip():
            parsed = _parse_gates_passed(passed_text.strip())
            if parsed is None:
                issues.append(
                    EvidenceIssue(
                        rule="accepted_gates_consistency",
                        path=rel_path,
                        message=f"gates_passed={passed_text!r} is not a valid N/M count",
                    )
                )
            elif parsed != (true_count, len(gates)):
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
        elif accepted:
            issues.append(
                EvidenceIssue(
                    rule="accepted_gates_consistency",
                    path=rel_path,
                    message="accepted=true but gates_passed is missing or empty",
                )
            )
        if accepted is False and all_passed:
            issues.append(
                EvidenceIssue(
                    rule="accepted_gates_consistency",
                    path=rel_path,
                    message="accepted=false despite all gates passing",
                )
            )

    if not accepted:
        issues.extend(validate_candidate_gate_metrics(rel_path, data))
        return issues
    if not isinstance(gates, Mapping) or not gates:
        issues.append(
            EvidenceIssue(
                rule="accepted_gates_consistency",
                path=rel_path,
                message="accepted=true but gates block is missing or empty",
            )
        )
        issues.extend(validate_candidate_gate_metrics(rel_path, data))
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
    issues.extend(validate_candidate_gate_metrics(rel_path, data))
    return issues


def _walk_strings(
    value: Any, *, field_path: str, issues: list[EvidenceIssue], rel_path: str
) -> None:
    if isinstance(value, str):
        if "file://" in value or _HOST_ABSOLUTE_RE.search(_HTTP_URL_RE.sub("", value)):
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


def _nonempty_taxonomy_str(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    return isinstance(value, str) and bool(value.strip())


def validate_receipt_taxonomy(
    rel_path: str, data: Mapping[str, Any]
) -> list[EvidenceIssue]:
    """Explicit engine/lane/capture metadata must not disagree with path heuristics."""
    issues: list[EvidenceIssue] = []
    missing_fields: list[str] = []
    if not _nonempty_taxonomy_str(data, "lane"):
        missing_fields.append("lane")
    if not (
        _nonempty_taxonomy_str(data, "engine")
        or _nonempty_taxonomy_str(data, "backend")
    ):
        missing_fields.append("engine")
    if not _nonempty_taxonomy_str(data, "capture"):
        missing_fields.append("capture")
    if missing_fields:
        issues.append(
            EvidenceIssue(
                rule="missing_taxonomy",
                path=rel_path,
                message=(
                    "receipt missing explicit taxonomy fields: "
                    + ", ".join(missing_fields)
                ),
            )
        )
    engine, lane, capture, _reason = classify_receipt(rel_path, data)
    explicit_lane = data.get("lane")
    explicit_engine = data.get("engine") or data.get("backend")
    explicit_capture = data.get("capture")
    if isinstance(explicit_lane, str) and explicit_lane.strip():
        explicit_lane_norm = explicit_lane.strip().lower()
        heuristic_lane = lane.lower()
        coarse_matched_bucket = heuristic_lane == "matched"
        if (
            lane != "unclassified"
            and not coarse_matched_bucket
            and explicit_lane_norm != heuristic_lane
        ):
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


def _valid_waiver_entry(entry: Mapping[str, Any]) -> bool:
    waiver_id = entry.get("id")
    rule = entry.get("rule")
    expires = entry.get("expires")
    if not isinstance(waiver_id, str) or not waiver_id.strip():
        return False
    if not isinstance(rule, str) or not rule.strip():
        return False
    if not isinstance(expires, str) or not expires.strip():
        return False
    try:
        date.fromisoformat(expires.strip())
    except ValueError:
        return False
    return True


def load_waivers(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    waivers = payload.get("waivers")
    if not isinstance(waivers, list):
        return []
    return [
        dict(entry)
        for entry in waivers
        if isinstance(entry, dict) and _valid_waiver_entry(entry)
    ]


def _waiver_matches(
    issue: EvidenceIssue, waiver: Mapping[str, Any], today: date
) -> bool:
    if waiver.get("rule") != issue.rule:
        return False
    waiver_path = waiver.get("path")
    if isinstance(waiver_path, str) and waiver_path != issue.path:
        return False
    expires = waiver.get("expires")
    if not isinstance(expires, str) or not expires.strip():
        return False
    try:
        if date.fromisoformat(expires.strip()) < today:
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
                raw_issues.extend(validate_receipt_taxonomy(rel, data))

    ledger_path = repo_root / "reports" / "matched_swing_ledger.json"
    if ledger_path.is_file():
        ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        rows = ledger_payload.get("rows")
        if isinstance(rows, list):
            row_maps = [row for row in rows if isinstance(row, Mapping)]
            raw_issues.extend(validate_ledger_unclassified_rows(row_maps))
            raw_issues.extend(validate_ledger_ratchets(row_maps))

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


def ledger_unverified_count(ledger_rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in ledger_rows if row.get("acceptance") is None)


def validate_ledger_ratchets(
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    max_unclassified: int = DEFAULT_MAX_LEDGER_UNCLASSIFIED,
    max_unverified: int = DEFAULT_MAX_LEDGER_UNVERIFIED,
) -> list[EvidenceIssue]:
    """Enforce shrink-only upper bounds on unclassified and unjudged ledger rows."""
    issues: list[EvidenceIssue] = []
    unclassified = ledger_unclassified_count(ledger_rows)
    if unclassified > max_unclassified:
        issues.append(
            EvidenceIssue(
                rule="ledger_unclassified_ratchet",
                path="reports/matched_swing_ledger.json",
                message=f"unclassified count {unclassified} exceeds ratchet {max_unclassified}",
            )
        )
    unverified = ledger_unverified_count(ledger_rows)
    if unverified > max_unverified:
        issues.append(
            EvidenceIssue(
                rule="ledger_unverified_ratchet",
                path="reports/matched_swing_ledger.json",
                message=f"missing-verdict count {unverified} exceeds ratchet {max_unverified}",
            )
        )
    return issues
