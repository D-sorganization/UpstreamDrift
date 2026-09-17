"""Validate the governed documentation and engine capability-evidence registries.

Both registries are data. This module is the single boundary that turns them
into companion manifest records: it derives documentation freshness from the
exact source commit, binds every capability claim to a resolvable test or
artifact, and renders the provider documentation page from the same machine
source. It never grants scientific qualification, copies calculations,
tolerances, or approval claims, and never reads wall-clock time.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

DOCUMENTATION_REGISTRY_PATH = Path("scripts/config/companion_documentation.v1.json")
CAPABILITY_REGISTRY_PATH = Path("scripts/config/companion_capability_evidence.v1.json")
DOCUMENTATION_REGISTRY_ID = "upstreamdrift-companion-documentation"
CAPABILITY_REGISTRY_ID = "upstreamdrift-companion-capability-evidence"
REGISTRY_VERSION = "1.0.0"
GENERATED_DOC_PATH = Path("docs/engines/engine_capability_evidence.md")
IMMUTABLE_URL_PREFIX = "https://github.com/D-sorganization/UpstreamDrift/blob/"
SCREENSHOT_BLOCKER = (
    "Screenshot inventory requires its governed provider slice (#9191)."
)
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_AUDIENCES = frozenset({"developer", "integrator", "reviewer", "user"})
_DOCUMENT_STATUSES = frozenset({"active", "deprecated", "missing"})
_FRESHNESS_STATES = ("current", "review_required", "stale", "unknown", "missing")
_AVAILABILITY_STATES = frozenset({"available", "conditional", "unavailable"})
_EVIDENCE_STATES = frozenset({"qualified", "unqualified"})
_EVIDENCE_KINDS = frozenset({"artifact", "test"})
_GAP_SCOPES = frozenset({"catalog", "documentation", "engine", "launcher", "parity"})
_GATE_PREFIX = PurePosixPath(".github/workflows")
_DOCUMENT_KEYS = frozenset(
    {
        "id",
        "title",
        "status",
        "source_path",
        "audiences",
        "topics",
        "program_ids",
        "engine_ids",
        "owner",
        "last_reviewed",
        "review_due",
        "reviewed_sha256",
        "reason",
    }
)
_ENGINE_KEYS = frozenset(
    {"id", "runtime_availability", "documentation_ids", "capabilities"}
)
_CAPABILITY_KEYS = frozenset(
    {"id", "title", "evidence_state", "evidence", "reason", "limitations"}
)
_EVIDENCE_KEYS = frozenset({"kind", "path", "selector", "gate"})
_GAP_KEYS = frozenset({"id", "issue", "scope", "summary", "summary_metric"})


class EvidenceContractError(ValueError):
    """Raised when registry data violates its fail-closed contract."""


@dataclass(frozen=True)
class SourceContext:
    """Exact source facts the derivations are bound to.

    ``read_input`` returns the committed bytes of a repository-relative path
    or ``None`` when the path is not a tracked file; it is the only I/O.
    """

    commit: str
    commit_date: str
    read_input: Callable[[str], bytes | None]

    def __post_init__(self) -> None:
        if not _COMMIT.fullmatch(self.commit):
            raise EvidenceContractError("source commit must be an exact commit")
        _parse_date(self.commit_date, label="source commit date")


def _exact_keys(value: Any, *, required: Set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceContractError(f"{label} must be an object")
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise EvidenceContractError(
            f"{label} has unknown or missing keys; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return value


def _non_empty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceContractError(f"{label} must be a non-empty string")
    return value


def _identifier(value: Any, *, label: str) -> str:
    text = _non_empty_string(value, label=label)
    if not _IDENTIFIER.fullmatch(text):
        raise EvidenceContractError(f"{label} is not a stable identifier: {text!r}")
    return text


def _identifier_list(value: Any, *, label: str, non_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise EvidenceContractError(f"{label} must be an array")
    items = [_identifier(item, label=f"{label} item") for item in value]
    if non_empty and not items:
        raise EvidenceContractError(f"{label} must not be empty")
    if len(set(items)) != len(items):
        raise EvidenceContractError(f"{label} contains duplicate values")
    return sorted(items)


def _resolvable(ids: Iterable[str], *, known: Set[str], label: str) -> list[str]:
    dangling = sorted(set(ids) - set(known))
    if dangling:
        raise EvidenceContractError(f"{label} references unknown ids: {dangling}")
    return sorted(ids)


def _repo_relative(value: Any, *, label: str) -> str:
    text = _non_empty_string(value, label=label).replace("\\", "/")
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or ".." in candidate.parts
        or candidate.parts in ((), (".",))
    ):
        raise EvidenceContractError(f"{label} must be a contained repo-relative path")
    return candidate.as_posix()


def _parse_date(value: Any, *, label: str) -> date:
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        raise EvidenceContractError(f"{label} must be an ISO calendar date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise EvidenceContractError(f"{label} is not a valid date: {exc}") from exc


def _tracked_sha256(context: SourceContext, path: str, *, label: str) -> str:
    payload = context.read_input(path)
    if payload is None:
        raise EvidenceContractError(f"{label} is not a tracked file: {path}")
    return hashlib.sha256(payload).hexdigest()


def _load_registry(
    payload: bytes, *, registry_id: str, collection: str, label: str
) -> Mapping[str, Any]:
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceContractError(f"{label} is not canonical JSON: {exc}") from exc
    keys = {"registry_id", "version", collection}
    if collection == "engines":
        keys.add("known_gaps")
    raw = _exact_keys(raw, required=keys, label=label)
    if raw["registry_id"] != registry_id or raw["version"] != REGISTRY_VERSION:
        raise EvidenceContractError(f"{label} identity or version is stale")
    if not isinstance(raw[collection], list) or not raw[collection]:
        raise EvidenceContractError(f"{label} must contain {collection} records")
    return raw


# ---------------------------------------------------------------------------
# Documentation registry
# ---------------------------------------------------------------------------


def _review_facts(
    raw: Mapping[str, Any], *, context: SourceContext, label: str
) -> tuple[str | None, str | None, str | None]:
    last_reviewed = raw["last_reviewed"]
    review_due = raw["review_due"]
    reviewed_sha256 = raw["reviewed_sha256"]
    if last_reviewed is None:
        if review_due is not None or reviewed_sha256 is not None:
            raise EvidenceContractError(
                f"{label}: review_due and reviewed_sha256 require last_reviewed"
            )
        return None, None, None
    reviewed = _parse_date(last_reviewed, label=f"{label} last_reviewed")
    due = _parse_date(review_due, label=f"{label} review_due")
    if due < reviewed:
        raise EvidenceContractError(
            f"{label}: review_due {review_due} precedes last_reviewed {last_reviewed}"
        )
    if reviewed > date.fromisoformat(context.commit_date):
        raise EvidenceContractError(
            f"{label}: last_reviewed {last_reviewed} is later than the source "
            f"commit date {context.commit_date}"
        )
    if not isinstance(reviewed_sha256, str) or not _SHA256.fullmatch(reviewed_sha256):
        raise EvidenceContractError(f"{label}: reviewed_sha256 must be an exact hash")
    return last_reviewed, review_due, reviewed_sha256


def _freshness(
    *,
    review_due: str | None,
    reviewed_sha256: str | None,
    source_sha256: str,
    commit_date: str,
) -> tuple[str, str | None]:
    """Derive freshness from exact facts only; no wall-clock time is read."""
    if review_due is None:
        return "unknown", "No governed review is recorded for this document."
    if date.fromisoformat(review_due) < date.fromisoformat(commit_date):
        return "stale", (
            f"Review was due {review_due}, before the source commit date {commit_date}."
        )
    if reviewed_sha256 != source_sha256:
        return "review_required", (
            "Committed content changed after the last recorded review."
        )
    return "current", None


def _documentation_record(
    raw: Any,
    *,
    context: SourceContext,
    program_ids: Set[str],
    engine_ids: Set[str],
) -> dict[str, Any]:
    raw = _exact_keys(raw, required=_DOCUMENT_KEYS, label="documentation record")
    record_id = _identifier(raw["id"], label="documentation id")
    label = f"documentation {record_id}"
    status = _non_empty_string(raw["status"], label=f"{label} status")
    if status not in _DOCUMENT_STATUSES:
        raise EvidenceContractError(f"{label}: unsupported status {status!r}")
    audiences = _identifier_list(raw["audiences"], label=f"{label} audiences")
    if not audiences or not set(audiences) <= _AUDIENCES:
        raise EvidenceContractError(f"{label}: audiences must be non-empty and known")
    declared_reason = raw["reason"]
    if declared_reason is not None:
        declared_reason = _non_empty_string(declared_reason, label=f"{label} reason")
    record: dict[str, Any] = {
        "id": record_id,
        "title": _non_empty_string(raw["title"], label=f"{label} title"),
        "status": status,
        "audiences": audiences,
        "topics": _identifier_list(
            raw["topics"], label=f"{label} topics", non_empty=True
        ),
        "program_ids": _resolvable(
            _identifier_list(raw["program_ids"], label=f"{label} program_ids"),
            known=program_ids,
            label=f"{label} program_ids",
        ),
        "engine_ids": _resolvable(
            _identifier_list(raw["engine_ids"], label=f"{label} engine_ids"),
            known=engine_ids,
            label=f"{label} engine_ids",
        ),
        "owner": _non_empty_string(raw["owner"], label=f"{label} owner"),
    }
    if status == "missing":
        if any(
            raw[key] is not None
            for key in ("source_path", "last_reviewed", "review_due", "reviewed_sha256")
        ):
            raise EvidenceContractError(
                f"{label}: missing documentation cannot declare a source or review"
            )
        if declared_reason is None:
            raise EvidenceContractError(
                f"{label}: missing documentation needs a reason"
            )
        record.update(
            {
                "source_path": None,
                "source_commit": None,
                "source_sha256": None,
                "url": None,
                "last_reviewed": None,
                "review_due": None,
                "reviewed_sha256": None,
                "freshness": "missing",
                "reason": declared_reason,
            }
        )
        return record
    source_path = _repo_relative(raw["source_path"], label=f"{label} source_path")
    source_sha256 = _tracked_sha256(context, source_path, label=f"{label} source")
    last_reviewed, review_due, reviewed_sha256 = _review_facts(
        raw, context=context, label=label
    )
    freshness, derived_reason = _freshness(
        review_due=review_due,
        reviewed_sha256=reviewed_sha256,
        source_sha256=source_sha256,
        commit_date=context.commit_date,
    )
    if status == "deprecated":
        if declared_reason is None:
            raise EvidenceContractError(
                f"{label}: deprecated documentation needs a reason"
            )
        reason: str | None = declared_reason
    elif declared_reason is not None and last_reviewed is not None:
        raise EvidenceContractError(
            f"{label}: reviewed active documentation derives its reason; "
            "declared reasons are only allowed while no review is recorded"
        )
    else:
        reason = declared_reason or derived_reason
    record.update(
        {
            "source_path": source_path,
            "source_commit": context.commit,
            "source_sha256": source_sha256,
            "url": f"{IMMUTABLE_URL_PREFIX}{context.commit}/{source_path}",
            "last_reviewed": last_reviewed,
            "review_due": review_due,
            "reviewed_sha256": reviewed_sha256,
            "freshness": freshness,
            "reason": reason,
        }
    )
    return record


def parse_documentation_registry(
    payload: bytes,
    *,
    context: SourceContext,
    program_ids: Set[str],
    engine_ids: Set[str],
    required_paths: Set[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Parse strict documentation registry bytes into manifest records.

    Postconditions: records are unique and sorted by id, every non-missing
    record carries an exact source commit, hash, and immutable URL, and every
    path in ``required_paths`` (workflow documentation) has a governed record.
    """
    raw = _load_registry(
        payload,
        registry_id=DOCUMENTATION_REGISTRY_ID,
        collection="documentation",
        label="documentation registry",
    )
    records = [
        _documentation_record(
            item, context=context, program_ids=program_ids, engine_ids=engine_ids
        )
        for item in raw["documentation"]
    ]
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise EvidenceContractError("documentation registry contains duplicate ids")
    governed = {
        record["source_path"] for record in records if record["source_path"] is not None
    }
    if len(governed) != len([r for r in records if r["source_path"] is not None]):
        raise EvidenceContractError("documentation registry maps one path twice")
    ungoverned = sorted(set(required_paths) - governed)
    if ungoverned:
        raise EvidenceContractError(
            f"workflow documentation paths have no governed record: {ungoverned}"
        )
    return sorted(records, key=lambda record: record["id"])


def documentation_ids_by_program(
    documentation: Sequence[Mapping[str, Any]],
    programs: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Map every program to the documentation records that route to it."""
    result: dict[str, list[str]] = {}
    for program in programs:
        ids = {
            record["id"]
            for record in documentation
            if record["status"] != "missing"
            and (
                program["id"] in record["program_ids"]
                or (
                    program["engine_id"] is not None
                    and program["engine_id"] in record["engine_ids"]
                )
            )
        }
        result[program["id"]] = sorted(ids)
    return result


# ---------------------------------------------------------------------------
# Capability-evidence registry
# ---------------------------------------------------------------------------


def _test_selector_resolves(payload: bytes, selector_tail: str, *, label: str) -> None:
    try:
        tree = ast.parse(payload.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise EvidenceContractError(
            f"{label}: test module is not parseable: {exc}"
        ) from exc
    scope: Sequence[ast.stmt] = tree.body
    for name in selector_tail.split("::"):
        match = next(
            (
                node
                for node in scope
                if isinstance(
                    node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                )
                and node.name == name
            ),
            None,
        )
        if match is None:
            raise EvidenceContractError(
                f"{label}: selector segment {name!r} does not exist in the test module"
            )
        scope = match.body if isinstance(match, ast.ClassDef) else ()


def _evidence_record(raw: Any, *, context: SourceContext, label: str) -> dict[str, Any]:
    raw = _exact_keys(raw, required=_EVIDENCE_KEYS, label=f"{label} evidence")
    kind = _non_empty_string(raw["kind"], label=f"{label} evidence kind")
    if kind not in _EVIDENCE_KINDS:
        raise EvidenceContractError(f"{label}: unsupported evidence kind {kind!r}")
    path = _repo_relative(raw["path"], label=f"{label} evidence path")
    payload = context.read_input(path)
    if payload is None:
        raise EvidenceContractError(f"{label}: evidence is not a tracked file: {path}")
    selector = raw["selector"]
    gate = raw["gate"]
    if kind == "test":
        selector = _non_empty_string(selector, label=f"{label} evidence selector")
        if not selector.startswith(f"{path}::"):
            raise EvidenceContractError(
                f"{label}: test selector must be '{path}::<node>'"
            )
        _test_selector_resolves(payload, selector[len(path) + 2 :], label=label)
        gate = _repo_relative(gate, label=f"{label} evidence gate")
        if _GATE_PREFIX not in PurePosixPath(gate).parents:
            raise EvidenceContractError(
                f"{label}: evidence gate must be a workflow under {_GATE_PREFIX}"
            )
        if context.read_input(gate) is None:
            raise EvidenceContractError(
                f"{label}: evidence gate is not tracked: {gate}"
            )
    elif selector is not None or gate is not None:
        raise EvidenceContractError(
            f"{label}: artifact evidence cannot declare a selector or gate"
        )
    return {
        "kind": kind,
        "path": path,
        "selector": selector,
        "gate": gate,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_commit": context.commit,
    }


def _capability_record(
    raw: Any, *, context: SourceContext, engine_id: str
) -> dict[str, Any]:
    raw = _exact_keys(raw, required=_CAPABILITY_KEYS, label=f"{engine_id} capability")
    capability_id = _identifier(raw["id"], label=f"{engine_id} capability id")
    label = f"engine {engine_id} capability {capability_id}"
    state = _non_empty_string(raw["evidence_state"], label=f"{label} evidence_state")
    if state not in _EVIDENCE_STATES:
        raise EvidenceContractError(f"{label}: unsupported evidence state {state!r}")
    if not isinstance(raw["evidence"], list):
        raise EvidenceContractError(f"{label}: evidence must be an array")
    evidence = [
        _evidence_record(item, context=context, label=label) for item in raw["evidence"]
    ]
    reason = raw["reason"]
    if state == "qualified":
        if not evidence or reason is not None:
            raise EvidenceContractError(
                f"{label}: qualified capabilities need evidence and no reason"
            )
    elif evidence:
        raise EvidenceContractError(
            f"{label}: unqualified capabilities cannot carry evidence"
        )
    else:
        reason = _non_empty_string(reason, label=f"{label} reason")
    limitations = raw["limitations"]
    if not isinstance(limitations, list) or len(set(limitations)) != len(limitations):
        raise EvidenceContractError(f"{label}: limitations must be unique strings")
    return {
        "id": capability_id,
        "title": _non_empty_string(raw["title"], label=f"{label} title"),
        "evidence_state": state,
        "evidence": sorted(
            evidence, key=lambda item: (item["path"], item["selector"] or "")
        ),
        "reason": reason,
        "limitations": sorted(
            _non_empty_string(item, label=f"{label} limitation") for item in limitations
        ),
    }


def _availability(raw: Any, *, label: str) -> dict[str, str | None]:
    raw = _exact_keys(raw, required={"state", "reason"}, label=label)
    state = _non_empty_string(raw["state"], label=f"{label} state")
    if state not in _AVAILABILITY_STATES:
        raise EvidenceContractError(f"{label}: unsupported state {state!r}")
    reason = raw["reason"]
    if state == "available" and reason is not None:
        raise EvidenceContractError(f"{label}: available runtime reason must be null")
    if state != "available":
        reason = _non_empty_string(reason, label=f"{label} reason")
    return {"state": state, "reason": reason}


def _engine_record(
    raw: Any,
    *,
    context: SourceContext,
    engine: Mapping[str, Any],
    documentation_ids: Set[str],
) -> dict[str, Any]:
    engine_id = engine["id"]
    raw = _exact_keys(raw, required=_ENGINE_KEYS, label=f"engine {engine_id}")
    if not isinstance(raw["capabilities"], list) or not raw["capabilities"]:
        raise EvidenceContractError(f"engine {engine_id}: capabilities are required")
    capabilities = [
        _capability_record(item, context=context, engine_id=engine_id)
        for item in raw["capabilities"]
    ]
    ids = [capability["id"] for capability in capabilities]
    if len(set(ids)) != len(ids):
        raise EvidenceContractError(f"engine {engine_id}: duplicate capability ids")
    return {
        "id": engine_id,
        "name": engine["name"],
        "support_tier": engine["support_tier"],
        "runtime_availability": _availability(
            raw["runtime_availability"],
            label=f"engine {engine_id} runtime_availability",
        ),
        "documentation_ids": _resolvable(
            _identifier_list(
                raw["documentation_ids"],
                label=f"engine {engine_id} documentation_ids",
                non_empty=True,
            ),
            known=documentation_ids,
            label=f"engine {engine_id} documentation_ids",
        ),
        "capabilities": sorted(capabilities, key=lambda item: item["id"]),
        "scientific_qualification": dict(engine["scientific_qualification"]),
    }


def _known_gap(raw: Any, *, summary: Mapping[str, int]) -> dict[str, Any]:
    raw = _exact_keys(raw, required=_GAP_KEYS, label="known gap")
    gap_id = _identifier(raw["id"], label="known gap id")
    issue = raw["issue"]
    if not isinstance(issue, int) or isinstance(issue, bool) or issue < 1:
        raise EvidenceContractError(
            f"known gap {gap_id}: issue must be a positive number"
        )
    scope = _non_empty_string(raw["scope"], label=f"known gap {gap_id} scope")
    if scope not in _GAP_SCOPES:
        raise EvidenceContractError(f"known gap {gap_id}: unsupported scope {scope!r}")
    metric = raw["summary_metric"]
    if metric is not None:
        metric = _non_empty_string(metric, label=f"known gap {gap_id} summary_metric")
        if metric not in summary:
            raise EvidenceContractError(
                f"known gap {gap_id}: summary_metric {metric!r} is not exported"
            )
        if summary[metric] <= 0:
            raise EvidenceContractError(
                f"known gap {gap_id}: {metric} is zero, so the gap is no longer observed"
            )
    return {
        "id": gap_id,
        "issue": issue,
        "scope": scope,
        "summary": _non_empty_string(
            raw["summary"], label=f"known gap {gap_id} summary"
        ),
        "summary_metric": metric,
    }


def parse_capability_registry(
    payload: bytes,
    *,
    context: SourceContext,
    engines: Sequence[Mapping[str, Any]],
    documentation_ids: Set[str],
    summary: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse the capability-evidence registry into engine and known-gap records.

    Support tier, name, and scientific qualification come only from the
    catalog's ``engines``; the registry cannot restate or promote them.
    Postconditions: engine ids equal the catalog engine set exactly, records are
    sorted by id, and every qualified capability names resolvable evidence.
    """
    raw = _load_registry(
        payload,
        registry_id=CAPABILITY_REGISTRY_ID,
        collection="engines",
        label="capability-evidence registry",
    )
    by_id = {engine["id"]: engine for engine in engines}
    declared = [
        _identifier(
            item.get("id") if isinstance(item, Mapping) else None, label="engine id"
        )
        for item in raw["engines"]
    ]
    if sorted(declared) != sorted(by_id) or len(set(declared)) != len(declared):
        raise EvidenceContractError(
            "capability-evidence registry engines contradict the catalog engine set; "
            f"declared={sorted(declared)}, catalog={sorted(by_id)}"
        )
    records = [
        _engine_record(
            item,
            context=context,
            engine=by_id[engine_id],
            documentation_ids=documentation_ids,
        )
        for engine_id, item in zip(declared, raw["engines"], strict=True)
    ]
    if not isinstance(raw["known_gaps"], list):
        raise EvidenceContractError("known_gaps must be an array")
    gaps = [_known_gap(item, summary=summary) for item in raw["known_gaps"]]
    gap_ids = [gap["id"] for gap in gaps]
    if len(set(gap_ids)) != len(gap_ids):
        raise EvidenceContractError("known_gaps contains duplicate ids")
    return (
        sorted(records, key=lambda record: record["id"]),
        sorted(gaps, key=lambda gap: gap["id"]),
    )


def publication_blockers(
    *,
    documentation: Sequence[Mapping[str, Any]],
    engines: Sequence[Mapping[str, Any]],
    known_gaps: Sequence[Mapping[str, Any]],
    undocumented_visible_programs: int,
) -> list[str]:
    """Derive the publication blockers from exported facts, deterministically."""
    blockers: list[str] = []
    capabilities = [c for engine in engines for c in engine["capabilities"]]
    unqualified = sum(c["evidence_state"] == "unqualified" for c in capabilities)
    if unqualified:
        blockers.append(
            f"{unqualified} of {len(capabilities)} engine capabilities remain "
            "unqualified pending exact passing evidence."
        )
    not_current = [r for r in documentation if r["freshness"] != "current"]
    if not_current:
        counts = ", ".join(
            f"{state}={sum(r['freshness'] == state for r in not_current)}"
            for state in _FRESHNESS_STATES[1:]
            if any(r["freshness"] == state for r in not_current)
        )
        blockers.append(
            f"{len(not_current)} of {len(documentation)} documentation records are "
            f"not current ({counts})."
        )
    if undocumented_visible_programs:
        blockers.append(
            f"{undocumented_visible_programs} visible programs have no governed "
            "documentation record."
        )
    if known_gaps:
        issues = ", ".join(f"#{gap['issue']}" for gap in known_gaps)
        blockers.append(f"Known gaps remain visible with owning issues: {issues}.")
    blockers.append(SCREENSHOT_BLOCKER)
    return blockers


def build_governed_records(
    *,
    documentation_payload: bytes,
    capability_payload: bytes,
    context: SourceContext,
    engines: Sequence[Mapping[str, Any]],
    programs: Sequence[dict[str, Any]],
    workflows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, int],
) -> dict[str, Any]:
    """Derive documentation, engine, gap, blocker, and summary records.

    Side effect: every program's ``documentation_ids`` is filled in so each
    public record carries an explicit route or an empty (undocumented) state.
    """
    documentation = parse_documentation_registry(
        documentation_payload,
        context=context,
        program_ids={program["id"] for program in programs},
        engine_ids={engine["id"] for engine in engines},
        required_paths={
            path for workflow in workflows for path in workflow["documentation_paths"]
        },
    )
    engine_records, known_gaps = parse_capability_registry(
        capability_payload,
        context=context,
        engines=engines,
        documentation_ids={record["id"] for record in documentation},
        summary=summary,
    )
    routes = documentation_ids_by_program(documentation, programs)
    for program in programs:
        program["documentation_ids"] = routes[program["id"]]
    capabilities = [c for engine in engine_records for c in engine["capabilities"]]
    undocumented = sum(
        not program["hidden"] and not program["documentation_ids"]
        for program in programs
    )
    return {
        "documentation": documentation,
        "engines": engine_records,
        "known_gaps": known_gaps,
        "blockers": publication_blockers(
            documentation=documentation,
            engines=engine_records,
            known_gaps=known_gaps,
            undocumented_visible_programs=undocumented,
        ),
        "summary": {
            "documentation_records": len(documentation),
            "current_documentation_records": sum(
                record["freshness"] == "current" for record in documentation
            ),
            "engine_capability_records": len(capabilities),
            "qualified_engine_capability_records": sum(
                c["evidence_state"] == "qualified" for c in capabilities
            ),
            "undocumented_visible_program_records": undocumented,
            "known_gap_records": len(known_gaps),
        },
    }


# ---------------------------------------------------------------------------
# Generated provider documentation (registry-only; no git, no HEAD facts)
# ---------------------------------------------------------------------------


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows), 3)
        for index in range(len(headers))
    ]

    def line(cells: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))
            + " |"
        )

    return [line(headers), line(["-" * w for w in widths]), *(line(r) for r in rows)]


def render_evidence_document(
    documentation_registry: Mapping[str, Any],
    capability_registry: Mapping[str, Any],
    *,
    engine_facts: Mapping[str, Mapping[str, str]],
) -> str:
    """Render the provider page from the raw registries; deterministic and pure.

    Freshness, hashes, URLs, and the source commit are HEAD-bound facts and are
    deliberately absent so the committed page never references its own commit.
    """
    lines = [
        "# Engine Capability Evidence and Documentation Governance",
        "",
        "<!-- AUTO-GENERATED — do not edit by hand. -->",
        "<!-- Regenerate with: python3 -m scripts.companion_evidence render-docs -->",
        "",
        f"Generated from `{CAPABILITY_REGISTRY_PATH.as_posix()}` and",
        f"`{DOCUMENTATION_REGISTRY_PATH.as_posix()}` (registry v{REGISTRY_VERSION},",
        "issue #9193). Every row states a software fact only: a capability is",
        "`qualified` when it names an exact test or artifact executed by the",
        "declared gate, and `unqualified` otherwise. Nothing here is scientific",
        "validation, a tolerance, a calculation, or engineering approval; those",
        "remain with their governed authorities (#9064, #9070). Review freshness,",
        "content hashes, and immutable links are exported per commit by",
        "`scripts/companion_catalog.py`, not copied into this page.",
        "",
        "## Engines",
    ]
    for engine in capability_registry["engines"]:
        facts = engine_facts[engine["id"]]
        availability = engine["runtime_availability"]
        lines += [
            "",
            f"### {facts['name']} (`{engine['id']}`)",
            "",
            f"- Support tier: `{facts['support_tier']}`",
            "- Runtime availability: `"
            + availability["state"]
            + "`"
            + (f" — {availability['reason']}" if availability["reason"] else ""),
            "- Documentation: "
            + ", ".join(
                f"`{doc_id}`" for doc_id in sorted(engine["documentation_ids"])
            ),
            "",
        ]
        rows = []
        for capability in sorted(engine["capabilities"], key=lambda item: item["id"]):
            evidence = "<br>".join(
                f"`{item['selector'] or item['path']}`"
                for item in sorted(capability["evidence"], key=lambda i: i["path"])
            )
            gates = ", ".join(
                sorted(
                    {
                        f"`{item['gate']}`"
                        for item in capability["evidence"]
                        if item["gate"]
                    }
                )
            )
            rows.append(
                [
                    f"`{capability['id']}`",
                    capability["title"],
                    capability["evidence_state"],
                    evidence or "—",
                    gates or "—",
                    capability["reason"] or "—",
                ]
            )
        lines += _table(
            ["Capability", "Title", "Evidence state", "Evidence", "Gate", "Reason"],
            rows,
        )
        limitations = sorted(
            {
                item
                for capability in engine["capabilities"]
                for item in capability["limitations"]
            }
        )
        if limitations:
            lines += ["", "Limitations:", ""]
            lines += [f"- {item}" for item in limitations]
    lines += ["", "## Known Gaps", ""]
    gaps = sorted(capability_registry["known_gaps"], key=lambda item: item["id"])
    if gaps:
        lines += _table(
            ["Gap", "Issue", "Scope", "Summary"],
            [
                [f"`{g['id']}`", f"#{g['issue']}", g["scope"], g["summary"]]
                for g in gaps
            ],
        )
    else:
        lines.append("No known gaps are declared.")
    lines += ["", "## Documentation Records", ""]
    lines += _table(
        ["ID", "Title", "Source", "Status", "Owner", "Last reviewed", "Review due"],
        [
            [
                f"`{record['id']}`",
                record["title"],
                f"`{record['source_path']}`" if record["source_path"] else "—",
                record["status"],
                record["owner"],
                record["last_reviewed"] or "—",
                record["review_due"] or "—",
            ]
            for record in sorted(
                documentation_registry["documentation"], key=lambda item: item["id"]
            )
        ],
    )
    return "\n".join(lines) + "\n"


def _read_registries(repo_root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    documentation = json.loads(
        (repo_root / DOCUMENTATION_REGISTRY_PATH).read_text(encoding="utf-8")
    )
    capability = json.loads(
        (repo_root / CAPABILITY_REGISTRY_PATH).read_text(encoding="utf-8")
    )
    return documentation, capability


def render_from_repository(repo_root: Path) -> str:
    """Render the provider page from the working-tree registries."""
    from scripts import companion_catalog

    documentation, capability = _read_registries(repo_root)
    return render_evidence_document(
        documentation, capability, engine_facts=companion_catalog.engine_facts()
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Render (or check) the generated provider documentation page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render-docs")
    render.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    rendered = render_from_repository(root)
    target = root / GENERATED_DOC_PATH
    if args.check:
        committed = target.read_text(encoding="utf-8") if target.is_file() else None
        if committed != rendered:
            sys.stderr.write(f"{GENERATED_DOC_PATH.as_posix()} is stale\n")
            return 1
        sys.stdout.write(f"{GENERATED_DOC_PATH.as_posix()} is current\n")
        return 0
    target.write_text(rendered, encoding="utf-8", newline="\n")
    sys.stdout.write(f"wrote {GENERATED_DOC_PATH.as_posix()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
