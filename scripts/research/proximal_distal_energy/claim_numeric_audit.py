"""Fail-closed numeric traceability for proximal-distal claim records.

This module checks a deliberately narrow contract: numerical literals in a
claim must agree with values addressed in declared JSON evidence.  It does not
establish that the JSON is scientifically correct.  Independent recomputation
from source arrays or equations is a separate release requirement.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, TypedDict

NUMERIC_LITERAL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])(?:(?<!-)[-+])?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:e[-+]?\d+)?(?![A-Za-z])",
    re.IGNORECASE,
)

EVIDENCE_SCOPES = frozenset(
    {
        "local_json_value",
        "reported_external_value",
        "registered_protocol_or_notation",
        "registered_claim_value_not_independently_recomputed",
    }
)


class ClaimNumericAuditResult(TypedDict):
    """Typed aggregate returned by one claim's numeric audit."""

    claim_id: str
    literal_count: int
    verified_count: int
    nondegenerate_comparison_count: int
    evidence_scope_counts: dict[str, int]


def extract_numeric_literals(statement: str) -> list[dict[str, str | float]]:
    """Return every numeric literal with a stable per-spelling occurrence ID."""
    if not isinstance(statement, str):
        raise TypeError("statement must be a string")
    occurrences: Counter[str] = Counter()
    literals: list[dict[str, str | float]] = []
    for match in NUMERIC_LITERAL_PATTERN.finditer(statement):
        text = match.group(0)
        occurrences[text] += 1
        value = float(text.replace(",", ""))
        if not math.isfinite(value):
            raise ValueError(f"non-finite numeric literal: {text!r}")
        literals.append(
            {
                "literal_id": f"{text}#{occurrences[text]}",
                "text": text,
                "value": value,
            }
        )
    return literals


def _finite_number(value: object, *, field: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context}: {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: {field} must be a finite number")
    return result


def _resolve_json_pointer(document: object, pointer: object, *, context: str) -> object:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"{context}: json_pointer must be an absolute JSON Pointer")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, list):
                if not token.isdigit():
                    raise KeyError(token)
                current = current[int(token)]
            elif isinstance(current, dict):
                current = current[token]
            else:
                raise KeyError(token)
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"{context}: JSON Pointer {pointer!r} does not resolve"
            ) from exc
    return current


def _evidence_document(
    artifact: object,
    *,
    claim_id: str,
    declared_artifacts: set[str],
    repository_root: Path,
    cache: dict[str, object],
) -> object:
    if not isinstance(artifact, str) or artifact not in declared_artifacts:
        raise ValueError(
            f"{claim_id}: numeric artifact {artifact!r} is not declared in "
            "evidence_artifacts"
        )
    if "#" in artifact or artifact.startswith(("http://", "https://")):
        raise ValueError(f"{claim_id}: numeric evidence must be a local JSON file")
    path = (repository_root / artifact).resolve()
    if not path.is_relative_to(repository_root) or path.suffix.lower() != ".json":
        raise ValueError(f"{claim_id}: numeric evidence must be contained JSON")
    if not path.is_file():
        raise ValueError(f"{claim_id}: missing numeric evidence artifact {artifact!r}")
    if artifact not in cache:
        cache[artifact] = json.loads(path.read_text(encoding="utf-8"))
    return cache[artifact]


def _validate_literal_entry(
    entry: object,
    *,
    claim_id: str,
    literal: dict[str, str | float],
    declared_artifacts: set[str],
    repository_root: Path,
    cache: dict[str, object],
) -> None:
    context = f"{claim_id}/{literal['literal_id']}"
    if not isinstance(entry, dict):
        raise ValueError(f"{context}: numeric_evidence entry must be an object")
    evidence_scope = entry.get("evidence_scope")
    if evidence_scope not in EVIDENCE_SCOPES:
        raise ValueError(
            f"{context}: evidence_scope must be one of {sorted(EVIDENCE_SCOPES)}"
        )
    document = _evidence_document(
        entry.get("artifact"),
        claim_id=claim_id,
        declared_artifacts=declared_artifacts,
        repository_root=repository_root,
        cache=cache,
    )
    actual = _resolve_json_pointer(document, entry.get("json_pointer"), context=context)
    actual_value = _finite_number(actual, field="resolved value", context=context)
    if evidence_scope != "local_json_value":
        pointer = entry.get("json_pointer")
        if not isinstance(pointer, str) or not pointer.endswith("/value"):
            raise ValueError(f"{context}: reported evidence pointer must end in /value")
        record = _resolve_json_pointer(
            document, pointer.rsplit("/", 1)[0], context=context
        )
        if not isinstance(record, dict):
            raise ValueError(f"{context}: reported evidence record must be an object")
        if record.get("literal_id") != literal["literal_id"]:
            raise ValueError(f"{context}: reported evidence literal_id mismatch")
        if record.get("evidence_scope") != evidence_scope:
            raise ValueError(f"{context}: reported evidence scope mismatch")
        if record.get("independent_validation") is not False:
            raise ValueError(
                f"{context}: reported evidence must deny independent validation"
            )
        for field in ("source_references", "source_locations"):
            values = record.get(field)
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(item, str) and item.strip() for item in values)
            ):
                raise ValueError(f"{context}: reported evidence needs {field}")
        boundary = record.get("boundary")
        if not isinstance(boundary, str) or not boundary.strip():
            raise ValueError(f"{context}: reported evidence needs boundary")
    scale = _finite_number(entry.get("scale"), field="scale", context=context)
    offset = _finite_number(entry.get("offset"), field="offset", context=context)
    atol = _finite_number(entry.get("atol"), field="atol", context=context)
    rtol = _finite_number(entry.get("rtol"), field="rtol", context=context)
    if atol < 0.0 or rtol < 0.0:
        raise ValueError(f"{context}: atol and rtol must be non-negative")
    transformed = actual_value * scale + offset
    expected = float(literal["value"])
    if not math.isclose(transformed, expected, abs_tol=atol, rel_tol=rtol):
        raise ValueError(
            f"{context}: numeric evidence mismatch; claim={expected!r}, "
            f"resolved={actual_value!r}, transformed={transformed!r}"
        )


def _flatten_numeric(value: object, *, context: str) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item, context=context))
        return flattened
    return [_finite_number(value, field="comparison value", context=context)]


def _validate_comparison(
    comparison: object,
    *,
    claim_id: str,
    declared_artifacts: set[str],
    repository_root: Path,
    cache: dict[str, object],
) -> None:
    if not isinstance(comparison, dict):
        raise ValueError(f"{claim_id}: numeric comparison must be an object")
    comparison_id = comparison.get("comparison_id")
    if not isinstance(comparison_id, str) or not comparison_id.strip():
        raise ValueError(f"{claim_id}: comparison_id must be non-empty")
    context = f"{claim_id}/{comparison_id}"
    document = _evidence_document(
        comparison.get("artifact"),
        claim_id=claim_id,
        declared_artifacts=declared_artifacts,
        repository_root=repository_root,
        cache=cache,
    )
    reference = _flatten_numeric(
        _resolve_json_pointer(
            document, comparison.get("reference_pointer"), context=context
        ),
        context=context,
    )
    candidate = _flatten_numeric(
        _resolve_json_pointer(
            document, comparison.get("candidate_pointer"), context=context
        ),
        context=context,
    )
    if not reference or len(reference) != len(candidate):
        raise ValueError(f"{context}: comparison arrays must be non-empty and aligned")
    atol = _finite_number(comparison.get("atol"), field="atol", context=context)
    rtol = _finite_number(comparison.get("rtol"), field="rtol", context=context)
    if atol < 0.0 or rtol < 0.0:
        raise ValueError(f"{context}: atol and rtol must be non-negative")
    if not all(
        math.isclose(left, right, abs_tol=atol, rel_tol=rtol)
        for left, right in zip(reference, candidate, strict=True)
    ):
        raise ValueError(f"{context}: parity comparison exceeds tolerance")
    if comparison.get("require_nondegenerate") is not True:
        raise ValueError(f"{context}: require_nondegenerate must be true")
    if all(left == right for left, right in zip(reference, candidate, strict=True)):
        raise ValueError(f"{context}: degenerate exact-zero comparison")


def audit_claim_numeric_evidence(
    claim: dict[str, object], *, repository_root: Path
) -> ClaimNumericAuditResult:
    """Validate all numeric literals and registered parity controls for one claim."""
    claim_id = claim.get("claim_id")
    if not isinstance(claim_id, str) or not claim_id.strip():
        raise ValueError("claim_id must be a non-empty string")
    statement = claim.get("statement")
    if not isinstance(statement, str):
        raise ValueError(f"{claim_id}: statement must be a string")
    artifacts = claim.get("evidence_artifacts")
    if not isinstance(artifacts, list) or not all(
        isinstance(item, str) for item in artifacts
    ):
        raise ValueError(f"{claim_id}: evidence_artifacts must be a string list")
    declared_artifacts = set(artifacts)
    literals = extract_numeric_literals(statement)
    entries = claim.get("numeric_evidence", [])
    if not isinstance(entries, list):
        raise ValueError(f"{claim_id}: numeric_evidence must be a list")
    entry_map: dict[str, object] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("literal_id"), str):
            raise ValueError(
                f"{claim_id}: each numeric_evidence entry needs literal_id"
            )
        literal_id = entry["literal_id"]
        if literal_id in entry_map:
            raise ValueError(f"{claim_id}: duplicate numeric evidence {literal_id!r}")
        entry_map[literal_id] = entry
    expected_ids = {str(literal["literal_id"]) for literal in literals}
    actual_ids = set(entry_map)
    if expected_ids != actual_ids:
        raise ValueError(
            f"{claim_id}: numeric evidence coverage mismatch; "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )
    cache: dict[str, object] = {}
    for literal in literals:
        _validate_literal_entry(
            entry_map[str(literal["literal_id"])],
            claim_id=claim_id,
            literal=literal,
            declared_artifacts=declared_artifacts,
            repository_root=repository_root.resolve(),
            cache=cache,
        )
    comparisons = claim.get("numeric_comparisons", [])
    if not isinstance(comparisons, list):
        raise ValueError(f"{claim_id}: numeric_comparisons must be a list")
    for comparison in comparisons:
        _validate_comparison(
            comparison,
            claim_id=claim_id,
            declared_artifacts=declared_artifacts,
            repository_root=repository_root.resolve(),
            cache=cache,
        )
    scope_counts = Counter(str(entry["evidence_scope"]) for entry in entries)
    return {
        "claim_id": claim_id,
        "literal_count": len(literals),
        "verified_count": len(literals),
        "nondegenerate_comparison_count": len(comparisons),
        "evidence_scope_counts": dict(sorted(scope_counts.items())),
    }


def audit_registry_numeric_evidence(
    registry_path: Path, *, repository_root: Path
) -> dict[str, object]:
    """Require complete numeric evidence for every claim in a registry."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    claims = registry.get("claims")
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    numeric_claim_count = 0
    literal_count = 0
    verified_count = 0
    comparison_count = 0
    scope_counts: Counter[str] = Counter()
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("each claim must be an object")
        result = audit_claim_numeric_evidence(
            claim, repository_root=repository_root.resolve()
        )
        count = result["literal_count"]
        numeric_claim_count += int(count > 0)
        literal_count += count
        verified_count += result["verified_count"]
        comparison_count += result["nondegenerate_comparison_count"]
        claim_scope_counts = result["evidence_scope_counts"]
        scope_counts.update(claim_scope_counts)
    return {
        "claim_count": len(claims),
        "numeric_claim_count": numeric_claim_count,
        "numeric_literal_count": literal_count,
        "verified_numeric_literal_count": verified_count,
        "nondegenerate_comparison_count": comparison_count,
        "evidence_scope_counts": dict(sorted(scope_counts.items())),
        "completion_status": "complete",
    }
