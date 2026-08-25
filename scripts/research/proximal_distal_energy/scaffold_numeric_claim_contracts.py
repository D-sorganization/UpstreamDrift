"""Scaffold reviewable numeric-claim contracts without granting authority.

This maintainer tool ranks matching numbers already present in declared JSON
evidence.  If no such number exists, it creates an explicitly weaker reported
or protocol/notation ledger record.  Its output remains a draft until reviewed
through the protected pull request that registers it; the release path never
runs this scaffold automatically.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
import math
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.claim_numeric_audit import (
    NUMERIC_LITERAL_PATTERN,
    extract_numeric_literals,
)
from scripts.research.proximal_distal_energy.register_numeric_claim_evidence import (
    ARTICLE,
    REGISTRY,
    statement_digest,
)
from scripts.research.proximal_distal_energy.numeric_contract_scaffold_matching import (
    SCALES,
    _has_semantic_pointer_match,
    _pointer_matches_declared_quantity,
    _rank_candidate,
    _rounding_tolerance,
    _scale_is_semantically_valid,
)


CONTRACT_PATH = ARTICLE / "data/claim_numeric_contracts.json"
REPORTED_PATH = ARTICLE / "data/claim_numeric_reported_values.json"
REPORTED_REL = str(REPORTED_PATH.relative_to(ARTICLE.parents[2])).replace("\\", "/")
REVIEWED_POINTER_OVERRIDES = {
    ("PD-CLAIM-099", "80#1"): {
        "artifact": (
            "docs/research/proximal_distal_energy_transfer/data/"
            "counterfactual_ensemble.json"
        ),
        "json_pointer": "/rows/66/horizon_s",
        "scale": 1000.0,
    },
}


def _walk_numbers(value: object, pointer: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(value, dict):
        for key, item in value.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            yield from _walk_numbers(item, f"{pointer}/{token}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_numbers(item, f"{pointer}/{index}")
    elif isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            yield pointer, number


def _json_numbers(
    claim: dict[str, Any], root: Path
) -> tuple[list[tuple[str, str, float]], dict[str, int]]:
    result: list[tuple[str, str, float]] = []
    artifact_order: dict[str, int] = {}
    for index, artifact in enumerate(claim["evidence_artifacts"]):
        if not isinstance(artifact, str):
            continue
        if artifact.startswith(("http://", "https://")) or "#" in artifact:
            continue
        if Path(artifact).name.startswith("claim_numeric_"):
            continue
        if not artifact.lower().endswith(".json"):
            continue
        path = root / artifact
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        artifact_order[artifact] = index
        result.extend(
            (artifact, pointer, value) for pointer, value in _walk_numbers(document)
        )
    return result, artifact_order


def _original_artifacts(claim: dict[str, Any]) -> list[str]:
    return [
        item
        for item in claim["evidence_artifacts"]
        if isinstance(item, str) and not Path(item).name.startswith("claim_numeric_")
    ]


def _override_entry(
    claim_id: str, literal: dict[str, Any], tolerance: float
) -> dict[str, Any] | None:
    override = REVIEWED_POINTER_OVERRIDES.get((claim_id, str(literal["literal_id"])))
    if override is None:
        return None
    return {
        "literal_id": literal["literal_id"],
        "artifact": override["artifact"],
        "json_pointer": override["json_pointer"],
        "evidence_scope": "local_json_value",
        "scale": override["scale"],
        "offset": 0.0,
        "atol": override.get("atol", tolerance),
        "rtol": 0.0,
    }


def _reviewed_numeric_authority() -> dict[
    tuple[str, str, str], tuple[dict[str, Any], dict[str, Any] | None]
]:
    """Load protected-review entries keyed by claim, statement, and literal."""

    if not CONTRACT_PATH.is_file() or not REPORTED_PATH.is_file():
        return {}
    contracts = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    reported = json.loads(REPORTED_PATH.read_text(encoding="utf-8"))
    authority = {}
    for contract in contracts.get("claims", []):
        claim_id = str(contract.get("claim_id", ""))
        statement_sha256 = str(contract.get("statement_sha256", ""))
        for entry in contract.get("numeric_evidence", []):
            retained = dict(entry)
            record = None
            if retained.get("artifact") == REPORTED_REL:
                parts = str(retained.get("json_pointer", "")).strip("/").split("/")
                if len(parts) != 4 or parts[:2] != ["claims", claim_id]:
                    raise ValueError(
                        f"{claim_id}: reviewed reported pointer is invalid"
                    )
                record = dict(reported["claims"][claim_id][int(parts[2])])
            key = (claim_id, statement_sha256, str(retained.get("literal_id", "")))
            authority[key] = (retained, record)
    return authority


def _retain_reviewed_entry(
    *,
    claim: dict[str, Any],
    statement_sha256: str,
    literal: dict[str, Any],
    authority: dict[tuple[str, str, str], tuple[dict[str, Any], dict[str, Any] | None]],
    reported: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    key = (claim["claim_id"], statement_sha256, str(literal["literal_id"]))
    reviewed = authority.get(key)
    if reviewed is None:
        return None
    entry, record = reviewed
    retained = dict(entry)
    if record is not None:
        records = reported.setdefault(claim["claim_id"], [])
        records.append(dict(record))
        retained["json_pointer"] = (
            f"/claims/{claim['claim_id']}/{len(records) - 1}/value"
        )
    return retained


def _local_numeric_entry(
    *,
    literal: dict[str, Any],
    literal_start: int,
    literal_end: int,
    statement: str,
    numbers: list[tuple[str, str, float]],
    artifact_order: dict[str, int],
    tolerance: float,
) -> dict[str, Any] | None:
    expected = float(literal["value"])
    text_length = len(str(literal["text"]))
    context = statement[max(0, literal_start - 60) : literal_start + text_length + 60]
    before = statement[max(0, literal_start - 40) : literal_start]
    after = statement[literal_end : literal_end + 40]
    candidates = [
        (artifact, pointer, value, scale)
        for artifact, pointer, value in numbers
        for scale in SCALES
        if _scale_is_semantically_valid(scale, pointer, context)
        and abs(value * scale - expected) <= tolerance
    ]
    if not candidates:
        return None
    artifact, pointer, _, scale = min(
        candidates,
        key=lambda item: _rank_candidate(
            item,
            statement=statement,
            context=context,
            artifact_order=artifact_order,
            expected=expected,
        ),
    )
    if not _has_semantic_pointer_match(pointer, context):
        return None
    if not _pointer_matches_declared_quantity(pointer, before=before, after=after):
        return None
    return {
        "literal_id": literal["literal_id"],
        "artifact": artifact,
        "json_pointer": pointer,
        "evidence_scope": "local_json_value",
        "scale": scale,
        "offset": 0.0,
        "atol": tolerance,
        "rtol": 0.0,
    }


def _reported_numeric_entry(
    *,
    claim: dict[str, Any],
    literal: dict[str, Any],
    original_artifacts: list[str],
    reported: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    external = [
        item for item in original_artifacts if item.startswith(("http://", "https://"))
    ]
    if external:
        scope = "reported_external_value"
    elif "identity" in str(claim.get("classification", "")):
        scope = "registered_protocol_or_notation"
    else:
        scope = "registered_claim_value_not_independently_recomputed"
    records = reported.setdefault(claim["claim_id"], [])
    record_index = len(records)
    records.append(
        {
            "literal_id": literal["literal_id"],
            "text": literal["text"],
            "value": float(literal["value"]),
            "evidence_scope": scope,
            "source_references": external or original_artifacts,
            "source_locations": claim["source_locations"],
            "independent_validation": False,
            "boundary": (
                "Reported-value transcription; the numeric pointer gate does not "
                "independently reproduce the cited study."
                if external
                else "Registered claim value lacking an unambiguous semantic JSON "
                "path; the numeric pointer gate does not independently recompute "
                "physics."
            ),
        }
    )
    return {
        "literal_id": literal["literal_id"],
        "artifact": REPORTED_REL,
        "json_pointer": f"/claims/{claim['claim_id']}/{record_index}/value",
        "evidence_scope": scope,
        "scale": 1.0,
        "offset": 0.0,
        "atol": 0.0,
        "rtol": 0.0,
    }


def _comparison_contract(claim_id: str) -> list[dict[str, Any]] | None:
    if claim_id != "PD-CLAIM-127":
        return None
    return [
        {
            "comparison_id": "spatial-forward-contact-cross-engine-couple",
            "artifact": (
                "docs/research/proximal_distal_energy_transfer/data/"
                "claim_numeric_comparison_evidence.json"
            ),
            "reference_pointer": "/spatial_forward_contact/reference",
            "candidate_pointer": "/spatial_forward_contact/candidate",
            "require_nondegenerate": True,
            "atol": 0.004,
            "rtol": 0.002,
        }
    ]


def _build_claim_contract(
    claim: dict[str, Any],
    root: Path,
    reported: dict[str, list[dict[str, Any]]],
    reviewed_authority: dict[
        tuple[str, str, str], tuple[dict[str, Any], dict[str, Any] | None]
    ],
) -> dict[str, Any] | None:
    statement = claim["statement"]
    statement_sha256 = statement_digest(statement)
    literals = extract_numeric_literals(statement)
    literal_matches = list(NUMERIC_LITERAL_PATTERN.finditer(statement))
    if len(literals) != len(literal_matches):
        raise ValueError(f"{claim['claim_id']}: literal extraction drift")
    if not literals:
        return None
    numbers, artifact_order = _json_numbers(claim, root)
    original_artifacts = _original_artifacts(claim)
    entries: list[dict[str, Any]] = []
    for literal, literal_match in zip(literals, literal_matches, strict=True):
        tolerance = _rounding_tolerance(str(literal["text"]))
        entry = _retain_reviewed_entry(
            claim=claim,
            statement_sha256=statement_sha256,
            literal=literal,
            authority=reviewed_authority,
            reported=reported,
        )
        if entry is None:
            entry = _override_entry(claim["claim_id"], literal, tolerance)
        if entry is None:
            entry = _local_numeric_entry(
                literal=literal,
                literal_start=literal_match.start(),
                literal_end=literal_match.end(),
                statement=statement,
                numbers=numbers,
                artifact_order=artifact_order,
                tolerance=tolerance,
            )
        if entry is None:
            entry = _reported_numeric_entry(
                claim=claim,
                literal=literal,
                original_artifacts=original_artifacts,
                reported=reported,
            )
        entries.append(entry)
    contract: dict[str, Any] = {
        "claim_id": claim["claim_id"],
        "statement_sha256": statement_sha256,
        "numeric_evidence": entries,
    }
    comparison = _comparison_contract(claim["claim_id"])
    if comparison is not None:
        contract["numeric_comparisons"] = comparison
    return contract


def build_scaffold(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    reviewed_authority = _reviewed_numeric_authority()
    contracts: list[dict[str, Any]] = []
    reported: dict[str, list[dict[str, Any]]] = {}
    for claim in registry["claims"]:
        contract = _build_claim_contract(claim, root, reported, reviewed_authority)
        if contract is not None:
            contracts.append(contract)
    contract_document = {
        "schema_version": "claim-numeric-contract-v1",
        "review_process": "protected_pull_request",
        "generation_boundary": (
            "Scaffolded candidates are not release authority until reviewed and "
            "registered through the protected pull request."
        ),
        "claims": contracts,
    }
    reported_document = {
        "schema_version": "claim-numeric-reported-values-v1",
        "independent_validation": False,
        "boundary": (
            "This ledger makes reported and protocol/notation values addressable. "
            "It does not independently validate cited studies or recompute model physics."
        ),
        "claims": reported,
    }
    return contract_document, reported_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-draft",
        action="store_true",
        help="write scaffolded contracts for explicit protected review",
    )
    args = parser.parse_args()
    contracts, reported = build_scaffold(ARTICLE.parents[2])
    summary = {
        "numeric_claim_count": len(contracts["claims"]),
        "reported_claim_count": len(reported["claims"]),
        "reported_literal_count": sum(
            len(items) for items in reported["claims"].values()
        ),
    }
    if args.write_draft:
        CONTRACT_PATH.write_text(
            json.dumps(contracts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        REPORTED_PATH.write_text(
            json.dumps(reported, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
