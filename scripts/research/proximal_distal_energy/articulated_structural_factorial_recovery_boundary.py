"""Cross-bind one attested structural recovery prefix without reading outcomes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from scripts.research.proximal_distal_energy.articulated_structural_factorial_recovery_registration import (
    validate_registered_slice,
)

BOUNDARY_SCHEMA = "articulated-structural-factorial-recovery-boundary/1.0.0"
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_SCHEMA = "articulated-structural-factorial-artifact-receipt/1.4.0"
_COLLECTION_SCHEMA = "articulated-structural-factorial-collection/1.3.0"
_RUNTIME_SCHEMA = "articulated-structural-factorial-runtime-replay-audit/1.0.0"
_CORRUPTION_SCHEMA = "articulated-structural-factorial-corruption-audit/1.0.0"
_PREFIX_SCHEMA = "articulated-structural-factorial-prefix-view/1.0.0"
_ENRICHMENT_SCHEMA = "articulated-structural-factorial-enrichment-audit/1.0.0"
_EXACT_ENRICHMENT_GATES = (
    "passes",
    "status_reproduction_exact",
    "completed_json_reproduction_exact",
    "legacy_array_reproduction_exact",
    "complete_evidence_sidecars_valid",
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(value: object, *, name: str) -> str:
    _require(
        isinstance(value, str) and _HEX_64.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256",
    )
    return str(value)


def _exact_schema(record: Mapping[str, object], expected: str, *, name: str) -> None:
    _require(record.get("schema_version") == expected, f"{name} schema is not exact")


def _successful_terminal(record: Mapping[str, object], *, name: str) -> None:
    _require(record.get("status") == "completed", f"{name} is not terminal")
    _require(record.get("conclusion") == "success", f"{name} did not succeed")


def _validate_collection_sources(
    *,
    sources: Sequence[object],
    case_stop: int,
    current_run_id: int,
    artifact_receipt_sha256: str,
) -> None:
    expected_start = 0
    last: Mapping[str, object] | None = None
    for raw in sources:
        source = _mapping(raw, name="collection source")
        requested = list(
            _sequence(source.get("requested_case_range"), name="requested range")
        )
        observed = list(
            _sequence(source.get("observed_case_range"), name="observed range")
        )
        requested_stop = requested[1] if len(requested) == 2 else None
        _require(
            len(requested) == 2
            and requested[0] == expected_start
            and isinstance(requested_stop, int)
            and not isinstance(requested_stop, bool),
            "collection sources are not gap-free",
        )
        if not isinstance(requested_stop, int) or isinstance(requested_stop, bool):
            raise ValueError("collection sources are not gap-free")
        _require(observed == requested, "collection observed range drifted")
        _require(source.get("run_conclusion") == "success", "source run failed")
        _require(
            source.get("artifact_receipt_schema") == _RECEIPT_SCHEMA,
            "collection source does not bind receipt 1.4",
        )
        expected_start = requested_stop
        last = source
    _require(expected_start == case_stop, "collection sources are not gap-free")
    if last is None:
        raise ValueError("collection must retain at least one source")
    _require(last.get("run_id") == current_run_id, "current run is not last source")
    _require(
        last.get("artifact_receipt_sha256") == artifact_receipt_sha256,
        "collection does not bind the exact current receipt",
    )


def audit_recovery_boundary(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    registration: Mapping[str, object],
    runtime_replay_audit: Mapping[str, object],
    artifact_receipt: Mapping[str, object],
    artifact_receipt_sha256: str,
    corruption_audit: Mapping[str, object],
    collection_manifest: Mapping[str, object],
    legacy_prefix_manifest: Mapping[str, object],
    enrichment_audit: Mapping[str, object],
    case_start: int,
    case_stop: int,
    input_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Return one outcome-blind exact boundary or fail closed."""

    validate_registered_slice(
        manifest=registration,
        plan=plan,
        launch=launch,
        case_start=case_start,
        case_stop=case_stop,
    )
    plan_identity = launch.get("plan_sha256")
    execution_revision = launch.get("execution_revision")
    _sha(artifact_receipt_sha256, name="artifact_receipt_sha256")
    for name, digest in input_sha256.items():
        _sha(digest, name=f"input_sha256.{name}")

    _exact_schema(runtime_replay_audit, _RUNTIME_SCHEMA, name="runtime replay")
    _require(
        runtime_replay_audit.get("classification") == "runtime_replay_contract_exact",
        "runtime replay is not exact",
    )
    runtime_gates = _mapping(runtime_replay_audit.get("gates"), name="runtime gates")
    _require(runtime_gates.get("passes") is True, "runtime replay gate failed")
    runtime_boundary = _mapping(
        runtime_replay_audit.get("claim_boundary"), name="runtime claim boundary"
    )
    _require(
        runtime_boundary.get("scientific_outcomes_inspected") is False,
        "runtime evidence is not outcome-blind",
    )

    _exact_schema(artifact_receipt, _RECEIPT_SCHEMA, name="receipt 1.4")
    _require(
        artifact_receipt.get("execution_revision") == execution_revision,
        "receipt execution identity drifted",
    )
    _require(
        artifact_receipt.get("requested_case_range") == [case_start, case_stop],
        "receipt range does not match the registered slice",
    )
    run = _mapping(artifact_receipt.get("run"), name="receipt run")
    job = _mapping(artifact_receipt.get("job"), name="receipt job")
    _successful_terminal(run, name="receipt run")
    _successful_terminal(job, name="receipt job")
    run_id = run.get("id")
    if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
        raise ValueError("receipt run id is invalid")

    _exact_schema(corruption_audit, _CORRUPTION_SCHEMA, name="corruption audit")
    corruption_identity = _mapping(
        corruption_audit.get("identity"), name="corruption identity"
    )
    _require(
        corruption_identity.get("plan_sha256") == plan_identity
        and corruption_identity.get("execution_revision") == execution_revision,
        "corruption audit identity drifted",
    )
    sentinel = _mapping(corruption_audit.get("sentinel"), name="corruption sentinel")
    _require(sentinel.get("passes") is True, "corruption killswitch failed")
    _require(
        sentinel.get("source_checkpoint_unchanged") is True,
        "corruption audit changed the source checkpoint",
    )

    _exact_schema(collection_manifest, _COLLECTION_SCHEMA, name="collection")
    _require(
        collection_manifest.get("plan_sha256") == plan_identity
        and collection_manifest.get("execution_revision") == execution_revision,
        "collection identity drifted",
    )
    _require(
        collection_manifest.get("combined_checkpoint_count") == case_stop
        and collection_manifest.get("next_missing_case_index") == case_stop,
        "collection does not expose the exact cumulative prefix",
    )
    collection_sources = _sequence(
        collection_manifest.get("sources"), name="collection sources"
    )
    _validate_collection_sources(
        sources=collection_sources,
        case_stop=case_stop,
        current_run_id=run_id,
        artifact_receipt_sha256=artifact_receipt_sha256,
    )

    _exact_schema(legacy_prefix_manifest, _PREFIX_SCHEMA, name="legacy prefix")
    _require(
        legacy_prefix_manifest.get("prefix_case_stop_exclusive") == case_stop,
        "legacy prefix does not match the cumulative boundary",
    )
    source_count = legacy_prefix_manifest.get("source_checkpoint_count")
    _require(
        isinstance(source_count, int)
        and not isinstance(source_count, bool)
        and source_count >= case_stop,
        "legacy prefix source is incomplete",
    )

    _exact_schema(enrichment_audit, _ENRICHMENT_SCHEMA, name="enrichment audit")
    enrichment_identity = _mapping(
        enrichment_audit.get("identity"), name="enrichment identity"
    )
    _require(
        enrichment_identity.get("enriched_plan_sha256") == plan_identity
        and enrichment_identity.get("enriched_execution_revision") == execution_revision
        and enrichment_identity.get("legacy_prefix_count") == case_stop,
        "enrichment audit identity drifted",
    )
    enrichment_gates = _mapping(enrichment_audit.get("gates"), name="enrichment gates")
    _require(
        all(enrichment_gates.get(name) is True for name in _EXACT_ENRICHMENT_GATES),
        "enrichment audit exact gate failed",
    )
    enrichment_boundary = _mapping(
        enrichment_audit.get("claim_boundary"), name="enrichment claim boundary"
    )
    _require(
        enrichment_boundary.get("scientific_outcomes_inspected") is False
        and enrichment_boundary.get("human_or_coaching_inference") is False,
        "enrichment evidence is not outcome-blind",
    )

    next_range: list[int] | None = None
    registered_slices = _sequence(registration.get("slices"), name="registered slices")
    for item in registered_slices:
        record = _mapping(item, name="registered slice")
        if record.get("case_start") == case_stop:
            next_range = [case_stop, int(record["case_stop"])]
            break
    return {
        "schema_version": BOUNDARY_SCHEMA,
        "classification": "attested_prefix_boundary_exact",
        "identity": {
            "plan_sha256": plan_identity,
            "execution_revision": execution_revision,
            "run_id": run_id,
            "registered_slice": [case_start, case_stop],
            "cumulative_prefix": [0, case_stop],
        },
        "input_sha256": dict(sorted(input_sha256.items())),
        "gates": {
            "registration_exact": True,
            "runtime_replay_exact": True,
            "receipt_1_4_exact": True,
            "corruption_killswitch_passes": True,
            "collection_gap_free": True,
            "legacy_projection_exact_boundary": True,
            "enrichment_exact": True,
            "passes": True,
        },
        "next_slice": {
            "registered_range": next_range,
            "authorized": False,
            "requirement": "separate_issue_preregistration_after_boundary_retention",
        },
        "claim_boundary": {
            "scientific_outcomes_inspected": False,
            "effect_direction_interpreted": False,
            "campaign_promotion_authorized": False,
            "human_or_coaching_inference_authorized": False,
        },
    }


def _read_mapping(path: Path) -> Mapping[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name=str(path))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Write one atomic post-run boundary audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "launch",
        "registration",
        "runtime-replay-audit",
        "artifact-receipt",
        "corruption-audit",
        "collection-manifest",
        "legacy-prefix-manifest",
        "enrichment-audit",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--case-start", type=int, required=True)
    parser.add_argument("--case-stop", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence_paths = {
        "registration": args.registration,
        "runtime_replay_audit": args.runtime_replay_audit,
        "artifact_receipt": args.artifact_receipt,
        "corruption_audit": args.corruption_audit,
        "collection_manifest": args.collection_manifest,
        "legacy_prefix_manifest": args.legacy_prefix_manifest,
        "enrichment_audit": args.enrichment_audit,
    }
    result = audit_recovery_boundary(
        plan=_read_mapping(args.plan),
        launch=_read_mapping(args.launch),
        registration=_read_mapping(args.registration),
        runtime_replay_audit=_read_mapping(args.runtime_replay_audit),
        artifact_receipt=_read_mapping(args.artifact_receipt),
        artifact_receipt_sha256=_file_sha256(args.artifact_receipt),
        corruption_audit=_read_mapping(args.corruption_audit),
        collection_manifest=_read_mapping(args.collection_manifest),
        legacy_prefix_manifest=_read_mapping(args.legacy_prefix_manifest),
        enrichment_audit=_read_mapping(args.enrichment_audit),
        case_start=args.case_start,
        case_stop=args.case_stop,
        input_sha256={
            name: _file_sha256(path) for name, path in evidence_paths.items()
        },
    )
    _write_atomic(args.output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["BOUNDARY_SCHEMA", "audit_recovery_boundary", "main"]
