"""Audit an evidence-enriched replay against a contiguous legacy prefix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_structural_factorial_evidence import (
    EVIDENCE_SIDECAR_SCHEMA,
    validate_structural_evidence_arrays,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    StructuralCheckpoint,
    build_registered_cases,
    load_available_checkpoints,
    plan_sha256,
)

AUDIT_SCHEMA = "articulated-structural-factorial-enrichment-audit/1.0.0"


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _outcome(checkpoint: StructuralCheckpoint) -> Mapping[str, Any]:
    payload = _mapping(
        json.loads(checkpoint.path.read_text(encoding="utf-8")), name="checkpoint"
    )
    return _mapping(payload.get("outcome"), name="checkpoint.outcome")


def _result_without_evidence_schema(checkpoint: StructuralCheckpoint) -> dict[str, Any]:
    result = dict(_mapping(_outcome(checkpoint).get("result"), name="result"))
    result.pop("evidence_sidecar_schema", None)
    return result


def _arrays(checkpoint: StructuralCheckpoint) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(checkpoint.path.with_suffix(".npz"), allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _contiguous_prefix(
    *,
    plan: Mapping[str, object],
    checkpoints: tuple[StructuralCheckpoint, ...],
    name: str,
) -> int:
    registered = build_registered_cases(plan)
    index = {case.case_key: position for position, case in enumerate(registered)}
    observed = [index[checkpoint.case.case_key] for checkpoint in checkpoints]
    if observed != list(range(len(observed))):
        raise ValueError(f"{name} checkpoints are not a contiguous prefix from zero")
    return len(observed)


def audit_enrichment_replay(
    *,
    legacy_plan: Mapping[str, object],
    legacy_launch: Mapping[str, object],
    legacy_checkpoint_dir: Path,
    enriched_plan: Mapping[str, object],
    enriched_launch: Mapping[str, object],
    enriched_checkpoint_dir: Path,
) -> dict[str, object]:
    """Require exact legacy reproduction plus the complete evidence contract."""

    legacy_cases = build_registered_cases(legacy_plan)
    enriched_cases = build_registered_cases(enriched_plan)
    if [case.case_key for case in legacy_cases] != [
        case.case_key for case in enriched_cases
    ]:
        raise ValueError("legacy and enriched plans do not register identical cases")
    legacy = load_available_checkpoints(
        plan=legacy_plan,
        launch=legacy_launch,
        checkpoint_dir=legacy_checkpoint_dir,
    )
    enriched = load_available_checkpoints(
        plan=enriched_plan,
        launch=enriched_launch,
        checkpoint_dir=enriched_checkpoint_dir,
    )
    legacy_count = _contiguous_prefix(
        plan=legacy_plan, checkpoints=legacy, name="legacy"
    )
    enriched_count = _contiguous_prefix(
        plan=enriched_plan, checkpoints=enriched, name="enriched"
    )
    if legacy_count == 0 or enriched_count < legacy_count:
        raise ValueError("enriched replay does not cover the complete legacy prefix")
    enriched_by_key = {checkpoint.case.case_key: checkpoint for checkpoint in enriched}
    completed_count = 0
    unavailable_count = 0
    compared_array_count = 0
    for legacy_checkpoint in legacy:
        enriched_checkpoint = enriched_by_key[legacy_checkpoint.case.case_key]
        if enriched_checkpoint.status != legacy_checkpoint.status:
            raise ValueError("enriched replay changes a legacy checkpoint status")
        if legacy_checkpoint.status == "completed":
            legacy_result = _result_without_evidence_schema(legacy_checkpoint)
            enriched_result = _result_without_evidence_schema(enriched_checkpoint)
            if enriched_result != legacy_result:
                raise ValueError("enriched replay changes a legacy JSON result")
            legacy_arrays = _arrays(legacy_checkpoint)
            enriched_arrays = _arrays(enriched_checkpoint)
            validate_structural_evidence_arrays(enriched_arrays)
            if (
                _mapping(
                    _outcome(enriched_checkpoint).get("result"), name="result"
                ).get("evidence_sidecar_schema")
                != EVIDENCE_SIDECAR_SCHEMA
            ):
                raise ValueError("enriched replay lacks the evidence sidecar schema")
            for name, legacy_array in legacy_arrays.items():
                if name not in enriched_arrays or not np.array_equal(
                    enriched_arrays[name], legacy_array
                ):
                    raise ValueError(
                        f"enriched replay changes legacy sidecar array: {name}"
                    )
                compared_array_count += 1
            completed_count += 1
        elif legacy_checkpoint.status == "unavailable":
            unavailable_count += 1
    identity_payload = {
        "legacy_plan_sha256": plan_sha256(legacy_plan),
        "legacy_execution_revision": legacy_launch["execution_revision"],
        "enriched_plan_sha256": plan_sha256(enriched_plan),
        "enriched_execution_revision": enriched_launch["execution_revision"],
        "legacy_prefix_count": legacy_count,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": AUDIT_SCHEMA,
        "identity": {**identity_payload, "audit_identity_sha256": identity},
        "legacy_prefix": {
            "checkpoint_count": legacy_count,
            "completed_count": completed_count,
            "unavailable_count": unavailable_count,
            "compared_array_count": compared_array_count,
        },
        "gates": {
            "case_registry_identical": True,
            "status_reproduction_exact": True,
            "completed_json_reproduction_exact": True,
            "legacy_array_reproduction_exact": True,
            "complete_evidence_sidecars_valid": True,
            "passes": True,
        },
        "claim_boundary": {
            "scientific_outcomes_inspected": False,
            "legacy_prefix_promotable_by_itself": False,
            "human_or_coaching_inference": False,
        },
    }


def main() -> None:
    """Write one deterministic audit for a completed enriched legacy prefix."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-plan", type=Path, required=True)
    parser.add_argument("--legacy-launch", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--enriched-plan", type=Path, required=True)
    parser.add_argument("--enriched-launch", type=Path, required=True)
    parser.add_argument("--enriched-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    legacy_plan = json.loads(args.legacy_plan.read_text(encoding="utf-8"))
    legacy_launch = json.loads(args.legacy_launch.read_text(encoding="utf-8"))
    enriched_plan = json.loads(args.enriched_plan.read_text(encoding="utf-8"))
    enriched_launch = json.loads(args.enriched_launch.read_text(encoding="utf-8"))
    if not all(
        isinstance(value, dict)
        for value in (legacy_plan, legacy_launch, enriched_plan, enriched_launch)
    ):
        raise ValueError("plan and launch documents must be mappings")
    audit = audit_enrichment_replay(
        legacy_plan=legacy_plan,
        legacy_launch=legacy_launch,
        legacy_checkpoint_dir=args.legacy_checkpoint_dir,
        enriched_plan=enriched_plan,
        enriched_launch=enriched_launch,
        enriched_checkpoint_dir=args.enriched_checkpoint_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()


__all__ = ["AUDIT_SCHEMA", "audit_enrichment_replay"]
