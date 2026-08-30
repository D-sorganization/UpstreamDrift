"""Classify one preregistered structural repeat without numerical tolerance."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    StructuralCheckpoint,
    build_registered_cases,
    load_available_checkpoints,
    plan_sha256,
)

AUDIT_SCHEMA = "articulated-structural-factorial-repeatability-audit/1.0.0"


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _result(checkpoint: StructuralCheckpoint) -> dict[str, Any]:
    payload = _mapping(
        json.loads(checkpoint.path.read_text(encoding="utf-8")), name="checkpoint"
    )
    outcome = _mapping(payload.get("outcome"), name="checkpoint.outcome")
    result = dict(_mapping(outcome.get("result"), name="checkpoint.result"))
    result.pop("evidence_sidecar_schema", None)
    return result


def _arrays(checkpoint: StructuralCheckpoint) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(checkpoint.path.with_suffix(".npz"), allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_identity(checkpoint: StructuralCheckpoint) -> dict[str, str]:
    return {
        "json_sha256": _sha256(checkpoint.path),
        "npz_sha256": _sha256(checkpoint.path.with_suffix(".npz")),
    }


def _checkpoint_for_case(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    checkpoint_dir: Path,
    case_key: str,
    require_single: bool,
) -> StructuralCheckpoint:
    checkpoints = load_available_checkpoints(
        plan=plan,
        launch=launch,
        checkpoint_dir=checkpoint_dir,
    )
    if require_single and len(checkpoints) != 1:
        raise ValueError("repeat checkpoint directory must contain exactly one case")
    by_key = {checkpoint.case.case_key: checkpoint for checkpoint in checkpoints}
    if case_key not in by_key:
        raise ValueError(
            "checkpoint directory does not contain the registered probe case"
        )
    checkpoint = by_key[case_key]
    if checkpoint.status != "completed":
        raise ValueError("repeatability probe requires completed checkpoints")
    return checkpoint


def _exact_match(
    *,
    authority: StructuralCheckpoint,
    repeat: StructuralCheckpoint,
    legacy_array_names: tuple[str, ...],
) -> dict[str, object]:
    authority_arrays = _arrays(authority)
    repeat_arrays = _arrays(repeat)
    missing = [name for name in legacy_array_names if name not in authority_arrays]
    if missing:
        raise ValueError("authority omits an array registered by the legacy checkpoint")
    missing = [name for name in legacy_array_names if name not in repeat_arrays]
    if missing:
        raise ValueError("repeat omits an array registered by the legacy checkpoint")
    json_exact = _result(repeat) == _result(authority)
    mismatched_arrays = tuple(
        name
        for name in legacy_array_names
        if not np.array_equal(repeat_arrays[name], authority_arrays[name])
    )
    return {
        "json_result_exact": json_exact,
        "legacy_arrays_exact": not mismatched_arrays,
        "compared_legacy_array_count": len(legacy_array_names),
        "mismatched_legacy_array_names": list(mismatched_arrays),
        "matches": json_exact and not mismatched_arrays,
    }


def audit_repeatability_probe(
    *,
    legacy_plan: Mapping[str, object],
    legacy_launch: Mapping[str, object],
    legacy_checkpoint_dir: Path,
    enriched_plan: Mapping[str, object],
    enriched_launch: Mapping[str, object],
    enriched_checkpoint_dir: Path,
    repeat_plan: Mapping[str, object],
    repeat_launch: Mapping[str, object],
    repeat_checkpoint_dir: Path,
    case_index: int,
) -> dict[str, object]:
    """Classify an exact repeat against frozen legacy and enriched authorities."""

    registries = [
        build_registered_cases(plan)
        for plan in (legacy_plan, enriched_plan, repeat_plan)
    ]
    case_keys = [[case.case_key for case in registry] for registry in registries]
    if case_keys[1:] != case_keys[:-1]:
        raise ValueError(
            "legacy, enriched, and repeat plans must register identical cases"
        )
    if (
        isinstance(case_index, bool)
        or not isinstance(case_index, int)
        or case_index < 0
        or case_index >= len(registries[0])
    ):
        raise ValueError("case index is outside the registered design")
    if enriched_launch.get("execution_revision") != repeat_launch.get(
        "execution_revision"
    ):
        raise ValueError(
            "enriched and repeat probes must use the same execution revision"
        )

    case_key = registries[0][case_index].case_key
    legacy = _checkpoint_for_case(
        plan=legacy_plan,
        launch=legacy_launch,
        checkpoint_dir=legacy_checkpoint_dir,
        case_key=case_key,
        require_single=False,
    )
    enriched = _checkpoint_for_case(
        plan=enriched_plan,
        launch=enriched_launch,
        checkpoint_dir=enriched_checkpoint_dir,
        case_key=case_key,
        require_single=False,
    )
    repeat = _checkpoint_for_case(
        plan=repeat_plan,
        launch=repeat_launch,
        checkpoint_dir=repeat_checkpoint_dir,
        case_key=case_key,
        require_single=True,
    )
    legacy_array_names = tuple(sorted(_arrays(legacy)))
    legacy_comparison = _exact_match(
        authority=legacy,
        repeat=repeat,
        legacy_array_names=legacy_array_names,
    )
    enriched_comparison = _exact_match(
        authority=enriched,
        repeat=repeat,
        legacy_array_names=legacy_array_names,
    )
    matches_legacy = bool(legacy_comparison["matches"])
    matches_enriched = bool(enriched_comparison["matches"])
    if matches_enriched and not matches_legacy:
        classification = "deterministic_source_delta_supported"
    elif matches_legacy and not matches_enriched:
        classification = "first_enriched_replay_anomaly_supported"
    elif not matches_legacy and not matches_enriched:
        classification = "cross_run_nonrepeatability_demonstrated"
    else:
        classification = "authorities_not_discriminating"

    identity_payload = {
        "case_index": case_index,
        "case_key": case_key,
        "legacy_plan_sha256": plan_sha256(legacy_plan),
        "legacy_execution_revision": legacy_launch["execution_revision"],
        "enriched_plan_sha256": plan_sha256(enriched_plan),
        "enriched_execution_revision": enriched_launch["execution_revision"],
        "repeat_plan_sha256": plan_sha256(repeat_plan),
        "repeat_execution_revision": repeat_launch["execution_revision"],
        "legacy_checkpoint": _checkpoint_identity(legacy),
        "enriched_checkpoint": _checkpoint_identity(enriched),
        "repeat_checkpoint": _checkpoint_identity(repeat),
    }
    audit_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": AUDIT_SCHEMA,
        "identity": {**identity_payload, "audit_identity_sha256": audit_identity},
        "comparison": {
            "legacy": legacy_comparison,
            "enriched": enriched_comparison,
        },
        "matches": {
            "enriched": matches_enriched,
            "legacy": matches_legacy,
        },
        "classification": classification,
        "gates": {
            "case_registry_identical": True,
            "same_enriched_execution_revision": True,
            "single_registered_repeat_checkpoint": True,
            "exact_equality_only": True,
            "classification_complete": True,
        },
        "claim_boundary": {
            "outcome_values_reported": False,
            "scientific_effect_direction_interpreted": False,
            "campaign_promotion_authorized": False,
            "human_or_coaching_inference": False,
        },
    }


def main() -> None:
    """Write one atomic audit for a preregistered repeatability probe."""

    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("legacy", "enriched", "repeat"):
        parser.add_argument(f"--{prefix}-plan", type=Path, required=True)
        parser.add_argument(f"--{prefix}-launch", type=Path, required=True)
        parser.add_argument(f"--{prefix}-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read_mapping(path: Path) -> Mapping[str, object]:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), name=str(path))

    audit = audit_repeatability_probe(
        legacy_plan=read_mapping(args.legacy_plan),
        legacy_launch=read_mapping(args.legacy_launch),
        legacy_checkpoint_dir=args.legacy_checkpoint_dir,
        enriched_plan=read_mapping(args.enriched_plan),
        enriched_launch=read_mapping(args.enriched_launch),
        enriched_checkpoint_dir=args.enriched_checkpoint_dir,
        repeat_plan=read_mapping(args.repeat_plan),
        repeat_launch=read_mapping(args.repeat_launch),
        repeat_checkpoint_dir=args.repeat_checkpoint_dir,
        case_index=args.case_index,
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


__all__ = ["AUDIT_SCHEMA", "audit_repeatability_probe"]
