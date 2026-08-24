"""Validate the biomechanics model-to-measurement evidence bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ARTICLE_REL = Path("docs/research/proximal_distal_energy_transfer")
BRIDGE_REL = ARTICLE_REL / "data/biomechanics_evidence_bridge.json"
EXTERNAL_REVIEW_REL = ARTICLE_REL / "data/external_source_review.json"
CLAIM_REGISTRY_REL = ARTICLE_REL / "data/claim_audit_registry.json"
SOURCE_REGISTER_REL = ARTICLE_REL / "data/biomechanics_source_register.json"
SCHEMA_VERSION = "proximal-distal-biomechanics-evidence-bridge/v1"

REQUIRED_MODALITIES = {
    "optical_motion_capture",
    "force_plates",
    "bilateral_six_axis_grip_wrenches",
    "surface_emg",
    "ultrasound",
    "shaft_strain_inertial_sensing",
    "launch_monitor",
}
REQUIRED_TRANSPORTABILITY_DIMENSIONS = {
    "anthropometry",
    "sex",
    "age",
    "skill",
    "injury",
    "handedness",
    "impairment",
    "club",
    "task",
}
BRIDGE_FIELDS = {
    "schema_version",
    "scope_statement",
    "measurement_boundary",
    "owner",
    "external_source_review_sha256",
    "claim_registry_sha256",
    "source_register_sha256",
    "human_validation_status",
    "modalities",
    "mechanisms",
    "transportability",
    "summary",
}
MODALITY_FIELDS = {
    "modality_id",
    "source_status",
    "source_ids",
    "observables",
    "frame_authority",
    "unit_authority",
    "calibration_requirement",
    "synchronization_requirement",
    "event_definition",
    "processing_method",
    "missingness_rule",
    "uncertainty_sources",
    "uncertainty_propagation",
    "directly_observed",
    "not_identifiable",
    "data_gate",
}
MECHANISM_FIELDS = {
    "mechanism_id",
    "statement",
    "model_quantity",
    "measurement_requirements",
    "identifiability",
    "competing_explanations",
    "observable_discriminator",
    "falsifier",
    "adverse_case",
    "claim_ids",
    "human_evidence_state",
}
TRANSPORTABILITY_FIELDS = {
    "dimension_id",
    "current_coverage",
    "limitation",
    "data_gate",
    "claim_ids",
}
SOURCE_STATES = {"registered", "gap"}
IDENTIFIABILITY_STATES = {
    "directly_observed",
    "derived_model_dependent",
    "structurally_unidentifiable",
    "practically_unqualified",
    "unavailable",
}
HUMAN_EVIDENCE_STATES = {
    "model_only",
    "empirical_context_only",
    "externally_blocked",
}
HUMAN_VALIDATION_STATES = {"qualified", "provisional", "externally_blocked"}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_text(record: dict[str, Any], field: str, record_id: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record_id}: {field} must be non-empty text")
    return value


def _require_text_list(record: dict[str, Any], field: str, record_id: str) -> list[str]:
    value = record.get(field)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(f"{record_id}: {field} must be a non-empty text list")
    if len(value) != len(set(value)):
        raise ValueError(f"{record_id}: {field} contains duplicates")
    return value


def _validate_modality(
    modality: dict[str, Any], known_sources: dict[str, dict[str, Any]]
) -> tuple[str, str]:
    modality_id = _require_text(modality, "modality_id", "modality")
    unknown_fields = sorted(set(modality) - MODALITY_FIELDS)
    if unknown_fields:
        raise ValueError(f"{modality_id}: unknown fields: {unknown_fields}")
    source_status = modality.get("source_status")
    if source_status not in SOURCE_STATES:
        raise ValueError(f"{modality_id}: invalid source_status")
    source_ids = modality.get("source_ids")
    if not isinstance(source_ids, list) or not all(
        isinstance(item, str) and item.strip() for item in source_ids
    ):
        raise ValueError(f"{modality_id}: source_ids must be a text list")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{modality_id}: source_ids contains duplicates")
    unknown = sorted(set(source_ids) - set(known_sources))
    if unknown:
        raise ValueError(f"{modality_id}: unknown source_ids: {unknown}")
    if source_status == "registered" and not source_ids:
        raise ValueError(f"{modality_id}: registered source_ids are required")
    if source_status == "registered":
        ineligible = sorted(
            source_id
            for source_id in source_ids
            if known_sources[source_id].get("independence") != "independent_of_project"
            or known_sources[source_id].get("evidence_disposition") != "eligible"
        )
        if ineligible:
            raise ValueError(
                f"{modality_id}: registered sources must be independent "
                f"evidence-eligible works: {ineligible}"
            )
    for field in (
        "observables",
        "uncertainty_sources",
        "directly_observed",
        "not_identifiable",
    ):
        _require_text_list(modality, field, modality_id)
    for field in (
        "frame_authority",
        "unit_authority",
        "calibration_requirement",
        "synchronization_requirement",
        "event_definition",
        "processing_method",
        "missingness_rule",
        "uncertainty_propagation",
        "data_gate",
    ):
        _require_text(modality, field, modality_id)
    if source_status == "gap" and not modality["data_gate"].strip():
        raise ValueError(f"{modality_id}: source gap requires a data_gate")
    return modality_id, source_status


def _validate_mechanism(
    mechanism: dict[str, Any], modality_ids: set[str], claim_ids: set[str]
) -> str:
    mechanism_id = _require_text(mechanism, "mechanism_id", "mechanism")
    unknown_fields = sorted(set(mechanism) - MECHANISM_FIELDS)
    if unknown_fields:
        raise ValueError(f"{mechanism_id}: unknown fields: {unknown_fields}")
    for field in (
        "statement",
        "model_quantity",
        "observable_discriminator",
        "falsifier",
        "adverse_case",
    ):
        _require_text(mechanism, field, mechanism_id)
    requirements = set(
        _require_text_list(mechanism, "measurement_requirements", mechanism_id)
    )
    unknown_modalities = sorted(requirements - modality_ids)
    if unknown_modalities:
        raise ValueError(
            f"{mechanism_id}: unknown measurement modalities: {unknown_modalities}"
        )
    _require_text_list(mechanism, "competing_explanations", mechanism_id)
    linked_claims = set(_require_text_list(mechanism, "claim_ids", mechanism_id))
    unknown_claims = sorted(linked_claims - claim_ids)
    if unknown_claims:
        raise ValueError(f"{mechanism_id}: unknown claim_ids: {unknown_claims}")
    identifiability = mechanism.get("identifiability")
    if identifiability not in IDENTIFIABILITY_STATES:
        raise ValueError(f"{mechanism_id}: invalid identifiability")
    if mechanism.get("human_evidence_state") not in HUMAN_EVIDENCE_STATES:
        raise ValueError(f"{mechanism_id}: invalid human_evidence_state")
    if (
        mechanism_id == "bilateral_hand_wrench_allocation"
        and identifiability != "structurally_unidentifiable"
    ):
        raise ValueError(
            "bilateral hand allocation remains structurally unidentifiable "
            "without independent bilateral measurements"
        )
    return mechanism_id


def _validate_transportability(records: Any, claim_ids: set[str]) -> set[str]:
    if not isinstance(records, list):
        raise ValueError("transportability must be a list")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each transportability record must be an object")
        dimension_id = _require_text(record, "dimension_id", "transportability")
        unknown_fields = sorted(set(record) - TRANSPORTABILITY_FIELDS)
        if unknown_fields:
            raise ValueError(f"{dimension_id}: unknown fields: {unknown_fields}")
        if dimension_id in seen:
            raise ValueError(f"duplicate transportability dimension: {dimension_id}")
        seen.add(dimension_id)
        if record.get("current_coverage") not in {
            "bounded_context",
            "unqualified",
        }:
            raise ValueError(f"{dimension_id}: invalid current_coverage")
        for field in ("limitation", "data_gate"):
            _require_text(record, field, dimension_id)
        linked_claims = set(_require_text_list(record, "claim_ids", dimension_id))
        unknown_claims = sorted(linked_claims - claim_ids)
        if unknown_claims:
            raise ValueError(f"{dimension_id}: unknown claim_ids: {unknown_claims}")
    if seen != REQUIRED_TRANSPORTABILITY_DIMENSIONS:
        raise ValueError("transportability coverage is incomplete or stale")
    return seen


def validate_biomechanics_evidence_bridge(
    root: str | Path,
    bridge: dict[str, Any],
    external_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless modalities, mechanisms, claims, and digests reconcile."""
    root_path = Path(root).resolve()
    external_path = root_path / EXTERNAL_REVIEW_REL
    claim_path = root_path / CLAIM_REGISTRY_REL
    source_register_path = root_path / SOURCE_REGISTER_REL
    review = external_review or json.loads(external_path.read_text(encoding="utf-8"))
    claim_registry = json.loads(claim_path.read_text(encoding="utf-8"))
    source_register = json.loads(source_register_path.read_text(encoding="utf-8"))

    if bridge.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("biomechanics evidence bridge schema version is stale")
    unknown_bridge_fields = sorted(set(bridge) - BRIDGE_FIELDS)
    if unknown_bridge_fields:
        raise ValueError(f"bridge contains unknown fields: {unknown_bridge_fields}")
    for field in ("scope_statement", "measurement_boundary", "owner"):
        _require_text(bridge, field, "bridge")
    if bridge.get("external_source_review_sha256") != _sha256(external_path):
        raise ValueError("external source review digest is stale")
    if bridge.get("claim_registry_sha256") != _sha256(claim_path):
        raise ValueError("claim registry digest is stale")
    if bridge.get("source_register_sha256") != _sha256(source_register_path):
        raise ValueError("biomechanics source register digest is stale")

    works = review.get("works")
    if not isinstance(works, list) or not works:
        raise ValueError("external source review contains no works")
    known_sources = {
        work["work_id"]: work
        for work in works
        if isinstance(work, dict) and isinstance(work.get("work_id"), str)
    }
    claims = claim_registry.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("claim registry contains no claims")
    claim_ids = {
        claim.get("claim_id")
        for claim in claims
        if isinstance(claim, dict) and isinstance(claim.get("claim_id"), str)
    }

    modalities = bridge.get("modalities")
    if not isinstance(modalities, list):
        raise ValueError("modalities must be a list")
    modality_states: dict[str, str] = {}
    for modality in modalities:
        if not isinstance(modality, dict):
            raise ValueError("each modality must be an object")
        modality_id, source_status = _validate_modality(modality, known_sources)
        if modality_id in modality_states:
            raise ValueError(f"duplicate modality_id: {modality_id}")
        modality_states[modality_id] = source_status
    if set(modality_states) != REQUIRED_MODALITIES:
        raise ValueError("required modality coverage is incomplete or stale")

    transportability_dimensions = _validate_transportability(
        bridge.get("transportability"), claim_ids
    )

    mechanisms = bridge.get("mechanisms")
    if not isinstance(mechanisms, list) or len(mechanisms) < 8:
        raise ValueError("at least eight mechanisms are required")
    mechanism_ids: set[str] = set()
    identifiability_counts: Counter[str] = Counter()
    linked_claim_ids: set[str] = set()
    for mechanism in mechanisms:
        if not isinstance(mechanism, dict):
            raise ValueError("each mechanism must be an object")
        mechanism_id = _validate_mechanism(mechanism, set(modality_states), claim_ids)
        if mechanism_id in mechanism_ids:
            raise ValueError(f"duplicate mechanism_id: {mechanism_id}")
        mechanism_ids.add(mechanism_id)
        identifiability_counts[mechanism["identifiability"]] += 1
        linked_claim_ids.update(mechanism["claim_ids"])
    for record in bridge["transportability"]:
        linked_claim_ids.update(record["claim_ids"])

    registered_source_claim_ids = {
        claim_id
        for source in source_register.get("sources", [])
        if isinstance(source, dict)
        for claim_id in source.get("linked_claim_ids", [])
        if isinstance(claim_id, str)
    }
    nonreciprocal_source_claims = sorted(registered_source_claim_ids - linked_claim_ids)
    if nonreciprocal_source_claims:
        raise ValueError(
            "source-register claim links are not reciprocated by the evidence "
            f"bridge: {nonreciprocal_source_claims}"
        )

    validation_status = bridge.get("human_validation_status")
    if validation_status not in HUMAN_VALIDATION_STATES:
        raise ValueError("invalid human_validation_status")
    expected_summary = {
        "modality_count": len(modalities),
        "source_registered_modality_count": sum(
            state == "registered" for state in modality_states.values()
        ),
        "source_gap_modality_count": sum(
            state == "gap" for state in modality_states.values()
        ),
        "mechanism_count": len(mechanisms),
        "linked_claim_count": len(linked_claim_ids),
        "transportability_dimension_count": len(transportability_dimensions),
        "identifiability_counts": {
            state: identifiability_counts[state]
            for state in sorted(IDENTIFIABILITY_STATES)
        },
        "human_validation_status": validation_status,
    }
    if bridge.get("summary") != expected_summary:
        raise ValueError("biomechanics evidence bridge summary is stale")
    return {"valid": True, **expected_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate",))
    parser.parse_args()
    root = _repository_root()
    bridge = json.loads((root / BRIDGE_REL).read_text(encoding="utf-8"))
    print(json.dumps(validate_biomechanics_evidence_bridge(root, bridge), indent=2))


if __name__ == "__main__":
    main()
