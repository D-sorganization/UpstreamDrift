"""Fail-closed source registry for measured golf-trajectory qualification."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "measured-trajectory-source-registry/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_TOP_KEYS = {
    "schema_version",
    "registry_id",
    "registered_at_utc",
    "search_scope",
    "sources",
    "readiness",
    "inference_boundary",
}
_SOURCE_KEYS = {
    "source_id",
    "title",
    "landing_url",
    "authority",
    "measurement_class",
    "human_observed",
    "access_status",
    "license_status",
    "participant_count",
    "trial_count",
    "body_kinematics",
    "club_kinematics",
    "raw_trajectory_available",
    "participant_grouping_available",
    "calibration_metadata_available",
    "synchronization_metadata_available",
    "content_digest_sha256",
    "decision",
    "blockers",
    "notes",
}
_MEASUREMENT_CLASSES = {
    "marker_based_motion_capture",
    "pose_keypoints",
    "video_event_annotations",
    "simulation_output",
    "software_test_fixture",
}
_ACCESS_STATUSES = {
    "public_download",
    "authorized_download",
    "authorization_required",
    "local_repository",
}
_LICENSE_STATUSES = {
    "explicit_reuse_license",
    "citation_guidance_no_explicit_reuse_license",
    "authorization_required",
    "repository_internal_only",
    "not_applicable_simulation",
}
_DECISIONS = {"qualification_candidate", "negative_control", "inadmissible"}
_TRISTATE_FIELDS = {
    "body_kinematics",
    "club_kinematics",
    "raw_trajectory_available",
    "participant_grouping_available",
    "calibration_metadata_available",
    "synchronization_metadata_available",
}


def _require_exact_keys(record: dict[str, Any], expected: set[str], name: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{name} fields do not match the registered schema")


def _positive_count(value: object, field: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 1
    ):
        raise ValueError(f"{field} must be null or a positive integer")


def _validate_source(source: dict[str, Any]) -> None:
    _require_exact_keys(source, _SOURCE_KEYS, "source")
    source_id = source["source_id"]
    if not isinstance(source_id, str) or _SOURCE_ID.fullmatch(source_id) is None:
        raise ValueError("source_id must be a lowercase hyphenated identifier")
    for field in ("title", "authority", "notes"):
        if not isinstance(source[field], str) or not source[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    url = source["landing_url"]
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("landing_url must use HTTPS")
    if source["measurement_class"] not in _MEASUREMENT_CLASSES:
        raise ValueError("measurement_class is not registered")
    if source["access_status"] not in _ACCESS_STATUSES:
        raise ValueError("access_status is not registered")
    if source["license_status"] not in _LICENSE_STATUSES:
        raise ValueError("license_status is not registered")
    if source["decision"] not in _DECISIONS:
        raise ValueError("decision is not registered")
    if not isinstance(source["human_observed"], bool):
        raise ValueError("human_observed must be Boolean")
    _positive_count(source["participant_count"], "participant_count")
    _positive_count(source["trial_count"], "trial_count")
    for field in _TRISTATE_FIELDS:
        if source[field] is not None and not isinstance(source[field], bool):
            raise ValueError(f"{field} must be Boolean or null")
    digest = source["content_digest_sha256"]
    if digest is not None and (
        not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
    ):
        raise ValueError("content_digest_sha256 must be null or lowercase SHA-256")
    blockers = source["blockers"]
    if (
        not isinstance(blockers, list)
        or any(not isinstance(row, str) or not row.strip() for row in blockers)
        or len(blockers) != len(set(blockers))
    ):
        raise ValueError("blockers must be a unique list of nonempty strings")
    unknown = [field for field in _TRISTATE_FIELDS if source[field] is None]
    if unknown and not blockers:
        raise ValueError("unknown fields require a blocker")
    if source["measurement_class"] == "simulation_output" and (
        source["human_observed"] or source["decision"] != "inadmissible"
    ):
        raise ValueError("simulation output cannot qualify as human measurement")


def _pipeline_probe_ready(source: dict[str, Any]) -> bool:
    return bool(
        source["decision"] == "qualification_candidate"
        and source["human_observed"]
        and source["access_status"] in {"public_download", "authorized_download"}
        and source["license_status"] == "explicit_reuse_license"
        and source["participant_count"] is not None
        and source["body_kinematics"] is True
        and source["club_kinematics"] is True
        and source["raw_trajectory_available"] is True
        and source["participant_grouping_available"] is True
        and source["calibration_metadata_available"] is True
        and source["synchronization_metadata_available"] is True
        and source["content_digest_sha256"] is not None
        and not source["blockers"]
    )


def compute_readiness(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate registered sources and derive motion-only readiness."""

    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a nonempty list")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("every source must be an object")
        _validate_source(source)
    ids = [row["source_id"] for row in sources]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("source_id values must be unique and sorted")
    pipeline = [row["source_id"] for row in sources if _pipeline_probe_ready(row)]
    held_out = [
        row["source_id"]
        for row in sources
        if _pipeline_probe_ready(row) and int(row["participant_count"]) >= 2
    ]
    return {
        "status": (
            "motion_only_held_out_authority_available"
            if held_out
            else "blocked_no_qualified_measured_trajectory_authority"
        ),
        "source_count": len(sources),
        "qualification_candidate_source_ids": [
            row["source_id"]
            for row in sources
            if row["decision"] == "qualification_candidate"
        ],
        "negative_control_source_ids": [
            row["source_id"] for row in sources if row["decision"] == "negative_control"
        ],
        "inadmissible_source_ids": [
            row["source_id"] for row in sources if row["decision"] == "inadmissible"
        ],
        "pipeline_probe_ready_source_ids": pipeline,
        "held_out_qualification_ready_source_ids": held_out,
        "human_inference_ready": False,
        "bilateral_wrench_gate_satisfied": False,
    }


def validate_registry(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one registry and return its independently recomputed readiness."""

    if not isinstance(record, dict):
        raise ValueError("registry must be an object")
    _require_exact_keys(record, _TOP_KEYS, "registry")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("registry schema_version is unsupported")
    if record["registry_id"] != "articulated-golf-trajectory-sources-v1":
        raise ValueError("registry_id is not registered")
    if not isinstance(record["registered_at_utc"], str):
        raise ValueError("registered_at_utc must be a string")
    if not isinstance(record["search_scope"], dict) or not record["search_scope"]:
        raise ValueError("search_scope must be a nonempty object")
    sources = record["sources"]
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    boundary = record["inference_boundary"]
    if not isinstance(boundary, str) or "cannot" not in boundary.lower():
        raise ValueError(
            "inference_boundary must state what this registry cannot prove"
        )
    readiness = compute_readiness(sources)
    if record["readiness"] != readiness:
        raise ValueError("committed readiness summary does not reproduce")
    return readiness


def load_and_validate_registry(path: Path) -> dict[str, Any]:
    """Load JSON without accepting duplicate keys and validate it."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    record = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    return validate_registry(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate",))
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(
            "docs/research/proximal_distal_energy_transfer/data/"
            "measured_trajectory_source_registry.json"
        ),
    )
    args = parser.parse_args()
    result = load_and_validate_registry(args.registry)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
