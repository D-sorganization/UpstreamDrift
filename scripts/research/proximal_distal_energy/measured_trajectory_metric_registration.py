"""Preregister golf-likeness metrics without inventing measured evidence."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.measured_trajectory_source_registry import (
    load_and_validate_registry,
)


SCHEMA_VERSION = "measured-trajectory-metric-registration/v1"
REQUIRED_METRIC_IDS = (
    "club_pose",
    "club_speed",
    "pelvis_pose_rate",
    "thorax_pose_rate",
    "segment_phase_ordering",
    "wrist_hand_path",
    "intersegment_geometry",
    "joint_range_margin",
    "closure_residual",
    "tracking_error",
    "residual_generalized_effort",
)
REQUIRED_NEGATIVE_CONTROL_IDS = (
    "phase_scrambled",
    "velocity_reversed",
    "scale_mismatched",
    "non_golf_synthetic",
)
REQUIRED_FRAME_IDS = ("lab", "anatomical", "model", "club")
REQUIRED_EVENT_IDS = ("downswing_start", "impact")
REQUIRED_UNCERTAINTY_IDS = (
    "time_alignment",
    "filtering",
    "coordinate_mapping",
    "marker_reconstruction",
    "event_detection",
    "anthropometric_scaling",
)

_TOP_KEYS = {
    "schema_version",
    "registration_id",
    "parent_issue",
    "registered_at_utc",
    "registered_before_outcomes",
    "source_registry_path",
    "authority_status",
    "estimand",
    "participant_split",
    "frames",
    "events",
    "metrics",
    "negative_controls",
    "uncertainty_analyses",
    "results_status",
    "readiness",
    "inference_boundary",
}
_SPLIT_KEYS = {
    "unit",
    "assignment",
    "grouping_key",
    "minimum_training_participants",
    "minimum_held_out_participants",
    "adverse_cohort_required",
    "framewise_random_split_prohibited",
    "freeze_before_outcomes",
}
_FRAME_KEYS = {
    "frame_id",
    "definition",
    "transform_authority",
    "uncertainty_required",
}
_EVENT_KEYS = {
    "event_id",
    "definition",
    "uncertainty_required",
    "missing_policy",
}
_METRIC_KEYS = {
    "metric_id",
    "family",
    "unit",
    "definition",
    "required_channels",
    "comparison",
    "normalization",
    "uncertainty_sources",
    "threshold_policy",
    "missing_data_policy",
    "falsifier",
}
_CONTROL_KEYS = {
    "control_id",
    "transformation",
    "must_reject_metrics",
    "pass_condition",
    "evidence_class",
}
_UNCERTAINTY_KEYS = {
    "analysis_id",
    "perturbation",
    "decision_rule",
}
_THRESHOLD_POLICIES = {
    "training_distribution_then_frozen",
    "registered_engineering_tolerance_then_frozen",
}
_COMPARISONS = {
    "distributional_coverage",
    "phase_order_confusion",
    "threshold_exceedance",
    "residual_distribution",
}
_NORMALIZATIONS = {
    "declared_units",
    "body_height",
    "club_length",
    "measured_range",
}


def _require_exact_keys(record: dict[str, Any], expected: set[str], name: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{name} fields do not match the registered schema")


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _unique_text_list(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{field} must be a unique nonempty string list")
    return value


def _ordered_ids(
    rows: object, keys: set[str], id_field: str, name: str
) -> tuple[str, ...]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} must be a nonempty list")
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"every {name} row must be an object")
        _require_exact_keys(row, keys, name)
        ids.append(_nonempty_text(row[id_field], id_field))
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} identifiers must be unique")
    return tuple(ids)


def validate_participant_split(
    training: Iterable[str], held_out: Iterable[str], *, adverse: Iterable[str] = ()
) -> dict[str, int]:
    """Require disjoint participant-level cohorts; frame-wise splits are invalid."""

    cohorts = {
        "training": tuple(training),
        "held_out": tuple(held_out),
        "adverse": tuple(adverse),
    }
    for name, values in cohorts.items():
        if name != "adverse" and not values:
            raise ValueError(f"{name} participant cohort cannot be empty")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{name} participant identifiers must be nonempty strings")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} participant identifiers must be unique")
    names = tuple(cohorts)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            if set(cohorts[left_name]) & set(cohorts[right_name]):
                raise ValueError("participant cohorts must be disjoint")
    return {name: len(values) for name, values in cohorts.items()}


def _validate_split(split: dict[str, Any]) -> None:
    _require_exact_keys(split, _SPLIT_KEYS, "participant_split")
    if split["unit"] != "participant":
        raise ValueError("participant split unit must be participant")
    if split["assignment"] != "deterministic_digest":
        raise ValueError("participant split assignment must be deterministic_digest")
    if split["grouping_key"] != "participant_id":
        raise ValueError("participant split grouping_key must be participant_id")
    for field in ("minimum_training_participants", "minimum_held_out_participants"):
        value = split[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in (
        "adverse_cohort_required",
        "framewise_random_split_prohibited",
        "freeze_before_outcomes",
    ):
        if split[field] is not True:
            raise ValueError(f"{field} must be true")


def _validate_frames(rows: list[dict[str, Any]]) -> None:
    ids = _ordered_ids(rows, _FRAME_KEYS, "frame_id", "frames")
    if ids != REQUIRED_FRAME_IDS:
        raise ValueError("frame identifiers do not match the registered frame set")
    for row in rows:
        _nonempty_text(row["definition"], "frame definition")
        _nonempty_text(row["transform_authority"], "transform_authority")
        if row["uncertainty_required"] is not True:
            raise ValueError("every frame transform requires uncertainty")


def _validate_events(rows: list[dict[str, Any]]) -> None:
    ids = _ordered_ids(rows, _EVENT_KEYS, "event_id", "events")
    if ids != REQUIRED_EVENT_IDS:
        raise ValueError("event identifiers do not match the registered event set")
    for row in rows:
        _nonempty_text(row["definition"], "event definition")
        if row["uncertainty_required"] is not True:
            raise ValueError("every event requires uncertainty")
        if row["missing_policy"] != "unavailable_not_zero":
            raise ValueError("missing events must remain unavailable, not zero")


def _validate_metrics(rows: list[dict[str, Any]]) -> None:
    ids = _ordered_ids(rows, _METRIC_KEYS, "metric_id", "metrics")
    if ids != REQUIRED_METRIC_IDS:
        raise ValueError("metric identifiers do not match the registered metric set")
    for row in rows:
        for field in ("family", "unit", "definition", "falsifier"):
            _nonempty_text(row[field], f"metric {field}")
        channels = _unique_text_list(row["required_channels"], "required_channels")
        if any(
            "missing" in channel or "bilateral_wrench" in channel
            for channel in channels
        ):
            raise ValueError("missing channels cannot be zero or force authority")
        uncertainty = _unique_text_list(
            row["uncertainty_sources"], "uncertainty_sources"
        )
        if not set(uncertainty) <= set(REQUIRED_UNCERTAINTY_IDS):
            raise ValueError("metric uncertainty source is not registered")
        if row["comparison"] not in _COMPARISONS:
            raise ValueError("metric comparison is not registered")
        if row["normalization"] not in _NORMALIZATIONS:
            raise ValueError("metric normalization is not registered")
        if row["threshold_policy"] not in _THRESHOLD_POLICIES:
            raise ValueError("metric threshold policy is not preregistered")
        if row["missing_data_policy"] != "unavailable_not_zero":
            raise ValueError("metric missing data must remain unavailable, not zero")


def _validate_controls(rows: list[dict[str, Any]]) -> None:
    ids = _ordered_ids(rows, _CONTROL_KEYS, "control_id", "negative_controls")
    if ids != REQUIRED_NEGATIVE_CONTROL_IDS:
        raise ValueError("negative-control identifiers do not match registration")
    for row in rows:
        _nonempty_text(row["transformation"], "control transformation")
        _nonempty_text(row["pass_condition"], "control pass_condition")
        metrics = _unique_text_list(row["must_reject_metrics"], "must_reject_metrics")
        if not set(metrics) <= set(REQUIRED_METRIC_IDS):
            raise ValueError("negative control references an unregistered metric")
        if row["evidence_class"] != "software_discrimination_only":
            raise ValueError("negative controls qualify software discrimination only")


def _validate_uncertainty(rows: list[dict[str, Any]]) -> None:
    ids = _ordered_ids(rows, _UNCERTAINTY_KEYS, "analysis_id", "uncertainty_analyses")
    if ids != REQUIRED_UNCERTAINTY_IDS:
        raise ValueError("uncertainty analyses do not match the registered set")
    for row in rows:
        _nonempty_text(row["perturbation"], "uncertainty perturbation")
        _nonempty_text(row["decision_rule"], "uncertainty decision_rule")


def _readiness(record: dict[str, Any]) -> dict[str, Any]:
    status = record["authority_status"]
    execution_ready = status == "motion_only_held_out_authority_available"
    return {
        "status": status,
        "metric_count": len(record["metrics"]),
        "negative_control_count": len(record["negative_controls"]),
        "frame_count": len(record["frames"]),
        "event_count": len(record["events"]),
        "participant_held_out_required": True,
        "thresholds_frozen_before_held_out": True,
        "execution_ready": execution_ready,
        "human_inference_ready": False,
        "bilateral_wrench_gate_satisfied": False,
    }


def validate_registration(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the preregistration and independently recompute readiness."""

    if not isinstance(record, dict):
        raise ValueError("registration must be an object")
    _require_exact_keys(record, _TOP_KEYS, "registration")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("registration schema_version is unsupported")
    if record["registration_id"] != "articulated-golf-likeness-v1":
        raise ValueError("registration_id is not registered")
    if (
        record["parent_issue"]
        != "https://github.com/D-sorganization/UpstreamDrift/issues/9004"
    ):
        raise ValueError("parent_issue must identify #9004")
    _nonempty_text(record["registered_at_utc"], "registered_at_utc")
    if record["registered_before_outcomes"] is not True:
        raise ValueError("metrics must be registered before outcomes")
    if record["source_registry_path"] != "measured_trajectory_source_registry.json":
        raise ValueError("source_registry_path is not registered")
    if record["authority_status"] not in {
        "blocked_no_qualified_measured_trajectory_authority",
        "motion_only_held_out_authority_available",
    }:
        raise ValueError("authority_status is not registered")
    _nonempty_text(record["estimand"], "estimand")
    _validate_split(record["participant_split"])
    _validate_frames(record["frames"])
    _validate_events(record["events"])
    _validate_metrics(record["metrics"])
    _validate_controls(record["negative_controls"])
    _validate_uncertainty(record["uncertainty_analyses"])
    expected_result = (
        "not_run_no_authority"
        if record["authority_status"]
        == "blocked_no_qualified_measured_trajectory_authority"
        else "registered_not_run"
    )
    if record["results_status"] != expected_result:
        raise ValueError("results_status is inconsistent with available authority")
    boundary = _nonempty_text(record["inference_boundary"], "inference_boundary")
    lower = boundary.lower()
    if (
        "cannot" not in lower
        or "coaching" not in lower
        or "bilateral wrench" not in lower
    ):
        raise ValueError("inference_boundary must preserve human and force limits")
    readiness = _readiness(record)
    if record["readiness"] != readiness:
        raise ValueError("committed readiness does not reproduce")
    return readiness


def load_and_validate_registration(path: Path) -> dict[str, Any]:
    """Load without duplicate keys, cross-check source authority, and validate."""

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
    readiness = validate_registration(record)
    source = load_and_validate_registry(path.parent / record["source_registry_path"])
    if source["status"] != record["authority_status"]:
        raise ValueError("source registry authority status does not reproduce")
    return readiness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate",))
    parser.add_argument(
        "--registration",
        type=Path,
        default=Path(
            "docs/research/proximal_distal_energy_transfer/data/"
            "measured_trajectory_metric_registration.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(load_and_validate_registration(args.registration), sort_keys=True))


if __name__ == "__main__":
    main()
