"""Fail-closed ingestion boundary for governed measured golf trajectories.

This module does not parse motion-capture formats. It verifies research
authority and immutable inputs before delegating to the canonical
``motion_pipeline.sources`` adapter stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .measured_trajectory_metric_registration import (
    validate_participant_split,
    validate_registration,
)
from .measured_trajectory_source_registry import validate_registry


SCHEMA_VERSION = "measured-trajectory-artifact/v1"
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS = {
    "schema_version",
    "manifest_id",
    "created_at_utc",
    "source_registry_id",
    "source_id",
    "participant_split",
    "artifact",
    "participant",
    "acquisition",
    "frames",
    "events",
    "channels",
    "uncertainties",
    "intended_use",
    "inference_boundary",
}
_PARTICIPANT_SPLIT_REFERENCE_KEYS = {"relative_path", "sha256"}
_PARTICIPANT_SPLIT_KEYS = {
    "schema_version",
    "split_id",
    "source_id",
    "assignment_method",
    "frozen_at_utc",
    "training_participant_ids",
    "held_out_participant_ids",
    "adverse_participant_ids",
}
_ARTIFACT_KEYS = {
    "source_package_relative_path",
    "source_package_sha256",
    "trajectory_relative_path",
    "trajectory_sha256",
    "format_hint",
}
_PARTICIPANT_KEYS = {"participant_id", "grouping_id", "cohort"}
_ACQUISITION_KEYS = {
    "trial_id",
    "sample_rate_hz",
    "spatial_unit",
    "angle_unit",
    "time_unit",
    "synchronization_method",
    "filtering_method",
    "marker_reconstruction_method",
    "anthropometric_source",
}
_FRAME_KEYS = {
    "frame_id",
    "definition",
    "transform_authority",
    "transform_sha256",
    "translation_uncertainty_m",
    "rotation_uncertainty_rad",
}
_EVENT_KEYS = {
    "event_id",
    "time_s",
    "detector_id",
    "detector_version",
    "uncertainty_s",
    "missing_policy",
}
_UNCERTAINTY_KEYS = {"analysis_id", "method", "lower", "upper", "unit"}
_FRAME_IDS = ("lab", "anatomical", "model", "club")
_EVENT_IDS = ("downswing_start", "impact")
_UNCERTAINTY_IDS = (
    "time_alignment",
    "filtering",
    "coordinate_mapping",
    "marker_reconstruction",
    "event_detection",
    "anthropometric_scaling",
)
_UNSAFE_PICKLE_SUFFIXES = {".pkl", ".pickle", ".joblib"}

IntendedUse = Literal["pipeline_probe", "held_out_qualification"]
PayloadLoader = Callable[[Path, str | None], Any]


@dataclass(frozen=True)
class GovernedTrajectoryArtifact:
    """A payload admitted through the measured-trajectory governance gate."""

    manifest_id: str
    source_id: str
    participant_id: str
    grouping_id: str
    cohort: str
    split_id: str
    split_manifest_sha256: str
    trial_id: str
    intended_use: IntendedUse
    source_package_sha256: str
    trajectory_sha256: str
    available_metric_ids: tuple[str, ...]
    unavailable_metric_ids: tuple[str, ...]
    missing_channel_ids: tuple[str, ...]
    payload: Any
    human_inference_ready: bool = False
    bilateral_wrench_gate_satisfied: bool = False


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    record = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates
    )
    if not isinstance(record, dict):
        raise ValueError("manifest must be a JSON object")
    return record


def _exact_keys(record: object, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError(f"{name} must contain exact keys {sorted(expected)}")
    return record


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def _digest_text(value: object, field: str) -> str:
    text = _text(value, field)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _utc_timestamp(value: object, field: str) -> str:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must identify UTC")
    return text


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return number


def _relative_safe_path(value: object, field: str) -> str:
    text = _text(value, field)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be contained beside the manifest")
    if path.suffix.lower() in _UNSAFE_PICKLE_SUFFIXES:
        raise ValueError(f"{field} must not use a pickle-based format")
    return text


def _validate_artifact(record: object) -> None:
    artifact = _exact_keys(record, _ARTIFACT_KEYS, "artifact")
    _relative_safe_path(
        artifact["source_package_relative_path"],
        "source_package_relative_path",
    )
    _relative_safe_path(
        artifact["trajectory_relative_path"], "trajectory_relative_path"
    )
    _digest_text(artifact["source_package_sha256"], "source_package_sha256")
    _digest_text(artifact["trajectory_sha256"], "trajectory_sha256")
    format_hint = _text(artifact["format_hint"], "format_hint")
    if re.fullmatch(r"[a-z0-9]+", format_hint) is None:
        raise ValueError("format_hint must be a lowercase format identifier")


def _validate_split_reference(record: object) -> None:
    reference = _exact_keys(
        record,
        _PARTICIPANT_SPLIT_REFERENCE_KEYS,
        "participant_split",
    )
    _relative_safe_path(reference["relative_path"], "participant split relative_path")
    _digest_text(reference["sha256"], "participant split sha256")


def _validate_participant(record: object) -> None:
    participant = _exact_keys(record, _PARTICIPANT_KEYS, "participant")
    for field in sorted(_PARTICIPANT_KEYS):
        _text(participant[field], field)


def _validate_acquisition(record: object) -> None:
    acquisition = _exact_keys(record, _ACQUISITION_KEYS, "acquisition")
    _text(acquisition["trial_id"], "trial_id")
    sample_rate = _finite_nonnegative(acquisition["sample_rate_hz"], "sample_rate_hz")
    if sample_rate == 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if acquisition["spatial_unit"] != "meters":
        raise ValueError("spatial_unit must be meters at the governed boundary")
    if acquisition["angle_unit"] != "radians":
        raise ValueError("angle_unit must be radians at the governed boundary")
    if acquisition["time_unit"] != "seconds":
        raise ValueError("time_unit must be seconds at the governed boundary")
    for field in (
        "synchronization_method",
        "filtering_method",
        "marker_reconstruction_method",
        "anthropometric_source",
    ):
        _text(acquisition[field], field)


def _validate_frames(rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("frames must be a list")
    ids: list[str] = []
    for row in rows:
        frame = _exact_keys(row, _FRAME_KEYS, "frame")
        frame_id = _text(frame["frame_id"], "frame_id")
        ids.append(frame_id)
        _text(frame["definition"], "definition")
        _text(frame["transform_authority"], "transform_authority")
        _digest_text(frame["transform_sha256"], "transform_sha256")
        _finite_nonnegative(
            frame["translation_uncertainty_m"], "translation_uncertainty_m"
        )
        _finite_nonnegative(
            frame["rotation_uncertainty_rad"], "rotation_uncertainty_rad"
        )
    if tuple(ids) != _FRAME_IDS:
        raise ValueError(f"frame order and coverage must be {_FRAME_IDS}")


def _validate_events(rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("events must be a list")
    ids: list[str] = []
    times: list[float] = []
    for row in rows:
        event = _exact_keys(row, _EVENT_KEYS, "event")
        ids.append(_text(event["event_id"], "event_id"))
        times.append(_finite_nonnegative(event["time_s"], "time_s"))
        _text(event["detector_id"], "detector_id")
        _text(event["detector_version"], "detector_version")
        _finite_nonnegative(event["uncertainty_s"], "uncertainty_s")
        if event["missing_policy"] != "unavailable_not_zero":
            raise ValueError("event missing_policy must be unavailable_not_zero")
    if tuple(ids) != _EVENT_IDS:
        raise ValueError(f"event order and coverage must be {_EVENT_IDS}")
    if times[1] <= times[0]:
        raise ValueError("impact time must follow downswing_start time")


def _validate_channels(channels: object) -> None:
    if (
        not isinstance(channels, list)
        or not channels
        or any(not isinstance(channel, str) or not channel for channel in channels)
        or len(channels) != len(set(channels))
    ):
        raise ValueError("channels must be a unique nonempty list of identifiers")
    if channels != sorted(channels):
        raise ValueError("channels must be sorted for deterministic comparison")
    if any(
        re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", item) is None
        for item in channels
    ):
        raise ValueError("channels must use lowercase underscore identifiers")


def _validate_uncertainties(rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("uncertainties must be a list")
    ids: list[str] = []
    for row in rows:
        uncertainty = _exact_keys(row, _UNCERTAINTY_KEYS, "uncertainty")
        ids.append(_text(uncertainty["analysis_id"], "analysis_id"))
        _text(uncertainty["method"], "method")
        lower = uncertainty["lower"]
        upper = uncertainty["upper"]
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) > float(upper)
        ):
            raise ValueError("uncertainty bounds must be finite and ordered")
        _text(uncertainty["unit"], "uncertainty unit")
    if tuple(ids) != _UNCERTAINTY_IDS:
        raise ValueError(f"uncertainty order and coverage must be {_UNCERTAINTY_IDS}")


def validate_artifact_manifest(path: Path) -> dict[str, Any]:
    """Validate one acquisition manifest loaded without duplicate JSON keys."""

    record = _load_json(Path(path))
    _exact_keys(record, _TOP_KEYS, "manifest")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("manifest schema_version is unsupported")
    manifest_id = _text(record["manifest_id"], "manifest_id")
    if _ID.fullmatch(manifest_id) is None:
        raise ValueError("manifest_id must be a lowercase hyphenated identifier")
    _utc_timestamp(record["created_at_utc"], "created_at_utc")
    if record["source_registry_id"] != "articulated-golf-trajectory-sources-v1":
        raise ValueError("source_registry_id is not registered")
    source_id = _text(record["source_id"], "source_id")
    if _ID.fullmatch(source_id) is None:
        raise ValueError("source_id must be a lowercase hyphenated identifier")
    _validate_split_reference(record["participant_split"])
    _validate_artifact(record["artifact"])
    _validate_participant(record["participant"])
    _validate_acquisition(record["acquisition"])
    _validate_frames(record["frames"])
    _validate_events(record["events"])
    _validate_channels(record["channels"])
    _validate_uncertainties(record["uncertainties"])
    if record["intended_use"] not in {"pipeline_probe", "held_out_qualification"}:
        raise ValueError("intended_use is not registered")
    boundary = _text(record["inference_boundary"], "inference_boundary").lower()
    for required in ("cannot", "human mechanism", "bilateral wrench", "coaching"):
        if required not in boundary:
            raise ValueError(f"inference_boundary must retain {required!r}")
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_file(manifest_path: Path, relative: str, field: str) -> Path:
    root = manifest_path.parent.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{field} must be contained beside the manifest")
    if not candidate.is_file():
        raise FileNotFoundError(f"{field} does not identify a file: {candidate}")
    return candidate


def _load_validated_registration(path: Path) -> dict[str, Any]:
    record = _load_json(path)
    validate_registration(record)
    return record


def _load_participant_split(
    manifest_path: Path,
    reference: dict[str, Any],
    *,
    source_id: str,
    registration: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    split_path = _contained_file(
        manifest_path,
        reference["relative_path"],
        "participant split relative_path",
    )
    split_digest = _sha256(split_path)
    if split_digest != reference["sha256"]:
        raise ValueError("participant split manifest digest does not match")
    split = _load_json(split_path)
    _exact_keys(split, _PARTICIPANT_SPLIT_KEYS, "participant split manifest")
    if split["schema_version"] != "measured-trajectory-participant-split/v1":
        raise ValueError("participant split schema_version is unsupported")
    split_id = _text(split["split_id"], "split_id")
    if _ID.fullmatch(split_id) is None:
        raise ValueError("split_id must be a lowercase hyphenated identifier")
    if split["source_id"] != source_id:
        raise ValueError(
            "participant split source_id does not match artifact source_id"
        )
    split_contract = registration["participant_split"]
    if split["assignment_method"] != split_contract["assignment"]:
        raise ValueError("participant split assignment method is not registered")
    _utc_timestamp(split["frozen_at_utc"], "participant split frozen_at_utc")
    cohorts = {
        "training": split["training_participant_ids"],
        "held_out": split["held_out_participant_ids"],
        "adverse": split["adverse_participant_ids"],
    }
    for name, participant_ids in cohorts.items():
        if not isinstance(participant_ids, list) or participant_ids != sorted(
            participant_ids
        ):
            raise ValueError(f"{name} participant identifiers must be a sorted list")
    counts = validate_participant_split(
        cohorts["training"],
        cohorts["held_out"],
        adverse=cohorts["adverse"],
    )
    if counts["training"] < split_contract["minimum_training_participants"]:
        raise ValueError("participant split has too few training participants")
    if counts["held_out"] < split_contract["minimum_held_out_participants"]:
        raise ValueError("participant split has too few held-out participants")
    if split_contract["adverse_cohort_required"] and counts["adverse"] == 0:
        raise ValueError("participant split requires an adverse cohort")
    return split, split_digest


def _require_participant_assignment(
    manifest: dict[str, Any], split: dict[str, Any]
) -> str:
    participant_id = manifest["participant"]["participant_id"]
    membership = {
        "training": participant_id in split["training_participant_ids"],
        "held_out": participant_id in split["held_out_participant_ids"],
        "adverse": participant_id in split["adverse_participant_ids"],
    }
    cohorts = [name for name, included in membership.items() if included]
    if len(cohorts) != 1:
        raise ValueError("participant must have exactly one split assignment")
    cohort = cohorts[0]
    if manifest["participant"]["cohort"] != cohort:
        raise ValueError("artifact cohort does not match participant split assignment")
    intended_use = manifest["intended_use"]
    if intended_use == "pipeline_probe" and cohort != "training":
        raise ValueError("pipeline probes require a training participant")
    if intended_use == "held_out_qualification" and cohort not in {
        "held_out",
        "adverse",
    }:
        raise ValueError("held-out qualification requires a held-out participant")
    return cohort


def _metric_coverage(
    registration: dict[str, Any], channels: set[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    available: list[str] = []
    unavailable: list[str] = []
    missing: set[str] = set()
    for metric in registration["metrics"]:
        required = set(metric["required_channels"])
        if required <= channels:
            available.append(metric["metric_id"])
        else:
            unavailable.append(metric["metric_id"])
            missing.update(required - channels)
    return tuple(available), tuple(unavailable), tuple(sorted(missing))


def load_governed_trajectory(
    manifest_path: Path,
    source_registry_path: Path,
    metric_registration_path: Path,
    *,
    payload_loader: PayloadLoader | None = None,
) -> GovernedTrajectoryArtifact:
    """Admit and load one measured trajectory after all authority checks pass."""

    manifest_path = Path(manifest_path)
    manifest = validate_artifact_manifest(manifest_path)
    source_registry = _load_json(Path(source_registry_path))
    source_readiness = validate_registry(source_registry)
    registration = _load_validated_registration(Path(metric_registration_path))
    if registration["authority_status"] != source_readiness["status"]:
        raise ValueError("metric registration and source authority status disagree")

    source_id = manifest["source_id"]
    source_record = next(
        (row for row in source_registry["sources"] if row["source_id"] == source_id),
        None,
    )
    if source_record is None:
        raise ValueError(f"source_id {source_id!r} is not registered")

    intended_use: IntendedUse = manifest["intended_use"]
    ready_key = (
        "pipeline_probe_ready_source_ids"
        if intended_use == "pipeline_probe"
        else "held_out_qualification_ready_source_ids"
    )
    if source_id not in source_readiness[ready_key]:
        label = (
            "pipeline probing"
            if intended_use == "pipeline_probe"
            else "held-out qualification"
        )
        raise ValueError(f"source {source_id!r} is not ready for {label}")

    split, split_digest = _load_participant_split(
        manifest_path,
        manifest["participant_split"],
        source_id=source_id,
        registration=registration,
    )
    cohort = _require_participant_assignment(manifest, split)

    artifact = manifest["artifact"]
    package_path = _contained_file(
        manifest_path,
        artifact["source_package_relative_path"],
        "source_package_relative_path",
    )
    trajectory_path = _contained_file(
        manifest_path,
        artifact["trajectory_relative_path"],
        "trajectory_relative_path",
    )
    package_digest = _sha256(package_path)
    trajectory_digest = _sha256(trajectory_path)
    if package_digest != artifact["source_package_sha256"]:
        raise ValueError("source package digest does not match artifact manifest")
    if package_digest != source_record["content_digest_sha256"]:
        raise ValueError("source package digest does not match source registry")
    if trajectory_digest != artifact["trajectory_sha256"]:
        raise ValueError("trajectory digest does not match artifact manifest")

    if payload_loader is None:
        from src.shared.python.motion_pipeline.sources.loader import load_source

        payload_loader = load_source
    payload = payload_loader(trajectory_path, artifact["format_hint"])
    available, unavailable, missing = _metric_coverage(
        registration, set(manifest["channels"])
    )
    participant = manifest["participant"]
    acquisition = manifest["acquisition"]
    return GovernedTrajectoryArtifact(
        manifest_id=manifest["manifest_id"],
        source_id=source_id,
        participant_id=participant["participant_id"],
        grouping_id=participant["grouping_id"],
        cohort=cohort,
        split_id=split["split_id"],
        split_manifest_sha256=split_digest,
        trial_id=acquisition["trial_id"],
        intended_use=intended_use,
        source_package_sha256=package_digest,
        trajectory_sha256=trajectory_digest,
        available_metric_ids=available,
        unavailable_metric_ids=unavailable,
        missing_channel_ids=missing,
        payload=payload,
    )
