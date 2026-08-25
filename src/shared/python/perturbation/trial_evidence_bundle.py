"""Atomic, digest-verifiable persistence for canonical variation trials."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .trial_evidence import (
    TRIAL_EVIDENCE_SCHEMA_VERSION,
    CanonicalTrialEvidence,
    ClosestApproach,
    ImpactObservation,
    SampledInput,
    TrialOutcome,
    TrialTrace,
)

BUNDLE_SCHEMA_VERSION = "upstream-tools-variation-bundle/v1"
_OUTCOMES: tuple[TrialOutcome, ...] = (
    "hit",
    "no_impact",
    "numerical_failure",
    "partial_valid_trace",
)
_IDENTITY_FIELDS = (
    "seed",
    "plan_sha256",
    "scenario_sha256",
    "execution_config_sha256",
    "tools_revision",
    "engine_id",
    "engine_revision",
    "model_id",
)
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "trial_schema_version",
    "identity",
    "sample_schema",
    "trial_count",
    "outcome_counts",
    "trials",
    "content_sha256",
}


@dataclass(frozen=True)
class TrialEvidenceBundleSummary:
    """Validated bundle location, content digest, and trial count."""

    path: Path
    content_sha256: str
    trial_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be Path")
        if (
            not isinstance(self.content_sha256, str)
            or len(self.content_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.content_sha256
            )
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if type(self.trial_count) is not int or self.trial_count <= 0:
            raise ValueError("trial_count must be a positive integer")


def write_trial_evidence_bundle(
    destination: Path,
    records: tuple[CanonicalTrialEvidence, ...],
) -> TrialEvidenceBundleSummary:
    """Write a new bundle atomically without replacing existing content.

    Postconditions: ``destination`` contains only the canonical manifest and
    registered ``.npy`` arrays, and loading it reproduces all typed records.
    """
    target = _validate_destination(destination)
    identity, sample_schema = _validate_records(records)
    staging = Path(tempfile.mkdtemp(prefix=".trial-bundle-", dir=target.parent))
    try:
        arrays_directory = staging / "arrays"
        arrays_directory.mkdir()
        trial_payloads = [
            _serialize_trial(record, arrays_directory) for record in records
        ]
        payload: dict[str, Any] = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "trial_schema_version": TRIAL_EVIDENCE_SCHEMA_VERSION,
            "identity": identity,
            "sample_schema": sample_schema,
            "trial_count": len(records),
            "outcome_counts": _outcome_counts(records),
            "trials": trial_payloads,
        }
        content_sha256 = hashlib.sha256(_canonical_json(payload)).hexdigest()
        manifest = {**payload, "content_sha256": content_sha256}
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return TrialEvidenceBundleSummary(target, content_sha256, len(records))


def load_trial_evidence_bundle(
    source: Path,
) -> tuple[CanonicalTrialEvidence, ...]:
    """Load a complete bundle after validating inventory, digests, and schema."""
    root, manifest = _load_manifest(source)
    _require_exact_keys(manifest, _TOP_LEVEL_FIELDS, "manifest")
    if manifest["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError("bundle schema version is incompatible")
    if manifest["trial_schema_version"] != TRIAL_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("trial schema version is incompatible")
    content_sha256 = manifest["content_sha256"]
    if not isinstance(content_sha256, str):
        raise ValueError("bundle content digest is invalid")
    unsigned = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != content_sha256:
        raise ValueError("bundle content digest does not match the manifest")

    trials = manifest["trials"]
    if not isinstance(trials, list) or not trials:
        raise ValueError("bundle trials must be a non-empty list")
    if type(manifest["trial_count"]) is not int or manifest["trial_count"] <= 0:
        raise ValueError("bundle trial_count must be a positive integer")
    outcome_counts = manifest["outcome_counts"]
    if not isinstance(outcome_counts, dict) or set(outcome_counts) != set(_OUTCOMES):
        raise ValueError("bundle outcome_counts fields are incompatible")
    if any(type(value) is not int or value < 0 for value in outcome_counts.values()):
        raise ValueError("bundle outcome_counts values must be non-negative integers")
    sample_schema = manifest["sample_schema"]
    if not isinstance(sample_schema, list) or not sample_schema:
        raise ValueError("bundle sample_schema must be a non-empty list")
    for item in sample_schema:
        if not isinstance(item, dict):
            raise ValueError("bundle sample_schema entries must be objects")
        _require_exact_keys(item, {"name", "unit"}, "sample_schema entry")
    manifest_identity = manifest["identity"]
    if not isinstance(manifest_identity, dict):
        raise ValueError("bundle identity must be an object")
    _require_exact_keys(manifest_identity, set(_IDENTITY_FIELDS), "identity")
    descriptors = _array_descriptors(trials)
    expected_files = {
        "manifest.json",
        *(descriptor["path"] for descriptor in descriptors),
    }
    observed_files = _observed_file_inventory(root)
    if observed_files != expected_files:
        raise ValueError("bundle file inventory does not match the manifest")
    arrays = {
        descriptor["path"]: _load_array(root, descriptor) for descriptor in descriptors
    }
    records = tuple(
        _deserialize_trial(item, arrays, manifest_identity) for item in trials
    )
    identity, sample_schema = _validate_records(records)
    if identity != manifest_identity or sample_schema != manifest["sample_schema"]:
        raise ValueError("bundle execution identity does not match trial records")
    if manifest["trial_count"] != len(records):
        raise ValueError("bundle trial_count does not match trial records")
    if manifest["outcome_counts"] != _outcome_counts(records):
        raise ValueError("bundle outcome_counts do not match trial records")
    return records


def validate_trial_evidence_bundle(source: Path) -> TrialEvidenceBundleSummary:
    """Validate ``source`` and return its immutable summary."""
    root, manifest = _load_manifest(source)
    records = load_trial_evidence_bundle(root)
    return TrialEvidenceBundleSummary(
        root,
        cast(str, manifest["content_sha256"]),
        len(records),
    )


def _validate_destination(destination: Path) -> Path:
    if not isinstance(destination, Path):
        raise TypeError("destination must be Path")
    target = destination.resolve()
    if target.exists():
        raise FileExistsError(f"destination already exists: {target}")
    if not target.parent.is_dir():
        raise ValueError("destination parent must be an existing directory")
    return target


def _validate_records(
    records: tuple[CanonicalTrialEvidence, ...],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    if not isinstance(records, tuple) or not records:
        raise ValueError("records must be a non-empty tuple")
    if any(not isinstance(record, CanonicalTrialEvidence) for record in records):
        raise TypeError("records must contain CanonicalTrialEvidence")
    indices = tuple(record.trial_index for record in records)
    if indices != tuple(range(len(records))):
        raise ValueError("trial indices must be contiguous and ordered from zero")
    first = records[0]
    identity = {field: getattr(first, field) for field in _IDENTITY_FIELDS}
    sample_schema = [
        {"name": value.name, "unit": value.unit} for value in first.sampled_inputs
    ]
    for record in records[1:]:
        if any(getattr(record, field) != identity[field] for field in _IDENTITY_FIELDS):
            raise ValueError("records must share one execution identity")
        current_schema = [
            {"name": value.name, "unit": value.unit} for value in record.sampled_inputs
        ]
        if current_schema != sample_schema:
            raise ValueError("records must share one sampled-input schema")
    return identity, sample_schema


def _outcome_counts(
    records: tuple[CanonicalTrialEvidence, ...],
) -> dict[str, int]:
    counts = Counter(record.outcome for record in records)
    return {outcome: counts[outcome] for outcome in _OUTCOMES}


def _serialize_trial(
    record: CanonicalTrialEvidence,
    arrays_directory: Path,
) -> dict[str, object]:
    trace = None
    if record.trace is not None:
        prefix = f"trial-{record.trial_index:06d}"
        trace = {
            "coordinate_ids": record.trace.coordinate_ids,
            "coordinate_units": record.trace.coordinate_units,
            "velocity_units": record.trace.velocity_units,
            "marker_ids": record.trace.marker_ids,
            "frame_id": record.trace.frame_id,
            "alignment_id": record.trace.alignment_id,
            "complete": record.trace.complete,
            "arrays": {
                "times_s": _write_array(
                    arrays_directory, prefix, "times_s", record.trace.times_s
                ),
                "q": _write_array(arrays_directory, prefix, "q", record.trace.q),
                "v": _write_array(arrays_directory, prefix, "v", record.trace.v),
                "markers_m": _write_array(
                    arrays_directory, prefix, "markers_m", record.trace.markers_m
                ),
            },
        }
    return {
        "trial_index": record.trial_index,
        "sampled_inputs": _serialize_samples(record.sampled_inputs),
        "outcome": record.outcome,
        "trace": trace,
        "impact": _serialize_impact(record.impact),
        "shot_result": (
            None
            if record.shot_result is None
            else _serialize_samples(record.shot_result)
        ),
        "closest_approach": _serialize_closest(record.closest_approach),
        "failure_reason": record.failure_reason,
    }


def _write_array(
    directory: Path,
    prefix: str,
    name: str,
    values: np.ndarray,
) -> dict[str, object]:
    filename = f"{prefix}-{name}.npy"
    path = directory / filename
    with path.open("wb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)
    return {
        "path": f"arrays/{filename}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def _serialize_samples(samples: tuple[SampledInput, ...]) -> list[dict[str, object]]:
    return [
        {"name": sample.name, "value": sample.value, "unit": sample.unit}
        for sample in samples
    ]


def _serialize_impact(impact: ImpactObservation | None) -> dict[str, object] | None:
    if impact is None:
        return None
    return {"time_s": impact.time_s, "state": _serialize_samples(impact.state)}


def _serialize_closest(closest: ClosestApproach | None) -> dict[str, object] | None:
    if closest is None:
        return None
    return {
        "time_s": closest.time_s,
        "distance_m": closest.distance_m,
        "source_marker_id": closest.source_marker_id,
        "target_id": closest.target_id,
        "contact_observed": closest.contact_observed,
    }


def _load_manifest(source: Path) -> tuple[Path, dict[str, Any]]:
    if not isinstance(source, Path):
        raise TypeError("source must be Path")
    root = source.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("source must be a non-symlink bundle directory")
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("bundle manifest cannot be read") from error
    if not isinstance(manifest, dict):
        raise ValueError("bundle manifest must be an object")
    return root, manifest


def _array_descriptors(trials: list[object]) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("bundle trial entries must be objects")
        trace = trial.get("trace")
        if trace is None:
            continue
        if not isinstance(trace, dict) or not isinstance(trace.get("arrays"), dict):
            raise ValueError("bundle trace arrays must be an object")
        arrays = trace["arrays"]
        if set(arrays) != {"times_s", "q", "v", "markers_m"}:
            raise ValueError("bundle trace array fields are incomplete")
        for descriptor in arrays.values():
            if not isinstance(descriptor, dict):
                raise ValueError("bundle array descriptor must be an object")
            _require_exact_keys(
                descriptor, {"path", "sha256", "shape", "dtype"}, "array descriptor"
            )
            descriptors.append(descriptor)
    paths = [descriptor["path"] for descriptor in descriptors]
    if any(not isinstance(path, str) for path in paths) or len(set(paths)) != len(
        paths
    ):
        raise ValueError("bundle array paths must be unique strings")
    return descriptors


def _observed_file_inventory(root: Path) -> set[str]:
    inventory: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("bundle must not contain symbolic links")
        if path.is_file():
            inventory.add(path.relative_to(root).as_posix())
    return inventory


def _load_array(root: Path, descriptor: dict[str, Any]) -> np.ndarray:
    relative = Path(cast(str, descriptor["path"]))
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != "arrays"
        or ".." in relative.parts
    ):
        raise ValueError("bundle array path is outside the arrays directory")
    path = root / relative
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("bundle array cannot be read") from error
    if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise ValueError("bundle array digest does not match the manifest")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("bundle array payload is invalid") from error
    if not isinstance(array, np.ndarray):
        raise ValueError("bundle array payload must be an ndarray")
    if (
        list(array.shape) != descriptor["shape"]
        or str(array.dtype) != descriptor["dtype"]
    ):
        raise ValueError("bundle array shape or dtype does not match the manifest")
    return array


def _deserialize_trial(
    raw: object,
    arrays: dict[str, np.ndarray],
    identity: dict[str, object],
) -> CanonicalTrialEvidence:
    if not isinstance(raw, dict):
        raise ValueError("bundle trial must be an object")
    _require_exact_keys(
        raw,
        {
            "trial_index",
            "sampled_inputs",
            "outcome",
            "trace",
            "impact",
            "shot_result",
            "closest_approach",
            "failure_reason",
        },
        "trial",
    )
    trace = _deserialize_trace(raw["trace"], arrays)
    return CanonicalTrialEvidence(
        seed=cast(int, identity["seed"]),
        plan_sha256=cast(str, identity["plan_sha256"]),
        scenario_sha256=cast(str, identity["scenario_sha256"]),
        execution_config_sha256=cast(str, identity["execution_config_sha256"]),
        tools_revision=cast(str, identity["tools_revision"]),
        engine_id=cast(str, identity["engine_id"]),
        engine_revision=cast(str, identity["engine_revision"]),
        model_id=cast(str, identity["model_id"]),
        trial_index=raw["trial_index"],
        sampled_inputs=_deserialize_samples(raw["sampled_inputs"]),
        outcome=cast(TrialOutcome, raw["outcome"]),
        trace=trace,
        impact=_deserialize_impact(raw["impact"]),
        shot_result=(
            None
            if raw["shot_result"] is None
            else _deserialize_samples(raw["shot_result"])
        ),
        closest_approach=_deserialize_closest(raw["closest_approach"]),
        failure_reason=raw["failure_reason"],
    )


def _deserialize_trace(raw: object, arrays: dict[str, np.ndarray]) -> TrialTrace | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("bundle trace must be an object")
    _require_exact_keys(
        raw,
        {
            "coordinate_ids",
            "coordinate_units",
            "velocity_units",
            "marker_ids",
            "frame_id",
            "alignment_id",
            "complete",
            "arrays",
        },
        "trace",
    )
    descriptors = raw["arrays"]
    if not isinstance(descriptors, dict):
        raise ValueError("bundle trace arrays must be an object")
    return TrialTrace(
        times_s=arrays[descriptors["times_s"]["path"]],
        q=arrays[descriptors["q"]["path"]],
        v=arrays[descriptors["v"]["path"]],
        coordinate_ids=tuple(raw["coordinate_ids"]),
        coordinate_units=tuple(raw["coordinate_units"]),
        velocity_units=tuple(raw["velocity_units"]),
        markers_m=arrays[descriptors["markers_m"]["path"]],
        marker_ids=tuple(raw["marker_ids"]),
        frame_id=raw["frame_id"],
        alignment_id=raw["alignment_id"],
        complete=raw["complete"],
    )


def _deserialize_samples(raw: object) -> tuple[SampledInput, ...]:
    if not isinstance(raw, list):
        raise ValueError("sample collection must be a list")
    samples: list[SampledInput] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("sample entry must be an object")
        _require_exact_keys(item, {"name", "value", "unit"}, "sample")
        samples.append(SampledInput(item["name"], item["value"], item["unit"]))
    return tuple(samples)


def _deserialize_impact(raw: object) -> ImpactObservation | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("impact must be an object")
    _require_exact_keys(raw, {"time_s", "state"}, "impact")
    return ImpactObservation(raw["time_s"], _deserialize_samples(raw["state"]))


def _deserialize_closest(raw: object) -> ClosestApproach | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("closest_approach must be an object")
    _require_exact_keys(
        raw,
        {
            "time_s",
            "distance_m",
            "source_marker_id",
            "target_id",
            "contact_observed",
        },
        "closest_approach",
    )
    return ClosestApproach(
        raw["time_s"],
        raw["distance_m"],
        raw["source_marker_id"],
        raw["target_id"],
        raw["contact_observed"],
    )


def _require_exact_keys(raw: dict[str, Any], expected: set[str], name: str) -> None:
    if set(raw) != expected:
        raise ValueError(f"{name} fields do not match the registered schema")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "TrialEvidenceBundleSummary",
    "load_trial_evidence_bundle",
    "validate_trial_evidence_bundle",
    "write_trial_evidence_bundle",
]
