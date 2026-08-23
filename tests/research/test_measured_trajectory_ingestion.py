from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research.proximal_distal_energy.measured_trajectory_ingestion import (
    load_governed_trajectory,
    validate_artifact_manifest,
)
from scripts.research.proximal_distal_energy.measured_trajectory_source_registry import (
    compute_readiness,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REGISTRY = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data"
    / "measured_trajectory_source_registry.json"
)
METRIC_REGISTRATION = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data"
    / "measured_trajectory_metric_registration.json"
)
pytestmark = pytest.mark.scientific


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualified_registry(tmp_path: Path, package_digest: str) -> Path:
    record = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    source = next(row for row in record["sources"] if row["source_id"] == "golfpose")
    source.update(
        {
            "access_status": "authorized_download",
            "license_status": "explicit_reuse_license",
            "participant_count": 6,
            "trial_count": 12,
            "content_digest_sha256": package_digest,
            "blockers": [],
        }
    )
    record["readiness"] = compute_readiness(record["sources"])
    path = tmp_path / "measured_trajectory_source_registry.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _qualified_registration(tmp_path: Path) -> Path:
    record = json.loads(METRIC_REGISTRATION.read_text(encoding="utf-8"))
    record["authority_status"] = "motion_only_held_out_authority_available"
    record["results_status"] = "registered_not_run"
    record["readiness"] = {
        **record["readiness"],
        "status": "motion_only_held_out_authority_available",
        "execution_ready": True,
    }
    path = tmp_path / "measured_trajectory_metric_registration.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _all_required_channels() -> list[str]:
    registration = json.loads(METRIC_REGISTRATION.read_text(encoding="utf-8"))
    return sorted(
        {
            channel
            for metric in registration["metrics"]
            for channel in metric["required_channels"]
        }
    )


def _split_manifest(tmp_path: Path, intended_use: str) -> Path:
    if intended_use == "pipeline_probe":
        training = ["p01", "p03"]
        held_out = ["p02"]
    else:
        training = ["p02", "p03"]
        held_out = ["p01"]
    path = tmp_path / "participant_split.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "measured-trajectory-participant-split/v1",
                "split_id": "golfpose-primary-split",
                "source_id": "golfpose",
                "assignment_method": "deterministic_digest",
                "frozen_at_utc": "2026-08-23T18:00:00Z",
                "training_participant_ids": training,
                "held_out_participant_ids": held_out,
                "adverse_participant_ids": ["p04"],
            }
        ),
        encoding="utf-8",
    )
    return path


def _manifest(
    package: Path,
    trajectory: Path,
    *,
    channels: list[str] | None = None,
    intended_use: str = "held_out_qualification",
) -> dict[str, Any]:
    split = _split_manifest(package.parent, intended_use)
    cohort = "training" if intended_use == "pipeline_probe" else "held_out"
    return {
        "schema_version": "measured-trajectory-artifact/v1",
        "manifest_id": "golfpose-p01-trial-01",
        "created_at_utc": "2026-08-23T18:00:00Z",
        "source_registry_id": "articulated-golf-trajectory-sources-v1",
        "source_id": "golfpose",
        "participant_split": {
            "relative_path": split.name,
            "sha256": _digest(split),
        },
        "artifact": {
            "source_package_relative_path": package.name,
            "source_package_sha256": _digest(package),
            "trajectory_relative_path": trajectory.name,
            "trajectory_sha256": _digest(trajectory),
            "format_hint": "c3d",
        },
        "participant": {
            "participant_id": "p01",
            "grouping_id": "participant-p01",
            "cohort": cohort,
        },
        "acquisition": {
            "trial_id": "trial-01",
            "sample_rate_hz": 240.0,
            "spatial_unit": "meters",
            "angle_unit": "radians",
            "time_unit": "seconds",
            "synchronization_method": "hardware_trigger",
            "filtering_method": "unfiltered_raw_then_registered_pipeline",
            "marker_reconstruction_method": "dataset_authority_rigid_body",
            "anthropometric_source": "participant_calibration",
        },
        "frames": [
            {
                "frame_id": frame_id,
                "definition": f"governed {frame_id} definition",
                "transform_authority": f"governed {frame_id} transform",
                "transform_sha256": "1" * 64,
                "translation_uncertainty_m": 0.002,
                "rotation_uncertainty_rad": 0.01,
            }
            for frame_id in ("lab", "anatomical", "model", "club")
        ],
        "events": [
            {
                "event_id": event_id,
                "time_s": time_s,
                "detector_id": f"{event_id}-detector",
                "detector_version": "1.0.0",
                "uncertainty_s": 0.002,
                "missing_policy": "unavailable_not_zero",
            }
            for event_id, time_s in (("downswing_start", 0.0), ("impact", 0.25))
        ],
        "channels": channels if channels is not None else _all_required_channels(),
        "uncertainties": [
            {
                "analysis_id": analysis_id,
                "method": f"governed {analysis_id} interval",
                "lower": -0.01,
                "upper": 0.01,
                "unit": "declared_by_method",
            }
            for analysis_id in (
                "time_alignment",
                "filtering",
                "coordinate_mapping",
                "marker_reconstruction",
                "event_detection",
                "anthropometric_scaling",
            )
        ],
        "intended_use": intended_use,
        "inference_boundary": (
            "This artifact cannot establish human mechanism, bilateral wrench "
            "allocation, or coaching guidance."
        ),
    }


def _write_manifest(tmp_path: Path, record: dict[str, Any]) -> Path:
    path = tmp_path / "artifact_manifest.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_governed_loader_checks_authority_and_digests_before_loading(
    tmp_path: Path,
) -> None:
    package = tmp_path / "authorized-source.zip"
    trajectory = tmp_path / "trial.c3d"
    package.write_bytes(b"authorized source package")
    trajectory.write_bytes(b"governed c3d payload")
    registry = _qualified_registry(tmp_path, _digest(package))
    registration = _qualified_registration(tmp_path)
    manifest = _write_manifest(tmp_path, _manifest(package, trajectory))
    sentinel = object()
    calls: list[tuple[Path, str | None]] = []

    def loader(path: Path, format_hint: str | None = None) -> object:
        calls.append((path, format_hint))
        return sentinel

    result = load_governed_trajectory(
        manifest,
        registry,
        registration,
        payload_loader=loader,
    )

    assert result.payload is sentinel
    assert calls == [(trajectory.resolve(), "c3d")]
    assert result.available_metric_ids
    assert result.unavailable_metric_ids == ()
    assert result.cohort == "held_out"
    assert result.split_id == "golfpose-primary-split"
    assert result.split_manifest_sha256 == _digest(tmp_path / "participant_split.json")
    assert result.human_inference_ready is False
    assert result.bilateral_wrench_gate_satisfied is False


def test_committed_registry_blocks_before_payload_loader(tmp_path: Path) -> None:
    package = tmp_path / "source.zip"
    trajectory = tmp_path / "trial.c3d"
    package.write_bytes(b"source")
    trajectory.write_bytes(b"trajectory")
    manifest = _write_manifest(tmp_path, _manifest(package, trajectory))
    called = False

    def loader(path: Path, format_hint: str | None = None) -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(ValueError, match="not ready for held-out qualification"):
        load_governed_trajectory(
            manifest,
            SOURCE_REGISTRY,
            METRIC_REGISTRATION,
            payload_loader=loader,
        )
    assert called is False


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda row: row["artifact"].update(trajectory_sha256="0" * 64), "digest"),
        (
            lambda row: row["artifact"].update(
                trajectory_relative_path="../outside.c3d"
            ),
            "contained",
        ),
        (
            lambda row: row["artifact"].update(trajectory_relative_path="trial.pkl"),
            "pickle",
        ),
        (lambda row: row["frames"].pop(), "frame"),
        (lambda row: row["events"].pop(), "event"),
        (lambda row: row["uncertainties"].pop(), "uncertainty"),
        (lambda row: row.update(channels=["club_position_lab_m"] * 2), "unique"),
        (lambda row: row.update(created_at_utc="not-a-time"), "ISO-8601"),
        (lambda row: row["artifact"].update(format_hint="C3D file"), "lowercase"),
        (lambda row: row.update(channels=["Club_Position"]), "lowercase"),
    ],
)
def test_manifest_and_loader_fail_closed(
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    package = tmp_path / "source.zip"
    trajectory = tmp_path / "trial.c3d"
    package.write_bytes(b"source")
    trajectory.write_bytes(b"trajectory")
    registry = _qualified_registry(tmp_path, _digest(package))
    registration = _qualified_registration(tmp_path)
    record = _manifest(package, trajectory)
    mutate(record)
    manifest = _write_manifest(tmp_path, record)

    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_governed_trajectory(
            manifest,
            registry,
            registration,
            payload_loader=lambda *_args, **_kwargs: object(),
        )


def test_missing_channels_are_unavailable_not_zero(tmp_path: Path) -> None:
    package = tmp_path / "source.zip"
    trajectory = tmp_path / "trial.c3d"
    package.write_bytes(b"source")
    trajectory.write_bytes(b"trajectory")
    registry = _qualified_registry(tmp_path, _digest(package))
    registration = _qualified_registration(tmp_path)
    record = _manifest(
        package,
        trajectory,
        channels=["club_orientation_lab_wxyz", "club_position_lab_m"],
        intended_use="pipeline_probe",
    )
    manifest = _write_manifest(tmp_path, record)

    result = load_governed_trajectory(
        manifest,
        registry,
        registration,
        payload_loader=lambda *_args, **_kwargs: object(),
    )

    assert result.available_metric_ids == ("club_pose",)
    assert "club_speed" in result.unavailable_metric_ids
    assert "club_linear_velocity_lab_m_s" in result.missing_channel_ids


def test_split_digest_and_participant_assignment_are_authoritative(
    tmp_path: Path,
) -> None:
    package = tmp_path / "source.zip"
    trajectory = tmp_path / "trial.c3d"
    package.write_bytes(b"source")
    trajectory.write_bytes(b"trajectory")
    registry = _qualified_registry(tmp_path, _digest(package))
    registration = _qualified_registration(tmp_path)
    record = _manifest(package, trajectory)
    split_path = tmp_path / record["participant_split"]["relative_path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["training_participant_ids"] = ["p01", "p03"]
    split["held_out_participant_ids"] = ["p02"]
    split_path.write_text(json.dumps(split), encoding="utf-8")
    record["participant_split"]["sha256"] = _digest(split_path)
    record["participant"]["cohort"] = "training"
    manifest = _write_manifest(tmp_path, record)
    called = False

    def loader(*_args: Any, **_kwargs: Any) -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(ValueError, match="held-out participant"):
        load_governed_trajectory(
            manifest,
            registry,
            registration,
            payload_loader=loader,
        )
    assert called is False


def test_split_digest_mismatch_blocks_before_payload_loader(tmp_path: Path) -> None:
    package = tmp_path / "source.zip"
    trajectory = tmp_path / "trial.c3d"
    package.write_bytes(b"source")
    trajectory.write_bytes(b"trajectory")
    registry = _qualified_registry(tmp_path, _digest(package))
    registration = _qualified_registration(tmp_path)
    record = _manifest(package, trajectory)
    record["participant_split"]["sha256"] = "0" * 64
    manifest = _write_manifest(tmp_path, record)
    called = False

    def loader(*_args: Any, **_kwargs: Any) -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(ValueError, match="split manifest digest"):
        load_governed_trajectory(
            manifest,
            registry,
            registration,
            payload_loader=loader,
        )
    assert called is False


def test_split_must_be_frozen_before_artifact_creation(tmp_path: Path) -> None:
    package = tmp_path / "source.zip"
    trajectory = tmp_path / "trial.c3d"
    package.write_bytes(b"source")
    trajectory.write_bytes(b"trajectory")
    registry = _qualified_registry(tmp_path, _digest(package))
    registration = _qualified_registration(tmp_path)
    record = _manifest(package, trajectory)
    split_path = tmp_path / record["participant_split"]["relative_path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["frozen_at_utc"] = "2026-08-24T18:00:00Z"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    record["participant_split"]["sha256"] = _digest(split_path)
    manifest = _write_manifest(tmp_path, record)

    with pytest.raises(ValueError, match="frozen before artifact creation"):
        load_governed_trajectory(
            manifest,
            registry,
            registration,
            payload_loader=lambda *_args, **_kwargs: object(),
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "duplicate.json"
    manifest.write_text(
        '{"schema_version":"measured-trajectory-artifact/v1",'
        '"schema_version":"measured-trajectory-artifact/v1"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_artifact_manifest(manifest)


def test_manifest_cannot_claim_human_or_wrench_authority(tmp_path: Path) -> None:
    package = tmp_path / "source.zip"
    trajectory = tmp_path / "trial.c3d"
    package.write_bytes(b"source")
    trajectory.write_bytes(b"trajectory")
    record = _manifest(package, trajectory)
    record["human_inference_ready"] = True
    manifest = _write_manifest(tmp_path, record)
    with pytest.raises(ValueError, match="exact keys"):
        validate_artifact_manifest(manifest)
