"""Regenerate registered anthropometric and joint-limit authority corners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    DATA,
    ScaledAuthority,
    ScaledAuthorityConfig,
    build_scaled_authority,
    load_scaled_authority,
    save_scaled_authority,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECORD = DATA / "articulated_structural_authority_campaign.json"
SOURCE_PATHS = (
    "scripts/research/proximal_distal_energy/articulated_structural_authority_campaign.py",
    "scripts/research/proximal_distal_energy/articulated_scaled_authority.py",
    "tests/research/test_articulated_structural_authority_campaign.py",
)
TERMINAL_STATUSES = {"feasible", "infeasible_retained", "failed_retained"}


@dataclass(frozen=True, slots=True)
class StructuralAuthorityCorner:
    """One registered nominal or one-at-a-time structural corner."""

    corner_id: str
    axis_name: str
    level: str
    value: float
    configuration: ScaledAuthorityConfig


def registered_corners() -> tuple[StructuralAuthorityCorner, ...]:
    """Return nominal plus the six issue-registered low/high OAT corners."""

    specifications = (
        ("height_scale", 0.90, 1.10),
        ("body_mass_scale", 0.85, 1.15),
        ("joint_limit_scale", 0.85, 1.15),
    )
    result = [
        StructuralAuthorityCorner(
            "nominal", "nominal", "nominal", 1.0, ScaledAuthorityConfig()
        )
    ]
    for axis_name, low, high in specifications:
        for level, value in (("low", low), ("high", high)):
            values = {axis_name: value}
            result.append(
                StructuralAuthorityCorner(
                    f"{axis_name}-{level}",
                    axis_name,
                    level,
                    value,
                    ScaledAuthorityConfig(**values),
                )
            )
    return tuple(result)


def _source_hashes() -> dict[str, str]:
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _design_digest() -> str:
    payload = [asdict(corner) for corner in registered_corners()]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _artifact_stem(corner: StructuralAuthorityCorner) -> str:
    return f"articulated_structural_authority_{corner.corner_id.replace('-', '_')}"


def _failure_distribution(authority: ScaledAuthority) -> dict[str, int]:
    selected = authority.selected_case_indices
    failed = ~authority.feasible[selected]
    values = authority.selected_failure_class[failed]
    classes, counts = np.unique(values, return_counts=True)
    return {str(name): int(count) for name, count in zip(classes, counts, strict=True)}


def _corner_record(
    corner: StructuralAuthorityCorner,
    authority: ScaledAuthority,
    record_path: Path,
    arrays_path: Path,
) -> dict[str, Any]:
    distribution = _failure_distribution(authority)
    return {
        "corner_id": corner.corner_id,
        "axis_name": corner.axis_name,
        "level": corner.level,
        "value": corner.value,
        "configuration": _jsonable(asdict(corner.configuration)),
        "status": "infeasible_retained" if distribution else "feasible",
        "failure_count": sum(distribution.values()),
        "failure_distribution": distribution,
        "authority_sha256": authority.authority_sha256,
        "record_artifact": record_path.name,
        "array_artifact": arrays_path.name,
    }


def _campaign_record(
    rows: list[dict[str, Any]],
    status: str,
    execution_sources: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "articulated-structural-authority-campaign/v1",
        "study_id": "articulated-structural-headline-authority-corners",
        "status": status,
        "design_sha256": _design_digest(),
        "design": _jsonable([asdict(corner) for corner in registered_corners()]),
        "corners": rows,
        "results": (
            {
                "corner_count": len(rows),
                "feasible_corner_count": sum(
                    row["status"] == "feasible" for row in rows
                ),
                "infeasible_corner_count": sum(
                    row["status"] == "infeasible_retained" for row in rows
                ),
                "failed_corner_count": sum(
                    row["status"] == "failed_retained" for row in rows
                ),
            }
            if status == "complete"
            else None
        ),
        "source_sha256": execution_sources,
        "limitations": {
            "bounds": "engineering OAT corners, not participant distributions",
            "dynamics": "authority regeneration is not headline atlas propagation",
            "human_inference": "no human, physiological, or coaching inference",
        },
    }


def _write_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _existing_rows(
    path: Path, expected_sources: dict[str, str]
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "articulated-structural-authority-campaign/v1":
        raise RuntimeError("structural authority checkpoint schema is unsupported")
    if record.get("design_sha256") != _design_digest():
        raise RuntimeError(
            "structural authority checkpoint design digest does not match"
        )
    if record.get("source_sha256") != expected_sources:
        return {}
    return {row["corner_id"]: row for row in record.get("corners", [])}


def _reusable_row(
    previous: dict[str, Any],
    corner: StructuralAuthorityCorner,
    artifact_directory: Path,
) -> dict[str, Any] | None:
    if previous.get("status") not in TERMINAL_STATUSES:
        return None
    if previous["status"] == "failed_retained":
        return previous
    record_name = previous.get("record_artifact")
    array_name = previous.get("array_artifact")
    if not isinstance(record_name, str) or not isinstance(array_name, str):
        return None
    try:
        authority = load_scaled_authority(
            artifact_directory / record_name,
            artifact_directory / array_name,
        )
    except (OSError, RuntimeError, ValueError):
        return None
    if authority.configuration != corner.configuration:
        return None
    if authority.authority_sha256 != previous.get("authority_sha256"):
        return None
    return previous


def run_campaign(
    checkpoint_path: Path = DEFAULT_RECORD,
    *,
    artifact_directory: Path = DATA,
) -> dict[str, Any]:
    """Regenerate every registered corner and retain failures with checkpoints."""

    execution_sources = _source_hashes()
    existing = _existing_rows(checkpoint_path, execution_sources)
    rows: list[dict[str, Any]] = []
    for corner in registered_corners():
        previous = existing.get(corner.corner_id)
        reusable = (
            _reusable_row(previous, corner, artifact_directory)
            if previous is not None
            else None
        )
        if reusable is not None:
            rows.append(reusable)
            continue
        stem = _artifact_stem(corner)
        record_path = artifact_directory / f"{stem}.json"
        arrays_path = artifact_directory / f"{stem}.npz"
        try:
            authority = build_scaled_authority(corner.configuration)
            save_scaled_authority(authority, record_path, arrays_path)
            row = _corner_record(corner, authority, record_path, arrays_path)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            row = {
                "corner_id": corner.corner_id,
                "axis_name": corner.axis_name,
                "level": corner.level,
                "value": corner.value,
                "configuration": _jsonable(asdict(corner.configuration)),
                "status": "failed_retained",
                "failure_count": None,
                "failure_distribution": {},
                "failure_class": type(error).__name__,
                "failure_message": str(error),
                "record_artifact": None,
                "array_artifact": None,
            }
        observed_sources = _source_hashes()
        if observed_sources != execution_sources:
            raise RuntimeError("structural authority campaign source drift detected")
        rows.append(row)
        _write_checkpoint(
            checkpoint_path,
            _campaign_record(rows, "in_progress", execution_sources),
        )
    record = _campaign_record(rows, "complete", execution_sources)
    _write_checkpoint(checkpoint_path, record)
    return record


def main() -> None:
    run_campaign()


if __name__ == "__main__":
    main()
