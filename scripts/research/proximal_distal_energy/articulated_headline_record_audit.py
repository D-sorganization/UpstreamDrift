"""Audit headline-campaign snapshots without promoting partial results."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.articulated_headline_uncertainty import (
    DATA,
    ROOT,
    SOURCE_PATHS,
    HeadlineUncertaintyConfig,
    RegisteredCorner,
    registered_corners,
)

SCHEMA_VERSION = "articulated-headline-uncertainty/v1"
STUDY_ID = "articulated-shaft-ground-headline-uncertainty"
PATHWAYS = ("shaft", "ground")
TOTAL_CELL_COUNT = 384
LIMITATIONS = {
    "interaction_order": (
        "one-at-a-time corners do not estimate higher-order parameter interactions"
    ),
    "calibration": (
        "bounds are engineering ranges, not measured participant or equipment "
        "properties"
    ),
    "human_inference": (
        "survival does not promote any result to a human or coaching claim"
    ),
}


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _source_hashes() -> dict[str, str]:
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _digest_mapping(value: Mapping[str, str]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _expected_design(config: HeadlineUncertaintyConfig) -> dict[str, Any]:
    return {
        "method": "registered_nominal_plus_one_at_a_time_low_high_corners",
        "corner_count": len(registered_corners(config)),
        "axes": _json_value([asdict(axis) for axis in config.axes]),
        "controls": (
            "each full atlas retains both engines, velocity reversal, timestep "
            "refinement, and pathway killswitches"
        ),
    }


def _config(record: Mapping[str, Any]) -> HeadlineUncertaintyConfig:
    raw = record.get("configuration")
    if not isinstance(raw, dict):
        raise RuntimeError("headline record configuration is invalid")
    worker_count = raw.get("worker_count")
    try:
        config = HeadlineUncertaintyConfig(worker_count=worker_count)
    except (TypeError, ValueError) as error:
        raise RuntimeError("headline record configuration is invalid") from error
    if raw != _json_value(asdict(config)):
        raise RuntimeError("headline record configuration drifted from registration")
    return config


def _integer_cell_count(value: Any) -> bool:
    return type(value) is int and 0 <= value <= TOTAL_CELL_COUNT


def _validate_pathway(
    result: Any,
    *,
    pathway: str,
    corner: RegisteredCorner,
    expected_sources: Mapping[str, str],
) -> None:
    if not isinstance(result, dict):
        raise RuntimeError("headline pathway result is invalid")
    status = result.get("status")
    affected = pathway in corner.pathways or corner.corner_id == "nominal"
    if status == "not_affected":
        if affected:
            raise RuntimeError("not_affected is invalid for a registered pathway")
        if (
            result.get("matched_cell_count") is not None
            or result.get("total_cell_count") != TOTAL_CELL_COUNT
            or result.get("all_registered_gates_passed") is not None
        ):
            raise RuntimeError("not_affected pathway fields are invalid")
        return
    if not affected:
        raise RuntimeError("unregistered pathway must be marked not_affected")
    if status == "completed":
        if (
            result.get("all_registered_gates_passed") is not True
            or result.get("total_cell_count") != TOTAL_CELL_COUNT
        ):
            raise RuntimeError("completed pathway did not pass its registered gates")
        if not _integer_cell_count(result.get("matched_cell_count")):
            raise RuntimeError("completed pathway matched cell count is invalid")
        if result.get("computed_source_sha256") != expected_sources:
            raise RuntimeError("completed pathway computed source hashes drifted")
        return
    if status == "failed_retained":
        if (
            result.get("matched_cell_count") is not None
            or result.get("total_cell_count") != TOTAL_CELL_COUNT
            or result.get("all_registered_gates_passed") is not False
            or not isinstance(result.get("failure_class"), str)
            or not result["failure_class"]
            or not isinstance(result.get("failure_message"), str)
            or not result["failure_message"]
            or result.get("computed_source_sha256") != expected_sources
        ):
            raise RuntimeError("failed_retained pathway fields are invalid")
        return
    raise RuntimeError("headline pathway status is not registered")


def _validate_row_identity(row: Any, corner: RegisteredCorner) -> None:
    if not isinstance(row, dict):
        raise RuntimeError("headline corner row is invalid")
    observed = {key: row.get(key) for key in asdict(corner)}
    if observed != _json_value(asdict(corner)):
        raise RuntimeError("headline corners must be an exact registered prefix")


def _expected_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for pathway in PATHWAYS:
        nominal = rows[0][pathway]["matched_cell_count"]
        evaluated = [
            row
            for row in rows
            if row[pathway]["status"] in {"completed", "failed_retained"}
        ]
        completed = [row for row in evaluated if row[pathway]["status"] == "completed"]
        failed = [
            row for row in evaluated if row[pathway]["status"] == "failed_retained"
        ]
        counts = [row[pathway]["matched_cell_count"] for row in completed]
        changes = [count - nominal for count in counts]
        expected[pathway] = {
            "nominal_matched_cell_count": nominal,
            "evaluated_corner_count": len(evaluated),
            "completed_corner_count": len(completed),
            "failed_corner_count": len(failed),
            "matched_cell_count_range": [min(counts), max(counts)],
            "matched_cell_count_change_range": [min(changes), max(changes)],
            "nonzero_change_corner_ids": [
                row["corner_id"]
                for row in completed
                if row[pathway]["matched_cell_count"] != nominal
            ],
            "failed_corner_ids": [row["corner_id"] for row in failed],
        }
    return expected


def _validate_complete_record(record: Mapping[str, Any], rows: list[dict]) -> None:
    for pathway in PATHWAYS:
        nominal = rows[0][pathway]["matched_cell_count"]
        for row in rows:
            count = row[pathway]["matched_cell_count"]
            expected_change = count - nominal if count is not None else None
            if row[pathway].get("matched_cell_count_change_from_nominal") != (
                expected_change
            ):
                raise RuntimeError("complete record movement fields are invalid")
    if record.get("results") != _expected_results(rows):
        raise RuntimeError("complete record results do not match its corner evidence")


def audit_headline_record(path: Path) -> dict[str, Any]:
    """Validate one immutable read of a live or completed campaign record."""

    try:
        raw = path.read_bytes()
        record = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("headline record is not a valid JSON snapshot") from error
    if not isinstance(record, dict):
        raise RuntimeError("headline record is not a JSON object")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("headline record schema is unsupported")
    if record.get("study_id") != STUDY_ID:
        raise RuntimeError("headline record study identity is invalid")
    status = record.get("status")
    if status not in {"in_progress", "complete"}:
        raise RuntimeError("headline record status is invalid")
    config = _config(record)
    if record.get("design") != _expected_design(config):
        raise RuntimeError("headline record design drifted from registration")
    expected_sources = _source_hashes()
    if record.get("source_sha256") != expected_sources:
        raise RuntimeError("headline record source hashes drifted")
    if record.get("limitations") != LIMITATIONS:
        raise RuntimeError("headline record limitations drifted")

    rows = record.get("corners")
    expected_corners = registered_corners(config)
    if not isinstance(rows, list) or not rows or len(rows) > len(expected_corners):
        raise RuntimeError("headline corner rows are incomplete or excessive")
    terminal_pathways = 0
    fully_accounted = 0
    for index, row in enumerate(rows):
        corner = expected_corners[index]
        _validate_row_identity(row, corner)
        present = [pathway for pathway in PATHWAYS if pathway in row]
        if present not in [["shaft"], ["shaft", "ground"]]:
            raise RuntimeError("headline pathway execution order is invalid")
        if index < len(rows) - 1 and len(present) != len(PATHWAYS):
            raise RuntimeError("only the active corner may be partially recorded")
        for pathway in present:
            _validate_pathway(
                row[pathway],
                pathway=pathway,
                corner=corner,
                expected_sources=expected_sources,
            )
            terminal_pathways += 1
        if len(present) == len(PATHWAYS):
            fully_accounted += 1

    nominal = rows[0]
    if (
        nominal.get("shaft", {}).get("status") != "completed"
        or nominal["shaft"].get("matched_cell_count") != 126
        or nominal.get("ground", {}).get("status") != "completed"
        or nominal["ground"].get("matched_cell_count") != 0
    ):
        raise RuntimeError("headline nominal authority does not match registration")

    all_complete = len(rows) == len(expected_corners) and fully_accounted == len(
        expected_corners
    )
    if status == "complete":
        if not all_complete:
            raise RuntimeError("headline record is prematurely complete")
        _validate_complete_record(record, rows)
    elif record.get("results") is not None:
        raise RuntimeError("in-progress headline results must remain unavailable")

    active_corner = rows[-1]["corner_id"] if fully_accounted < len(rows) else None
    return {
        "schema_version": "articulated-headline-record-audit/v1",
        "status": "complete" if status == "complete" else "partial",
        "campaign_status": status,
        "corner_count": len(rows),
        "expected_corner_count": len(expected_corners),
        "fully_accounted_corner_count": fully_accounted,
        "terminal_pathway_count": terminal_pathways,
        "active_corner_id": active_corner,
        "release_evidence": status == "complete",
        "source_set_sha256": _digest_mapping(expected_sources),
        "record_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DATA / "articulated_headline_uncertainty.json",
    )
    args = parser.parse_args(argv)
    print(json.dumps(audit_headline_record(args.path), indent=2))


if __name__ == "__main__":
    main()


__all__ = ["audit_headline_record", "main"]
