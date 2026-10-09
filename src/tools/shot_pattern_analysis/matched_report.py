"""Integrity-checked assembly of matched-study reports without rerunning physics."""

from __future__ import annotations

import hashlib
import json
import csv
import gzip
import math
import re
from collections import Counter
from pathlib import Path

from .provenance import assert_source_unchanged, manifest_files_match

_PATTERNS = ("Straight", "Draw", "Fade")


def verify_nominal_deliveries_hit_target(
    summary: dict,
    target_x_m: float,
    tolerance_m: float,
    *,
    context: str = "paired",
) -> None:
    """Require all three finite zero-error deliveries to hit one raw target."""
    if summary.get("status") != "completed":
        raise ValueError(f"{context} nominal verification requires a completed cell")
    actual_target = summary.get("target_x_m")
    if not isinstance(actual_target, (int, float)) or not math.isfinite(actual_target):
        raise ValueError(f"{context} target must be finite")
    if not math.isclose(float(actual_target), target_x_m, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{context} target differs from requested target")
    nominals = summary.get("nominals")
    if not isinstance(nominals, dict) or set(nominals) != set(_PATTERNS):
        raise ValueError(f"{context} source lacks nominal deliveries for all patterns")
    for pattern in _PATTERNS:
        outcome = nominals[pattern].get("outcome", {})
        try:
            x, y = float(outcome["carry_x_m"]), float(outcome["carry_y_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{context} nominal coordinates must be finite") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"{context} nominal coordinates must be finite")
        if math.hypot(x - target_x_m, y) > tolerance_m:
            raise ValueError(f"{context} nominal misses target tolerance: {pattern}")


def load_individually_supported_endpoints(
    csv_path: Path, n_shots: int
) -> tuple[dict[str, list[list[float]]], dict[str, int]]:
    """Load each pattern's supported endpoints without cross-pattern filtering."""
    points: dict[str, list[list[float]]] = {name: [] for name in _PATTERNS}
    totals = dict.fromkeys(_PATTERNS, 0)
    observed: dict[str, set[int]] = {name: set() for name in _PATTERNS}
    with gzip.open(csv_path, "rt", newline="") as stream:
        rows = csv.DictReader(stream)
        required = {"pattern", "shot_index", "status", "carry_x_m", "carry_y_m"}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError("saved shots lack supported endpoint fields")
        for row in rows:
            pattern = row["pattern"]
            if pattern not in points:
                raise ValueError("saved shot has an unknown physical pattern")
            index = int(row["shot_index"])
            if index < 0 or index >= n_shots or index in observed[pattern]:
                raise ValueError("saved shot indices must be distinct and in range")
            observed[pattern].add(index)
            totals[pattern] += 1
            if row["status"] == "supported":
                x, y = float(row["carry_x_m"]), float(row["carry_y_m"])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("supported endpoint must have finite coordinates")
                points[pattern].append([x, y])
    if any(indices != set(range(n_shots)) for indices in observed.values()):
        raise ValueError("saved population must contain every shot index once")
    return points, totals


def validate_original_matrix_protocol(
    report: dict,
    *,
    protocol: str,
    common_targets: dict[str, float] | None = None,
) -> None:
    """Require the exact reviewed factorial cell/config contract for originals."""
    if protocol not in {"original_primary", "original_control"}:
        raise ValueError("unknown original matched-matrix protocol")
    if protocol == "original_control":
        if not isinstance(common_targets, dict) or set(common_targets) != {
            "driver",
            "seven_iron",
            "pitching_wedge",
        }:
            raise ValueError("original control requires the canonical common target map")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in common_targets.values()
        ):
            raise ValueError("canonical common targets must be positive and finite")
    from dataclasses import asdict

    from .matched_matrix import matched_matrix_cells

    expected = {
        name: json.loads(json.dumps(asdict(config)))
        for name, config in matched_matrix_cells(
            n_shots=10_000,
            seed=20261009,
            common_targets=common_targets if protocol == "original_control" else None,
            profiles=("shaft_only", "combined2")
            if protocol == "original_control"
            else None,
        )
    }
    actual_cells = report.get("cells")
    if not isinstance(actual_cells, list):
        raise ValueError("original matrix report lacks its explicit cell inventory")
    actual = {cell.get("cell"): cell.get("config") for cell in actual_cells}
    if actual != expected:
        raise ValueError(
            f"{protocol} cells/configurations differ from the reviewed protocol"
        )


def load_reviewed_common_targets(phase2_root: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Read targets only from the pilot plan bound by final acceptance evidence."""
    root = Path(phase2_root)
    acceptance_path = root / "FINAL_RUN_ACCEPTANCE.md"
    audit_path = root / "validation/independent_control_pilot_audit.json"
    pilot_plan_path = root / "control_pilot_v1/matrix_plan.json"
    target_path = root / "common_targets.json"
    acceptance = acceptance_path.read_text(encoding="utf-8")
    match = re.search(
        r"validation/independent_control_pilot_audit\.json`?:\s*`?([0-9a-f]{64})",
        acceptance,
    )
    if match is None or _sha256(audit_path) != match.group(1):
        raise ValueError("control-pilot audit is not bound by final acceptance")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("pass") is not True or audit.get("failures") != []:
        raise ValueError("control-pilot audit did not pass")
    plan_bytes = pilot_plan_path.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != audit.get("plan_sha256"):
        raise ValueError("control-pilot matrix plan differs from its accepted audit")
    plan = json.loads(plan_bytes)
    if plan.get("stage") != "pilot" or len(plan.get("cells", {})) != 72:
        raise ValueError("accepted control-pilot plan has the wrong protocol shape")
    targets: dict[str, float] = {}
    for config in plan["cells"].values():
        club_id = config["club"]["club_id"]
        target = config.get("target_x_m")
        if (
            club_id not in {"driver", "seven_iron", "pitching_wedge"}
            or isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not math.isfinite(target)
            or target <= 0
        ):
            raise ValueError("accepted control-pilot plan has invalid targets")
        if club_id in targets and not math.isclose(
            targets[club_id], float(target), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("accepted control-pilot targets vary within a club")
        targets[club_id] = float(target)
    from dataclasses import asdict

    from .matched_matrix import matched_matrix_cells

    expected = {
        name: json.loads(json.dumps(asdict(config)))
        for name, config in matched_matrix_cells(
            n_shots=32,
            seed=20261009,
            common_targets=targets,
            profiles=("shaft_only", "combined2"),
        )
    }
    normalized_plan = json.loads(json.dumps(plan["cells"]))
    if normalized_plan != expected:
        raise ValueError("accepted control-pilot plan differs from its factorial protocol")
    if json.loads(target_path.read_text(encoding="utf-8")) != targets:
        raise ValueError("common_targets.json differs from the accepted pilot targets")
    return targets, {
        "acceptance_sha256": _sha256(acceptance_path),
        "audit_sha256": _sha256(audit_path),
        "pilot_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "common_targets_sha256": _sha256(target_path),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_validated_cell(
    directory: Path,
    expected_config: dict | None = None,
    expected_execution: dict | None = None,
) -> dict:
    """Read a saved cell only after manifest, inputs, and source bind together."""
    directory = Path(directory)
    if not manifest_files_match(directory / "manifest.json"):
        raise ValueError(f"cell integrity failure: {directory.name}")
    start = json.loads((directory / "run_start.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    config = start["config"]
    if summary.get("config") != config or (
        expected_config is not None and config != expected_config
    ):
        raise ValueError(f"cell input mismatch: {directory.name}")
    if expected_execution is not None:
        assert_source_unchanged(expected_execution, start["execution"])
    if summary.get("status") not in ("completed", "unavailable"):
        raise ValueError(f"unrecognized cell status: {directory.name}")
    return summary


def load_matrix_report(root: Path, *, require_complete: bool = True) -> dict:
    """Load every planned cell; reject tampering and preserve unavailable outcomes.

    Reports bind to the saved execution snapshot, not the current installation.
    Missing cells can be shown only when explicitly requesting a partial report.
    """
    root = Path(root)
    plan_path = root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text())
    if not isinstance(plan.get("cells"), dict) or not plan["cells"]:
        raise ValueError("matrix plan must contain explicit cells")
    cells, manifests = [], {}
    for name, config in plan["cells"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("matrix cell name must be a simple directory name")
        directory = root / name
        manifest = directory / "manifest.json"
        if not directory.exists():
            cells.append({"cell": name, "config": config, "status": "not_run"})
            continue
        summary = load_validated_cell(directory, config, plan["execution"])
        if summary["status"] == "completed":
            for item in summary["summary"].values():
                if item["total_n"] != config["n_shots"]:
                    raise ValueError(f"population count mismatch: {name}")
        cells.append({**summary, "cell": name})
        manifests[name] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    counts = dict.fromkeys(("completed", "unavailable", "not_run"), 0)
    counts.update(Counter(cell["status"] for cell in cells))
    if require_complete and counts["not_run"]:
        raise ValueError("matrix incomplete: planned cells have not run")
    return {
        "schema": "matched-study-report/1",
        "stage": plan["stage"],
        "execution": plan["execution"],
        "planned_cells": len(cells),
        "counts": counts,
        "cells": cells,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "cell_manifest_sha256": manifests,
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "conditional geometric simulation; not empirical qualification",
    }
