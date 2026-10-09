"""Integrity-checked inventory for the separately versioned bounds supplement."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

from .matched_config import matched_preset
from .matched_matrix import PROFILES, noise_profile
from .matched_report import load_matrix_report
from .provenance import manifest_files_match

_PROTOCOLS = {
    "pw_within_lean_face": "pw-primary-face-bounds/1",
    "pw_fixed_target_face": "pw-control-face-bounds/1",
    "driver_fixed_target_speed": "driver-control-speed-bounds/1",
}
_PROTOCOL_BY_ID = {value: key for key, value in _PROTOCOLS.items()}
_PATTERNS = ("Straight", "Draw", "Fade")
_EXPECTED_ALIAS_COUNT = 4
_EXPECTED_CELL_COUNT = 26
_NUMERIC_TOLERANCE = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: Any, expected: float, *, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE
    ):
        raise ValueError(f"{label} differs from the declared supplement inventory")


def _cell_parts(cell_name: str) -> tuple[str, float, float, float, str]:
    club_id = next(
        (
            name
            for name in ("pitching_wedge", "driver")
            if cell_name.startswith(name + "_")
        ),
        None,
    )
    if club_id is None:
        raise ValueError(f"supplement cell has an unknown club: {cell_name}")
    parts = cell_name[len(club_id) + 1 :].split("_")
    if len(parts) < 4:
        raise ValueError(f"supplement cell name is malformed: {cell_name}")
    try:
        lean = float(parts[0].removeprefix("lean"))
        curve = float(parts[1].removeprefix("curve"))
        rotation = float(parts[2].removeprefix("rotation"))
    except ValueError as exc:
        raise ValueError(f"supplement cell name is malformed: {cell_name}") from exc
    profile = "_".join(parts[3:])
    if profile not in PROFILES:
        raise ValueError(f"supplement cell has an unknown profile: {cell_name}")
    return club_id, lean, curve, rotation, profile


def _inventory_entries(
    inventory: dict[str, Any],
) -> tuple[dict[str, dict], dict[str, str]]:
    entries = inventory.get("entries")
    aliases = inventory.get("control_aliases_to_primary")
    if (
        inventory.get("schema") != "shot-pattern-supplement-execution-inventory/1"
        or not isinstance(entries, list)
        or len(entries) != _EXPECTED_CELL_COUNT
        or inventory.get("unique_executed_cells") != _EXPECTED_CELL_COUNT
    ):
        raise ValueError("supplement inventory must declare exactly 26 unique cells")
    if not isinstance(aliases, dict) or len(aliases) != _EXPECTED_ALIAS_COUNT:
        raise ValueError(
            "supplement inventory must declare exactly four control aliases"
        )
    by_cell: dict[str, dict] = {}
    for entry in entries:
        name = entry.get("cell")
        if not isinstance(name, str) or not name or name in by_cell:
            raise ValueError(
                "supplement inventory contains a missing or duplicate cell"
            )
        if entry.get("protocol") not in _PROTOCOLS:
            raise ValueError(f"unknown supplement protocol category for {name}")
        _cell_parts(name)
        by_cell[name] = entry
    if set(aliases) != {name for name in aliases if aliases[name] == name}:
        raise ValueError(
            "control aliases must point to identically named unique primary cells"
        )
    if not set(aliases.values()).issubset(by_cell):
        raise ValueError(
            "control alias points outside the unique supplemental inventory"
        )
    return by_cell, aliases


def _validate_cell_config(
    name: str, entry: dict[str, Any], config: dict[str, Any]
) -> None:
    club_id, lean, curve, rotation, profile = _cell_parts(name)
    if entry["protocol"] not in _PROTOCOLS:
        raise ValueError(f"unknown protocol category for {name}")
    if config.get("club") != json.loads(
        json.dumps(
            {
                **vars(matched_preset(club_id)),
                "shaft_lean_deg": lean,
            }
        )
    ):
        raise ValueError(f"supplement cell club configuration differs: {name}")
    expected_noise = json.loads(
        json.dumps(
            {
                "standard_deviations": noise_profile(
                    profile, rotation
                ).standard_deviations,
                "correlation": noise_profile(profile, rotation).correlation,
            }
        )
    )
    if config.get("noise") != expected_noise:
        raise ValueError(
            f"supplement noise config differs from the declared profile: {name}"
        )
    for key, expected in (
        ("n_shots", entry["n_shots"]),
        ("seed", entry["seed"]),
        ("match_tolerance_m", entry["match_tolerance_m"]),
    ):
        if config.get(key) != expected:
            raise ValueError(
                f"supplement {key} differs from the declared inventory: {name}"
            )
    _close(config.get("curve_scale"), curve, label=f"{name} curve scale")
    if config.get("target_x_m") is None:
        if entry.get("target_x_m") is not None:
            raise ValueError(
                f"{name} target request differs from the declared inventory"
            )
    else:
        _close(config["target_x_m"], float(entry["target_x_m"]), label=f"{name} target")
    bounds = entry["face_bounds_deg"]
    actual_bounds = config.get("match_face_bounds_deg")
    if (
        not isinstance(actual_bounds, list)
        or len(actual_bounds) != 2
        or any(
            not math.isclose(
                float(a), float(b), rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE
            )
            for a, b in zip(actual_bounds, bounds, strict=True)
        )
    ):
        raise ValueError(
            f"supplement face bounds differ from the declared inventory: {name}"
        )
    speed = config.get("match_speed_factors")
    club_speed = float(config["club"]["club_speed_mps"])
    expected_speed = [float(value) / club_speed for value in entry["speed_bounds_mps"]]
    if (
        not isinstance(speed, list)
        or len(speed) != 2
        or any(
            not math.isclose(
                float(a), float(b), rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE
            )
            for a, b in zip(speed, expected_speed, strict=True)
        )
    ):
        raise ValueError(
            f"supplement speed bounds differ from the declared inventory: {name}"
        )
    expected_lean = 0.0 if entry["protocol"] == "pw_within_lean_face" else 5.0
    _close(lean, expected_lean, label=f"{name} lean")
    expected_club = (
        "driver"
        if entry["protocol"] == "driver_fixed_target_speed"
        else "pitching_wedge"
    )
    if club_id != expected_club:
        raise ValueError(f"supplement club differs from its protocol: {name}")
    if curve != 2.0:
        raise ValueError(f"supplement curve scale must be 2: {name}")
    expected = {
        "club": json.loads(
            json.dumps({**vars(matched_preset(club_id)), "shaft_lean_deg": lean})
        ),
        "noise": expected_noise,
        "n_shots": entry["n_shots"],
        "seed": entry["seed"],
        "curve_scale": curve,
        "target_x_m": entry.get("target_x_m"),
        "match_tolerance_m": entry["match_tolerance_m"],
        "target_radius_m": 15.0,
        "match_face_bounds_deg": [float(value) for value in bounds],
        "match_speed_factors": expected_speed,
    }
    _assert_exact_config(config, expected, name)


def _assert_exact_config(
    actual: Any, expected: Any, cell_name: str, path: str = "config"
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{cell_name} {path} has missing or unknown fields")
        for key, value in expected.items():
            _assert_exact_config(actual[key], value, cell_name, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{cell_name} {path} differs from the declared config")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_exact_config(
                actual_value, expected_value, cell_name, f"{path}[{index}]"
            )
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        _close(actual, float(expected), label=f"{cell_name} {path}")
    elif actual != expected:
        raise ValueError(f"{cell_name} {path} differs from the declared config")


def _cell_reference(
    root: Path, report: dict, cell: dict, protocol_id: str
) -> dict[str, Any]:
    name = cell["cell"]
    cell_root = root / name
    csv_path = cell_root / "shots.csv.gz"
    return {
        "cell": name,
        "source_protocol": protocol_id,
        "source_root": str(root.resolve()),
        "execution_kind": "unique",
        "status": cell["status"],
        "config": cell["config"],
        "summary": cell,
        "source_plan_sha256": report["plan_sha256"],
        "source_cell_manifest_sha256": report["cell_manifest_sha256"].get(name),
        "source_csv_sha256": _sha256(csv_path) if csv_path.is_file() else None,
    }


def _verify_actual_nominal_target(
    cell: dict, expected_target: float, tolerance: float
) -> None:
    if cell.get("status") != "completed":
        raise ValueError(f"control alias source is not completed: {cell['cell']}")
    _close(
        cell.get("target_x_m"), expected_target, label=f"{cell['cell']} actual target"
    )
    nominals = cell.get("nominals")
    if not isinstance(nominals, dict) or set(nominals) != set(_PATTERNS):
        raise ValueError(
            f"control alias source lacks all nominal patterns: {cell['cell']}"
        )
    for pattern in _PATTERNS:
        outcome = nominals[pattern].get("outcome", {})
        try:
            x = float(outcome["carry_x_m"])
            y = float(outcome["carry_y_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"control alias nominal coordinates are invalid: {cell['cell']} {pattern}"
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f"control alias nominal coordinates are nonfinite: {cell['cell']} {pattern}"
            )
        if math.hypot(x - expected_target, y) > tolerance:
            raise ValueError(
                f"control alias nominal misses the fixed target: {cell['cell']} {pattern}"
            )


def load_supplement_sources(
    protocol_roots: dict[str, Path], inventory_path: Path, *, stage: str = "final"
) -> dict[str, Any]:
    """Load and bind the three supplemental matrix roots to their exact inventory.

    The original primary/control matrix reports are deliberately not merged or
    mutated by this loader. Four control aliases are references to unique
    supplemental primary cells and never create a second run or receipt.
    """
    inventory_path = Path(inventory_path)
    if stage not in {"pilot", "final"}:
        raise ValueError("supplement stage must be pilot or final")
    inventory_hash = _sha256(inventory_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected_by_cell, aliases = _inventory_entries(inventory)
    expected_plan_hashes = inventory.get("protocol_plan_sha256")
    if not isinstance(expected_plan_hashes, dict):
        raise ValueError("supplement inventory lacks predeclaration plan hashes")
    for relative, expected_hash in expected_plan_hashes.items():
        path = inventory_path.parent / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(
                f"supplement predeclaration source hash mismatch: {relative}"
            )
    if set(protocol_roots) != set(_PROTOCOL_BY_ID):
        raise ValueError(
            "supplement source roots must cover the three versioned protocol IDs"
        )
    reports: dict[str, Any] = {}
    unique_cells: list[dict[str, Any]] = []
    expected_by_protocol: dict[str, set[str]] = {
        _PROTOCOLS[category]: {
            name
            for name, entry in expected_by_cell.items()
            if entry["protocol"] == category
        }
        for category in _PROTOCOLS
    }
    for protocol_id, root_value in protocol_roots.items():
        root = Path(root_value).resolve()
        plan_path = root / "matrix_plan.json"
        if not plan_path.is_file():
            raise ValueError(f"supplement matrix plan is missing: {protocol_id}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("supplement_protocol") != protocol_id:
            raise ValueError(f"supplement protocol ID mismatch: {protocol_id}")
        if plan.get("stage") != stage:
            raise ValueError(
                f"supplement stage mismatch: expected {stage}, found {plan.get('stage')}"
            )
        expected_names = expected_by_protocol[protocol_id]
        selected = plan.get("selected_cell_names")
        cells = plan.get("cells")
        if not isinstance(selected, list) or len(selected) != len(set(selected)):
            raise ValueError(
                f"supplement plan has duplicate selected cell names: {protocol_id}"
            )
        if (
            not isinstance(cells, dict)
            or set(cells) != expected_names
            or set(selected) != expected_names
        ):
            raise ValueError(
                f"supplement selected cell inventory differs: {protocol_id}"
            )
        for name in expected_names:
            entry = dict(expected_by_cell[name])
            entry["n_shots"] = 32 if stage == "pilot" else 10_000
            _validate_cell_config(name, entry, cells[name])
        report = load_matrix_report(root, require_complete=True)
        if report["plan_sha256"] != _sha256(plan_path):
            raise ValueError(f"supplement report plan hash mismatch: {protocol_id}")
        report_cells = {cell["cell"]: cell for cell in report["cells"]}
        if set(report_cells) != expected_names:
            raise ValueError(
                f"loaded supplement cells differ from inventory: {protocol_id}"
            )
        protocol_cells = []
        for name in sorted(expected_names):
            ref = _cell_reference(root, report, report_cells[name], protocol_id)
            if ref["source_cell_manifest_sha256"] is None:
                raise ValueError(f"supplement cell has no manifest hash: {name}")
            protocol_cells.append(ref)
            unique_cells.append(ref)
        reports[protocol_id] = {
            "source_root": str(root),
            "plan_sha256": report["plan_sha256"],
            "report_source_sha256": report["report_source_sha256"],
            "cells": protocol_cells,
        }

    refs_by_name = {ref["cell"]: ref for ref in unique_cells}
    fixed_target_by_zero_name = {
        entry["cell"].replace("_lean5_", "_lean0_"): float(entry["target_x_m"])
        for entry in inventory["entries"]
        if entry["protocol"] == "pw_fixed_target_face"
    }
    alias_rows: list[dict[str, Any]] = []
    for alias_name, source_name in sorted(aliases.items()):
        source = refs_by_name.get(source_name)
        if (
            source is None
            or source["source_protocol"] != _PROTOCOLS["pw_within_lean_face"]
        ):
            raise ValueError(
                f"control alias does not resolve to a PW primary source: {alias_name}"
            )
        expected_target = fixed_target_by_zero_name.get(alias_name)
        if expected_target is None:
            raise ValueError(
                f"control alias has no predeclared PW fixed target: {alias_name}"
            )
        source_config = source["config"]
        if source_config.get("target_x_m") is not None:
            raise ValueError(
                f"control alias source is not a within-lean request: {alias_name}"
            )
        requested_config = json.loads(json.dumps(source_config))
        requested_config["target_x_m"] = expected_target
        alias_status = "unverified"
        reason = "source_not_completed"
        if source["status"] == "completed":
            _verify_actual_nominal_target(
                source["summary"],
                expected_target,
                float(source_config["match_tolerance_m"]),
            )
            alias_status, reason = "verified", ""
        alias_rows.append(
            {
                "alias_cell": alias_name,
                "alias_of_primary": source_name,
                "source_protocol": _PROTOCOLS["pw_within_lean_face"],
                "execution_kind": "provenance_alias",
                "proof_status": alias_status,
                "reason": reason,
                "source_root": source["source_root"],
                "requested_config": requested_config,
                "source_plan_sha256": source["source_plan_sha256"],
                "source_cell_manifest_sha256": source["source_cell_manifest_sha256"],
                "source_csv_sha256": source["source_csv_sha256"],
            }
        )
    # Recheck every source byte after aggregation so no artifact can change
    # while the report is being assembled.
    for protocol_id, report in reports.items():
        root = Path(report["source_root"])
        if _sha256(root / "matrix_plan.json") != report["plan_sha256"]:
            raise ValueError(
                f"supplement plan changed during aggregation: {protocol_id}"
            )
    for ref in unique_cells:
        cell_root = Path(ref["source_root"]) / ref["cell"]
        if (
            not manifest_files_match(cell_root / "manifest.json")
            or _sha256(cell_root / "manifest.json")
            != ref["source_cell_manifest_sha256"]
            or _sha256(cell_root / "shots.csv.gz") != ref["source_csv_sha256"]
        ):
            raise ValueError(
                f"supplement source changed during aggregation: {ref['cell']}"
            )
    if _sha256(inventory_path) != inventory_hash:
        raise ValueError("supplement inventory changed during aggregation")
    counts = {protocol: len(reports[protocol]["cells"]) for protocol in _PROTOCOL_BY_ID}
    return {
        "schema": "matched-supplement-report/1",
        "stage": stage,
        "issue": 12000,
        "inventory_sha256": inventory_hash,
        "unique_executed_cells": len(unique_cells),
        "protocol_counts": counts,
        "protocols": reports,
        "unique_cells": unique_cells,
        "aliases": alias_rows,
        "control_aliases": {
            "schema": "matched-study-control-aliases/1",
            "references": [
                {
                    "protocol_id": "supplement_control_alias",
                    "alias_cell": row["alias_cell"],
                    "alias_of_primary": row["alias_of_primary"],
                    "execution_kind": "provenance_alias",
                    "source_root": row["source_root"],
                    "source_plan_sha256": row["source_plan_sha256"],
                    "source_cell_manifest_sha256": row["source_cell_manifest_sha256"],
                    "source_csv_sha256": row["source_csv_sha256"],
                    "requested_config": row["requested_config"],
                }
                for row in alias_rows
            ],
            "status_by_alias": {
                row["alias_cell"]: {
                    "proof_status": row["proof_status"],
                    "reason": row["reason"],
                }
                for row in alias_rows
            },
        },
        "interpretation": "separate bounded supplement; aliases reference unique primary cells and are not executions",
    }


def render_supplement_inventory(report: dict[str, Any]) -> str:
    """Return a distinct inventory CSV without rewriting original matrix rows."""
    if report.get("schema") != "matched-supplement-report/1":
        raise ValueError("supplement report has an unexpected schema")
    fields = (
        "row_kind",
        "status",
        "source_protocol",
        "execution_kind",
        "cell",
        "alias_of_primary",
        "club",
        "lean_deg",
        "curve_scale",
        "shaft_rotation_sd_deg",
        "target_request_m",
        "target_actual_m",
        "n_shots_per_pattern",
        "seed",
        "face_bounds_deg",
        "speed_factors",
        "source_root",
        "source_plan_sha256",
        "source_cell_manifest_sha256",
        "source_csv_sha256",
        "reason",
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for ref in report["unique_cells"]:
        config = ref["config"]
        club = config["club"]
        writer.writerow(
            {
                "row_kind": "unique_cell",
                "status": ref["status"],
                "source_protocol": ref["source_protocol"],
                "execution_kind": "unique",
                "cell": ref["cell"],
                "club": club["club_id"],
                "lean_deg": club["shaft_lean_deg"],
                "curve_scale": config["curve_scale"],
                "shaft_rotation_sd_deg": config["noise"]["standard_deviations"][0],
                "target_request_m": config["target_x_m"],
                "target_actual_m": ref["summary"].get("target_x_m"),
                "n_shots_per_pattern": config["n_shots"],
                "seed": config["seed"],
                "face_bounds_deg": json.dumps(config["match_face_bounds_deg"]),
                "speed_factors": json.dumps(config["match_speed_factors"]),
                "source_root": ref["source_root"],
                "source_plan_sha256": ref["source_plan_sha256"],
                "source_cell_manifest_sha256": ref["source_cell_manifest_sha256"],
                "source_csv_sha256": ref["source_csv_sha256"],
            }
        )
    for alias in report["aliases"]:
        config = alias["requested_config"]
        club = config["club"]
        writer.writerow(
            {
                "row_kind": "control_alias",
                "status": alias["proof_status"],
                "source_protocol": alias["source_protocol"],
                "execution_kind": alias["execution_kind"],
                "cell": alias["alias_cell"],
                "alias_of_primary": alias["alias_of_primary"],
                "club": club["club_id"],
                "lean_deg": club["shaft_lean_deg"],
                "curve_scale": config["curve_scale"],
                "shaft_rotation_sd_deg": config["noise"]["standard_deviations"][0],
                "target_request_m": config["target_x_m"],
                "n_shots_per_pattern": config["n_shots"],
                "seed": config["seed"],
                "face_bounds_deg": json.dumps(config["match_face_bounds_deg"]),
                "speed_factors": json.dumps(config["match_speed_factors"]),
                "source_root": alias["source_root"],
                "source_plan_sha256": alias["source_plan_sha256"],
                "source_cell_manifest_sha256": alias["source_cell_manifest_sha256"],
                "source_csv_sha256": alias["source_csv_sha256"],
                "reason": alias["reason"],
            }
        )
    return output.getvalue()


def write_supplement_artifacts(
    report: dict[str, Any], output_dir: Path
) -> dict[str, Path]:
    """Write a separately named inventory and exact alias-reference envelope."""
    if report.get("schema") != "matched-supplement-report/1":
        raise ValueError("supplement report has an unexpected schema")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "REPORT_SUPPLEMENT.csv"
    aliases_path = output_dir / "control_aliases.json"
    inventory_path.write_text(render_supplement_inventory(report), encoding="utf-8")
    aliases_path.write_text(
        json.dumps(report["control_aliases"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"inventory": inventory_path, "control_aliases": aliases_path}
