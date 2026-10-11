"""Integrity-checked course sensitivity from saved matched landing populations."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

from .matched_report import load_validated_cell
from .provenance import manifest_files_match
from .scenario_scoring import score_course_endpoints


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).parent
    root = Path(__file__).resolve().parents[3]
    files = {
        name: package / name
        for name in (
            "matched_sensitivity.py",
            "matched_report.py",
            "scenario_scoring.py",
            "scoring_cache.py",
            "scoring.py",
            "uncertainty.py",
        )
    }
    provider = "vendor/ud-tools/src/shared/python/launch_monitor/strokes_gained.py"
    files[provider] = root / provider
    return {name: _sha256(path) for name, path in files.items()}


def _supported_endpoints(
    csv_path: Path, n_shots: int
) -> tuple[dict[str, list[tuple[int, float, float]]], dict[str, int]]:
    names = ("Straight", "Draw", "Fade")
    supported: dict[str, dict[int, tuple[float, float]]] = {name: {} for name in names}
    observed: dict[str, set[int]] = {name: set() for name in names}
    invalid = dict.fromkeys(names, 0)
    with gzip.open(csv_path, "rt", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"pattern", "shot_index", "status", "carry_x_m", "carry_y_m"}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError("saved shots lack the required paired endpoint fields")
        for row in rows:
            name = row["pattern"]
            if name not in observed:
                raise ValueError("saved shot has an unknown pattern")
            index = int(row["shot_index"])
            if index < 0 or index >= n_shots or index in observed[name]:
                raise ValueError("saved shot indices must be distinct and in range")
            observed[name].add(index)
            if row["status"] == "supported":
                x, y = float(row["carry_x_m"]), float(row["carry_y_m"])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("supported endpoint must have finite coordinates")
                supported[name][index] = (x, y)
            else:
                invalid[name] += 1
    if any(indices != set(range(n_shots)) for indices in observed.values()):
        raise ValueError("saved population must contain each paired shot exactly once")
    common = sorted(set.intersection(*(set(rows) for rows in supported.values())))
    return {
        name: [(index, *supported[name][index]) for index in common] for name in names
    }, invalid


def _sensitivity_cases(club_id: str) -> Iterator[tuple[str, dict[str, float]]]:
    """Return the ordered hypothetical-course grid for one scoring preset."""
    if club_id == "driver":
        return (
            (
                f"width{half:g}_hole{hole:g}",
                {"fairway_half_width_m": half, "hole_distance_m": hole},
            )
            for half in (10.0, 15.0, 20.0)
            for hole in (350.0, 400.0, 450.0)
        )
    if club_id in ("seven_iron", "pitching_wedge"):
        return (
            (f"radius{radius:g}", {"green_radius_m": radius})
            for radius in (10.0, 15.0, 20.0)
        )
    raise ValueError("saved club is not a supported scoring preset")


def _recheck_sensitivity_inputs(
    cell_dir: Path, plan_path: Path, provenance: dict
) -> None:
    """Reject source or saved-input drift after all scenarios have been scored."""
    if _source_hashes() != provenance["source_sha256"]:
        raise RuntimeError("scoring source changed during sensitivity analysis")
    if not manifest_files_match(cell_dir / "manifest.json") or any(
        _sha256(plan_path if name == "matrix_plan.json" else cell_dir / name) != digest
        for name, digest in provenance["input_sha256"].items()
    ):
        raise RuntimeError("saved input changed during sensitivity analysis")


def analyze_saved_sensitivity(cell_dir: Path, plan_path: Path) -> dict:
    """Re-score common supported landings; retain every unsupported count."""
    cell_dir = Path(cell_dir)
    plan_path = Path(plan_path)
    if cell_dir.parent.resolve() != plan_path.parent.resolve():
        raise ValueError("cell and matrix plan must share the saved matrix root")
    plan = json.loads(plan_path.read_text())
    cells = plan.get("cells")
    if not isinstance(cells, dict) or cell_dir.name not in cells:
        raise ValueError("cell must be explicitly listed in the saved matrix plan")
    summary = load_validated_cell(cell_dir, cells[cell_dir.name], plan["execution"])
    if summary["status"] != "completed":
        raise ValueError("course sensitivity requires a completed saved cell")
    config = cells[cell_dir.name]
    n_shots = config["n_shots"]
    if isinstance(n_shots, bool) or not isinstance(n_shots, int) or n_shots < 2:
        raise ValueError("saved shot count must be an integer >=2")
    endpoints, invalid = _supported_endpoints(cell_dir / "shots.csv.gz", n_shots)
    common_n = len(endpoints["Straight"])
    input_files = ("manifest.json", "run_start.json", "summary.json", "shots.csv.gz")
    provenance = {
        "schema": "matched-course-sensitivity/1",
        "cell": cell_dir.name,
        "input_sha256": {
            "matrix_plan.json": _sha256(plan_path),
            **{name: _sha256(cell_dir / name) for name in input_files},
        },
        "execution": plan["execution"],
        "source_sha256": _source_hashes(),
        "population": {
            "total_per_pattern": n_shots,
            "common_supported_n": common_n,
            "unsupported_by_pattern": invalid,
            "supported_by_pattern": {
                name: n_shots - count for name, count in invalid.items()
            },
            "supported_but_unpaired_by_pattern": {
                name: n_shots - count - common_n for name, count in invalid.items()
            },
            "conditioning": "common_supported_indices_only",
            "full_population_status": (
                "available"
                if common_n == n_shots
                else "unavailable_unsupported_terminal_lies"
            ),
        },
        "limitations": (
            "Historical source-backed strokes gained on hypothetical flat courses; "
            "carry treated as final lie, no roll or hazards; unsupported terminal "
            "lies are counted but cannot be scored. Intervals cover paired "
            "sampling, not model or course uncertainty."
        ),
    }
    if common_n < 2:
        return {
            **provenance,
            "status": "unavailable",
            "reason": "fewer_than_two_common_supported_shots",
            "scenarios": {},
        }
    club_id = config["club"]["club_id"]
    target_x_m = float(summary["target_x_m"])
    seed = config["seed"]
    scenarios = {}
    cases = _sensitivity_cases(club_id)
    for name, dimensions in cases:
        try:
            scenarios[name] = score_course_endpoints(
                endpoints,
                club_id=club_id,
                target_x_m=target_x_m,
                seed=seed,
                **dimensions,
            )
        except ValueError as exc:
            scenarios[name] = {"status": "unavailable", "reason": str(exc)}
    _recheck_sensitivity_inputs(cell_dir, plan_path, provenance)
    return {
        **provenance,
        "status": "available" if common_n == n_shots else "conditional_only",
        "scenarios": scenarios,
    }


def write_saved_sensitivity(cell_dir: Path, plan_path: Path, output_dir: Path) -> Path:
    """Atomically publish a separate result, leaving frozen cell files untouched."""
    cell_dir = Path(cell_dir)
    output_dir = Path(output_dir)
    if (
        output_dir.resolve() == cell_dir.resolve()
        or cell_dir.resolve() in output_dir.resolve().parents
    ):
        raise ValueError("sensitivity output must be outside the frozen cell")
    report = analyze_saved_sensitivity(cell_dir, plan_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{cell_dir.name}.course_sensitivity.json"
    if destination.exists():
        raise FileExistsError("sensitivity result already exists")
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
