"""Supplemental trajectories replayed from immutable matched-shot records.

Planning reads frozen data only. Execution uses an injected matched-shot physics
provider and refuses publication unless runtime, saved-path, and row parity pass.
The original study files are never changed.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.tools.shot_pattern_analysis.matched_config import MatchedConfig

PATTERNS = ("Straight", "Draw", "Fade")
INPUTS = ("shots.csv.gz", "flight_samples.json", "summary.json", "run_start.json")
TOLERANCES = {
    "trajectory_m": 1e-6,
    "endpoint_m": 1e-6,
    "launch_scalar": 1e-6,
    "spin_rpm": 1e-5,
    "geometry_deg": 1e-6,
}
PHYSICS_SOURCES = (
    "src/tools/shot_pattern_analysis/matched_physics.py",
    "src/tools/shot_pattern_analysis/matched_config.py",
    "src/tools/shot_pattern_analysis/club_geometry.py",
    "src/tools/shot_pattern_analysis/physics.py",
    "src/shared/python/physics/ball_simulator.py",
    "src/shared/python/physics/ball_launch_conditions.py",
    "src/shared/python/physics/incremental_contact.py",
    "src/shared/python/physics/incremental_contact_kernel.py",
    "src/shared/python/physics/rigid_contact.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ReplayPlan:
    """Frozen cells and one common shot-index selection across all populations."""

    cells: dict[str, Path]
    shot_indices: tuple[int, ...]
    input_sha256: dict[str, dict[str, str]]


def _rows(cell: Path) -> dict[str, dict[int, dict[str, str]]]:
    rows: dict[str, dict[int, dict[str, str]]] = {p: {} for p in PATTERNS}
    with gzip.open(cell / "shots.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            pattern = row["pattern"]
            if pattern not in rows:
                raise ValueError(f"unknown pattern {pattern}")
            index = int(row["shot_index"])
            if index in rows[pattern]:
                raise ValueError("duplicate shot index")
            rows[pattern][index] = row
    return rows


def _verify_cell(cell: Path) -> dict[str, str]:
    manifest = _read_json(cell / "manifest.json")["files_sha256"]
    hashes = {name: _sha(cell / name) for name in INPUTS}
    if any(manifest.get(name) != digest for name, digest in hashes.items()):
        raise ValueError(f"frozen input hash differs from manifest: {cell}")
    flights = _read_json(cell / "flight_samples.json")["trajectories"]
    if set(flights) != set(PATTERNS) or any(len(flights[p]) != 6 for p in PATTERNS):
        raise ValueError("frozen flights must contain six paths per pattern")
    return hashes


def plan_replay(cells: dict[str, Path], *, target: int = 100) -> ReplayPlan:
    """Select evenly spaced supported IDs shared by every cell and pattern.

    IDs 0–4 are reserved for strict saved-path parity. No simulation occurs.
    """
    if (
        not cells
        or isinstance(target, bool)
        or not isinstance(target, int)
        or target <= 0
    ):
        raise ValueError("cells and positive integer target required")
    paths = {name: Path(path) for name, path in cells.items()}
    hashes = {name: _verify_cell(cell) for name, cell in paths.items()}
    populations = [_rows(cell) for cell in paths.values()]
    supported = [
        {i for i, row in population[pattern].items() if row["status"] == "supported"}
        for population in populations
        for pattern in PATTERNS
    ]
    common = sorted(set.intersection(*supported) - set(range(5)))
    if len(common) < target:
        raise ValueError(f"only {len(common)} common supported replay candidates")
    # Even ranks sample the full supported population without inventing paths.
    ranks = (
        [i * (len(common) - 1) // (target - 1) for i in range(target)]
        if target > 1
        else [len(common) // 2]
    )
    indices = tuple(common[i] for i in ranks)
    if len(set(indices)) != target:
        raise ValueError("selection did not yield distinct IDs")
    return ReplayPlan(paths, indices, hashes)


def _runtime_matches(recorded: dict[str, Any], actual: dict[str, Any]) -> None:
    for key in ("native_binary_sha256", "numerical_runtime", "contact_backend"):
        if key not in recorded or key not in actual or recorded[key] != actual[key]:
            raise ValueError(f"replay runtime differs: {key}")
    old_sources = recorded.get("source_sha256")
    new_sources = actual.get("source_sha256")
    if not isinstance(old_sources, dict) or not isinstance(new_sources, dict):
        raise ValueError("replay physics source identity missing")
    for name in PHYSICS_SOURCES:
        if (
            name not in old_sources
            or name not in new_sources
            or old_sources[name] != new_sources[name]
        ):
            raise ValueError(f"replay physics source differs: {name}")


def _path(path: Any) -> list[list[float]]:
    points = np.asarray(path, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) != 41
        or not np.isfinite(points).all()
    ):
        raise ValueError("replay produced invalid trajectory")
    return points.tolist()


def _same_path(actual: Any, saved: Any) -> bool:
    a, b = np.asarray(actual, dtype=float), np.asarray(saved, dtype=float)
    return (
        a.shape == b.shape == (41, 2)
        and bool(np.isfinite(a).all() and np.isfinite(b).all())
        and bool(np.allclose(a, b, atol=TOLERANCES["trajectory_m"], rtol=0))
    )


def _shot(
    simulator: Any, config: MatchedConfig, nominal_face: float, row: dict[str, str]
) -> Any:
    club = replace(
        config,
        club_speed_mps=float(row["speed_mps"]),
        shaft_lean_deg=float(row["shaft_lean_deg"]),
        attack_angle_deg=float(row["attack_angle_deg"]),
    )
    return simulator.simulate_delivery(
        nominal_face_deg=nominal_face,
        path_deg=float(row["path_deg"]),
        config=club,
        shaft_rotation_deg=float(row["shaft_rotation_deg"]),
        contact_horizontal_m=float(row["contact_horizontal_m"]),
        contact_vertical_m=float(row["contact_vertical_m"]),
        sample_trajectory=True,
    )


def _check_row(delivered: Any, row: dict[str, str]) -> None:
    outcome = delivered.outcome
    fields = (
        ("carry_x_m", TOLERANCES["endpoint_m"]),
        ("carry_y_m", TOLERANCES["endpoint_m"]),
        ("spin_rpm", TOLERANCES["spin_rpm"]),
        ("launch_azimuth_deg", TOLERANCES["launch_scalar"]),
        ("spin_axis_tilt_deg", TOLERANCES["launch_scalar"]),
        ("launch_elevation_deg", TOLERANCES["launch_scalar"]),
        ("ball_speed_mps", TOLERANCES["launch_scalar"]),
    )
    for name, tolerance in fields:
        saved = float(row[name])
        actual = float(getattr(outcome, name))
        if (
            not math.isfinite(saved)
            or not math.isfinite(actual)
            or abs(actual - saved) > tolerance
        ):
            raise ValueError(f"replay row parity differs: {name}")
    for saved_name, actual_name in (
        ("face_deg", "delivered_face_deg"),
        ("loft_deg", "delivered_loft_deg"),
        ("contact_face_deg", "contact_face_deg"),
        ("contact_loft_deg", "contact_loft_deg"),
    ):
        saved = float(row[saved_name])
        actual = float(getattr(delivered, actual_name))
        if (
            not math.isfinite(saved)
            or not math.isfinite(actual)
            or abs(actual - saved) > TOLERANCES["geometry_deg"]
        ):
            raise ValueError(f"replay row parity differs: {saved_name}")
    path = _path(outcome.trajectory_xy_m)
    if (
        abs(path[-1][0] - float(row["carry_x_m"])) > TOLERANCES["endpoint_m"]
        or abs(path[-1][1] - float(row["carry_y_m"])) > TOLERANCES["endpoint_m"]
    ):
        raise ValueError("replay path endpoint differs from saved row")


def _replay_pattern(
    simulator: Any,
    club: MatchedConfig,
    nominal: dict[str, Any],
    rows: dict[int, dict[str, str]],
    flights: list[Any],
    indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    face = float(nominal["face_deg"])
    baseline = simulator.simulate(
        face_deg=face,
        path_deg=float(nominal["path_deg"]),
        config=replace(club, club_speed_mps=float(nominal["speed_mps"])),
        sample_trajectory=True,
    )
    if not _same_path(baseline.trajectory_xy_m, flights[0]):
        raise ValueError("saved path parity differs: nominal")
    for index in range(5):
        row = rows[index]
        if row["status"] != "supported":
            raise ValueError("saved sampled path has unsupported row")
        delivered = _shot(simulator, club, face, row)
        if not _same_path(delivered.outcome.trajectory_xy_m, flights[index + 1]):
            raise ValueError("saved path parity differs: sampled")
        _check_row(delivered, row)
    result = []
    for index in indices:
        delivered = _shot(simulator, club, face, rows[index])
        _check_row(delivered, rows[index])
        result.append(
            {"shot_index": index, "path_xy_m": _path(delivered.outcome.trajectory_xy_m)}
        )
    return result


def _replay_cell(
    cell: Path, simulator: Any, indices: tuple[int, ...]
) -> dict[str, list[dict[str, Any]]]:
    rows = _rows(cell)
    nominals = _read_json(cell / "summary.json")["nominals"]
    flights = _read_json(cell / "flight_samples.json")["trajectories"]
    club = MatchedConfig(**_read_json(cell / "run_start.json")["config"]["club"])
    return {
        pattern: _replay_pattern(
            simulator, club, nominals[pattern], rows[pattern], flights[pattern], indices
        )
        for pattern in PATTERNS
    }


def execute_replay(
    plan: ReplayPlan, simulator: Any, output_dir: Path, *, runtime: dict[str, Any]
) -> dict[str, Any]:
    """Replay after strict qualification; write separate traces and receipt."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    for name, cell in plan.cells.items():
        if _verify_cell(cell) != plan.input_sha256[name]:
            raise ValueError("frozen source hash changed after planning")
        _runtime_matches(_read_json(cell / "run_start.json")["execution"], runtime)
    traces = {
        name: _replay_cell(cell, simulator, plan.shot_indices)
        for name, cell in plan.cells.items()
    }
    receipt = {
        "schema": "shot-pattern-supplemental-replay/1",
        "qualification": "saved six paths and selected shot rows reproduced",
        "source_sha256": plan.input_sha256,
        "runtime": runtime,
        "shot_indices": list(plan.shot_indices),
        "selected_paths_per_pattern": len(plan.shot_indices),
        "conditioning": "same supported shot indices in every cell and pattern",
        "replay_source_sha256": _sha(Path(__file__)),
        "tolerances": TOLERANCES,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "trajectories.json").write_text(
        json.dumps(traces, indent=2, allow_nan=False) + "\n"
    )
    receipt["trajectories_sha256"] = _sha(output_dir / "trajectories.json")
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    return receipt
