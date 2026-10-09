"""Read-only paired comparisons across actual shaft-lean study cells."""

from __future__ import annotations

import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .matched_matrix import PROFILES
from .matched_report import load_validated_cell
from .matched_sensitivity import _source_hashes, _supported_endpoints
from .provenance import manifest_files_match
from .scenario_scoring import score_course_endpoints
from .uncertainty import paired_mean_difference

PATTERNS = ("Straight", "Draw", "Fade")
LEANS = (0.0, 5.0, 10.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(name: str) -> str:
    matches = [profile for profile in PROFILES if name.endswith("_" + profile)]
    if len(matches) != 1:
        raise ValueError("saved cell name must identify exactly one noise profile")
    return matches[0]


def _group_key(name: str, config: dict) -> tuple:
    club = dict(config["club"])
    club.pop("shaft_lean_deg")
    return (
        club["club_id"],
        _profile(name),
        config["noise"]["standard_deviations"][0],
        config["curve_scale"],
        config["target_x_m"],
        config["n_shots"],
        config["seed"],
        config["target_radius_m"],
        json.dumps(club, sort_keys=True),
        json.dumps(config["noise"], sort_keys=True),
    )


def _paired_pattern_metrics(
    before: list[tuple[int, float, float]],
    after: list[tuple[int, float, float]],
    *,
    club_id: str,
    target_x_m: float,
    radius_m: float,
    seed: int,
) -> dict:
    a = np.asarray([(x, y) for _, x, y in before], dtype=float)
    b = np.asarray([(x, y) for _, x, y in after], dtype=float)
    radial_a = np.hypot(a[:, 0] - target_x_m, a[:, 1])
    radial_b = np.hypot(b[:, 0] - target_x_m, b[:, 1])
    result = {
        "radial_error_difference_m": paired_mean_difference(
            radial_b, radial_a, seed=seed
        ),
        "squared_target_error_difference_m2": paired_mean_difference(
            radial_b**2, radial_a**2, seed=seed
        ),
        "containment_fraction_difference": paired_mean_difference(
            (radial_b <= radius_m).astype(float),
            (radial_a <= radius_m).astype(float),
            seed=seed,
        ),
        "role_mapping": {
            "Straight": "physical pattern at earlier lean",
            "Draw": "same physical pattern at later lean",
            "Fade": "duplicate of earlier lean for scorer API shape only; not an observation",
        },
    }
    # The public course scorer owns the historical baseline and lie geometry.
    # Its Draw-vs-Straight paired score is relabeled only after evaluation.
    endpoints = {"Straight": before, "Draw": after, "Fade": before}
    try:
        scored = score_course_endpoints(
            endpoints,
            club_id=club_id,
            target_x_m=target_x_m,
            green_radius_m=radius_m,
            seed=seed,
        )
    except ValueError as exc:
        result["course_sg"] = {"status": "unavailable", "reason": str(exc)}
    else:
        result["course_sg_difference"] = scored["paired_benefit_vs_straight"]["Draw"]
        result["course_sg_means"] = {
            "earlier_lean": scored["patterns"]["Straight"]["mean_strokes_gained"],
            "later_lean": scored["patterns"]["Draw"]["mean_strokes_gained"],
        }
        result["course_sg"] = {
            "status": "available",
            "scenario": scored["scenario"],
            "baseline": scored["baseline"],
            "api_evaluated_states": scored["api_evaluated_states"],
        }
    return result


def analyze_across_lean(matrix_root: Path) -> dict:
    """Compare lean 0/5/10 on one six-population index intersection per pair."""
    matrix_root = Path(matrix_root)
    plan_path = matrix_root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = _sha256(plan_path)
    cells = plan.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise ValueError("saved matrix plan must contain cells")
    if any(config.get("target_x_m") is None for config in cells.values()):
        raise ValueError("across-lean comparison requires a fixed common target")
    groups: dict[tuple, dict[float, tuple[str, dict]]] = {}
    loaded: dict[str, dict] = {}
    manifest_hashes: dict[str, str] = {}
    for name, config in cells.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("matrix cell name must be a simple directory name")
        lean = config["club"]["shaft_lean_deg"]
        if lean not in LEANS:
            raise ValueError("across-lean study supports lean 0, 5, and 10 degrees")
        group = groups.setdefault(_group_key(name, config), {})
        if lean in group:
            raise ValueError("duplicate lean in a matched study group")
        group[lean] = (name, config)
        directory = matrix_root / name
        if directory.exists():
            loaded[name] = load_validated_cell(directory, config, plan["execution"])
            manifest_hashes[name] = _sha256(directory / "manifest.json")
        else:
            loaded[name] = {"status": "not_run"}
    source_hashes = {
        **_source_hashes(),
        "matched_lean_control.py": _sha256(Path(__file__)),
    }
    pairs = []
    for key, members in groups.items():
        club_id, profile, rotation_sd, curve, target, n_shots, seed, radius = key[:8]
        if (
            not isinstance(target, (int, float))
            or not math.isfinite(target)
            or target <= 0
        ):
            raise ValueError("common target must be finite and positive")
        for lean_a, lean_b in combinations(LEANS, 2):
            descriptor = {
                "club_id": club_id,
                "profile": profile,
                "rotation_sd_deg": rotation_sd,
                "curve_scale": curve,
                "target_x_m": target,
                "target_radius_m": radius,
                "leans_deg": [lean_a, lean_b],
                "difference_direction": "later_lean_minus_earlier_lean",
                "total_per_physical_population": n_shots,
            }
            if lean_a not in members or lean_b not in members:
                pairs.append(
                    {
                        **descriptor,
                        "status": "unavailable",
                        "reason": "lean_not_planned",
                    }
                )
                continue
            name_a, _ = members[lean_a]
            name_b, _ = members[lean_b]
            descriptor["cells"] = [name_a, name_b]
            if (
                loaded[name_a]["status"] != "completed"
                or loaded[name_b]["status"] != "completed"
            ):
                pairs.append(
                    {
                        **descriptor,
                        "status": "unavailable",
                        "reason": "one_or_both_cells_unavailable",
                        "cell_statuses": [
                            loaded[name_a]["status"],
                            loaded[name_b]["status"],
                        ],
                    }
                )
                continue
            summary_a, summary_b = loaded[name_a], loaded[name_b]
            if (
                float(summary_a["target_x_m"]) != target
                or float(summary_b["target_x_m"]) != target
            ):
                raise ValueError("saved cells do not share the declared common target")
            endpoints_a, invalid_a = _supported_endpoints(
                matrix_root / name_a / "shots.csv.gz", n_shots
            )
            endpoints_b, invalid_b = _supported_endpoints(
                matrix_root / name_b / "shots.csv.gz", n_shots
            )
            common = sorted(
                {i for i, _, _ in endpoints_a["Straight"]}
                & {i for i, _, _ in endpoints_b["Straight"]}
            )
            if len(common) < 2:
                pairs.append(
                    {
                        **descriptor,
                        "status": "unavailable",
                        "reason": "fewer_than_two_common_supported_shots",
                        "common_supported_n": len(common),
                        "unsupported_by_cell_and_pattern": {
                            name_a: invalid_a,
                            name_b: invalid_b,
                        },
                    }
                )
                continue
            metrics = {}
            for pattern in PATTERNS:
                before_map = {i: (x, y) for i, x, y in endpoints_a[pattern]}
                after_map = {i: (x, y) for i, x, y in endpoints_b[pattern]}
                before = [(i, *before_map[i]) for i in common]
                after = [(i, *after_map[i]) for i in common]
                metrics[pattern] = _paired_pattern_metrics(
                    before,
                    after,
                    club_id=club_id,
                    target_x_m=target,
                    radius_m=radius,
                    seed=seed,
                )
                metrics[pattern].update(
                    physical_pattern=pattern,
                    earlier_lean_deg=lean_a,
                    later_lean_deg=lean_b,
                )
            pairs.append(
                {
                    **descriptor,
                    "status": "available"
                    if len(common) == n_shots
                    else "conditional_only",
                    "common_supported_n": len(common),
                    "physical_eligible_rows": 6 * len(common),
                    "unsupported_by_cell_and_pattern": {
                        name_a: invalid_a,
                        name_b: invalid_b,
                    },
                    "conditioning": "same shot indices supported in all six physical populations",
                    "patterns": metrics,
                }
            )
    if _sha256(plan_path) != plan_hash or any(
        not manifest_files_match(matrix_root / name / "manifest.json")
        or _sha256(matrix_root / name / "manifest.json") != digest
        for name, digest in manifest_hashes.items()
    ):
        raise RuntimeError("saved matrix inputs changed during lean comparison")
    if {
        **_source_hashes(),
        "matched_lean_control.py": _sha256(Path(__file__)),
    } != source_hashes:
        raise RuntimeError("scoring source changed during lean comparison")
    return {
        "schema": "matched-across-lean/1",
        "status": "complete"
        if all(p["status"] != "unavailable" for p in pairs)
        else "partial",
        "input_sha256": {
            "matrix_plan.json": plan_hash,
            "cell_manifests": manifest_hashes,
        },
        "source_sha256": source_hashes,
        "execution": plan["execution"],
        "population": {"planned_groups": len(groups), "planned_pairs": len(pairs)},
        "pairs": pairs,
        "limitations": "Conditional geometric model; shared supported indices only; historical hypothetical course SG with sampling-only paired intervals.",
    }


def write_across_lean_report(matrix_root: Path, destination: Path) -> Path:
    """Publish one atomic report outside every immutable matrix cell directory."""
    matrix_root = Path(matrix_root)
    destination = Path(destination)
    plan = json.loads((matrix_root / "matrix_plan.json").read_text())
    if any(
        (matrix_root / name).resolve() in destination.resolve().parents
        for name in plan["cells"]
    ):
        raise ValueError("lean report must be outside immutable matrix cells")
    if destination.exists():
        raise FileExistsError("lean report already exists")
    result = analyze_across_lean(matrix_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
