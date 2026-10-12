"""Read-only paired comparisons across actual shaft-lean study cells."""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .core import PATTERN_NAMES
from .matched_matrix import PROFILES
from .matched_report import load_validated_cell, verify_nominal_deliveries_hit_target
from .matched_sensitivity import _source_hashes, _supported_endpoints
from .provenance import manifest_files_match, publish_new_text, sha256_file
from .scenario_scoring import score_course_endpoints
from .course_score_vectors import score_course_endpoint_vectors
from .uncertainty import paired_mean_difference

PATTERNS = PATTERN_NAMES
LEANS = (0.0, 5.0, 10.0)


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
    a: np.ndarray = np.asarray([(x, y) for _, x, y in before], dtype=float)
    b: np.ndarray = np.asarray([(x, y) for _, x, y in after], dtype=float)
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


_PAIR_ALLOWED_CONFIG_DIFFERENCES = {
    "club.shaft_lean_deg",
    "curve_scale",
    "target_x_m",
    "match_face_bounds_deg",
    "match_speed_factors",
    "noise.standard_deviations[0]",
}
_PAIR_ERROR_FIELDS = (
    "shaft_rotation_deg",
    "path_deg",
    "speed_mps",
    "contact_horizontal_m",
    "contact_vertical_m",
    "shaft_lean_deg",
    "attack_angle_deg",
)
_TRANSITION_NONPROJECTION_PATHS = {
    f"src/tools/shot_pattern_analysis/{name}"
    for name in (
        "matched_runner.py",
        "matched_matrix.py",
        "matched_supplement.py",
        "matched_export.py",
        "nominal_matching.py",
        "density.py",
        "academic_plots.py",
        "matched_report_plots.py",
        "provenance.py",
        "scenario_scoring.py",
        "course_score_vectors.py",
    )
}
_TRANSITION_LEGACY_PATH = "src/shared/python/physics/impact_model/utils.py"


def _verify_actual_nominal_target(
    summary: dict, target_x_m: float, tolerance: float
) -> None:
    verify_nominal_deliveries_hit_target(
        summary, target_x_m, tolerance, context="paired"
    )


def _config_differences(first: dict, second: dict, prefix: str = "") -> set[str]:
    differences: set[str] = set()
    for key in first.keys() | second.keys():
        name = f"{prefix}.{key}" if prefix else key
        if key not in first or key not in second:
            differences.add(name)
        elif isinstance(first[key], dict) and isinstance(second[key], dict):
            differences.update(_config_differences(first[key], second[key], name))
        elif isinstance(first[key], list) and isinstance(second[key], list):
            if len(first[key]) != len(second[key]):
                differences.add(name)
            elif (
                key in {"match_face_bounds_deg", "match_speed_factors"}
                and first[key] != second[key]
            ):
                # Matcher bounds are reviewed as a single declared vector;
                # ordinary scientific vectors retain index-level differences.
                differences.add(name)
            else:
                for index, (left, right) in enumerate(
                    zip(first[key], second[key], strict=True)
                ):
                    if left != right:
                        differences.add(f"{name}[{index}]")
        elif first[key] != second[key]:
            differences.add(name)
    return differences


def _comparison_config(config: dict) -> dict:
    """Normalize known pre-bounds defaults while retaining raw source configs."""
    normalized = json.loads(json.dumps(config))
    if "match_face_bounds_deg" not in normalized:
        normalized["match_face_bounds_deg"] = (
            "legacy_reference_centered_plus_minus_3_deg"
        )
    if "match_speed_factors" not in normalized:
        normalized["match_speed_factors"] = [0.75, 1.25]
    return normalized


def _speed_factors(config: dict) -> list[float]:
    return [float(value) for value in config.get("match_speed_factors", [0.75, 1.25])]


def _is_reviewed_driver_speed_control(
    first_config: dict, second_config: dict, first_plan: dict, second_plan: dict
) -> bool:
    """Allow only the predeclared driver 0→5° and 1.25→1.30 control pair."""
    first_club, second_club = first_config["club"], second_config["club"]
    return (
        first_club["club_id"] == second_club["club_id"] == "driver"
        and (float(first_club["shaft_lean_deg"]), float(second_club["shaft_lean_deg"]))
        == (0.0, 5.0)
        and _speed_factors(first_config) == [0.75, 1.25]
        and _speed_factors(second_config) == [0.75, 1.30]
        and first_plan.get("supplement_protocol") is None
        and second_plan.get("supplement_protocol") == "driver-control-speed-bounds/1"
    )


def _is_reviewed_driver_curve_control(
    first_config: dict,
    second_config: dict,
    first_plan: dict,
    second_plan: dict,
    first_protocol: str | None,
) -> bool:
    """Qualify the fixed-target Driver5° curve1→2 speed-bound extension only."""
    first_club, second_club = first_config["club"], second_config["club"]
    return (
        first_protocol == "original_control/1"
        and first_plan.get("supplement_protocol") is None
        and second_plan.get("supplement_protocol") == "driver-control-speed-bounds/1"
        and first_club["club_id"] == second_club["club_id"] == "driver"
        and first_club["shaft_lean_deg"] == second_club["shaft_lean_deg"] == 5.0
        and (first_config["curve_scale"], second_config["curve_scale"]) == (1.0, 2.0)
        and _speed_factors(first_config) == [0.75, 1.25]
        and _speed_factors(second_config) == [0.75, 1.30]
        and first_config.get("match_face_bounds_deg") is None
        and second_config.get("match_face_bounds_deg") == [-6.0, 6.0]
        and first_config.get("target_x_m") is not None
        and first_config["target_x_m"] == second_config.get("target_x_m")
    )


def _execution_transition(
    first: dict,
    second: dict,
    receipt_path: Path | None,
    expected_sha256: str | None,
) -> str | None:
    """Validate an independently accepted, hash-bound source transition."""
    if first == second:
        if receipt_path is not None or expected_sha256 is not None:
            raise ValueError(
                "transition receipt is unnecessary for identical executions"
            )
        return None
    if receipt_path is None or expected_sha256 is None:
        raise ValueError(
            "cross-snapshot comparison requires an accepted source-transition receipt"
        )
    receipt_path = Path(receipt_path).resolve()
    if not receipt_path.is_file() or sha256_file(receipt_path) != expected_sha256:
        raise ValueError("source-transition receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        set(receipt)
        != {
            "schema",
            "status",
            "qualified_scope",
            "old_execution",
            "new_execution",
            "changed_sources",
            "evidence_files",
        }
        or receipt.get("schema") != "matched-delivery-source-transition/1"
        or receipt.get("status") != "independently_accepted"
        or receipt.get("qualified_scope") != "matched-delivery-comparisons"
    ):
        raise ValueError(
            "source-transition receipt is not accepted for matched comparisons"
        )
    old, new = receipt.get("old_execution"), receipt.get("new_execution")
    if not ((old == first and new == second) or (old == second and new == first)):
        raise ValueError(
            "source-transition receipt does not bind both source executions"
        )
    old_sources, new_sources = old.get("source_sha256"), new.get("source_sha256")
    if not isinstance(old_sources, dict) or not isinstance(new_sources, dict):
        raise ValueError("source-transition receipt lacks source hash maps")
    if {key: value for key, value in old.items() if key != "source_sha256"} != {
        key: value for key, value in new.items() if key != "source_sha256"
    }:
        raise ValueError(
            "source-transition receipt changes runtime or backend metadata"
        )
    changed = {
        path
        for path in old_sources.keys() | new_sources.keys()
        if old_sources.get(path) != new_sources.get(path)
    }
    classifications = receipt.get("changed_sources")
    if not isinstance(classifications, dict) or set(classifications) != changed:
        raise ValueError(
            "source-transition changed-source inventory is incomplete or has extras"
        )
    for path, record in classifications.items():
        if not isinstance(record, dict) or set(record) != {
            "old_sha256",
            "new_sha256",
            "classification",
        }:
            raise ValueError(f"source-transition record is malformed for {path}")
        old_hash, new_hash = old_sources.get(path), new_sources.get(path)
        if record.get("old_sha256") != old_hash or record.get("new_sha256") != new_hash:
            raise ValueError(f"source-transition hash pair differs for {path}")
        classification = record.get("classification")
        expected_class = (
            "unreachable_legacy"
            if path == _TRANSITION_LEGACY_PATH
            else "approved_nonprojection"
        )
        allowed = _TRANSITION_NONPROJECTION_PATHS | {_TRANSITION_LEGACY_PATH}
        if path not in allowed or classification != expected_class:
            raise ValueError(
                f"source-transition includes an unapproved source change: {path}"
            )
    evidence_files = receipt.get("evidence_files")
    if not isinstance(evidence_files, dict) or not evidence_files:
        raise ValueError("source-transition receipt lacks hash-bound review evidence")
    for relative, expected_hash in evidence_files.items():
        evidence_path = (receipt_path.parent / relative).resolve()
        if (
            not evidence_path.is_relative_to(receipt_path.parent)
            or not evidence_path.is_file()
        ):
            raise ValueError(
                "source-transition evidence file is missing or escapes its directory"
            )
        if sha256_file(evidence_path) != expected_hash:
            raise ValueError("source-transition evidence file hash mismatch")
    return expected_sha256


def _reload_explicit_source(reference: dict) -> tuple[Path, dict, dict, dict]:
    """Revalidate one unique source or a no-execution alias reference."""
    root = Path(reference["source_root"]).resolve()
    plan_path = root / "matrix_plan.json"
    if not plan_path.is_file() or sha256_file(plan_path) != reference.get(
        "source_plan_sha256"
    ):
        raise ValueError("explicit comparison source plan hash changed")
    plan = json.loads(plan_path.read_text())
    is_alias = reference.get("execution_kind") == "provenance_alias"
    if is_alias:
        if reference.get("proof_status") != "verified":
            raise ValueError("unverified alias cannot enter a paired comparison")
        cell_name = reference.get("alias_of_primary")
    else:
        if reference.get("execution_kind") != "unique":
            raise ValueError("source must identify a unique run or verified alias")
        cell_name = reference.get("cell")
    if not isinstance(cell_name, str) or cell_name not in plan.get("cells", {}):
        raise ValueError("explicit comparison source cell is not in its source plan")
    protocol = reference.get("source_protocol")
    if (
        plan.get("supplement_protocol") is not None
        and plan["supplement_protocol"] != protocol
    ):
        raise ValueError("explicit comparison source protocol differs from its plan")
    config = plan["cells"][cell_name]
    directory = root / cell_name
    manifest = directory / "manifest.json"
    csv_path = directory / "shots.csv.gz"
    if (
        not manifest.is_file()
        or sha256_file(manifest) != reference.get("source_cell_manifest_sha256")
        or not csv_path.is_file()
        or sha256_file(csv_path) != reference.get("source_csv_sha256")
    ):
        raise ValueError("explicit comparison source artifact hash changed")
    summary = load_validated_cell(directory, config, plan["execution"])
    if summary["status"] != "completed":
        raise ValueError("explicit paired comparison requires completed source cells")
    if is_alias:
        requested = reference.get("requested_config")
        expected = json.loads(json.dumps(config))
        expected["target_x_m"] = summary.get("target_x_m")
        if requested != expected:
            raise ValueError(
                "control alias request differs beyond the target-mode reference"
            )
    return directory, config, summary, plan


def _recheck_explicit_source(reference: dict, directory: Path) -> None:
    root = Path(reference["source_root"]).resolve()
    if sha256_file(root / "matrix_plan.json") != reference["source_plan_sha256"]:
        raise ValueError("explicit comparison source plan changed during analysis")
    manifest = directory / "manifest.json"
    if (
        not manifest_files_match(manifest)
        or sha256_file(manifest) != reference["source_cell_manifest_sha256"]
        or sha256_file(directory / "shots.csv.gz") != reference["source_csv_sha256"]
    ):
        raise ValueError("explicit comparison source changed during analysis")


def _primitive_errors_by_pattern(
    csv_path: Path, config: dict, summary: dict
) -> tuple[dict[str, dict[int, tuple[float, ...]]], dict[str, int]]:
    n_shots = config["n_shots"]
    nominals = summary.get("nominals")
    if not isinstance(nominals, dict) or set(nominals) != set(PATTERNS):
        raise ValueError("saved cell lacks nominal delivery baselines for all patterns")
    observed: dict[str, dict[int, tuple[float, ...]]] = {
        pattern: {} for pattern in PATTERNS
    }
    unsupported = dict.fromkeys(PATTERNS, 0)
    required = {"pattern", "shot_index", "status", *_PAIR_ERROR_FIELDS}
    with gzip.open(csv_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("saved shots lack primitive delivery inputs")
        for row in reader:
            pattern = row["pattern"]
            if pattern not in observed:
                raise ValueError("saved shot has an unknown pattern")
            try:
                index = int(row["shot_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError("saved shot index is invalid") from exc
            if index < 0 or index >= n_shots or index in observed[pattern]:
                raise ValueError("saved shot indices must be distinct and in range")
            try:
                path_error = float(row["path_deg"]) - float(
                    nominals[pattern]["path_deg"]
                )
                speed_error = float(row["speed_mps"]) - float(
                    nominals[pattern]["speed_mps"]
                )
                lean_error = float(row["shaft_lean_deg"]) - float(
                    config["club"]["shaft_lean_deg"]
                )
                attack_error = float(row["attack_angle_deg"]) - float(
                    config["club"]["attack_angle_deg"]
                )
                values = (
                    float(row["shaft_rotation_deg"]),
                    path_error,
                    speed_error,
                    float(row["contact_horizontal_m"]),
                    float(row["contact_vertical_m"]),
                    lean_error,
                    attack_error,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("saved primitive delivery input is invalid") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError("saved primitive delivery input must be finite")
            observed[pattern][index] = values
            if row["status"] != "supported":
                unsupported[pattern] += 1
    expected_indices = set(range(n_shots))
    if any(set(rows) != expected_indices for rows in observed.values()):
        raise ValueError("saved population must contain every shot index per pattern")
    return observed, unsupported


def _verify_common_primitive_errors(
    first: dict[str, dict[int, tuple[float, ...]]],
    second: dict[str, dict[int, tuple[float, ...]]],
) -> float:
    maximum = 0.0
    for pattern in PATTERNS:
        if set(first[pattern]) != set(second[pattern]):
            raise ValueError("paired cells have different shot IDs")
        for index in first[pattern]:
            for left, right in zip(
                first[pattern][index], second[pattern][index], strict=True
            ):
                maximum = max(maximum, abs(left - right))
        for index in first[pattern]:
            reference = first[pattern][index]
            for other_pattern in PATTERNS[1:]:
                if any(
                    abs(a - b) > 1e-12
                    for a, b in zip(reference, first[other_pattern][index], strict=True)
                ):
                    raise ValueError(
                        "source patterns do not reuse common primitive input errors"
                    )
            for other_pattern in PATTERNS:
                if any(
                    abs(a - b) > 1e-12
                    for a, b in zip(
                        reference, second[other_pattern][index], strict=True
                    )
                ):
                    raise ValueError("paired cells have primitive input error mismatch")
    if maximum > 1e-12:
        raise ValueError("paired cells have primitive input error mismatch")
    return maximum


def _normalized_rotation_errors(
    errors: dict[str, dict[int, tuple[float, ...]]],
    rotation_sd_deg: float,
) -> dict[str, dict[int, tuple[float, ...]]]:
    if not math.isfinite(rotation_sd_deg) or rotation_sd_deg <= 0:
        raise ValueError(
            "shaft rotation standard deviation must be finite and positive"
        )
    return {
        pattern: {
            index: (values[0] / rotation_sd_deg, *values[1:])
            for index, values in rows.items()
        }
        for pattern, rows in errors.items()
    }


def _paired_scoring_source_hashes() -> dict[str, str]:
    """Bind the scoring adapters and uncertainty helper across pair analysis."""
    package = Path(__file__).parent
    hashes = _source_hashes()
    hashes["matched_lean_control.py"] = sha256_file(Path(__file__))
    hashes["course_score_vectors.py"] = sha256_file(package / "course_score_vectors.py")
    return hashes


def _require_rotation_sd_order(first_sd: float, second_sd: float) -> None:
    if (first_sd, second_sd) != (1.0, 2.0):
        raise ValueError("shaft-SD pair must order 1 degree before 2 degrees")


def _paired_effect_modification(
    straight_a: np.ndarray,
    curved_a: np.ndarray,
    straight_b: np.ndarray,
    curved_b: np.ndarray,
    *,
    seed: int,
) -> dict:
    """Bootstrap (curve−straight)_B − (curve−straight)_A by shared index."""
    arrays = tuple(
        np.asarray(value, dtype=float)
        for value in (straight_a, curved_a, straight_b, curved_b)
    )
    if any(array.ndim != 1 or array.size < 2 for array in arrays):
        raise ValueError("effect modification requires at least two paired indices")
    if len({array.shape for array in arrays}) != 1 or any(
        not np.all(np.isfinite(array)) for array in arrays
    ):
        raise ValueError("effect modification inputs must be equal finite vectors")
    interaction_b = arrays[3] - arrays[2]
    interaction_a = arrays[1] - arrays[0]
    return paired_mean_difference(interaction_b, interaction_a, seed=seed)


@dataclass(frozen=True)
class _SourceCell:
    reference: dict
    directory: Path
    config: dict
    summary: dict
    plan: dict


@dataclass(frozen=True)
class _PairedPopulation:
    endpoints_a: dict
    endpoints_b: dict
    common_indices: list[int]
    unsupported_a: dict
    unsupported_b: dict
    max_error: float
    input_error_note: str


def _validate_pair_differences(
    first: _SourceCell, second: _SourceCell, comparison_kind: str
) -> set[str]:
    config_a, config_b = first.config, second.config
    plan_a, plan_b = first.plan, second.plan
    differences = _config_differences(
        _comparison_config(config_a), _comparison_config(config_b)
    )
    if differences - _PAIR_ALLOWED_CONFIG_DIFFERENCES:
        raise ValueError(
            f"source config differences are not allowed: {sorted(differences)}"
        )
    if comparison_kind == "across_lean":
        expected_differences = {
            "club.shaft_lean_deg",
            "target_x_m",
            "match_face_bounds_deg",
            "match_speed_factors",
        }
    elif comparison_kind == "curve_strength":
        expected_differences = {"curve_scale", "target_x_m", "match_face_bounds_deg"}
        if _is_reviewed_driver_curve_control(
            config_a, config_b, plan_a, plan_b, first.reference.get("source_protocol")
        ):
            expected_differences.add("match_speed_factors")
    elif comparison_kind == "shaft_rotation_sd":
        expected_differences = {"noise.standard_deviations[0]", "target_x_m"}
    else:
        expected_differences = {
            "match_speed_factors",
            "target_x_m",
            "match_face_bounds_deg",
        }
    if not differences.issubset(expected_differences):
        raise ValueError(
            f"source config differences are not valid for {comparison_kind}: {sorted(differences)}"
        )
    if (
        comparison_kind == "across_lean"
        and "match_speed_factors" in differences
        and not _is_reviewed_driver_speed_control(config_a, config_b, plan_a, plan_b)
    ):
        raise ValueError(
            "speed bounds differ outside the declared driver control protocol"
        )
    return differences


def _validate_pair_conditions(
    config_a: dict, config_b: dict, comparison_kind: str
) -> None:
    club_a, club_b = config_a["club"], config_b["club"]
    if comparison_kind == "across_lean":
        if (
            club_a["shaft_lean_deg"] == club_b["shaft_lean_deg"]
            or config_a["curve_scale"] != config_b["curve_scale"]
        ):
            raise ValueError(
                "across-lean pair must change lean and hold curve scale fixed"
            )
        if (float(club_a["shaft_lean_deg"]), float(club_b["shaft_lean_deg"])) not in (
            (0.0, 5.0),
            (0.0, 10.0),
            (5.0, 10.0),
        ):
            raise ValueError(
                "across-lean pair must order supported leans 0, 5, and 10 ascending"
            )
    else:
        if comparison_kind == "curve_strength" and (
            club_a["shaft_lean_deg"] != club_b["shaft_lean_deg"]
            or (config_a["curve_scale"], config_b["curve_scale"]) != (1.0, 2.0)
        ):
            raise ValueError(
                "curve-strength pair must hold lean and compare curve scales 1 and 2"
            )
        if comparison_kind == "shaft_rotation_sd":
            sd_a = float(config_a["noise"]["standard_deviations"][0])
            sd_b = float(config_b["noise"]["standard_deviations"][0])
            if (
                club_a["shaft_lean_deg"] != club_b["shaft_lean_deg"]
                or config_a["curve_scale"] != config_b["curve_scale"]
            ):
                raise ValueError("shaft-SD pair must hold lean and curve scale fixed")
            _require_rotation_sd_order(sd_a, sd_b)
        if comparison_kind == "speed_bound":
            speed_a, speed_b = _speed_factors(config_a), _speed_factors(config_b)
            if (
                len(speed_a) != 2
                or len(speed_b) != 2
                or not math.isclose(
                    float(speed_a[0]), float(speed_b[0]), rel_tol=0.0, abs_tol=1e-12
                )
                or (float(speed_a[1]), float(speed_b[1])) != (1.25, 1.30)
            ):
                raise ValueError(
                    "speed-bound pair must compare the shared lower bound and 1.25/1.30 upper factors"
                )


def _validate_pair_target(
    first: _SourceCell, second: _SourceCell, target_x_m: float
) -> None:
    config_a, config_b = first.config, second.config
    summary_a, summary_b = first.summary, second.summary
    if (
        config_a["n_shots"] != config_b["n_shots"]
        or config_a["seed"] != config_b["seed"]
    ):
        raise ValueError("paired cells must have identical shot count and seed")
    target_a = float(summary_a["target_x_m"])
    target_b = float(summary_b["target_x_m"])
    if not math.isclose(
        target_a, target_x_m, rel_tol=0.0, abs_tol=1e-12
    ) or not math.isclose(target_b, target_x_m, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("paired cells do not share the declared physical target")
    tolerance = max(
        float(config_a["match_tolerance_m"]), float(config_b["match_tolerance_m"])
    )
    for summary in (summary_a, summary_b):
        _verify_actual_nominal_target(summary, target_x_m, tolerance)


def _paired_population(
    first: _SourceCell, second: _SourceCell, comparison_kind: str
) -> _PairedPopulation:
    config_a, config_b = first.config, second.config
    summary_a, summary_b = first.summary, second.summary
    directory_a, directory_b = first.directory, second.directory
    errors_a, unsupported_a = _primitive_errors_by_pattern(
        directory_a / "shots.csv.gz", config_a, summary_a
    )
    errors_b, unsupported_b = _primitive_errors_by_pattern(
        directory_b / "shots.csv.gz", config_b, summary_b
    )
    if comparison_kind == "shaft_rotation_sd":
        sd_a = float(config_a["noise"]["standard_deviations"][0])
        sd_b = float(config_b["noise"]["standard_deviations"][0])
        normalized_a = _normalized_rotation_errors(errors_a, sd_a)
        normalized_b = _normalized_rotation_errors(errors_b, sd_b)
        max_error = _verify_common_primitive_errors(normalized_a, normalized_b)
        input_error_note = "shaft_rotation_deg divided by its configured SD; remaining centered primitive errors compared directly"
    else:
        max_error = _verify_common_primitive_errors(errors_a, errors_b)
        input_error_note = "all centered primitive input errors compared directly"
    endpoints_a, _ = _supported_endpoints(
        directory_a / "shots.csv.gz", config_a["n_shots"]
    )
    endpoints_b, _ = _supported_endpoints(
        directory_b / "shots.csv.gz", config_b["n_shots"]
    )
    common_indices = sorted(
        set.intersection(
            *(
                {index for index, _, _ in endpoints[pattern]}
                for endpoints in (endpoints_a, endpoints_b)
                for pattern in PATTERNS
            )
        )
    )
    return _PairedPopulation(
        endpoints_a,
        endpoints_b,
        common_indices,
        unsupported_a,
        unsupported_b,
        max_error,
        input_error_note,
    )


def _explicit_source_receipt(source: dict) -> dict:
    return {
        "protocol_id": source["source_protocol"],
        "cell": source.get("alias_of_primary", source.get("cell")),
        "execution_kind": source["execution_kind"],
        "alias_cell": source.get("alias_cell"),
        "source_plan_sha256": source["source_plan_sha256"],
        "source_cell_manifest_sha256": source["source_cell_manifest_sha256"],
        "source_csv_sha256": source["source_csv_sha256"],
    }


def _explicit_pair_receipt(
    sources: tuple[dict, dict],
    n_total: int,
    population: _PairedPopulation,
    comparison_kind: str,
    target_x_m: float,
    scoring_sources: dict,
    transition_sha: str | None,
    differences: set[str],
) -> dict:
    first_source, second_source = sources
    common_indices = population.common_indices
    unsupported_a, unsupported_b = population.unsupported_a, population.unsupported_b
    max_error, input_error_note = population.max_error, population.input_error_note
    n_common = len(common_indices)
    status = "available" if n_common == n_total else "conditional_only"
    if n_common < 2:
        status = "unavailable"
    protocol_a = first_source["source_protocol"]
    protocol_b = second_source["source_protocol"]
    result = {
        "schema": "matched-explicit-source-pair/1",
        "comparison_kind": comparison_kind,
        "status": status,
        "reason": "fewer_than_two_common_supported_shots" if n_common < 2 else "",
        "target_x_m": target_x_m,
        "total_n_per_population": n_total,
        "common_supported_n": n_common,
        "conditioning": "same_indices_supported_in_all_six_populations",
        "input_error_tolerance": 1e-12,
        "input_error_max_abs_difference": max_error,
        "input_error_comparison": input_error_note,
        "analysis_source_sha256": scoring_sources,
        "source_protocols": [protocol_a, protocol_b],
        "source_transition_receipt_sha256": transition_sha,
        "sources": [
            _explicit_source_receipt(first_source),
            _explicit_source_receipt(second_source),
        ],
        "unsupported_by_source_and_pattern": {
            "first": unsupported_a,
            "second": unsupported_b,
        },
        "allowed_config_differences": sorted(differences),
        "patterns": {},
        "limitations": "Paired conditional geometric simulation; matched-condition changes include nominal re-matching and do not isolate causal curvature; no inference of mirrored outcome symmetry; source protocols remain separate.",
    }
    return result


def _populate_explicit_metrics(
    result: dict,
    first: _SourceCell,
    second: _SourceCell,
    population: _PairedPopulation,
    target_x_m: float,
    comparison_kind: str,
) -> None:
    config_a, config_b = first.config, second.config
    club_a, club_b = config_a["club"], config_b["club"]
    endpoints_a, endpoints_b = population.endpoints_a, population.endpoints_b
    common_indices = population.common_indices
    common_index_set = set(common_indices)
    endpoint_maps = []
    for endpoints in (endpoints_a, endpoints_b):
        endpoint_maps.append(
            {
                pattern: [
                    (index, x, y)
                    for index, x, y in sorted(endpoints[pattern])
                    if index in common_index_set
                ]
                for pattern in PATTERNS
            }
        )
    for pattern in PATTERNS:
        map_a = {index: (x, y) for index, x, y in endpoints_a[pattern]}
        map_b = {index: (x, y) for index, x, y in endpoints_b[pattern]}
        before = [(index, *map_a[index]) for index in common_indices]
        after = [(index, *map_b[index]) for index in common_indices]
        metrics = _paired_pattern_metrics(
            before,
            after,
            club_id=club_a["club_id"],
            target_x_m=target_x_m,
            radius_m=float(config_a["target_radius_m"]),
            seed=int(config_a["seed"]),
        )
        metrics["physical_pattern"] = pattern
        metrics["first_condition"] = {
            "shaft_lean_deg": club_a["shaft_lean_deg"],
            "curve_scale": config_a["curve_scale"],
            "shaft_rotation_sd_deg": config_a["noise"]["standard_deviations"][0],
        }
        metrics["second_condition"] = {
            "shaft_lean_deg": club_b["shaft_lean_deg"],
            "curve_scale": config_b["curve_scale"],
            "shaft_rotation_sd_deg": config_b["noise"]["standard_deviations"][0],
        }
        metrics["role_mapping"] = {
            "Straight": "physical pattern at first condition",
            "Draw": "same physical pattern at second condition",
            "Fade": "duplicate of first condition for paired scorer shape only; not an observation",
        }
        metrics["first_curve_scale"] = config_a["curve_scale"]
        metrics["second_curve_scale"] = config_b["curve_scale"]
        result["patterns"][pattern] = metrics
    if comparison_kind in ("across_lean", "curve_strength", "shaft_rotation_sd"):
        _populate_effect_modification(
            result, endpoint_maps, config_a, config_b, target_x_m
        )


def _populate_effect_modification(
    result: dict,
    endpoint_maps: list[dict],
    config_a: dict,
    config_b: dict,
    target_x_m: float,
) -> None:
    club_a, club_b = config_a["club"], config_b["club"]
    try:
        score_a = score_course_endpoint_vectors(
            endpoint_maps[0],
            club_id=club_a["club_id"],
            target_x_m=target_x_m,
            green_radius_m=float(config_a["target_radius_m"]),
        )
        score_b = score_course_endpoint_vectors(
            endpoint_maps[1],
            club_id=club_b["club_id"],
            target_x_m=target_x_m,
            green_radius_m=float(config_b["target_radius_m"]),
        )
        effects = {
            pattern: _paired_effect_modification(
                score_a.scores["Straight"],
                score_a.scores[pattern],
                score_b.scores["Straight"],
                score_b.scores[pattern],
                seed=int(config_a["seed"]),
            )
            for pattern in ("Draw", "Fade")
        }
        result["course_scoring_effect_modification"] = effects
        result["course_scoring_effect_modification_scope"] = (
            "(curve−Straight) in the second condition minus (curve−Straight) in the first; paired on the same six-population shot-index intersection; positive means increased relative benefit"
        )
        result["course_scoring_provenance"] = {
            "source_backed_status": "available",
            "scenario_id": score_a.scenario_id,
            "baseline": score_a.baseline,
            "api_evaluated_states_per_condition": score_a.api_evaluated_states,
            "index_order": "ascending_common_shot_index",
        }
    except ValueError as exc:
        result["course_scoring_effect_modification"] = {
            "status": "unavailable",
            "reason": str(exc),
        }


def analyze_explicit_source_pair(
    first_source: dict,
    second_source: dict,
    *,
    comparison_kind: str,
    target_x_m: float,
    transition_receipt_path: Path | None = None,
    transition_receipt_sha256: str | None = None,
) -> dict:
    """Pair two validated source cells without conflating their protocols.

    Delivered values are centered on each pattern's saved nominal and club
    preset before the underlying Gaussian errors are compared. Comparisons use
    only the intersection supported in all six pattern-by-source populations.
    """
    if comparison_kind not in (
        "across_lean",
        "curve_strength",
        "shaft_rotation_sd",
        "speed_bound",
    ):
        raise ValueError("unknown explicit source comparison kind")
    if not math.isfinite(target_x_m) or target_x_m <= 0:
        raise ValueError("comparison target must be finite and positive")
    first = _SourceCell(first_source, *_reload_explicit_source(first_source))
    second = _SourceCell(second_source, *_reload_explicit_source(second_source))
    scoring_sources = _paired_scoring_source_hashes()
    execution_a, execution_b = first.plan.get("execution"), second.plan.get("execution")
    if not isinstance(execution_a, dict) or not isinstance(execution_b, dict):
        raise ValueError("paired source plans must bind execution snapshots")
    transition_sha = _execution_transition(
        execution_a, execution_b, transition_receipt_path, transition_receipt_sha256
    )
    differences = _validate_pair_differences(first, second, comparison_kind)
    _validate_pair_conditions(first.config, second.config, comparison_kind)
    _validate_pair_target(first, second, target_x_m)
    population = _paired_population(first, second, comparison_kind)
    result = _explicit_pair_receipt(
        (first_source, second_source),
        first.config["n_shots"],
        population,
        comparison_kind,
        target_x_m,
        scoring_sources,
        transition_sha,
        differences,
    )
    if len(population.common_indices) >= 2:
        _populate_explicit_metrics(
            result, first, second, population, target_x_m, comparison_kind
        )
    _recheck_explicit_source(first_source, first.directory)
    _recheck_explicit_source(second_source, second.directory)
    if _paired_scoring_source_hashes() != scoring_sources:
        raise RuntimeError("paired scoring source changed during analysis")
    _execution_transition(
        execution_a, execution_b, transition_receipt_path, transition_receipt_sha256
    )
    return result


def _lean_pair_metrics(
    endpoints_a: dict, endpoints_b: dict, common: list[int], descriptor: dict, seed: int
) -> dict:
    """Reduce paired endpoints in the historical physical-pattern order."""
    club_id = descriptor["club_id"]
    target, radius = descriptor["target_x_m"], descriptor["target_radius_m"]
    lean_a, lean_b = descriptor["leans_deg"]
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
    return metrics


def _analyze_lean_pair(
    matrix_root: Path, descriptor: dict, members: dict, loaded: dict, seed: int
) -> dict:
    """Resolve one planned pair and retain its six-population eligibility."""
    lean_a, lean_b = descriptor["leans_deg"]
    target = descriptor["target_x_m"]
    n_shots = descriptor["total_per_physical_population"]
    if lean_a not in members or lean_b not in members:
        return {
            **descriptor,
            "status": "unavailable",
            "reason": "lean_not_planned",
        }
    name_a, _ = members[lean_a]
    name_b, _ = members[lean_b]
    descriptor["cells"] = [name_a, name_b]
    if (
        loaded[name_a]["status"] != "completed"
        or loaded[name_b]["status"] != "completed"
    ):
        return {
            **descriptor,
            "status": "unavailable",
            "reason": "one_or_both_cells_unavailable",
            "cell_statuses": [
                loaded[name_a]["status"],
                loaded[name_b]["status"],
            ],
        }
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
        return {
            **descriptor,
            "status": "unavailable",
            "reason": "fewer_than_two_common_supported_shots",
            "common_supported_n": len(common),
            "unsupported_by_cell_and_pattern": {
                name_a: invalid_a,
                name_b: invalid_b,
            },
        }
    metrics = _lean_pair_metrics(endpoints_a, endpoints_b, common, descriptor, seed)
    return {
        **descriptor,
        "status": "available" if len(common) == n_shots else "conditional_only",
        "common_supported_n": len(common),
        "physical_eligible_rows": 6 * len(common),
        "unsupported_by_cell_and_pattern": {
            name_a: invalid_a,
            name_b: invalid_b,
        },
        "conditioning": "same shot indices supported in all six physical populations",
        "patterns": metrics,
    }


def analyze_across_lean(matrix_root: Path) -> dict:
    """Compare lean 0/5/10 on one six-population index intersection per pair."""
    matrix_root = Path(matrix_root)
    plan_path = matrix_root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = sha256_file(plan_path)
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
            manifest_hashes[name] = sha256_file(directory / "manifest.json")
        else:
            loaded[name] = {"status": "not_run"}
    source_hashes = {
        **_source_hashes(),
        "matched_lean_control.py": sha256_file(Path(__file__)),
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
            pairs.append(
                _analyze_lean_pair(matrix_root, descriptor, members, loaded, seed)
            )
    if sha256_file(plan_path) != plan_hash or any(
        not manifest_files_match(matrix_root / name / "manifest.json")
        or sha256_file(matrix_root / name / "manifest.json") != digest
        for name, digest in manifest_hashes.items()
    ):
        raise RuntimeError("saved matrix inputs changed during lean comparison")
    if {
        **_source_hashes(),
        "matched_lean_control.py": sha256_file(Path(__file__)),
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
    return publish_new_text(
        destination, json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
