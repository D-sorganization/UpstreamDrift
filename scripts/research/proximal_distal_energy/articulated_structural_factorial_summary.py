"""Deterministic gates and contrasts for the structural-factorial campaign."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_structural_factorial_corruption_audit import (
    validate_corruption_audit,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    StructuralCheckpoint,
    load_registered_checkpoints,
    plan_sha256,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_launcher import (
    validate_execution_session,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runtime_audit import (
    validate_runtime_audit,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_evidence import (
    EVIDENCE_SIDECAR_SCHEMA,
    validate_structural_evidence_arrays,
)

FloatArray = NDArray[np.float64]
_DIRECT_OUTCOMES = (
    "final_club_translation_speed_m_s",
    "club_linear_momentum_change_kg_m_s",
    "signed_contact_impulse_n_s",
    "signed_contact_work_j",
)
_DISSIPATION_COMPONENTS = (
    "terminal_contact_dissipation_j",
    "terminal_shaft_dissipation_j",
    "terminal_ground_dissipation_j",
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _relative_error(left: FloatArray, right: FloatArray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return float(np.max(np.abs(left - right)) / scale)


def _checkpoint_payload(checkpoint: StructuralCheckpoint) -> Mapping[str, Any]:
    return _mapping(
        json.loads(checkpoint.path.read_text(encoding="utf-8")),
        name="checkpoint",
    )


def _result(checkpoint: StructuralCheckpoint) -> Mapping[str, Any]:
    payload = _checkpoint_payload(checkpoint)
    outcome = _mapping(payload.get("outcome"), name="checkpoint.outcome")
    return _mapping(outcome.get("result"), name="checkpoint.outcome.result")


def _sidecar(checkpoint: StructuralCheckpoint) -> dict[str, NDArray[Any]]:
    result = _result(checkpoint)
    if result.get("evidence_sidecar_schema") != EVIDENCE_SIDECAR_SCHEMA:
        raise ValueError(
            "completed checkpoint lacks the registered evidence sidecar schema"
        )
    with np.load(checkpoint.path.with_suffix(".npz"), allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    validate_structural_evidence_arrays(arrays)
    return arrays


def _horizon_rows(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = result.get("horizons")
    if not isinstance(raw, list) or not raw:
        raise ValueError("completed result must contain horizon rows")
    return tuple(_mapping(row, name="horizon row") for row in raw)


def _outcome(row: Mapping[str, Any], name: str) -> float:
    if name == "terminal_total_dissipation_j":
        value = sum(float(row[component]) for component in _DISSIPATION_COMPONENTS)
    else:
        value = float(row[name])
    if not np.isfinite(value):
        raise ValueError(f"outcome {name} must be finite")
    return value


def _individual_numerical_pass(
    result: Mapping[str, Any], gates: Mapping[str, Any]
) -> bool:
    numerical = _mapping(result.get("numerical"), name="result.numerical")
    return bool(
        float(numerical["normalized_work_energy_residual"])
        <= float(gates["normalized_work_energy_residual_max"])
        and float(numerical["maximum_small_deflection_ratio"]) <= 0.05
        and float(numerical["maximum_twist_angle_rad"]) <= np.deg2rad(10.0)
        and float(numerical["maximum_base_translation_m"])
        <= float(gates["base_translation_max_m"])
        and float(numerical["maximum_base_pitch_rad"])
        <= float(gates["base_pitch_max_rad"])
    )


def _refinement_records(
    checkpoints: Sequence[StructuralCheckpoint], gates: Mapping[str, Any]
) -> tuple[list[dict[str, object]], set[tuple[object, ...]]]:
    grouped: dict[tuple[object, ...], list[tuple[float, float]]] = defaultdict(list)
    for checkpoint in checkpoints:
        if checkpoint.status != "completed":
            continue
        case = checkpoint.case
        numerical = _mapping(_result(checkpoint).get("numerical"), name="numerical")
        key = (
            case.source_case_index,
            case.source_sample_index,
            case.cell_id,
            case.velocity_factor,
            case.engine,
        )
        grouped[key].append(
            (case.time_step_s, float(numerical["normalized_work_energy_residual"]))
        )
    records: list[dict[str, object]] = []
    passed: set[tuple[object, ...]] = set()
    limit = float(gates["three_step_refinement_ratio_max"])
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        ordered = sorted(values, reverse=True)
        residuals = [value for _, value in ordered]
        complete = len(ordered) == 3
        successive_ratios = [
            later / max(earlier, np.finfo(float).tiny)
            for earlier, later in zip(residuals, residuals[1:], strict=False)
        ]
        maximum_successive_ratio = max(successive_ratios, default=float("inf"))
        fine_to_coarse_ratio = residuals[-1] / max(residuals[0], np.finfo(float).tiny)
        passes = (
            complete
            and all(
                later <= earlier + 1e-12
                for earlier, later in zip(residuals, residuals[1:], strict=False)
            )
            and maximum_successive_ratio <= limit
        )
        if passes:
            passed.add(key)
        records.append(
            {
                "group": list(key),
                "time_steps_s": [step for step, _ in ordered],
                "normalized_residuals": residuals,
                "successive_refinement_ratios": successive_ratios,
                "maximum_successive_refinement_ratio": maximum_successive_ratio,
                "fine_to_coarse_ratio": fine_to_coarse_ratio,
                "passes": passes,
            }
        )
    return records, passed


def _parity_records(
    checkpoints: Sequence[StructuralCheckpoint], gates: Mapping[str, Any]
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, StructuralCheckpoint]] = defaultdict(
        dict
    )
    for checkpoint in checkpoints:
        case = checkpoint.case
        key = (
            case.source_case_index,
            case.source_sample_index,
            case.cell_id,
            case.velocity_factor,
            case.time_step_s,
        )
        grouped[key][case.engine] = checkpoint
    records = []
    for key, engines in sorted(grouped.items(), key=lambda item: str(item[0])):
        available = {
            name: checkpoint
            for name, checkpoint in engines.items()
            if checkpoint.status == "completed"
        }
        if set(available) != {"mujoco", "pinocchio"}:
            records.append(
                {"group": list(key), "status": "unavailable", "passes": False}
            )
            continue
        left, right = _sidecar(available["mujoco"]), _sidecar(available["pinocchio"])
        trajectory = max(
            _relative_error(
                np.asarray(left[name], dtype=float),
                np.asarray(right[name], dtype=float),
            )
            for name in ("q", "qd", "elastic_coordinates", "base_coordinates")
        )
        force = max(
            _relative_error(
                np.asarray(left[name], dtype=float),
                np.asarray(right[name], dtype=float),
            )
            for name in (
                "net_club_force_n",
                "maximum_station_force_n",
                "ground_force_n",
            )
        )
        active = np.array_equal(
            left["active_station_count"], right["active_station_count"]
        )
        passes = bool(
            trajectory <= float(gates["cross_engine_trajectory_relative_error_max"])
            and force <= float(gates["cross_engine_force_relative_error_max"])
            and active
        )
        records.append(
            {
                "group": list(key),
                "status": "compared",
                "trajectory_relative_error": trajectory,
                "force_relative_error": force,
                "active_set_parity": active,
                "passes": passes,
            }
        )
    return records


def _contrast_records(
    *,
    plan: Mapping[str, object],
    checkpoints: Sequence[StructuralCheckpoint],
    individual_pass: Mapping[str, bool],
    refinement_pass: set[tuple[object, ...]],
) -> list[dict[str, object]]:
    design = _mapping(plan.get("design"), name="design")
    analysis = _mapping(plan.get("analysis"), name="analysis")
    factors = tuple(str(value) for value in design["factors"])
    contrasts = tuple(analysis["primary_contrasts"]) + tuple(
        analysis["exploratory_higher_order_contrasts"]
    )
    grouped: dict[tuple[object, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for checkpoint in checkpoints:
        if (
            checkpoint.status != "completed"
            or not individual_pass[checkpoint.case.case_key]
        ):
            continue
        case = checkpoint.case
        refinement_key = (
            case.source_case_index,
            case.source_sample_index,
            case.cell_id,
            case.velocity_factor,
            case.engine,
        )
        if refinement_key not in refinement_pass:
            continue
        for row in _horizon_rows(_result(checkpoint)):
            key = (
                case.source_case_index,
                case.source_sample_index,
                case.velocity_factor,
                case.time_step_s,
                case.engine,
                float(row["horizon_s"]),
            )
            grouped[key][case.cell_id] = row
    records = []
    outcomes = (*_DIRECT_OUTCOMES, "terminal_total_dissipation_j")
    for block, cells in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(cells) != 16:
            continue
        for raw_contrast in contrasts:
            contrast = _mapping(raw_contrast, name="contrast")
            selected = tuple(factors.index(str(name)) for name in contrast["factors"])
            for outcome_name in outcomes:
                signed = []
                for cell_id, row in cells.items():
                    sign = int(
                        np.prod(
                            [1 if cell_id[index] == "1" else -1 for index in selected]
                        )
                    )
                    signed.append(sign * _outcome(row, outcome_name))
                coefficient = float(np.mean(signed))
                records.append(
                    {
                        "block": list(block),
                        "contrast_id": contrast["contrast_id"],
                        "order": len(selected),
                        "outcome": outcome_name,
                        "walsh_coefficient": coefficient,
                        "high_minus_low_effect": 2.0 * coefficient,
                    }
                )
    return records


def _checkpoint_set_sha256(checkpoints: Sequence[StructuralCheckpoint]) -> str:
    digest = hashlib.sha256()
    for checkpoint in checkpoints:
        for path in (checkpoint.path, checkpoint.path.with_suffix(".npz")):
            if path.is_file():
                digest.update(path.name.encode("utf-8"))
                digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _runtime_session_identity(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    checkpoint_dir: Path,
    plan_path: Path | None,
    launch_path: Path | None,
    runtime_audit_path: Path | None,
) -> str | None:
    supplied = (plan_path, launch_path, runtime_audit_path)
    if all(path is None for path in supplied):
        return None
    if any(path is None for path in supplied):
        raise ValueError("runtime qualification requires plan, launch, and audit paths")
    assert plan_path is not None
    assert launch_path is not None
    assert runtime_audit_path is not None
    file_plan = _mapping(json.loads(plan_path.read_text(encoding="utf-8")), name="plan")
    file_launch = _mapping(
        json.loads(launch_path.read_text(encoding="utf-8")), name="launch"
    )
    if file_plan != plan or file_launch != launch:
        raise ValueError("summary mappings do not match their supplied files")
    audit = _mapping(
        json.loads(runtime_audit_path.read_text(encoding="utf-8")),
        name="runtime audit",
    )
    runtime_identity = validate_runtime_audit(plan=plan, launch=launch, audit=audit)
    validate_execution_session(
        plan_path=plan_path,
        launch_path=launch_path,
        runtime_audit_path=runtime_audit_path,
        launch=dict(launch),
        runtime_identity=runtime_identity,
        checkpoint_dir=checkpoint_dir,
    )
    return runtime_identity


def _corruption_audit_identity(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    checkpoint_dir: Path,
    corruption_audit_path: Path | None,
) -> str | None:
    if corruption_audit_path is None:
        return None
    audit = _mapping(
        json.loads(corruption_audit_path.read_text(encoding="utf-8")),
        name="corruption audit",
    )
    validate_corruption_audit(
        plan=plan,
        launch=launch,
        checkpoint_dir=checkpoint_dir,
        audit=audit,
    )
    return hashlib.sha256(corruption_audit_path.read_bytes()).hexdigest()


def _required_list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    return value


def _contrast_aggregates(
    *, plan: Mapping[str, object], contrasts: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    design = _mapping(plan.get("design"), name="design")
    analysis = _mapping(plan.get("analysis"), name="analysis")
    expected_blocks = int(
        np.prod(
            [
                len(_required_list(design.get("states"), name="design.states")),
                len(
                    _required_list(
                        design.get("velocity_factors"),
                        name="design.velocity_factors",
                    )
                ),
                len(_required_list(design.get("engines"), name="design.engines")),
                len(
                    _required_list(
                        design.get("time_steps_s"), name="design.time_steps_s"
                    )
                ),
                len(_required_list(design.get("horizons_s"), name="design.horizons_s")),
            ]
        )
    )
    definitions: list[tuple[str, Mapping[str, Any]]] = []
    for estimand_class, key in (
        ("primary", "primary_contrasts"),
        ("exploratory", "exploratory_higher_order_contrasts"),
    ):
        for raw in _required_list(analysis.get(key), name=f"analysis.{key}"):
            definitions.append(
                (estimand_class, _mapping(raw, name=f"analysis.{key} item"))
            )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in contrasts:
        grouped[(str(row["contrast_id"]), str(row["outcome"]))].append(
            float(row["walsh_coefficient"])
        )
    aggregates: list[dict[str, object]] = []
    for estimand_class, definition in definitions:
        contrast_id = str(definition["contrast_id"])
        order = len(_required_list(definition.get("factors"), name="contrast.factors"))
        for outcome in (*_DIRECT_OUTCOMES, "terminal_total_dissipation_j"):
            values = np.asarray(grouped[(contrast_id, outcome)], dtype=float)
            count = int(values.size)
            coefficient_range: dict[str, float | None]
            effect_range: dict[str, float | None]
            if count:
                coefficient_range = {
                    "minimum": float(np.min(values)),
                    "median": float(np.median(values)),
                    "maximum": float(np.max(values)),
                }
                effect_range = {
                    "minimum": float(2.0 * np.min(values)),
                    "median": float(2.0 * np.median(values)),
                    "maximum": float(2.0 * np.max(values)),
                }
            else:
                coefficient_range = {"minimum": None, "median": None, "maximum": None}
                effect_range = {"minimum": None, "median": None, "maximum": None}
            aggregates.append(
                {
                    "contrast_id": contrast_id,
                    "estimand_class": estimand_class,
                    "order": order,
                    "outcome": outcome,
                    "expected_block_count": expected_blocks,
                    "eligible_block_count": count,
                    "missing_block_count": expected_blocks - count,
                    "support_fraction": count / expected_blocks,
                    "exact_sign_counts": {
                        "negative": int(np.count_nonzero(values < 0.0)),
                        "zero": int(np.count_nonzero(values == 0.0)),
                        "positive": int(np.count_nonzero(values > 0.0)),
                    },
                    "sign_reversal": bool(
                        np.any(values < 0.0) and np.any(values > 0.0)
                    ),
                    "walsh_coefficient": coefficient_range,
                    "high_minus_low_effect": effect_range,
                }
            )
    return aggregates


def summarize_structural_factorial(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    checkpoint_dir: Path,
    plan_path: Path | None = None,
    launch_path: Path | None = None,
    runtime_audit_path: Path | None = None,
    corruption_audit_path: Path | None = None,
) -> dict[str, object]:
    """Validate a complete run and compute preregistered factorial coefficients."""

    runtime_identity = _runtime_session_identity(
        plan=plan,
        launch=launch,
        checkpoint_dir=checkpoint_dir,
        plan_path=plan_path,
        launch_path=launch_path,
        runtime_audit_path=runtime_audit_path,
    )
    corruption_identity = _corruption_audit_identity(
        plan=plan,
        launch=launch,
        checkpoint_dir=checkpoint_dir,
        corruption_audit_path=corruption_audit_path,
    )
    checkpoints = load_registered_checkpoints(
        plan=plan, launch=launch, checkpoint_dir=checkpoint_dir
    )
    for checkpoint in checkpoints:
        if checkpoint.status == "completed":
            _sidecar(checkpoint)
    gates = _mapping(plan.get("gates"), name="gates")
    counts = Counter(checkpoint.status for checkpoint in checkpoints)
    individual_pass = {
        checkpoint.case.case_key: (
            checkpoint.status == "completed"
            and _individual_numerical_pass(_result(checkpoint), gates)
        )
        for checkpoint in checkpoints
    }
    refinement, refinement_pass = _refinement_records(checkpoints, gates)
    parity = _parity_records(checkpoints, gates)
    contrasts = _contrast_records(
        plan=plan,
        checkpoints=checkpoints,
        individual_pass=individual_pass,
        refinement_pass=refinement_pass,
    )
    parity_complete = bool(parity) and all(bool(row["passes"]) for row in parity)
    all_numerical = all(individual_pass.values())
    all_refinement = bool(refinement) and all(bool(row["passes"]) for row in refinement)
    promotion = bool(
        runtime_identity is not None
        and corruption_identity is not None
        and counts.get("completed", 0) == len(checkpoints)
        and all_numerical
        and all_refinement
        and parity_complete
        and contrasts
    )
    aggregates = _contrast_aggregates(plan=plan, contrasts=contrasts)
    return {
        "schema_version": "articulated-structural-factorial-summary/1.4.0",
        "identity": {
            "plan_sha256": plan_sha256(plan),
            "execution_revision": launch["execution_revision"],
            "checkpoint_set_sha256": _checkpoint_set_sha256(checkpoints),
            "runtime_identity_sha256": runtime_identity,
            "corruption_audit_sha256": corruption_identity,
        },
        "inventory": {
            "registered_case_count": len(checkpoints),
            "status_counts": dict(sorted(counts.items())),
            "completed_sidecar_count": counts.get("completed", 0),
        },
        "gates": {
            "all_individual_numerical_pass": all_numerical,
            "all_refinement_groups_pass": all_refinement,
            "cross_engine_parity_complete_and_passed": parity_complete,
            "runtime_session_qualified": runtime_identity is not None,
            "corruption_sentinel_passed": corruption_identity is not None,
            "promotion_eligible": promotion,
        },
        "refinement": refinement,
        "cross_engine_parity": parity,
        "contrast_convention": {
            "walsh_coefficient": "mean(outcome * coded contrast sign)",
            "high_minus_low_effect": "two times the Walsh coefficient",
            "sign_counts": "exact algebraic sign with no tolerance",
        },
        "factorial_contrasts": contrasts,
        "contrast_aggregates": aggregates,
        "claim_boundary": {
            "causal_scope": "declared synthetic pathway interventions only",
            "human_or_coaching_inference": False,
            "equipment_optimization": False,
        },
    }


def main() -> None:
    """Validate a complete checkpoint set and atomically write its summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path)
    parser.add_argument("--corruption-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    launch = json.loads(args.launch.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(launch, dict):
        raise ValueError("plan and launch manifests must be mappings")
    summary = summarize_structural_factorial(
        plan=plan,
        launch=launch,
        checkpoint_dir=args.checkpoint_dir,
        plan_path=args.plan if args.runtime_audit is not None else None,
        launch_path=args.launch if args.runtime_audit is not None else None,
        runtime_audit_path=args.runtime_audit,
        corruption_audit_path=args.corruption_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()


__all__ = ["summarize_structural_factorial"]
