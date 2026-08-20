"""Assemble complete, deterministic articulated structural-propagation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_structural_cell_evidence import (
    load_structural_cell_evidence,
)

CORNER_IDS = (
    "nominal",
    "height_scale-low",
    "height_scale-high",
    "body_mass_scale-low",
    "body_mass_scale-high",
    "joint_limit_scale-low",
    "joint_limit_scale-high",
)
CORNER_PATHWAYS = tuple(
    (corner_id, pathway) for corner_id in CORNER_IDS for pathway in ("shaft", "ground")
)
AXIS_PATHWAYS = tuple(
    (axis, pathway)
    for axis in ("height_scale", "body_mass_scale", "joint_limit_scale")
    for pathway in ("shaft", "ground")
)
AXIS_SCALE_KEYS = {
    "height_scale": "height",
    "body_mass_scale": "body_mass",
    "joint_limit_scale": "joint_limit",
}


def _serialized(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, indent=2) + "\n").encode("utf-8")


def _digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_corner(record: dict[str, Any]) -> None:
    required = {
        "corner_id",
        "pathway",
        "cell_evidence_artifact",
        "cell_evidence_sha256",
        "requested_state_count",
        "feasible_state_count",
        "retained_failures",
        "planned_headline_cell_count",
        "feasible_headline_cell_count",
        "executed_headline_cell_count",
        "matched_cell_count",
        "matched_fraction_of_feasible",
        "all_registered_gates_passed",
        "authority",
    }
    if not required.issubset(record):
        raise ValueError("corner record is missing required fields")
    feasible = int(record["feasible_headline_cell_count"])
    executed = int(record["executed_headline_cell_count"])
    matched = int(record["matched_cell_count"])
    if executed != feasible or not bool(record["all_registered_gates_passed"]):
        raise ValueError("corner record must contain complete feasible execution")
    if not 0 <= matched <= executed:
        raise ValueError("matched cells must lie within executed support")
    expected_fraction = matched / executed if executed else 0.0
    if not np.isclose(record["matched_fraction_of_feasible"], expected_fraction):
        raise ValueError("matched fraction does not reproduce its denominator")
    requested_states = int(record["requested_state_count"])
    feasible_states = int(record["feasible_state_count"])
    if record["planned_headline_cell_count"] != requested_states * 32 or (
        feasible != feasible_states * 32
    ):
        raise ValueError("corner state and headline denominators do not agree")
    failure_states = {
        (int(value["case_index"]), int(value["phase_index"]))
        for value in record["retained_failures"]
    }
    if len(failure_states) != requested_states - feasible_states:
        raise ValueError("retained failures do not account for infeasible states")
    authority = record["authority"]
    if not {"authority_sha256", "scales", "model_sha256"}.issubset(authority):
        raise ValueError("corner authority record is incomplete")
    artifact_text = str(record["cell_evidence_artifact"])
    artifact = PurePosixPath(artifact_text)
    if (
        not artifact_text
        or artifact.is_absolute()
        or ".." in artifact.parts
        or artifact.suffix != ".npz"
        or artifact.as_posix() != artifact_text
    ):
        raise ValueError(
            "corner cell-evidence artifact must be a safe relative NPZ path"
        )
    evidence_sha256 = str(record["cell_evidence_sha256"])
    try:
        valid_digest = len(evidence_sha256) == 64 and int(evidence_sha256, 16) >= 0
    except ValueError:
        valid_digest = False
    if not valid_digest:
        raise ValueError("corner cell-evidence digest must be SHA-256")


def _validate_axis(record: dict[str, Any]) -> None:
    required = {
        "axis_name",
        "pathway",
        "low_scale",
        "nominal_scale",
        "high_scale",
        "shared_persistent_cell_count",
        "summary_statistic",
        "low_to_nominal_secant_m_s_per_unit_scale",
        "nominal_to_high_secant_m_s_per_unit_scale",
        "low_to_nominal_secant_range_m_s_per_unit_scale",
        "nominal_to_high_secant_range_m_s_per_unit_scale",
        "cell_classification_counts",
        "nonmonotonic_classification",
    }
    if not required.issubset(record):
        raise ValueError("axis record is missing required fields")
    scales = np.asarray(
        [record["low_scale"], record["nominal_scale"], record["high_scale"]],
        dtype=float,
    )
    if not np.all(np.isfinite(scales)) or not scales[0] < scales[1] < scales[2]:
        raise ValueError("axis scales must be finite and ordered")
    support = int(record["shared_persistent_cell_count"])
    values = (
        record["low_to_nominal_secant_m_s_per_unit_scale"],
        record["nominal_to_high_secant_m_s_per_unit_scale"],
        record["low_to_nominal_secant_range_m_s_per_unit_scale"],
        record["nominal_to_high_secant_range_m_s_per_unit_scale"],
    )
    if support == 0 and any(value is not None for value in values):
        raise ValueError("empty shared support must emit null secant summaries")
    if support > 0 and any(value is None for value in values):
        raise ValueError("nonempty shared support must retain secant summaries")


def assemble_structural_propagation_result(
    *,
    plan_contract_sha256: str,
    corner_records: tuple[dict[str, Any], ...],
    axis_records: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Require the complete registered design and bind it to the plan contract."""

    if len(plan_contract_sha256) != 64:
        raise ValueError("plan_contract_sha256 must be a SHA-256 digest")
    corners = {
        (str(value.get("corner_id")), str(value.get("pathway"))): value
        for value in corner_records
    }
    if len(corners) != len(corner_records) or set(corners) != set(CORNER_PATHWAYS):
        raise ValueError(
            "result must contain exactly the registered corner-pathway set"
        )
    axes = {
        (str(value.get("axis_name")), str(value.get("pathway"))): value
        for value in axis_records
    }
    if len(axes) != len(axis_records) or set(axes) != set(AXIS_PATHWAYS):
        raise ValueError("result must contain exactly the registered axis pathways")
    ordered_corners = [dict(corners[value]) for value in CORNER_PATHWAYS]
    artifacts = [value["cell_evidence_artifact"] for value in ordered_corners]
    if len(set(artifacts)) != len(artifacts):
        raise ValueError("corner cell-evidence artifact paths must be unique")
    ordered_axes = [dict(axes[value]) for value in AXIS_PATHWAYS]
    for record in ordered_corners:
        _validate_corner(record)
    for record in ordered_axes:
        _validate_axis(record)
    result = {
        "schema_version": "articulated-structural-propagation/v2",
        "status": "complete",
        "plan_contract_sha256": plan_contract_sha256,
        "corners": ordered_corners,
        "axes": ordered_axes,
        "limitations": {
            "scope": "registered synthetic engineering one-at-a-time corners",
            "causal_inference": "none",
            "population_inference": "none",
            "human_inference": "none",
            "coaching_inference": "none",
        },
    }
    return {**result, "result_sha256": _digest(result)}


def write_structural_propagation_result(record: dict[str, Any], output: Path) -> None:
    """Write one complete result atomically after exact reconstruction."""

    rebuilt = assemble_structural_propagation_result(
        plan_contract_sha256=record["plan_contract_sha256"],
        corner_records=tuple(record["corners"]),
        axis_records=tuple(record["axes"]),
    )
    if rebuilt != record:
        raise RuntimeError("structural propagation result does not reproduce")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(_serialized(record))
    temporary.replace(output)


def validate_structural_propagation_result(
    path: Path, plan_contract_sha256: str
) -> dict[str, Any]:
    """Validate design completeness, plan binding, digest, and exact bytes."""

    raw = path.read_bytes()
    record = json.loads(raw.decode("utf-8"))
    try:
        rebuilt = assemble_structural_propagation_result(
            plan_contract_sha256=plan_contract_sha256,
            corner_records=tuple(record["corners"]),
            axis_records=tuple(record["axes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "structural propagation result does not reproduce"
        ) from error
    if rebuilt != record or raw != _serialized(rebuilt):
        raise RuntimeError("structural propagation result does not reproduce")
    return rebuilt


def validate_structural_propagation_bundle(
    path: Path, plan_contract_sha256: str
) -> dict[str, Any]:
    """Reopen every referenced cell pack and reconcile it to the result."""

    record = validate_structural_propagation_result(path, plan_contract_sha256)
    for corner in record["corners"]:
        artifact = PurePosixPath(corner["cell_evidence_artifact"])
        evidence_path = path.parent.joinpath(*artifact.parts)
        try:
            evidence = load_structural_cell_evidence(evidence_path)
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                "structural cell-evidence artifact is invalid"
            ) from error
        if str(evidence["pathway"].item()) != corner["pathway"]:
            raise RuntimeError("structural cell-evidence pathway does not agree")
        if str(evidence["evidence_sha256"].item()) != corner["cell_evidence_sha256"]:
            raise RuntimeError("structural cell-evidence digest does not agree")
        if evidence["cell_identity"].size != corner["executed_headline_cell_count"]:
            raise RuntimeError(
                "structural cell-evidence execution count does not agree"
            )
        if (
            int(np.count_nonzero(evidence["matched_load_work"]))
            != corner["matched_cell_count"]
        ):
            raise RuntimeError("structural cell-evidence matched count does not agree")
    return record


def reconcile_structural_result_to_plan(
    record: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Require result denominators, authorities, and scales to match the plan."""

    if plan.get("status") != "ready" or plan.get("contract_sha256") != record.get(
        "plan_contract_sha256"
    ):
        raise RuntimeError("structural result does not bind the registered plan")
    planned = {str(value.get("corner_id")): value for value in plan.get("corners", [])}
    if set(planned) != set(CORNER_IDS) or len(planned) != len(plan.get("corners", [])):
        raise RuntimeError("structural plan corner set is incomplete or duplicated")
    for corner in record["corners"]:
        expected = planned[corner["corner_id"]]
        pathway = corner["pathway"]
        comparisons = {
            "requested_state_count": expected["requested_state_count"],
            "feasible_state_count": expected["feasible_state_count"],
            "retained_failures": expected["retained_failures"],
            "planned_headline_cell_count": expected["requested_state_count"] * 32,
            "feasible_headline_cell_count": expected[
                f"expected_{pathway}_headline_cell_count"
            ],
            "executed_headline_cell_count": expected[
                f"expected_{pathway}_headline_cell_count"
            ],
            "authority": expected["authority"],
        }
        if any(corner[name] != value for name, value in comparisons.items()):
            raise RuntimeError("structural corner evidence does not match the plan")
    for axis in record["axes"]:
        scale_key = AXIS_SCALE_KEYS[axis["axis_name"]]
        expected_scales = (
            planned[f"{axis['axis_name']}-low"]["authority"]["scales"][scale_key],
            planned["nominal"]["authority"]["scales"][scale_key],
            planned[f"{axis['axis_name']}-high"]["authority"]["scales"][scale_key],
        )
        observed_scales = (
            axis["low_scale"],
            axis["nominal_scale"],
            axis["high_scale"],
        )
        if observed_scales != expected_scales:
            raise RuntimeError("structural axis scales do not match the plan")
    return record


def validate_structural_propagation_bundle_against_plan(
    result_path: Path, plan_path: Path
) -> dict[str, Any]:
    """Validate the exact governed plan, result JSON, and all cell packs."""

    from scripts.research.proximal_distal_energy.articulated_structural_propagation_plan import (
        validate_structural_propagation_plan,
    )

    plan = validate_structural_propagation_plan(plan_path)
    record = validate_structural_propagation_bundle(
        result_path, plan["contract_sha256"]
    )
    return reconcile_structural_result_to_plan(record, plan)


__all__ = [
    "AXIS_PATHWAYS",
    "AXIS_SCALE_KEYS",
    "CORNER_IDS",
    "CORNER_PATHWAYS",
    "assemble_structural_propagation_result",
    "reconcile_structural_result_to_plan",
    "validate_structural_propagation_bundle",
    "validate_structural_propagation_bundle_against_plan",
    "validate_structural_propagation_result",
    "write_structural_propagation_result",
]
