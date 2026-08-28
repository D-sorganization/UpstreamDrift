"""Fail-closed aggregation and publication for the stateful #9153 smoke."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    CaseCheckpoint,
    load_registered_checkpoints,
    manifest_sha256,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_summary import (
    _checkpoint_inventory,
    _finite_number,
    _mapping,
    _payload,
    _unavailable_group,
    _write_atomic,
)


def _refinement_ratios(values: list[float]) -> tuple[list[float | None], bool]:
    ratios: list[float | None] = []
    indeterminate = False
    for coarse, fine in zip(values, values[1:], strict=False):
        if coarse == 0.0:
            ratios.append(0.0 if fine == 0.0 else None)
            indeterminate = indeterminate or fine != 0.0
        else:
            ratios.append(fine / coarse)
    return ratios, indeterminate


def _result_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    outcome = _mapping(payload.get("outcome"), name="outcome")
    return _mapping(outcome.get("result"), name="outcome.result")


def _completed_group(
    checkpoints: list[CaseCheckpoint],
    payloads: list[Mapping[str, Any]],
    tolerances: Mapping[str, Any],
) -> dict[str, object]:
    ordered = sorted(
        zip(checkpoints, payloads, strict=True),
        key=lambda pair: pair[0].case.time_step_s,
        reverse=True,
    )
    energy: list[float] = []
    coupling: list[float] = []
    failures: list[str] = []
    results: list[Mapping[str, Any]] = []
    for _, payload in ordered:
        result = _result_payload(payload)
        results.append(result)
        closure = _mapping(result.get("closure"), name="result.closure")
        energy.append(
            _finite_number(
                closure.get("trajectory_energy_relative_residual"),
                name="trajectory_energy_relative_residual",
            )
        )
        coupling.append(
            _finite_number(
                closure.get("tangential_coupling_work_relative_residual"),
                name="tangential_coupling_work_relative_residual",
            )
        )
        codes = closure.get("failure_codes")
        if not isinstance(codes, list) or any(
            not isinstance(code, str) for code in codes
        ):
            raise ValueError("closure.failure_codes must be a string list")
        if closure.get("passes_registered_tolerances") is not (not codes):
            raise ValueError("closure pass flag disagrees with failure codes")
        failures.extend(codes)
    energy_ratios, energy_indeterminate = _refinement_ratios(energy)
    coupling_ratios, coupling_indeterminate = _refinement_ratios(coupling)
    limit = _finite_number(
        tolerances.get("refinement_ratio_limit"), name="refinement_ratio_limit"
    )
    if energy_indeterminate:
        failures.append("passive_energy_refinement_indeterminate")
    elif any(ratio is not None and ratio > limit for ratio in energy_ratios):
        failures.append("passive_energy_refinement")
    if coupling_indeterminate:
        failures.append("coupling_work_refinement_indeterminate")
    elif any(ratio is not None and ratio > limit for ratio in coupling_ratios):
        failures.append("coupling_work_refinement")
    fine = results[-1]
    regimes = _mapping(fine.get("regimes"), name="result.regimes")
    variant = ordered[0][0].case.variant
    if variant == "low_friction_slip_probe" and "coulomb_slip" not in regimes:
        failures.append("expected_slip_not_observed")
    if variant == "opening_probe" and "open" not in regimes:
        failures.append("expected_opening_not_observed")
    return {
        "status": "completed",
        "time_steps_s": [pair[0].case.time_step_s for pair in ordered],
        "trajectory_energy_relative_residuals": energy,
        "coupling_work_relative_residuals": coupling,
        "passive_energy_refinement_ratios": energy_ratios,
        "coupling_work_refinement_ratios": coupling_ratios,
        "fine_step_regime_counts": dict(regimes),
        "fine_step_outcomes": dict(
            _mapping(fine.get("outcomes"), name="result.outcomes")
        ),
        "failure_codes": list(dict.fromkeys(failures)),
        "passes": not failures,
    }


def _grouped_checkpoints(
    checkpoints: tuple[CaseCheckpoint, ...],
) -> dict[tuple[int, int, float, str, str], list[CaseCheckpoint]]:
    grouped: dict[tuple[int, int, float, str, str], list[CaseCheckpoint]] = {}
    for checkpoint in checkpoints:
        case = checkpoint.case
        key = (
            case.source_case_index,
            case.source_sample_index,
            case.source_time_s,
            case.engine,
            case.variant,
        )
        grouped.setdefault(key, []).append(checkpoint)
    return grouped


def _counterfactuals(groups: list[dict[str, object]]) -> list[dict[str, object]]:
    completed = [group for group in groups if group.get("status") == "completed"]
    baselines = {
        (
            group["source_case_index"],
            group["source_sample_index"],
            group["engine"],
        ): group
        for group in completed
        if group["variant"] == "nominal"
    }
    rows: list[dict[str, object]] = []
    for group in completed:
        key = (
            group["source_case_index"],
            group["source_sample_index"],
            group["engine"],
        )
        baseline = baselines.get(key)
        if baseline is None or group["variant"] == "nominal":
            continue
        outcome = _mapping(group.get("fine_step_outcomes"), name="fine outcomes")
        base = _mapping(baseline.get("fine_step_outcomes"), name="baseline outcomes")
        rows.append(
            {
                "source_case_index": group["source_case_index"],
                "source_sample_index": group["source_sample_index"],
                "engine": group["engine"],
                "variant": group["variant"],
                "reference_variant": "nominal",
                "clubhead_speed_difference_m_s": _finite_number(
                    outcome.get("clubhead_speed_m_s"), name="clubhead speed"
                )
                - _finite_number(
                    base.get("clubhead_speed_m_s"), name="baseline clubhead speed"
                ),
                "estimand": "matched_separately_integrated_counterfactual",
            }
        )
    return rows


def aggregate_stateful_smoke(
    *,
    manifest: Mapping[str, object],
    execution_revision: str,
    checkpoint_dir: Path,
) -> dict[str, object]:
    """Aggregate a complete stateful run while retaining every failed gate."""

    checkpoints = load_registered_checkpoints(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
    )
    payloads = {checkpoint.path: _payload(checkpoint) for checkpoint in checkpoints}
    tolerances = _mapping(manifest.get("tolerances"), name="tolerances")
    groups: list[dict[str, object]] = []
    for key, items in _grouped_checkpoints(checkpoints).items():
        item_payloads = [payloads[item.path] for item in items]
        statuses = {item.status for item in items}
        if statuses == {"completed"}:
            result = _completed_group(items, item_payloads, tolerances)
        elif statuses == {"unavailable"}:
            result = _unavailable_group(item_payloads)
        else:
            result = {
                "status": "failed",
                "failure_codes": ["mixed_or_failed_case_status"],
                "passes": False,
            }
        groups.append(
            {
                "source_case_index": key[0],
                "source_sample_index": key[1],
                "source_time_s": key[2],
                "engine": key[3],
                "variant": key[4],
                **result,
            }
        )
    counts = Counter(checkpoint.status for checkpoint in checkpoints)
    codes = [code for group in groups for code in group.get("failure_codes", [])]
    promotion_failures: list[str] = []
    if counts["unavailable"]:
        promotion_failures.extend(
            ("native_engine_unavailable", "cross_engine_parity_unavailable")
        )
    if counts["failed"]:
        promotion_failures.append("failed_case")
    if any("refinement" in code for code in codes):
        promotion_failures.append("refinement_failure")
    if any("expected_" in code for code in codes):
        promotion_failures.append("adverse_regime_not_observed")
    return {
        "schema_version": "stateful-distributed-forward-summary/v1",
        "identity": {
            "plan_sha256": manifest_sha256(manifest),
            "execution_revision": execution_revision,
        },
        "counts": {
            "registered": len(checkpoints),
            "completed": counts["completed"],
            "unavailable": counts["unavailable"],
            "failed": counts["failed"],
        },
        "checkpoint_inventory": _checkpoint_inventory(checkpoints),
        "groups": groups,
        "counterfactuals": _counterfactuals(groups),
        "promotion": {
            "eligible": not promotion_failures,
            "failure_codes": list(dict.fromkeys(promotion_failures)),
        },
        "claim_boundary": {
            "human_or_anatomical_inference": False,
            "human_or_coaching_inference": False,
            "cross_engine_parity_established": False,
            "screening_population_evaluated": False,
        },
    }


def publish_stateful_smoke_evidence(
    *,
    manifest: Mapping[str, object],
    execution_revision: str,
    checkpoint_dir: Path,
    output_dir: Path,
) -> Path:
    """Publish all identity-valid checkpoints and a deterministic summary."""

    checkpoints = load_registered_checkpoints(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
    )
    summary = aggregate_stateful_smoke(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
    )
    destination = output_dir / "checkpoints"
    expected = {checkpoint.path.name for checkpoint in checkpoints}
    if destination.exists():
        extras = {path.name for path in destination.glob("case-*.json")} - expected
        if extras:
            raise ValueError("publication contains unregistered checkpoints")
    for checkpoint in checkpoints:
        _write_atomic(destination / checkpoint.path.name, checkpoint.path.read_bytes())
    summary_path = output_dir / "summary.json"
    content = (
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    _write_atomic(summary_path, content)
    return summary_path


__all__ = ["aggregate_stateful_smoke", "publish_stateful_smoke_evidence"]
