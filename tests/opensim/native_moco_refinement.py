"""Pure integrity and grid planning for one bounded synthetic Moco refinement."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

DEFAULT_ACTIVATION_INDICATOR = 1e-8
MAX_REFINED_CELLS = 4
MAX_MESH_INTERVALS = 24


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_normalized_mesh(mesh: Sequence[float]) -> tuple[float, ...]:
    values = tuple(mesh)
    if len(values) < 2 or len(values) - 1 > MAX_MESH_INTERVALS:
        raise ValueError("normalized mesh interval count is outside the supported cap")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("normalized mesh points must be finite numbers")
    points = tuple(float(value) for value in values)
    if points[0] != 0.0 or points[-1] != 1.0:
        raise ValueError("normalized mesh endpoints must be exactly zero and one")
    if any(right <= left for left, right in zip(points, points[1:], strict=False)):
        raise ValueError("normalized mesh points must be strictly increasing")
    return points


def build_refinement_plan(
    parent: Path,
    analysis_path: Path,
    *,
    indicator_tolerance: float = DEFAULT_ACTIVATION_INDICATOR,
    maximum_cells: int = MAX_REFINED_CELLS,
) -> dict[str, Any]:
    """Validate retained evidence and split flagged HS cells under fixed caps."""
    _validate_planner_limits(indicator_tolerance, maximum_cells)
    request, analysis, report_inputs, digests = _load_bound_parent(
        parent, analysis_path
    )
    intervals = _parent_interval_count(request, analysis)
    flagged = _flagged_cells(analysis, float(indicator_tolerance))
    if len(flagged) > maximum_cells:
        raise ValueError("refinement cell cap exceeded; no child attempt was created")
    parent_mesh, normalized_mesh = make_refined_mesh(intervals, flagged)
    return _refinement_plan_document(
        parent,
        analysis_path,
        request,
        report_inputs,
        digests,
        flagged,
        parent_mesh,
        normalized_mesh,
        float(indicator_tolerance),
        maximum_cells,
    )


def _validate_planner_limits(indicator_tolerance: float, maximum_cells: int) -> None:
    if (
        isinstance(indicator_tolerance, bool)
        or not isinstance(indicator_tolerance, (int, float))
        or not math.isfinite(indicator_tolerance)
        or indicator_tolerance <= 0
    ):
        raise ValueError("activation indicator threshold must be finite and positive")
    if (
        isinstance(maximum_cells, bool)
        or not isinstance(maximum_cells, int)
        or not 1 <= maximum_cells <= MAX_REFINED_CELLS
    ):
        raise ValueError("refinement cell cap must be between one and four")


def _load_bound_parent(
    parent: Path, analysis_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    paths = {
        "source_osim": parent / "source.osim",
        "request": parent / "request.json",
        "solution_sto": parent / "solution.sto",
        "time_only_controls_sto": parent / "time_only_controls.sto",
    }
    digests = {name: sha256(path) for name, path in paths.items()}
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    report_inputs = analysis.get("input_sha256")
    expected_names = {
        "source_osim": "source_osim",
        "request": "request",
        "solution_sto": "solution_sto",
        "time_only_controls_sto": "time_only_controls_sto",
    }
    if not isinstance(report_inputs, dict) or any(
        report_inputs.get(report_name) != digests[name]
        for name, report_name in expected_names.items()
    ):
        raise ValueError("activation analysis does not bind the retained parent inputs")
    audit_path = (
        analysis_path.parent
        / "diagnostic_retained_replay_fauldump_3e195b0e"
        / "native_audit"
        / "fresh_replay_diagnostic.json"
    )
    if not audit_path.is_file() or sha256(audit_path) != report_inputs.get(
        "replay_audit"
    ):
        raise ValueError(
            "activation analysis replay audit identity is missing or stale"
        )
    if request.get("source_sha256") != digests["source_osim"]:
        raise ValueError("parent request source does not match retained source bytes")
    if request.get("transcription_scheme") != "hermite-simpson":
        raise ValueError("refinement supports only the retained Hermite-Simpson parent")
    if request.get("replay_control_interpolation") != "piecewise_linear":
        raise ValueError("refinement requires the unchanged piecewise-linear controls")
    if request.get("physical_qualification") != "unqualified":
        raise ValueError("refinement cannot change physical qualification")
    return request, analysis, report_inputs, digests


def _parent_interval_count(request: dict[str, Any], analysis: dict[str, Any]) -> int:
    intervals = request.get("mesh_intervals")
    horizon = request.get("horizon_seconds")
    if (
        isinstance(intervals, bool)
        or not isinstance(intervals, int)
        or intervals < 1
        or isinstance(horizon, bool)
        or not isinstance(horizon, (int, float))
        or not math.isfinite(horizon)
        or horizon <= 0
    ):
        raise ValueError("parent mesh or horizon is invalid")
    channels = analysis.get("channels")
    if not isinstance(channels, dict) or not channels:
        raise ValueError("activation analysis has no channels")
    first_channel = next(iter(channels.values()))
    allocation = first_channel.get("full_cell_local_vs_accumulated_allocation", {})
    if allocation.get("cell_count") != intervals:
        raise ValueError("full-cell activation allocations differ from parent HS mesh")
    if analysis.get("sample_count") != 2 * intervals + 1:
        raise ValueError("analysis trajectory sample count differs from parent HS grid")
    return intervals


def _refinement_plan_document(
    parent: Path,
    analysis_path: Path,
    request: dict[str, Any],
    report_inputs: dict[str, Any],
    digests: dict[str, str],
    flagged: list[int],
    parent_mesh: tuple[float, ...],
    normalized_mesh: tuple[float, ...],
    indicator_tolerance: float,
    maximum_cells: int,
) -> dict[str, Any]:
    identity = {
        "parent_hashes": digests | {"replay_audit": report_inputs["replay_audit"]},
        "analysis_sha256": sha256(analysis_path),
        "flagged_parent_cells": flagged,
        "status": "planned" if flagged else "no_refinement_indicated",
        "normalized_mesh_points": normalized_mesh,
    }
    attempt_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "native-moco-bounded-nonuniform-refinement-plan-v1",
        "attempt_id": attempt_id,
        "parent_directory": str(parent.resolve()),
        "analysis_path": str(analysis_path.resolve()),
        "parent_hashes": identity["parent_hashes"],
        "parent_request": request,
        "analysis_sha256": identity["analysis_sha256"],
        "indicator": {
            "metric": "local_scalar_activation_minus_optimized_at_hs_midpoint_and_endpoint",
            "threshold": indicator_tolerance,
            "purpose": "refinement indication only; not a solve or qualification criterion",
        },
        "flagged_parent_cells": flagged,
        "parent_normalized_mesh_points": parent_mesh,
        "normalized_mesh_points": normalized_mesh,
        "maximum_refined_parent_cells": maximum_cells,
        "maximum_child_intervals": MAX_MESH_INTERVALS,
        "preserved_contract": _preserved_contract(request),
        "control_seed": {
            "source": "retained_time_only_controls.sto",
            "interpolation": "piecewise_linear",
            "sample_at": "every child Hermite-Simpson endpoint and midpoint",
            "forbidden_operations": ["resampleControlsTrajectory", "resampleWith"],
        },
        "interpretation": (
            "A lower local activation indicator does not guarantee acceptance and "
            "cannot qualify fiber, kinematic, or physical behavior."
        ),
    }


def make_refined_mesh(
    parent_intervals: int,
    flagged_cells: Sequence[int],
    *,
    maximum_intervals: int = MAX_MESH_INTERVALS,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if (
        isinstance(parent_intervals, bool)
        or not isinstance(parent_intervals, int)
        or not 1 <= parent_intervals <= maximum_intervals
    ):
        raise ValueError("parent mesh interval count exceeds the refinement cap")
    cells = tuple(flagged_cells)
    if len(cells) > MAX_REFINED_CELLS:
        raise ValueError("refinement cell cap exceeded")
    if len(set(cells)) != len(cells):
        raise ValueError("flagged parent cell indices must be unique")
    if any(
        isinstance(cell, bool)
        or not isinstance(cell, int)
        or not 0 <= cell < parent_intervals
        for cell in cells
    ):
        raise ValueError("flagged parent cell index is outside the mesh")
    parent = tuple(index / parent_intervals for index in range(parent_intervals + 1))
    child = list(parent)
    child.extend((cell + 0.5) / parent_intervals for cell in cells)
    normalized = validate_normalized_mesh(sorted(child))
    if len(normalized) - 1 > maximum_intervals:
        raise ValueError("refined mesh exceeds the fixed interval cap")
    return parent, normalized


def _flagged_cells(analysis: dict[str, Any], threshold: float) -> list[int]:
    channels = analysis.get("channels")
    if not isinstance(channels, dict) or not channels:
        raise ValueError("activation analysis has no channels")
    flags: set[int] = set()
    cell_counts: set[int] = set()
    for channel in channels.values():
        allocation = channel.get("full_cell_local_vs_accumulated_allocation")
        if not isinstance(allocation, dict):
            raise ValueError("activation analysis is missing local cell allocations")
        errors = allocation.get("local_scalar_minus_optimized_midpoint_endpoint")
        if not isinstance(errors, list):
            raise ValueError("activation analysis has no per-cell error pairs")
        cell_counts.add(len(errors))
        if any(not isinstance(pair, list) or len(pair) != 2 for pair in errors):
            raise ValueError(
                "activation cell errors must contain midpoint and endpoint"
            )
        for index, pair in enumerate(errors):
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in pair
            ):
                raise ValueError("activation cell error is non-finite")
            if max(abs(float(value)) for value in pair) > threshold:
                # Each row is one full HS cell: its pair is midpoint/end.
                flags.add(index)
    declared_cells = next(iter(cell_counts)) if len(cell_counts) == 1 else None
    if declared_cells is None or any(
        channel["full_cell_local_vs_accumulated_allocation"].get("cell_count")
        != declared_cells
        for channel in channels.values()
    ):
        raise ValueError("channel activation allocation lengths do not match analysis")
    return sorted(flags)


def _preserved_contract(request: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "horizon_seconds",
        "target_coordinate",
        "target_q",
        "initial_q",
        "transcription_scheme",
        "interpolate_control_mesh_interior_points",
        "replay_control_interpolation",
        "parallel",
        "maximum_iterations",
        "convergence_tolerance",
        "constraint_tolerance",
        "native_control_bounds",
        "native_control_bounds_source",
        "cold_start_named_state",
        "cold_start_state_units",
        "cold_start_chart_bounds",
        "cold_start_constraint_enforcement",
        "cold_start_lock_targets",
        "objective",
        "physical_qualification",
        "solution_serialization_max_ulps",
    )
    contract = {key: request[key] for key in keys if key in request}
    if "objective" not in contract:
        contract["objective"] = "MocoControlGoal(effort)"
    if request.get("interpolate_control_mesh_interior_points") is not True:
        raise ValueError(
            "refinement requires retained Hermite-Simpson control interpolation"
        )
    return contract


def piecewise_linear_samples(
    source_times: Sequence[float],
    source_values: Sequence[float],
    target_times: Sequence[float],
) -> tuple[float, ...]:
    """Evaluate a finite persisted control channel by exact PWL interpolation."""
    times = validate_times(source_times, "source")
    values = tuple(source_values)
    targets = validate_times(target_times, "target")
    if len(values) != len(times) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("control samples must be finite and aligned with source times")
    if targets[0] < times[0] or targets[-1] > times[-1]:
        raise ValueError("child control grid falls outside the saved control clock")
    result = []
    for target in targets:
        right = next(
            (i for i, time in enumerate(times) if time >= target), len(times) - 1
        )
        if times[right] == target or right == 0:
            result.append(float(values[right]))
            continue
        left = right - 1
        fraction = (target - times[left]) / (times[right] - times[left])
        result.append(float(values[left] + fraction * (values[right] - values[left])))
    return tuple(result)


def validate_times(values: Sequence[float], label: str) -> tuple[float, ...]:
    times = tuple(values)
    if len(times) < 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in times
    ):
        raise ValueError(f"{label} times must contain at least two finite values")
    result = tuple(float(value) for value in times)
    if any(right <= left for left, right in zip(result, result[1:], strict=False)):
        raise ValueError(f"{label} times must be strictly increasing")
    return result


def child_hs_grid(
    normalized_mesh: Sequence[float], horizon: float
) -> tuple[float, ...]:
    mesh = validate_normalized_mesh(normalized_mesh)
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, (int, float))
        or not math.isfinite(horizon)
        or horizon <= 0
    ):
        raise ValueError("horizon must be finite and positive")
    points: list[float] = []
    for left, right in zip(mesh, mesh[1:], strict=False):
        points.extend((left * horizon, ((left + right) / 2) * horizon))
    points.append(horizon)
    return tuple(points)
