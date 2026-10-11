"""One-rung native Moco solve/export/fresh-replay worker.

Launch only through ``native_moco_ladder_harness.run_ladder`` so the worker
has a separate process and a hard parent-enforced 30-second deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tests.opensim.native_moco_ladder_harness import DEFAULT_RUNGS, build_synthetic_rung


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_start() -> tuple[float, float]:
    return time.perf_counter(), time.process_time()


def _stage_finish(start: tuple[float, float]) -> dict[str, float]:
    return {
        "wall_seconds": time.perf_counter() - start[0],
        "cpu_seconds": time.process_time() - start[1],
    }


def _write_progress(
    output: Path, rung: str, stages: dict[str, dict[str, float]]
) -> None:
    path = output / "progress.json"
    _write_json(path, {"rung": rung, "stage_metrics": stages})


def run_rung(rung_name: str, output: Path) -> None:
    import opensim as osim

    rung = next(item for item in DEFAULT_RUNGS if item.name == rung_name)
    # The supervisor creates the per-rung directory before launching us.
    output.mkdir(parents=True, exist_ok=True)
    stages: dict[str, dict[str, float]] = {}
    prepared = _prepare_rung(rung, output, stages, osim)
    solved = _solve_and_export(prepared, stages, osim)
    replay_path = _fresh_native_replay(prepared, solved, stages)
    _write_success_receipt(prepared, solved, replay_path, stages, osim)


def _prepare_rung(rung: Any, output: Path, stages: dict, osim: Any) -> dict[str, Any]:
    from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
        declare_native_mechanics_execution_profile,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_reference_conventions import (
        ReferenceStateDeclaration,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_wrapped_path import (
        audit_wrapped_path_source,
        declare_wrapped_path_source,
    )
    from tests.opensim.test_native_constrained_muscle import _moco_problem

    started = _stage_start()
    built_path, declaration = build_synthetic_rung(rung, output / "build")
    source = output / "source.osim"
    shutil.copyfile(built_path, source)
    _flush_file(source)
    declaration = replace(declaration, model_path=source)
    source_hash = _sha256(source)
    if source_hash != declaration.source_sha256:
        raise ValueError("staged source differs from the fresh declaration")
    stages["source_build_and_state_equilibration"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)

    started = _stage_start()
    profile = declare_native_mechanics_execution_profile(source, source_hash)
    wrapped = (
        declare_wrapped_path_source(source)
        if {"intersecting_wrap", "conditional_path_point"} & set(rung.features)
        else None
    )
    if "intersecting_wrap" in rung.features:
        _verify_native_wrap(
            source,
            wrapped,
            source_hash,
            rung.name,
            ReferenceStateDeclaration,
            audit_wrapped_path_source,
        )
    target_q = declaration.named_state["/jointset/base/q/value"] + 5e-6
    initial_q = declaration.named_state["/jointset/base/q/value"]
    config = _rung_config(rung, target_q, source_hash, initial_q)
    config_path = output / "request.json"
    _write_json(config_path, config)
    study = _moco_problem(source, declaration, target_q=target_q)
    solver = osim.MocoCasADiSolver.safeDownCast(study.initCasADiSolver())
    _configure_solver(solver)
    stages["preflight_and_solver_setup"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)
    return {
        "rung": rung,
        "source": source,
        "source_hash": source_hash,
        "declaration": declaration,
        "profile": profile,
        "wrapped": wrapped,
        "target_q": target_q,
        "config": config,
        "config_path": config_path,
        "study": study,
    }


def _verify_native_wrap(
    source: Path,
    wrapped: Any,
    source_hash: str,
    rung: str,
    reference_type: Any,
    audit_fn: Any,
) -> None:
    reference = reference_type(
        reference_id=f"synthetic-{rung}-post-init",
        provenance="fresh synthetic worker initialization",
        mode="post-init-system",
        expected_source_sha256=source_hash,
    )
    audit = audit_fn(source, wrapped, reference)
    flexor = next(
        item
        for item in audit.reference_observation.muscles
        if item.path == "/forceset/flexor"
    )
    if not any(
        point.concrete_class == "PathWrapPoint" for point in flexor.current_route
    ):
        raise ValueError("synthetic wrap did not produce a native wrapped route")


def _rung_config(
    rung: Any, target_q: float, source_sha256: str, initial_q: float
) -> dict[str, Any]:
    features = rung.features
    return {
        "rung": rung.name,
        "features": features,
        "source_sha256": source_sha256,
        "target_coordinate": "/jointset/base/q/value",
        "target_q": target_q,
        "initial_q": initial_q,
        "horizon_seconds": 0.01,
        "mesh_intervals": 10,
        "parallel": 0,
        "maximum_iterations": 300,
        "convergence_tolerance": 1e-6,
        "constraint_tolerance": 1e-7,
        "physical_qualification": "unqualified",
        "custom_joint_simm_spline_knots": [-1.0, 0.0, 1.0]
        if "nonlinear_custom_joint" in features
        else [],
        "custom_joint_simm_spline_values": [-0.2, 0.0, 0.3]
        if "nonlinear_custom_joint" in features
        else [],
        "moving_path_simm_spline_knots": [-1.0, 0.0, 1.0]
        if "nonlinear_mpp" in features
        else [],
        "moving_path_simm_spline_values": [-0.008, 0.0, 0.012]
        if "nonlinear_mpp" in features
        else [],
        "conditional_point_active_range": [0.2, 0.4]
        if "conditional_path_point" in features
        else [],
        "coupler_function": "dependent = 2 * q"
        if "coordinate_coupler" in features
        else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _flush_file(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _configure_solver(solver: Any) -> None:
    solver.set_num_mesh_intervals(10)
    solver.set_parallel(0)
    solver.set_optim_max_iterations(300)
    solver.set_optim_convergence_tolerance(1e-6)
    solver.set_optim_constraint_tolerance(1e-7)


def _solve_and_export(
    prepared: dict[str, Any], stages: dict, osim: Any
) -> dict[str, Any]:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
    )

    rung = prepared["rung"]
    output = prepared["config_path"].parent
    started = _stage_start()
    solution = prepared["study"].solve()
    stages["solve"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)
    if not solution.success():
        raise RuntimeError(f"Moco solve failed: {solution.getStatus()}")
    solution_path = output / "solution.sto"
    solution.write(str(solution_path))

    started = _stage_start()
    grid = np.asarray(solution.getTimeMat(), dtype=np.float64).reshape(-1)
    controls = {
        name: np.asarray(
            solution.getControlMat(f"/forceset/{name}"), dtype=np.float64
        ).reshape(-1)
        for name in ("flexor", "extensor")
    }
    controls_path = output / "time_only_controls.sto"
    _write_time_only_controls(osim, grid, controls, controls_path)
    stages["solution_export_and_replay_input_staging"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)

    started = _stage_start()
    saved_grid, saved_controls = _load_time_only_controls(
        osim, controls_path, grid, tuple(controls)
    )
    stages["saved_replay_input_reload"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)

    started = _stage_start()
    bundle = build_constrained_muscle_bundle(
        prepared["declaration"],
        saved_grid,
        saved_controls,
        wrapped_path_declaration=prepared["wrapped"],
        native_mechanics_profile=prepared["profile"],
    )
    stages["replay_input_preflight"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)
    return {
        "solution": solution,
        "grid": saved_grid,
        "controls": saved_controls,
        "solution_path": solution_path,
        "controls_path": controls_path,
        "bundle": bundle,
    }


def _write_time_only_controls(
    osim: Any, grid: np.ndarray, controls: dict, path: Path
) -> None:
    table = osim.TimeSeriesTable()
    labels = osim.StdVectorString()
    for name in controls:
        labels.append(f"/forceset/{name}")
    table.setColumnLabels(labels)
    for row_index, t in enumerate(grid):
        row = osim.RowVector(len(controls))
        for column_index, values in enumerate(controls.values()):
            row[column_index] = float(values[row_index])
        table.appendRow(float(t), row)
    osim.STOFileAdapter.write(table, str(path))


def _load_time_only_controls(
    osim: Any,
    path: Path,
    expected_grid: np.ndarray,
    muscle_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    table = osim.TimeSeriesTable(str(path))
    independent = table.getIndependentColumn()
    grid = np.asarray(
        [independent[index] for index in range(table.getNumRows())],
        dtype=np.float64,
    )
    labels = tuple(str(label) for label in table.getColumnLabels())
    controls = {
        label.removeprefix("/forceset/"): np.asarray(
            [
                table.getDependentColumn(label)[index]
                for index in range(table.getNumRows())
            ],
            dtype=np.float64,
        )
        for label in labels
        if label.startswith("/forceset/")
    }
    expected_labels = tuple(f"/forceset/{name}" for name in muscle_names)
    _validate_saved_control_data(grid, labels, controls, expected_grid, muscle_names)
    if labels != expected_labels:
        raise ValueError("saved time-only controls changed ordered muscle channels")
    return grid, controls


def _validate_saved_control_data(
    grid: np.ndarray,
    labels: tuple[str, ...],
    controls: dict[str, np.ndarray],
    expected_grid: np.ndarray,
    muscle_names: tuple[str, ...],
) -> None:
    expected_labels = tuple(f"/forceset/{name}" for name in muscle_names)
    if labels != expected_labels:
        raise ValueError("saved time-only controls changed ordered muscle channels")
    if (
        grid.ndim != 1
        or len(grid) == 0
        or len(grid) != len(expected_grid)
        or not np.isfinite(grid).all()
        or not np.array_equal(grid, expected_grid)
        or grid[0] != 0.0
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("saved time-only control knots differ from the solved grid")
    if set(controls) != set(muscle_names):
        raise ValueError("saved controls do not exactly cover the declared muscles")
    for name in muscle_names:
        values = controls[name]
        if (
            values.ndim != 1
            or len(values) != len(grid)
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(f"saved controls for {name} are invalid or out of bounds")


def _fresh_native_replay(prepared: dict[str, Any], solved: dict, stages: dict) -> Path:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        replay_constrained_muscle_bundle,
    )

    output = prepared["config_path"].parent
    started = _stage_start()
    replay = replay_constrained_muscle_bundle(
        solved["bundle"],
        prepared["declaration"],
        wrapped_path_declaration=prepared["wrapped"],
        native_mechanics_profile=prepared["profile"],
    )
    grid = solved["grid"]
    if not np.isfinite(replay.states).all() or not np.array_equal(replay.times, grid):
        raise ValueError("fresh replay differs from saved time-only controls")
    state_audit = _compare_replayed_states(
        solved["solution"],
        replay.states,
        replay.state_names,
        grid,
        prepared["target_q"],
        prepared["declaration"].named_state[prepared["config"]["target_coordinate"]],
    )
    replay_path = output / "fresh_replay.json"
    _write_json(
        replay_path,
        {
            "fresh_model": True,
            "input_kind": "time_only_controls",
            "input_sha256": replay.input_sha256,
            "time_count": len(grid),
            "state_names": replay.state_names,
            **state_audit,
            "state_matrix_sha256": hashlib.sha256(
                np.asarray(replay.states, dtype=np.float64).tobytes()
            ).hexdigest(),
            "native_constraint_audit_count": len(replay.constraint_audits),
            "physical_qualification": "unqualified",
        },
    )
    stages["fresh_native_replay"] = _stage_finish(started)
    _write_progress(output, prepared["rung"].name, stages)
    return replay_path


def _compare_replayed_states(
    solution: Any,
    replay_states: np.ndarray,
    state_names: tuple[str, ...],
    saved_grid: np.ndarray,
    target_q: float,
    initial_q: float,
) -> dict[str, Any]:
    solution_grid = np.asarray(solution.getTimeMat(), dtype=np.float64).reshape(-1)
    optimized = np.column_stack(
        [
            np.asarray(solution.getStateMat(name), dtype=np.float64).reshape(-1)
            for name in state_names
        ]
    )
    if (
        not np.array_equal(solution_grid, saved_grid)
        or optimized.shape != replay_states.shape
        or optimized.shape[0] != len(saved_grid)
        or not np.isfinite(optimized).all()
        or not np.isfinite(replay_states).all()
    ):
        raise ValueError(
            "optimized and fresh replay states differ in shape or finiteness"
        )
    groups = {
        "coordinates_rad": tuple(
            i for i, name in enumerate(state_names) if name.endswith("/value")
        ),
        "speeds_rad_s": tuple(
            i for i, name in enumerate(state_names) if name.endswith("/speed")
        ),
        "activations": tuple(
            i for i, name in enumerate(state_names) if name.endswith("/activation")
        ),
        "fiber_lengths_m": tuple(
            i for i, name in enumerate(state_names) if name.endswith("/fiber_length")
        ),
    }
    if not all(groups.values()) or set(sum(groups.values(), ())) != set(
        range(len(state_names))
    ):
        raise ValueError("replay states include an unclassified native state variable")
    difference = np.abs(optimized - replay_states)
    limits = {
        "coordinates_rad": 2e-7,
        "speeds_rad_s": 1e-7,
        "activations": 1e-8,
        "fiber_lengths_m": 1e-7,
    }
    errors = {
        group: float(np.max(difference[:, indices]))
        for group, indices in groups.items()
    }
    if any(errors[group] >= limit for group, limit in limits.items()):
        raise ValueError(f"fresh replay state error exceeds limits: {errors}")
    coordinate_index = state_names.index("/jointset/base/q/value")
    optimized_endpoint = float(optimized[-1, coordinate_index])
    replay_endpoint = float(replay_states[-1, coordinate_index])
    endpoints = (optimized_endpoint, replay_endpoint)
    if any(abs(value - target_q) >= 2e-7 for value in endpoints):
        raise ValueError("optimized or replay endpoint misses requested target")
    if any((value - initial_q) * (target_q - initial_q) <= 0 for value in endpoints):
        raise ValueError("optimized or replay endpoint moves in the wrong direction")
    if any(abs(value - initial_q) <= 2e-6 for value in endpoints):
        raise ValueError("optimized or replay endpoint did not move to nonzero target")
    return {
        "max_state_error_by_unit": errors,
        "optimized_endpoint_q": optimized_endpoint,
        "replayed_endpoint_q": replay_endpoint,
        "target_endpoint_within_tolerance": True,
    }


def _write_success_receipt(
    prepared: dict[str, Any], solved: dict, replay_path: Path, stages: dict, osim: Any
) -> None:
    solution = solved["solution"]
    coordinate = prepared["config"]["target_coordinate"]
    achieved_q = float(solution.getStateMat(coordinate)[-1])
    output = prepared["config_path"].parent
    replay_evidence = json.loads(
        (output / "fresh_replay.json").read_text(encoding="utf-8")
    )
    runtime_path = _write_runtime_fingerprint(osim, output)
    objective = float(solution.getObjective())
    if not np.isfinite(objective):
        raise ValueError("native Moco solver objective must be finite")
    receipt = {
        "rung": prepared["rung"].name,
        "features": list(prepared["rung"].features),
        "status": "solved_and_replayed",
        "fresh_model_replay": True,
        "nonzero_target_reached": replay_evidence.get(
            "target_endpoint_within_tolerance"
        )
        is True,
        "target_q": prepared["target_q"],
        "initial_q": prepared["config"]["initial_q"],
        "achieved_q": achieved_q,
        "objective": objective,
        "solver_status": str(solution.getStatus()),
        "source_sha256": prepared["source_hash"],
        "worker_sha256": _sha256(Path(__file__).resolve()),
        "opensim_runtime_fingerprint_sha256": _sha256(runtime_path),
        "request_sha256": _sha256(prepared["config_path"]),
        "saved_solution_sha256": _sha256(solved["solution_path"]),
        "saved_time_only_controls_sha256": _sha256(solved["controls_path"]),
        "fresh_replay_sha256": _sha256(replay_path),
        "stage_metrics": stages,
        "physical_qualification": "unqualified",
    }
    _write_json(output / "receipt.json", receipt)


def _write_runtime_fingerprint(osim: Any, output: Path) -> Path:
    components = {}
    package = Path(osim.__file__).resolve().parent
    for filename in ("_common.pyd", "_simulation.pyd", "_moco.pyd", "osimMoco.dll"):
        path = package / filename
        if not path.is_file():
            raise ValueError(f"OpenSim runtime component is missing: {filename}")
        components[filename] = _sha256(path)
    path = output / "opensim_runtime_fingerprint.json"
    _write_json(
        path,
        {
            "schema": "opensim-bounded-runtime-fingerprint-v1",
            "version": str(osim.GetVersionAndDate()),
            "components": components,
            "coverage_note": "bounded Python extension and Moco DLL fingerprint; "
            "not a closed dependency-runtime identity",
        },
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_rung(args.rung, args.output)


if __name__ == "__main__":
    main()
