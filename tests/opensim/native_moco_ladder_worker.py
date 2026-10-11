"""One-rung native Moco solve/export/fresh-replay worker.

Launch only through ``native_moco_ladder_harness.run_ladder`` so the worker
has a separate process and a hard parent-enforced 30-second deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from tests.opensim.native_moco_ladder_harness import (
    DEFAULT_RUNGS,
    _validate_mesh_intervals,
    build_synthetic_rung,
)
from tests.opensim.native_moco_refinement import (
    build_refinement_plan,
    child_hs_grid,
    piecewise_linear_samples,
)

_SERIALIZATION_MAX_ULPS = 8


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


def run_rung(
    rung_name: str,
    output: Path,
    mesh_intervals: int = 10,
    refinement_plan_path: Path | None = None,
) -> None:
    import opensim as osim

    rung = next(item for item in DEFAULT_RUNGS if item.name == rung_name)
    # The supervisor creates the per-rung directory before launching us.
    output.mkdir(parents=True, exist_ok=True)
    stages: dict[str, dict[str, float]] = {}
    refinement = (
        json.loads(refinement_plan_path.read_text(encoding="utf-8"))
        if refinement_plan_path is not None
        else None
    )
    prepared = _prepare_rung(
        rung, output, stages, osim, mesh_intervals, refinement=refinement
    )
    solved = _solve_and_export(prepared, stages, osim)
    replay_path = _fresh_native_replay(prepared, solved, stages)
    _write_success_receipt(prepared, solved, replay_path, stages, osim)


def run_saved_replay_diagnostic(source_run: Path, output: Path) -> None:
    """Replay retained solution inputs without constructing or running a solver."""
    _start_replay_diagnostic(output)
    import opensim as osim

    _write_diagnostic_stage(output, "opensim_imported")
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        replay_constrained_muscle_bundle,
    )

    _write_diagnostic_stage(output, "replay_provider_imported")
    prepared, solution, grid, controls, bundle = _load_saved_replay_inputs(
        source_run, output, osim
    )
    _write_diagnostic_stage(output, "saved_inputs_loaded")
    _write_diagnostic_stage(output, "native_replay_started")
    replay = _run_with_periodic_replay_stack_dumps(
        output,
        lambda: replay_constrained_muscle_bundle(
            bundle,
            prepared["declaration"],
            native_mechanics_profile=prepared["profile"],
        ),
    )
    _write_diagnostic_stage(output, "native_replay_completed")
    audit = _make_replay_state_audit(
        solution,
        replay.states,
        replay.state_names,
        grid,
        prepared["target_q"],
        prepared["initial_q"],
        prepared["declaration"].named_state,
    )
    audit.update(
        {
            "schema": "native-moco-fresh-replay-diagnostic-v1",
            "fresh_model": True,
            "input_kind": "retained_time_only_controls",
            "source_sha256": prepared["source_sha256"],
            "request_sha256": prepared["request_sha256"],
            "solution_sha256": prepared["solution_sha256"],
            "saved_controls_sha256": prepared["controls_sha256"],
            "input_sha256": bundle.applied_input_sha256,
            "native_constraint_audit_count": len(replay.constraint_audits),
            "physical_qualification": "unqualified",
            **_replay_control_audit(replay, controls),
        }
    )
    _record_replay_audit(
        output / "fresh_replay_diagnostic.json", audit, require_pass=False
    )
    _write_json(
        output / "diagnostic_status.json",
        {
            "status": audit["status"],
            "acceptance_failures": audit["acceptance_failures"],
        },
    )
    _write_diagnostic_stage(output, "audit_written")


def _validate_refinement_request(
    refinement: dict[str, Any], staged_source_sha256: str, child_config: dict[str, Any]
) -> None:
    _validate_refinement_plan(refinement)
    if staged_source_sha256 != refinement["parent_hashes"].get("source_osim"):
        raise ValueError("refinement child source differs from parent source bytes")
    parent_request = refinement["parent_request"]
    preserved = refinement["preserved_contract"]
    if any(
        child_config.get(key) != value
        for key, value in preserved.items()
        if key != "objective"
    ):
        raise ValueError("refinement changed a bound objective or solver contract")
    if (
        parent_request.get("objective", "MocoControlGoal(effort)")
        != preserved["objective"]
    ):
        raise ValueError("refinement objective differs from retained parent")


def _validate_refinement_plan(refinement: dict[str, Any]) -> None:
    parent = Path(refinement["parent_directory"])
    analysis = Path(refinement["analysis_path"])
    checked = build_refinement_plan(parent, analysis)
    try:
        checked_json = json.dumps(checked, sort_keys=True, allow_nan=False)
        supplied_json = json.dumps(refinement, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("refinement plan is not canonical JSON data") from exc
    if checked_json != supplied_json:
        raise ValueError("refinement plan no longer matches its retained evidence")


def _build_mesh_grid_audit(
    refinement: dict[str, Any],
    child_request_refinement: dict[str, Any],
    solution_grid: Any,
) -> dict[str, Any]:
    plan_points = refinement.get("normalized_mesh_points")
    request_points = child_request_refinement.get("normalized_mesh_points")
    if (
        not isinstance(plan_points, (list, tuple))
        or not isinstance(request_points, (list, tuple))
        or child_request_refinement.get("attempt_id") != refinement.get("attempt_id")
        or child_request_refinement.get("parent_hashes")
        != refinement.get("parent_hashes")
        or list(request_points) != list(plan_points)
    ):
        raise ValueError("child request changed the retained refinement plan")
    seed_grid = child_request_refinement.get("native_guess_time_grid_seconds")
    if not isinstance(seed_grid, list) or not seed_grid:
        raise ValueError("child request omitted its seeded native guess grid")
    expected_grid = np.asarray(
        child_hs_grid(
            plan_points,
            refinement["parent_request"]["horizon_seconds"],
        ),
        dtype=np.float64,
    )
    actual_grid = np.asarray(solution_grid, dtype=np.float64)
    if not np.array_equal(np.asarray(seed_grid, dtype=np.float64), expected_grid):
        raise ValueError("native seed grid differs from the planned HS grid")
    return {
        "schema": "native-moco-refinement-solution-grid-v1",
        "attempt_id": refinement["attempt_id"],
        "parent_hashes": refinement["parent_hashes"],
        "planned_normalized_mesh_points": refinement["normalized_mesh_points"],
        "planned_time_grid_seconds": expected_grid.tolist(),
        "seed_time_grid_seconds": seed_grid,
        "solution_time_grid_seconds": actual_grid.tolist(),
        "exact_solution_grid_match": bool(np.array_equal(actual_grid, expected_grid)),
        "physical_qualification": "unqualified",
    }


def _set_parent_solution_guess(
    solver: Any, parent: Path, refinement: dict[str, Any], osim: Any
) -> tuple[float, ...]:
    """Seed states from the parent trajectory and controls by exact PWL values."""
    hashes = refinement["parent_hashes"]
    solution_path = parent / "solution.sto"
    controls_path = parent / "time_only_controls.sto"
    if (
        _sha256(solution_path) != hashes["solution_sto"]
        or _sha256(controls_path) != hashes["time_only_controls_sto"]
    ):
        raise ValueError("parent solution or persisted controls changed before seeding")
    solution = osim.MocoTrajectory(str(solution_path))
    controls = osim.TimeSeriesTable(str(controls_path))
    guess = solver.createGuess()
    state_table = solution.exportToStatesTable()
    guess.insertStatesTrajectory(state_table, True)
    child_times = _native_vector_values(guess.getTime(), guess.getNumTimes())
    expected_times = child_hs_grid(
        refinement["normalized_mesh_points"],
        refinement["parent_request"]["horizon_seconds"],
    )
    if child_times != expected_times:
        raise ValueError("native child guess grid differs from the planned HS grid")
    parent_times = tuple(float(value) for value in controls.getIndependentColumn())
    labels = tuple(str(value) for value in controls.getColumnLabels())
    if not labels or set(labels) != set(guess.getControlNames()):
        raise ValueError(
            "parent saved controls do not bind every child control channel"
        )
    bounds = refinement["parent_request"].get("native_control_bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(labels):
        raise ValueError("parent request does not bind every persisted control bound")
    for name in labels:
        column = controls.getDependentColumn(name)
        source_values = tuple(
            float(column[index]) for index in range(controls.getNumRows())
        )
        lower, upper = bounds[name]
        if any(not lower <= value <= upper for value in source_values):
            raise ValueError("persisted control seed falls outside its native bounds")
        values = piecewise_linear_samples(parent_times, source_values, child_times)
        vector = osim.Vector(len(values), 0.0)
        for index, value in enumerate(values):
            vector[index] = value
        guess.setControl(name, vector)
        readback = _native_vector_values(guess.getControl(name), len(values))
        if readback != values:
            raise ValueError("Moco guess changed an exact PWL control seed sample")
    if set(guess.getStateNames()) != set(solution.getStateNames()):
        raise ValueError("parent state channels do not bind every child state")
    _restore_declared_initial_states(
        guess,
        refinement["preserved_contract"]["cold_start_named_state"],
        osim,
        len(child_times),
    )
    solver.setGuess(guess)
    return child_times


def _restore_declared_initial_states(
    guess: Any, named_state: dict[str, float], osim: Any, time_count: int
) -> None:
    state_names = set(guess.getStateNames())
    if not set(named_state).issubset(state_names):
        raise ValueError("declared cold states are absent from the parent trajectory")
    for name, initial_value in named_state.items():
        original = _native_vector_values(guess.getState(name), time_count)
        if not original or not math.isfinite(initial_value):
            raise ValueError("declared cold state is empty or non-finite")
        values = osim.Vector(len(original), 0.0)
        for index, value in enumerate(original):
            values[index] = float(initial_value) if index == 0 else value
        guess.setState(name, values)
        readback = _native_vector_values(guess.getState(name), time_count)
        if not readback or readback[0] != float(initial_value):
            raise ValueError(
                "Moco guess did not preserve the exact declared initial state"
            )


def _native_vector_values(vector: Any, size: int) -> tuple[float, ...]:
    if size < 0:
        raise ValueError("native vector size cannot be negative")
    return tuple(float(vector[index]) for index in range(size))


def _run_with_periodic_replay_stack_dumps(
    output: Path,
    replay_call: Any,
    *,
    dump_later: Any = None,
    cancel_dump: Any = None,
) -> Any:
    """Capture periodic Python stacks while a native replay is running."""
    import faulthandler

    schedule = dump_later or faulthandler.dump_traceback_later
    cancel = cancel_dump or faulthandler.cancel_dump_traceback_later
    trace_path = output / "native_replay_stack_traces.log"
    with trace_path.open("a", encoding="utf-8", buffering=1) as traces:
        schedule(10, repeat=True, file=traces)
        try:
            return replay_call()
        finally:
            cancel()
            traces.flush()
            os.fsync(traces.fileno())


def _start_replay_diagnostic(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    _write_diagnostic_stage(output, "worker_started")


def _write_diagnostic_stage(output: Path, stage: str) -> None:
    _write_json(output / "diagnostic_progress.json", {"stage": stage})


def _load_saved_replay_inputs(
    source_run: Path, output: Path, osim: Any
) -> tuple[dict[str, Any], Any, np.ndarray, dict[str, np.ndarray], Any]:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
        declare_native_mechanics_execution_profile,
    )

    request_path = source_run / "request.json"
    source_path = source_run / "source.osim"
    solution_path = source_run / "solution.sto"
    controls_path = source_run / "time_only_controls.sto"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source_hash = _sha256(source_path)
    if request.get("source_sha256") != source_hash:
        raise ValueError("retained replay source differs from its request identity")
    built_path, declaration = build_synthetic_rung(DEFAULT_RUNGS[0], output / "build")
    if _sha256(built_path) != source_hash:
        raise ValueError(
            "retained replay source differs from the declared plain source"
        )
    if dict(declaration.named_state) != request.get("cold_start_named_state"):
        raise ValueError(
            "retained cold state differs from freshly declared source state"
        )
    declaration = replace(declaration, model_path=source_path)
    profile = declare_native_mechanics_execution_profile(source_path, source_hash)
    if profile.profile_sha256 != request.get("native_mechanics_profile_sha256"):
        raise ValueError("retained replay mechanics profile identity changed")
    solution = osim.MocoTrajectory(str(solution_path))
    arrays = _trajectory_arrays(solution)
    muscle_names = tuple(
        name.removeprefix("/forceset/") for name in arrays["control_names"]
    )
    expected_controls = {
        name: arrays["controls"][:, index] for index, name in enumerate(muscle_names)
    }
    grid, controls = _load_time_only_controls(
        osim, controls_path, arrays["time"], muscle_names, expected_controls
    )
    bundle = build_constrained_muscle_bundle(
        declaration,
        grid,
        controls,
        experiment_id="native-moco-retained-solution-diagnostic-v1",
        native_mechanics_profile=profile,
    )
    prepared = {
        "declaration": declaration,
        "profile": profile,
        "source_sha256": source_hash,
        "request_sha256": _sha256(request_path),
        "solution_sha256": _sha256(solution_path),
        "controls_sha256": _sha256(controls_path),
        "target_q": float(request["target_q"]),
        "initial_q": float(request["initial_q"]),
    }
    return prepared, solution, grid, controls, bundle


def _prepare_rung(
    rung: Any,
    output: Path,
    stages: dict,
    osim: Any,
    mesh_intervals: int = 10,
    *,
    refinement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source, declaration, source_hash = _stage_rung_source(
        rung, output, stages, refinement
    )
    prepared = _prepare_rung_model(
        rung, output, stages, source, declaration, source_hash, osim
    )
    config = _build_rung_config(
        rung,
        source,
        source_hash,
        declaration,
        prepared["profile"],
        prepared["witness"],
        prepared["control_bounds"],
        mesh_intervals,
        refinement,
    )
    study, solver, config_path = _configure_rung_problem(
        rung,
        source,
        declaration,
        prepared["witness"],
        prepared["target_q"],
        prepared["control_bounds"],
        config,
        output,
        refinement,
        osim,
        stages,
        mesh_intervals,
    )
    return {
        "rung": rung,
        "source": source,
        "source_hash": source_hash,
        "declaration": declaration,
        "profile": prepared["profile"],
        "wrapped": prepared["wrapped"],
        "target_q": prepared["target_q"],
        "config": config,
        "config_path": config_path,
        "study": study,
        "solver": solver,
        "refinement": refinement,
        "witness": prepared["witness"],
    }


def _stage_rung_source(
    rung: Any, output: Path, stages: dict, refinement: dict[str, Any] | None
) -> tuple[Path, Any, str]:
    from tests.opensim.native_moco_ladder_harness import (
        _derive_synthetic_cold_start,
    )

    started = _stage_start()
    source = output / "source.osim"
    if refinement is None:
        built_path, declaration = build_synthetic_rung(rung, output / "build")
        shutil.copyfile(built_path, source)
    else:
        parent_source = Path(refinement["parent_directory"]) / "source.osim"
        if _sha256(parent_source) != refinement["parent_hashes"]["source_osim"]:
            raise ValueError("refinement parent source changed before child staging")
        shutil.copyfile(parent_source, source)
        derived_source, declaration = _derive_synthetic_cold_start(source, rung)
        if Path(derived_source).resolve() != source.resolve():
            raise ValueError("cold-start helper returned a different staged source")
    _flush_file(source)
    declaration = replace(declaration, model_path=source)
    source_hash = _sha256(source)
    if source_hash != declaration.source_sha256:
        raise ValueError("staged source differs from the fresh declaration")
    stages["source_build_and_state_equilibration"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)
    return source, declaration, source_hash


def _prepare_rung_model(
    rung: Any,
    output: Path,
    stages: dict,
    source: Path,
    declaration: Any,
    source_hash: str,
    osim: Any,
) -> dict[str, Any]:
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
    started = _stage_start()
    witness = _build_feasible_witness(
        source, declaration, profile, wrapped, output / "witness", osim
    )
    stages["fixed_excitation_forward_witness"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)
    return {
        "profile": profile,
        "wrapped": wrapped,
        "witness": witness,
        "target_q": witness["target_q"],
        "control_bounds": _native_control_bounds(
            osim.Model(str(source)),
            {f"/forceset/{name}" for name in witness["controls"]},
        ),
    }


def _build_rung_config(
    rung: Any,
    source: Path,
    source_hash: str,
    declaration: Any,
    profile: Any,
    witness: dict[str, Any],
    control_bounds: dict[str, tuple[float, float]],
    mesh_intervals: int,
    refinement: dict[str, Any] | None,
) -> dict[str, Any]:
    config = _rung_config(
        rung,
        witness["target_q"],
        source_hash,
        witness["initial_q"],
        mesh_intervals=mesh_intervals,
    )
    if refinement is not None:
        config["mesh_refinement"] = _refinement_request_fields(refinement)
    config["native_control_bounds"] = {
        name: list(bounds) for name, bounds in control_bounds.items()
    }
    config["native_control_bounds_source"] = {
        "source_sha256": source_hash,
        "model_path": source.name,
        "property": "Muscle.min_control/max_control",
    }
    config.update(_witness_request_metadata(declaration, profile, witness))
    if refinement is not None:
        _validate_refinement_request(refinement, source_hash, config)
    return config


def _refinement_request_fields(refinement: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_id": refinement["attempt_id"],
        "parent_hashes": refinement["parent_hashes"],
        "analysis_sha256": refinement["analysis_sha256"],
        "flagged_parent_cells": refinement["flagged_parent_cells"],
        # Native readback is represented as a JSON list; canonicalize the
        # planner's tuple at this request boundary for exact in-memory equality.
        "normalized_mesh_points": list(refinement["normalized_mesh_points"]),
        "preserved_contract": refinement["preserved_contract"],
        "control_seed": refinement["control_seed"],
        "status": "attempted_unqualified_until_fresh_replay_passes",
    }


def _configure_rung_problem(
    rung: Any,
    source: Path,
    declaration: Any,
    witness: dict[str, Any],
    target_q: float,
    control_bounds: dict[str, tuple[float, float]],
    config: dict[str, Any],
    output: Path,
    refinement: dict[str, Any] | None,
    osim: Any,
    stages: dict,
    mesh_intervals: int,
) -> tuple[Any, Any, Path]:
    started = _stage_start()
    study, solver = _build_witness_seeded_problem(
        source,
        declaration,
        witness,
        target_q,
        osim,
        control_bounds,
        mesh_intervals=mesh_intervals,
        transcription_scheme=config["transcription_scheme"],
        interpolate_control_mesh_interior_points=config[
            "interpolate_control_mesh_interior_points"
        ],
        normalized_mesh_points=(
            refinement["normalized_mesh_points"] if refinement is not None else None
        ),
    )
    if refinement is not None:
        parent = Path(refinement["parent_directory"])
        guess_grid = _set_parent_solution_guess(solver, parent, refinement, osim)
        config["mesh_refinement"]["native_guess_time_grid_seconds"] = list(guess_grid)
    config["actual_solver_settings"] = _actual_solver_settings(
        solver,
        refinement["normalized_mesh_points"] if refinement is not None else None,
    )
    config["solver_settings_validation_status"] = "pending"
    config_path = output / "request.json"
    # Keep the attempted request and native readback even when strict binding
    # validation rejects them; this makes preparation failures diagnosable.
    _write_json(config_path, config)
    _validate_actual_solver_settings(config)
    config["solver_settings_validation_status"] = "passed"
    _write_json(config_path, config)
    stages["preflight_and_solver_setup"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)
    return study, solver, config_path


def _build_feasible_witness(
    source: Path,
    declaration: Any,
    profile: Any,
    wrapped: Any,
    output: Path,
    osim: Any,
) -> dict[str, Any]:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )

    source_hash = _sha256(source)
    if source_hash != declaration.source_sha256 or profile.source_sha256 != source_hash:
        raise ValueError("forward witness source identity differs from declarations")
    grid = np.linspace(0.0, 0.01, 11, dtype=np.float64)
    muscle_names = _source_muscle_names(source, osim)
    controls = {
        name: np.full(grid.shape, 0.05, dtype=np.float64) for name in muscle_names
    }
    bundle = build_constrained_muscle_bundle(
        declaration,
        grid,
        controls,
        experiment_id="synthetic-fixed-excitation-witness-v1",
        wrapped_path_declaration=wrapped,
        native_mechanics_profile=profile,
    )
    replay = replay_constrained_muscle_bundle(
        bundle,
        declaration,
        wrapped_path_declaration=wrapped,
        native_mechanics_profile=profile,
    )
    witness = _persist_and_validate_witness(
        replay, grid, controls, source_hash, declaration, output, osim
    )
    provider_source = inspect.getsourcefile(build_constrained_muscle_bundle)
    if provider_source is None:
        raise ValueError("forward witness provider source identity is unavailable")
    witness.update(
        {
            "provider_sha256": _sha256(Path(provider_source)),
            "worker_sha256": _sha256(Path(__file__)),
            "builder_sha256": _sha256(
                Path(__file__).with_name("native_moco_ladder_harness.py")
            ),
        }
    )
    return witness


def _source_muscle_names(source: Path, osim: Any) -> tuple[str, ...]:
    model = osim.Model(str(source))
    muscles = model.getMuscles()
    names = tuple(
        str(muscles.get(index).getName()) for index in range(muscles.getSize())
    )
    if not names or len(set(names)) != len(names):
        raise ValueError("witness source has no unique muscle control channels")
    return names


def _persist_and_validate_witness(
    replay: Any,
    grid: np.ndarray,
    controls: dict[str, np.ndarray],
    source_hash: str,
    declaration: Any,
    output: Path,
    osim: Any,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    times = np.asarray(replay.times, dtype=np.float64)
    states = np.asarray(replay.states, dtype=np.float64)
    names = tuple(replay.state_names)
    if times.shape != grid.shape or not np.array_equal(times, grid):
        raise ValueError("forward witness clock differs from its declared knots")
    if states.shape != (len(grid), len(names)) or not np.isfinite(states).all():
        raise ValueError("forward witness state matrix is incomplete or non-finite")
    for name, expected in declaration.named_state.items():
        actual = float(states[0, names.index(name)])
        if abs(actual - expected) > 1e-12:
            raise ValueError("forward witness initial state differs from cold start")
    constraint_errors = [
        float(error)
        for audit in replay.constraint_audits
        for error in (*audit.position_errors, *audit.velocity_errors)
    ]
    if any(
        not math.isfinite(error) or abs(error) > declaration.residual_tolerance
        for error in constraint_errors
    ):
        raise ValueError("forward witness does not preserve native constraints")
    q_column = names.index("/jointset/base/q/value")
    initial_q = float(states[0, q_column])
    target_q = float(states[-1, q_column])
    if target_q - initial_q <= 2e-6:
        raise ValueError("fixed excitation witness has no positive target motion")
    fiber_columns = [
        index for index, name in enumerate(names) if name.endswith("/fiber_length")
    ]
    if not fiber_columns or np.any(states[:, fiber_columns] <= 0.0):
        raise ValueError("forward witness contains non-positive fiber length")
    state_path = output / "witness_states.sto"
    control_path = output / "witness_time_only_controls.sto"
    _write_witness_states(osim, times, names, states, state_path)
    _write_time_only_controls(osim, times, controls, control_path)
    return {
        "times": times,
        "states": states,
        "state_names": names,
        "controls": controls,
        "source_sha256": source_hash,
        "initial_q": initial_q,
        "target_q": target_q,
        "state_path": state_path,
        "control_path": control_path,
        "states_sha256": _sha256(state_path),
        "controls_sha256": _sha256(control_path),
    }


def _write_witness_states(
    osim: Any, times: np.ndarray, names: tuple[str, ...], values: np.ndarray, path: Path
) -> None:
    table = osim.TimeSeriesTable()
    labels = osim.StdVectorString()
    for name in names:
        labels.append(name)
    table.setColumnLabels(labels)
    for row_index, time_value in enumerate(times):
        row = osim.RowVector(len(names))
        for column_index in range(len(names)):
            row[column_index] = float(values[row_index, column_index])
        table.appendRow(float(time_value), row)
    osim.STOFileAdapter.write(table, str(path))


def _synthetic_state_bounds(
    initial: dict[str, float], optimal_fiber_lengths: dict[str, float]
) -> dict[str, tuple[float, float]]:
    bounds = {}
    for name, value in initial.items():
        if name.endswith("/activation"):
            bound = (max(0.01, value - 0.1), min(0.15, value + 0.1))
        elif name.endswith("/fiber_length"):
            muscle_path = name.removesuffix("/fiber_length")
            optimal = optimal_fiber_lengths.get(muscle_path)
            if optimal is None or not math.isfinite(optimal) or optimal <= 0.0:
                raise ValueError("synthetic fiber bounds lack a positive source scale")
            bound = (0.5 * optimal, 2.0 * optimal)
        else:
            bound = (value - 1.0, value + 1.0)
        if not all(math.isfinite(item) for item in bound) or bound[0] >= bound[1]:
            raise ValueError("synthetic Moco state bounds are invalid")
        if not bound[0] <= value <= bound[1]:
            raise ValueError("cold-start state is outside synthetic Moco bounds")
        bounds[name] = bound
    return bounds


def _state_units(named_state: dict[str, float]) -> dict[str, str]:
    units = {}
    for name in named_state:
        if name.endswith("/value"):
            unit = "rad"
        elif name.endswith("/speed"):
            unit = "rad/s"
        elif name.endswith("/activation"):
            unit = "1"
        elif name.endswith("/fiber_length"):
            unit = "m"
        else:
            raise ValueError(
                f"synthetic witness has an unclassified state unit: {name}"
            )
        units[name] = unit
    return units


def _witness_request_metadata(
    declaration: Any, profile: Any, witness: dict[str, Any]
) -> dict[str, Any]:
    return {
        "witness_controls_sha256": witness["controls_sha256"],
        "witness_states_sha256": witness["states_sha256"],
        "witness_provider_sha256": witness["provider_sha256"],
        "witness_worker_sha256": witness["worker_sha256"],
        "witness_builder_sha256": witness["builder_sha256"],
        "native_mechanics_profile_sha256": profile.profile_sha256,
        "witness_time_knots": witness["times"].tolist(),
        "witness_excitation_by_channel": dict.fromkeys(witness["controls"], 0.05),
        "witness_profile": "constant-0.05-forward-integrated-v1",
        "witness_source_sha256": witness["source_sha256"],
        "cold_start_named_state": dict(declaration.named_state),
        "cold_start_state_units": _state_units(declaration.named_state),
        "cold_start_time_seconds": declaration.time_seconds,
        "cold_start_lock_targets": dict(declaration.lock_targets),
        "cold_start_chart_bounds": {
            name: list(bounds) for name, bounds in declaration.chart_bounds.items()
        },
        "cold_start_constraint_enforcement": dict(declaration.constraint_enforcement),
        "cold_start_residual_tolerance": declaration.residual_tolerance,
        "native_mechanics_profile": asdict(profile),
    }


def _build_witness_seeded_problem(
    source: Path,
    declaration: Any,
    witness: dict[str, Any],
    target_q: float,
    osim: Any,
    control_bounds: dict[str, tuple[float, float]] | None = None,
    *,
    mesh_intervals: int = 10,
    transcription_scheme: str = "hermite-simpson",
    interpolate_control_mesh_interior_points: bool = True,
    normalized_mesh_points: list[float] | None = None,
) -> tuple[Any, Any]:
    from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
        MocoInitialBindings,
        apply_moco_initial_bindings,
    )

    if _sha256(source) != witness["source_sha256"]:
        raise ValueError("witness source changed before Moco problem construction")
    model = osim.Model(str(source))
    observed_bounds = _native_control_bounds(
        model, {f"/forceset/{name}" for name in witness["controls"]}
    )
    if control_bounds is not None and control_bounds != observed_bounds:
        raise ValueError("declared Moco control bounds differ from native source")
    native_bounds = observed_bounds
    optimal = {
        f"/forceset/{model.getMuscles().get(index).getName()}": float(
            model.getMuscles().get(index).getOptimalFiberLength()
        )
        for index in range(model.getMuscles().getSize())
    }
    state_bounds = _synthetic_state_bounds(declaration.named_state, optimal)
    _validate_witness_inside_bounds(witness, state_bounds)
    _validate_witness_control_bounds(witness, native_bounds)
    bindings = MocoInitialBindings(
        state_bounds,
        declaration.named_state,
        native_bounds,
    )
    study = osim.MocoStudy()
    problem = study.updProblem()
    problem.setModelProcessor(osim.ModelProcessor(str(source)))
    problem.setTimeBounds(float(witness["times"][0]), float(witness["times"][-1]))
    apply_moco_initial_bindings(problem, str(source), bindings, osim)
    problem.setStateInfo(
        "/jointset/base/q/value",
        osim.MocoBounds(0.2, 0.4),
        osim.MocoInitialBounds(witness["initial_q"]),
        osim.MocoFinalBounds(target_q),
    )
    problem.addGoal(osim.MocoControlGoal("effort"))
    solver = osim.MocoCasADiSolver.safeDownCast(study.initCasADiSolver())
    _configure_solver(
        solver,
        mesh_intervals,
        transcription_scheme,
        interpolate_control_mesh_interior_points,
        normalized_mesh_points=normalized_mesh_points,
    )
    _set_witness_guess(solver, witness, state_bounds, optimal, osim)
    return study, solver


def _native_control_bounds(
    model: Any, expected_control_paths: set[str]
) -> dict[str, tuple[float, float]]:
    """Read complete scalar muscle excitation bounds from the unchanged model."""
    muscles = model.getMuscles()
    result = {}
    for index in range(muscles.getSize()):
        muscle = muscles.get(index)
        path = str(muscle.getAbsolutePathString())
        lower = _finite_control_bound(muscle.getMinControl(), path, "min_control")
        upper = _finite_control_bound(muscle.getMaxControl(), path, "max_control")
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError(f"native muscle control bounds are invalid for {path}")
        result[path] = (lower, upper)
    if set(result) != expected_control_paths:
        raise ValueError("native muscle bounds do not cover exact witness controls")
    return result


def _finite_control_bound(value: Any, path: str, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"native {label} must be a real scalar for {path}")
    bound = float(value)
    if not math.isfinite(bound):
        raise ValueError(f"native {label} must be finite for {path}")
    return bound


def _validate_witness_inside_bounds(
    witness: dict[str, Any], state_bounds: dict[str, tuple[float, float]]
) -> None:
    for column, name in enumerate(witness["state_names"]):
        low, high = state_bounds[name]
        values = witness["states"][:, column]
        if np.any(values < low) or np.any(values > high):
            raise ValueError(f"forward witness leaves synthetic bounds for {name}")


def _validate_witness_control_bounds(
    witness: dict[str, Any], control_bounds: dict[str, tuple[float, float]]
) -> None:
    if set(witness["controls"]) != {
        name.removeprefix("/forceset/") for name in control_bounds
    }:
        raise ValueError("forward witness controls differ from native bound channels")
    for path, (lower, upper) in control_bounds.items():
        values = np.asarray(witness["controls"][path.removeprefix("/forceset/")])
        if (
            not np.isfinite(values).all()
            or np.any(values < lower)
            or np.any(values > upper)
        ):
            raise ValueError(
                f"forward witness excitation leaves native bounds for {path}"
            )


def _set_witness_guess(
    solver: Any,
    witness: dict[str, Any],
    state_bounds: dict[str, tuple[float, float]],
    optimal_fiber_lengths: dict[str, float],
    osim: Any,
) -> Any:
    if _sha256(witness["state_path"]) != witness["states_sha256"]:
        raise ValueError("saved witness states changed before Moco guess loading")
    if _sha256(witness["control_path"]) != witness["controls_sha256"]:
        raise ValueError("saved witness controls changed before Moco guess loading")
    states_table = osim.TimeSeriesTable(str(witness["state_path"]))
    controls_table = osim.TimeSeriesTable(str(witness["control_path"]))
    state_names = tuple(str(value) for value in states_table.getColumnLabels())
    control_names = tuple(str(value) for value in controls_table.getColumnLabels())
    expected_controls = tuple(optimal_fiber_lengths)
    if (
        state_names != tuple(witness["state_names"])
        or set(state_names) != set(state_bounds)
        or control_names != expected_controls
    ):
        raise ValueError(
            "saved witness state/control channels differ from Moco bindings"
        )
    _validate_saved_witness_tables(states_table, controls_table, witness)
    guess = solver.createGuess()
    guess.insertStatesTrajectory(states_table, True)
    guess.insertControlsTrajectory(controls_table, True)
    if set(guess.getStateNames()) != set(state_bounds) or set(
        guess.getControlNames()
    ) != set(expected_controls):
        raise ValueError("Moco witness guess channels differ from request")
    solver.setGuess(guess)
    return guess


def _validate_saved_witness_tables(
    states_table: Any, controls_table: Any, witness: dict[str, Any]
) -> None:
    expected = (
        (states_table, tuple(witness["state_names"])),
        (
            controls_table,
            tuple(f"/forceset/{name}" for name in witness["controls"]),
        ),
    )
    for table, expected_names in expected:
        labels = tuple(str(name) for name in table.getColumnLabels())
        independent = table.getIndependentColumn()
        times = np.asarray(
            [independent[index] for index in range(table.getNumRows())],
            dtype=np.float64,
        )
        if labels != expected_names or not np.array_equal(times, witness["times"]):
            raise ValueError("persisted witness channels or clock changed")
    for name, expected_values in witness["controls"].items():
        saved = controls_table.getDependentColumn(f"/forceset/{name}")
        actual = np.asarray(
            [saved[index] for index in range(controls_table.getNumRows())],
            dtype=np.float64,
        )
        if not np.array_equal(actual, expected_values):
            raise ValueError("persisted witness excitation values changed")
    for index, name in enumerate(witness["state_names"]):
        saved = states_table.getDependentColumn(name)
        actual = np.asarray(
            [saved[row] for row in range(states_table.getNumRows())],
            dtype=np.float64,
        )
        if not np.isfinite(actual).all() or not np.allclose(
            actual, witness["states"][:, index], rtol=0.0, atol=1e-14
        ):
            raise ValueError("persisted witness state values changed")


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
    rung: Any,
    target_q: float,
    source_sha256: str,
    initial_q: float,
    *,
    mesh_intervals: int = 10,
) -> dict[str, Any]:
    mesh_intervals = _validate_mesh_intervals(mesh_intervals)
    features = rung.features
    return {
        "rung": rung.name,
        "features": features,
        "source_sha256": source_sha256,
        "target_coordinate": "/jointset/base/q/value",
        "target_q": target_q,
        "initial_q": initial_q,
        "horizon_seconds": 0.01,
        "mesh_intervals": mesh_intervals,
        "transcription_scheme": "hermite-simpson",
        "interpolate_control_mesh_interior_points": True,
        "replay_control_interpolation": "piecewise_linear",
        "parallel": 0,
        "maximum_iterations": 300,
        "solution_serialization_max_ulps": _SERIALIZATION_MAX_ULPS,
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
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _flush_file(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _configure_solver(
    solver: Any,
    mesh_intervals: int = 10,
    transcription_scheme: str = "hermite-simpson",
    interpolate_control_mesh_interior_points: bool = True,
    *,
    normalized_mesh_points: list[float] | None = None,
) -> None:
    mesh_intervals = _validate_mesh_intervals(mesh_intervals)
    if transcription_scheme != "hermite-simpson":
        raise ValueError("synthetic ladder requires Hermite-Simpson transcription")
    if not isinstance(interpolate_control_mesh_interior_points, bool):
        raise TypeError("control midpoint interpolation setting must be boolean")
    if normalized_mesh_points is None:
        solver.set_num_mesh_intervals(mesh_intervals)
    else:
        from tests.opensim.native_moco_refinement import validate_normalized_mesh

        points = validate_normalized_mesh(normalized_mesh_points)
        if len(points) - 1 != mesh_intervals:
            raise ValueError("custom mesh interval count differs from its request")
        setter = getattr(solver, "setMesh", None)
        if not callable(setter):
            raise RuntimeError("installed Moco solver does not expose setMesh")
        setter(list(points))
    solver.set_transcription_scheme(transcription_scheme)
    solver.set_interpolate_control_mesh_interior_points(
        interpolate_control_mesh_interior_points
    )
    solver.set_parallel(0)
    solver.set_optim_max_iterations(300)
    solver.set_optim_convergence_tolerance(1e-6)
    solver.set_optim_constraint_tolerance(1e-7)


def _actual_solver_settings(
    solver: Any, normalized_mesh_points: list[float] | None = None
) -> dict[str, Any]:
    result = {
        "transcription_scheme": str(solver.get_transcription_scheme()),
        "interpolate_control_mesh_interior_points": bool(
            solver.get_interpolate_control_mesh_interior_points()
        ),
        "parallel": int(solver.get_parallel()),
    }
    if normalized_mesh_points is None:
        result["mesh_intervals"] = int(solver.get_num_mesh_intervals())
    else:
        # OpenSim's uniform-count getter does not report the custom setMesh grid.
        result["uniform_mesh_interval_getter_ignored"] = int(
            solver.get_num_mesh_intervals()
        )
        result["set_mesh_requested_normalized_points"] = list(normalized_mesh_points)
        result["set_mesh_requested_interval_count"] = len(normalized_mesh_points) - 1
    return result


def _preparation_witness_evidence(witness: dict[str, Any]) -> dict[str, Any]:
    """Return source-bound fields produced by the fixed-excitation witness."""
    state_path = Path(witness["state_path"])
    control_path = Path(witness["control_path"])
    if (
        not state_path.is_file()
        or not control_path.is_file()
        or _sha256(state_path) != witness["states_sha256"]
        or _sha256(control_path) != witness["controls_sha256"]
    ):
        raise ValueError("prepared witness files differ from their recorded hashes")
    return {
        "source_sha256": witness["source_sha256"],
        "initial_q": float(witness["initial_q"]),
        "target_q": float(witness["target_q"]),
        "state_path": str(state_path),
        "control_path": str(control_path),
        "states_sha256": witness["states_sha256"],
        "controls_sha256": witness["controls_sha256"],
        "time_count": int(len(witness["times"])),
        "control_channels": sorted(witness["controls"]),
    }


def _validate_actual_solver_settings(config: dict[str, Any]) -> None:
    expected = {
        "mesh_intervals": config["mesh_intervals"],
        "transcription_scheme": config["transcription_scheme"],
        "interpolate_control_mesh_interior_points": config[
            "interpolate_control_mesh_interior_points"
        ],
        "parallel": config["parallel"],
    }
    refinement = config.get("mesh_refinement")
    if isinstance(refinement, dict):
        expected.pop("mesh_intervals")
        expected["set_mesh_requested_normalized_points"] = refinement[
            "normalized_mesh_points"
        ]
        expected["set_mesh_requested_interval_count"] = config["mesh_intervals"]
        actual = config.get("actual_solver_settings")
        if (
            not isinstance(actual, dict)
            or isinstance(actual.get("uniform_mesh_interval_getter_ignored"), bool)
            or not isinstance(actual.get("uniform_mesh_interval_getter_ignored"), int)
            or actual["uniform_mesh_interval_getter_ignored"] < 0
        ):
            raise ValueError("custom mesh diagnostic getter is malformed")
        comparable = {
            key: value
            for key, value in actual.items()
            if key != "uniform_mesh_interval_getter_ignored"
        }
    else:
        comparable = config.get("actual_solver_settings")
    if comparable != expected:
        raise ValueError("native solver settings differ from the bound request")


def _validate_preparation_request_bindings(
    plan: dict[str, Any], config: dict[str, Any]
) -> None:
    """Bind wrapper observations to the exact request and refinement plan."""
    from tests.opensim.native_moco_refinement import child_hs_grid

    _validate_actual_solver_settings(config)
    points = list(plan["normalized_mesh_points"])
    settings = config["actual_solver_settings"]
    refinement = config.get("mesh_refinement")
    if (
        settings.get("set_mesh_requested_normalized_points") != points
        or settings.get("set_mesh_requested_interval_count") != len(points) - 1
        or not isinstance(refinement, dict)
        or refinement.get("normalized_mesh_points") != points
    ):
        raise ValueError("prepared custom mesh differs from the reviewed plan")
    expected_grid = child_hs_grid(
        plan["normalized_mesh_points"],
        plan["preserved_contract"]["horizon_seconds"],
    )
    actual_grid = refinement.get("native_guess_time_grid_seconds")
    if not isinstance(actual_grid, list) or tuple(actual_grid) != expected_grid:
        raise ValueError("prepared native guess clock differs from the reviewed plan")
    if (
        config.get("interpolate_control_mesh_interior_points") is not True
        or config.get("replay_control_interpolation") != "piecewise_linear"
    ):
        raise ValueError(
            "prepared control interpolation differs from the reviewed plan"
        )


def _trajectory_arrays(trajectory: Any) -> dict[str, Any]:
    state_names = tuple(str(name) for name in trajectory.getStateNames())
    control_names = tuple(str(name) for name in trajectory.getControlNames())
    return {
        "state_names": state_names,
        "control_names": control_names,
        "time": np.asarray(trajectory.getTimeMat(), dtype=np.float64).reshape(-1),
        "states": np.column_stack(
            [
                np.asarray(trajectory.getStateMat(name), dtype=np.float64).reshape(-1)
                for name in state_names
            ]
        ),
        "controls": np.column_stack(
            [
                np.asarray(trajectory.getControlMat(name), dtype=np.float64).reshape(-1)
                for name in control_names
            ]
        ),
    }


def _audit_solution_roundtrip(
    in_memory: dict[str, Any], persisted: dict[str, Any]
) -> dict[str, Any]:
    _validate_roundtrip_channels(in_memory)
    _validate_roundtrip_channels(persisted)
    for names_key in ("state_names", "control_names"):
        if in_memory[names_key] != persisted[names_key]:
            raise ValueError(f"serialized solution changed {names_key}")
    arrays = {}
    max_ulp_error = 0
    for name in ("time", "states", "controls"):
        left = np.asarray(in_memory[name], dtype=np.float64)
        right = np.asarray(persisted[name], dtype=np.float64)
        if left.shape != right.shape or left.size == 0:
            raise ValueError(f"serialized solution changed {name} shape")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"serialized solution contains non-finite {name}")
        delta = np.abs(left - right)
        ulp_error = _float64_ulp_distance(left, right)
        maximum_ulp = int(np.max(ulp_error))
        if not np.isfinite(delta).all():
            raise ValueError(f"serialized solution has invalid {name} round-trip error")
        max_ulp_error = max(max_ulp_error, maximum_ulp)
        arrays[name] = {
            "max_absolute_error": float(np.max(delta)),
            "max_ulp_error": maximum_ulp,
            "in_memory_sha256": _array_sha256(left, in_memory, name),
            "persisted_sha256": _array_sha256(right, persisted, name),
        }
    for trajectory in (in_memory, persisted):
        time = np.asarray(trajectory["time"], dtype=np.float64)
        if time[0] != 0.0 or time[-1] != 0.01:
            raise ValueError("solution does not retain the exact requested clock")
    return {
        "schema": "synthetic-solution-sto-roundtrip-v1",
        "maximum_allowed_ulps": _SERIALIZATION_MAX_ULPS,
        "maximum_ulp_error": max_ulp_error,
        "arrays": arrays,
        "state_names": list(in_memory["state_names"]),
        "control_names": list(in_memory["control_names"]),
    }


def _float64_ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return exact representable-value distances, treating signed zeros equally."""
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("float64 ULP operands must have matching shapes")
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("float64 ULP operands must be finite")
    sign_mask = np.uint64(1 << 63)

    def ordered_keys(values: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(values).view(np.uint64).copy()
        bits[values == 0.0] = np.uint64(0)
        negative = (bits & sign_mask) != 0
        keys = bits | sign_mask
        keys[negative] = ~bits[negative] + np.uint64(1)
        return keys

    left_keys = ordered_keys(left_values)
    right_keys = ordered_keys(right_values)
    return np.maximum(left_keys, right_keys) - np.minimum(left_keys, right_keys)


def _array_sha256(values: np.ndarray, trajectory: dict[str, Any], name: str) -> str:
    digest = hashlib.sha256()
    if name == "states":
        labels = trajectory["state_names"]
    elif name == "controls":
        labels = trajectory["control_names"]
    else:
        labels = ("time",)
    digest.update(json.dumps(labels, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _validate_roundtrip_channels(trajectory: dict[str, Any]) -> None:
    state_names = trajectory["state_names"]
    control_names = trajectory["control_names"]
    time = np.asarray(trajectory["time"])
    states = np.asarray(trajectory["states"])
    controls = np.asarray(trajectory["controls"])
    if (
        not state_names
        or not control_names
        or len(set(state_names)) != len(state_names)
        or len(set(control_names)) != len(control_names)
        or time.ndim != 1
        or states.shape != (len(time), len(state_names))
        or controls.shape != (len(time), len(control_names))
    ):
        raise ValueError("serialized solution changed trajectory shape or channels")


def _require_solution_roundtrip(audit: dict[str, Any]) -> None:
    if (
        not math.isfinite(audit["maximum_ulp_error"])
        or audit["maximum_ulp_error"] > _SERIALIZATION_MAX_ULPS
    ):
        raise ValueError(
            "serialized solution exceeds the predeclared 8 float64 ULP budget"
        )


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
    in_memory = _trajectory_arrays(solution)
    persisted_solution = osim.MocoTrajectory(str(solution_path))
    persisted = _trajectory_arrays(persisted_solution)
    roundtrip_audit = _audit_solution_roundtrip(in_memory, persisted)
    audit_path = output / "solution_serialization_audit.json"
    _write_json(audit_path, roundtrip_audit)
    _require_solution_roundtrip(roundtrip_audit)
    grid = persisted["time"]
    mesh_grid_audit_path = None
    if prepared.get("refinement") is not None:
        refinement = prepared["refinement"]
        child_request_refinement = prepared["config"].get("mesh_refinement")
        if not isinstance(child_request_refinement, dict):
            raise ValueError("child request omitted its refinement grid binding")
        grid_audit = _build_mesh_grid_audit(refinement, child_request_refinement, grid)
        mesh_grid_audit_path = output / "mesh_grid_audit.json"
        _write_json(mesh_grid_audit_path, grid_audit)
        if not grid_audit["exact_solution_grid_match"]:
            raise ValueError("optimized trajectory differs from the planned HS grid")
    control_names = persisted["control_names"]
    controls = {
        name.removeprefix("/forceset/"): persisted["controls"][:, index]
        for index, name in enumerate(control_names)
    }
    controls_path = output / "time_only_controls.sto"
    _write_time_only_controls(osim, grid, controls, controls_path)
    stages["solution_export_and_replay_input_staging"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)

    started = _stage_start()
    saved_grid, saved_controls = _load_time_only_controls(
        osim, controls_path, grid, tuple(controls), controls
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
        "solution": persisted_solution,
        "solver_solution": solution,
        "grid": saved_grid,
        "controls": saved_controls,
        "solution_path": solution_path,
        "controls_path": controls_path,
        "bundle": bundle,
        "serialization_audit_path": audit_path,
        "mesh_grid_audit_path": mesh_grid_audit_path,
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
    expected_controls: dict[str, np.ndarray],
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
    _validate_saved_control_data(
        grid, labels, controls, expected_grid, muscle_names, expected_controls
    )
    if labels != expected_labels:
        raise ValueError("saved time-only controls changed ordered muscle channels")
    return grid, controls


def _validate_saved_control_data(
    grid: np.ndarray,
    labels: tuple[str, ...],
    controls: dict[str, np.ndarray],
    expected_grid: np.ndarray,
    muscle_names: tuple[str, ...],
    expected_controls: dict[str, np.ndarray],
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
        if name not in expected_controls or not np.array_equal(
            values, expected_controls[name]
        ):
            raise ValueError(
                f"saved controls for {name} differ from persisted solution"
            )


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
    state_audit = _make_replay_state_audit(
        solved["solution"],
        replay.states,
        replay.state_names,
        grid,
        prepared["target_q"],
        prepared["declaration"].named_state[prepared["config"]["target_coordinate"]],
        prepared["declaration"].named_state,
    )
    replay_path = output / "fresh_replay.json"
    _record_replay_audit(
        replay_path,
        {
            "schema": "native-moco-fresh-replay-diagnostic-v1",
            "status": state_audit["status"],
            "fresh_model": True,
            "input_kind": "time_only_controls",
            "input_sha256": replay.input_sha256,
            "source_sha256": prepared["source_hash"],
            "request_sha256": _sha256(prepared["config_path"]),
            "solution_sha256": _sha256(solved["solution_path"]),
            "saved_controls_sha256": _sha256(solved["controls_path"]),
            "time_count": len(grid),
            **state_audit,
            **_replay_control_audit(replay, solved["controls"]),
            "native_constraint_audit_count": len(replay.constraint_audits),
            "physical_qualification": "unqualified",
        },
        require_pass=True,
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
    audit = _make_replay_state_audit(
        solution, replay_states, state_names, saved_grid, target_q, initial_q
    )
    _require_replay_audit(audit)
    return audit


def _make_replay_state_audit(
    solution: Any,
    replay_states: np.ndarray,
    state_names: tuple[str, ...],
    saved_grid: np.ndarray,
    target_q: float,
    initial_q: float,
    cold_start_named_state: dict[str, float] | None = None,
) -> dict[str, Any]:
    solution_grid = np.asarray(solution.getTimeMat(), dtype=np.float64).reshape(-1)
    optimized = np.column_stack(
        [
            np.asarray(solution.getStateMat(name), dtype=np.float64).reshape(-1)
            for name in state_names
        ]
    )
    replay_values = np.asarray(replay_states, dtype=np.float64)
    if optimized.shape != replay_values.shape or optimized.shape[0] != len(saved_grid):
        raise ValueError("optimized and fresh replay states differ in shape")
    if not np.isfinite(optimized).all() or not np.isfinite(replay_values).all():
        raise ValueError("optimized and fresh replay states fail finiteness checks")
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
    difference = np.abs(optimized - replay_values)
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
    max_error_times = {
        group: float(saved_grid[int(np.argmax(np.max(difference[:, indices], axis=1)))])
        for group, indices in groups.items()
    }
    coordinate_index = state_names.index("/jointset/base/q/value")
    optimized_endpoint = float(optimized[-1, coordinate_index])
    replay_endpoint = float(replay_values[-1, coordinate_index])
    endpoints = (optimized_endpoint, replay_endpoint)
    endpoint_residuals = {
        "target_q": target_q,
        "initial_q": initial_q,
        "optimized_endpoint_q": optimized_endpoint,
        "replayed_endpoint_q": replay_endpoint,
        "optimized_target_residual_rad": optimized_endpoint - target_q,
        "replayed_target_residual_rad": replay_endpoint - target_q,
        "optimized_initial_motion_rad": optimized_endpoint - initial_q,
        "replayed_initial_motion_rad": replay_endpoint - initial_q,
    }
    initial_state_residuals = None
    if cold_start_named_state is not None:
        if set(cold_start_named_state) != set(state_names):
            raise ValueError("cold-start state does not exactly cover replay channels")
        initial_state_residuals = {
            name: {
                "optimized_minus_declared": float(
                    optimized[0, index] - cold_start_named_state[name]
                ),
                "replayed_minus_declared": float(
                    replay_values[0, index] - cold_start_named_state[name]
                ),
            }
            for index, name in enumerate(state_names)
        }
    failures = [
        f"unit_error_exceeds:{group}"
        for group, limit in limits.items()
        if errors[group] >= limit
    ]
    if not np.array_equal(solution_grid, saved_grid):
        failures.append("optimized_clock_differs_from_saved_clock")
    if any(abs(value - target_q) >= 2e-7 for value in endpoints):
        failures.append("endpoint_target_residual_exceeds_2e-7_rad")
    if any((value - initial_q) * (target_q - initial_q) <= 0 for value in endpoints):
        failures.append("endpoint_moves_in_wrong_direction")
    if any(abs(value - initial_q) <= 2e-6 for value in endpoints):
        failures.append("endpoint_motion_not_nonzero")
    return {
        "status": "failed" if failures else "passed",
        "acceptance_failures": failures,
        "state_names": list(state_names),
        "times_seconds": saved_grid.tolist(),
        "replayed_state_matrix": replay_values.tolist(),
        "optimized_state_matrix_sha256": hashlib.sha256(
            np.ascontiguousarray(optimized, dtype=np.float64).tobytes()
        ).hexdigest(),
        "replayed_state_matrix_sha256": hashlib.sha256(
            np.ascontiguousarray(replay_values, dtype=np.float64).tobytes()
        ).hexdigest(),
        "absolute_state_error_by_time": difference.tolist(),
        "max_state_error_time_by_unit_seconds": max_error_times,
        "absolute_state_error_matrix_sha256": hashlib.sha256(
            np.ascontiguousarray(difference, dtype=np.float64).tobytes()
        ).hexdigest(),
        "initial_state_residuals_by_name": initial_state_residuals,
        "final_absolute_state_error_by_name": {
            name: float(difference[-1, index]) for index, name in enumerate(state_names)
        },
        "max_state_error_by_unit": errors,
        "state_error_limits_by_unit": limits,
        "optimized_endpoint_q": optimized_endpoint,
        "replayed_endpoint_q": replay_endpoint,
        "endpoint_residuals": endpoint_residuals,
        "target_endpoint_within_tolerance": not any(
            failure.startswith("endpoint_") for failure in failures
        ),
    }


def _require_replay_audit(audit: dict[str, Any]) -> None:
    failures = audit["acceptance_failures"]
    if failures:
        raise ValueError(
            "fresh replay failed acceptance checks: "
            f"{failures}; max state errors {audit['max_state_error_by_unit']}"
        )


def _record_replay_audit(
    path: Path, audit: dict[str, Any], *, require_pass: bool = True
) -> None:
    _write_json(path, audit)
    if require_pass:
        _require_replay_audit(audit)


def _replay_control_audit(
    replay: Any, saved_controls: dict[str, np.ndarray]
) -> dict[str, Any]:
    names = tuple(replay.muscle_names)
    expected = np.column_stack([saved_controls[name] for name in names])
    applied = np.asarray(replay.applied_excitations, dtype=np.float64)
    if applied.shape != expected.shape or not np.isfinite(applied).all():
        raise ValueError("replayed control samples have invalid shape or finiteness")
    difference = np.abs(applied - expected)
    return {
        "control_names": [f"/forceset/{name}" for name in names],
        "replayed_applied_controls": applied.tolist(),
        "saved_control_samples": expected.tolist(),
        "applied_controls_equal_saved_samples_exactly": bool(
            np.array_equal(applied, expected)
        ),
        "max_applied_control_sample_error": float(np.max(difference)),
        "replayed_applied_controls_sha256": hashlib.sha256(
            np.ascontiguousarray(applied, dtype=np.float64).tobytes()
        ).hexdigest(),
        "saved_control_samples_sha256": hashlib.sha256(
            np.ascontiguousarray(expected, dtype=np.float64).tobytes()
        ).hexdigest(),
    }


def _write_success_receipt(
    prepared: dict[str, Any], solved: dict, replay_path: Path, stages: dict, osim: Any
) -> None:
    solution = solved["solution"]
    solver_solution = solved["solver_solution"]
    coordinate = prepared["config"]["target_coordinate"]
    achieved_q = float(solution.getStateMat(coordinate)[-1])
    output = prepared["config_path"].parent
    replay_evidence = json.loads(
        (output / "fresh_replay.json").read_text(encoding="utf-8")
    )
    serialization_audit = json.loads(
        solved["serialization_audit_path"].read_text(encoding="utf-8")
    )
    runtime_path = _write_runtime_fingerprint(osim, output)
    objective = float(solver_solution.getObjective())
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
        "solver_status": str(solver_solution.getStatus()),
        "source_sha256": prepared["source_hash"],
        "worker_sha256": _sha256(Path(__file__).resolve()),
        "harness_builder_sha256": prepared["witness"]["builder_sha256"],
        "witness_provider_sha256": prepared["witness"]["provider_sha256"],
        "witness_states_sha256": prepared["witness"]["states_sha256"],
        "witness_controls_sha256": prepared["witness"]["controls_sha256"],
        "opensim_runtime_fingerprint_sha256": _sha256(runtime_path),
        "request_sha256": _sha256(prepared["config_path"]),
        "saved_solution_sha256": _sha256(solved["solution_path"]),
        "saved_time_only_controls_sha256": _sha256(solved["controls_path"]),
        "solution_serialization_audit_sha256": _sha256(
            solved["serialization_audit_path"]
        ),
        "solution_serialization_audit": serialization_audit,
        "fresh_replay_sha256": _sha256(replay_path),
        "stage_metrics": stages,
        "physical_qualification": "unqualified",
    }
    if prepared.get("refinement") is not None:
        refinement = prepared["refinement"]
        grid_path = solved.get("mesh_grid_audit_path")
        if grid_path is None or not grid_path.is_file():
            raise ValueError("refinement success omitted its native mesh-grid audit")
        receipt["mesh_refinement"] = {
            "attempt_id": refinement["attempt_id"],
            "parent_hashes": refinement["parent_hashes"],
            "analysis_sha256": refinement["analysis_sha256"],
            "mesh_grid_audit_sha256": _sha256(grid_path),
            "flagged_parent_cells": refinement["flagged_parent_cells"],
            "normalized_mesh_points": refinement["normalized_mesh_points"],
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
    parser.add_argument("--rung")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-existing", type=Path)
    parser.add_argument("--mesh-intervals", type=int, default=10)
    parser.add_argument("--refinement-plan", type=Path)
    args = parser.parse_args()
    if args.replay_existing is not None:
        if args.rung is not None:
            parser.error("--rung cannot be combined with --replay-existing")
        run_saved_replay_diagnostic(args.replay_existing, args.output)
        return
    if args.rung is None:
        parser.error("--rung is required unless --replay-existing is supplied")
    run_rung(
        args.rung,
        args.output,
        _validate_mesh_intervals(args.mesh_intervals),
        args.refinement_plan,
    )


if __name__ == "__main__":
    main()
