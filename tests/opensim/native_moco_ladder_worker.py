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
    started = _stage_start()
    witness = _build_feasible_witness(
        source, declaration, profile, wrapped, output / "witness", osim
    )
    stages["fixed_excitation_forward_witness"] = _stage_finish(started)
    _write_progress(output, rung.name, stages)
    target_q = witness["target_q"]
    initial_q = witness["initial_q"]
    config = _rung_config(rung, target_q, source_hash, initial_q)
    config.update(_witness_request_metadata(declaration, profile, witness))
    config_path = output / "request.json"
    _write_json(config_path, config)
    study, solver = _build_witness_seeded_problem(
        source, declaration, witness, target_q, osim
    )
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
        "solver": solver,
        "witness": witness,
    }


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
        "witness_time_knots": witness["times"].tolist(),
        "witness_excitation_by_channel": dict.fromkeys(witness["controls"], 0.05),
        "witness_profile": "constant-0.05-forward-integrated-v1",
        "witness_source_sha256": witness["source_sha256"],
        "cold_start_named_state": declaration.named_state,
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
    source: Path, declaration: Any, witness: dict[str, Any], target_q: float, osim: Any
) -> tuple[Any, Any]:
    from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
        MocoInitialBindings,
        apply_moco_initial_bindings,
    )

    if _sha256(source) != witness["source_sha256"]:
        raise ValueError("witness source changed before Moco problem construction")
    model = osim.Model(str(source))
    optimal = {
        f"/forceset/{model.getMuscles().get(index).getName()}": float(
            model.getMuscles().get(index).getOptimalFiberLength()
        )
        for index in range(model.getMuscles().getSize())
    }
    state_bounds = _synthetic_state_bounds(declaration.named_state, optimal)
    _validate_witness_inside_bounds(witness, state_bounds)
    bindings = MocoInitialBindings(
        state_bounds,
        declaration.named_state,
        dict.fromkeys(optimal, (0.0, 1.0)),
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
    _configure_solver(solver)
    _set_witness_guess(solver, witness, state_bounds, optimal, osim)
    return study, solver


def _validate_witness_inside_bounds(
    witness: dict[str, Any], state_bounds: dict[str, tuple[float, float]]
) -> None:
    for column, name in enumerate(witness["state_names"]):
        low, high = state_bounds[name]
        values = witness["states"][:, column]
        if np.any(values < low) or np.any(values > high):
            raise ValueError(f"forward witness leaves synthetic bounds for {name}")


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
        "harness_builder_sha256": prepared["witness"]["builder_sha256"],
        "witness_provider_sha256": prepared["witness"]["provider_sha256"],
        "witness_states_sha256": prepared["witness"]["states_sha256"],
        "witness_controls_sha256": prepared["witness"]["controls_sha256"],
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
