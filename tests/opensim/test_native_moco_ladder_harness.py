"""Contract tests for isolated Moco ladder orchestration (no native solve)."""

from __future__ import annotations

import sys
import os
import time
import hashlib
import json
from pathlib import Path

import pytest

from tests.opensim.native_moco_ladder_harness import (
    DEFAULT_RUNGS,
    LadderRung,
    _WorkerTimeout,
    _run_one_rung,
    _validate_bounded_runtime_fingerprint,
    _validate_request_rung_binding,
    _process_stop_reason,
    run_ladder,
)

pytestmark = pytest.mark.unit


def test_default_ladder_adds_one_feature_per_rung() -> None:
    assert tuple(rung.name for rung in DEFAULT_RUNGS) == (
        "plain",
        "nonlinear_joint",
        "moving_path",
        "conditional_path",
        "wrap",
        "coupler",
    )
    assert all(
        len(current.features) - len(previous.features) == 1
        and set(previous.features) < set(current.features)
        for previous, current in zip(DEFAULT_RUNGS, DEFAULT_RUNGS[1:], strict=False)
    )


def test_mesh_refinement_request_keeps_other_solver_contract_fixed() -> None:
    from tests.opensim.native_moco_ladder_harness import worker_command
    from tests.opensim.native_moco_ladder_worker import _rung_config

    baseline = _rung_config(DEFAULT_RUNGS[0], 0.300005, "a" * 64, 0.3)
    refined = _rung_config(DEFAULT_RUNGS[0], 0.300005, "a" * 64, 0.3, mesh_intervals=20)
    assert baseline["mesh_intervals"] == 10
    assert refined["mesh_intervals"] == 20
    assert "--mesh-intervals" not in worker_command(
        DEFAULT_RUNGS[0], Path("plain"), "worker-python", 10
    )
    assert worker_command(DEFAULT_RUNGS[0], Path("plain"), "worker-python", 20)[
        -2:
    ] == (
        "--mesh-intervals",
        "20",
    )
    for config in (baseline, refined):
        assert config["horizon_seconds"] == 0.01
        assert config["target_q"] == baseline["target_q"]
        assert config["initial_q"] == baseline["initial_q"]
        assert config["transcription_scheme"] == "hermite-simpson"
        assert config["interpolate_control_mesh_interior_points"] is True
        assert config["replay_control_interpolation"] == "piecewise_linear"
        assert config["convergence_tolerance"] == 1e-6
        assert config["constraint_tolerance"] == 1e-7


def test_bounded_refinement_marks_one_parent_cell_and_preserves_contract(
    tmp_path: Path,
) -> None:
    from tests.opensim.native_moco_refinement import build_refinement_plan

    parent = tmp_path / "parent"
    parent.mkdir()
    source = parent / "source.osim"
    request_path = parent / "request.json"
    solution = parent / "solution.sto"
    controls = parent / "time_only_controls.sto"
    source.write_bytes(b"source")
    solution.write_bytes(b"solution")
    controls.write_bytes(b"controls")
    request = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mesh_intervals": 20,
        "horizon_seconds": 0.01,
        "target_coordinate": "/jointset/base/q/value",
        "target_q": 0.300005,
        "initial_q": 0.3,
        "transcription_scheme": "hermite-simpson",
        "interpolate_control_mesh_interior_points": True,
        "replay_control_interpolation": "piecewise_linear",
        "parallel": 0,
        "maximum_iterations": 300,
        "convergence_tolerance": 1e-6,
        "constraint_tolerance": 1e-7,
        "native_control_bounds": {"/forceset/flexor": [0.01, 1.0]},
        "native_control_bounds_source": {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
        },
        "cold_start_named_state": {"/forceset/flexor/activation": 0.05},
        "physical_qualification": "unqualified",
        "solution_serialization_max_ulps": 8,
    }
    request_bytes = json.dumps(request).encode("utf-8")
    request_path.write_bytes(request_bytes)
    report_dir = tmp_path / "analysis"
    audit_dir = (
        report_dir / "diagnostic_retained_replay_fauldump_3e195b0e" / "native_audit"
    )
    audit_dir.mkdir(parents=True)
    audit = audit_dir / "fresh_replay_diagnostic.json"
    audit.write_bytes(b"audit")
    channels = {}
    for name in ("flexor", "extensor"):
        pairs = [[0.0, 0.0] for _ in range(40)]
        channels[name] = {
            "full_cell_local_vs_accumulated_allocation": {
                "local_scalar_minus_optimized_midpoint_endpoint": pairs[:20],
                "cell_count": 20,
            }
        }
    channels["flexor"]["full_cell_local_vs_accumulated_allocation"][
        "local_scalar_minus_optimized_midpoint_endpoint"
    ][0] = [1.15e-4, 9.4e-5]
    report = {
        "input_sha256": {
            "source_osim": hashlib.sha256(source.read_bytes()).hexdigest(),
            "request": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "solution_sto": hashlib.sha256(solution.read_bytes()).hexdigest(),
            "time_only_controls_sto": hashlib.sha256(controls.read_bytes()).hexdigest(),
            "replay_audit": hashlib.sha256(audit.read_bytes()).hexdigest(),
        },
        "sample_count": 41,
        "interval_count": 40,
        "channels": channels,
    }
    report_path = report_dir / "activation_residual_analysis.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    plan = build_refinement_plan(parent, report_path)
    from tests.opensim.native_moco_ladder_worker import (
        _actual_solver_settings,
        _refinement_request_fields,
        _rung_config,
        _validate_preparation_request_bindings,
        _validate_refinement_request,
    )
    from tests.opensim.native_moco_refinement import child_hs_grid
    from types import SimpleNamespace

    _validate_refinement_request(
        plan, request["source_sha256"], plan["preserved_contract"]
    )
    reloaded_plan = json.loads(json.dumps(plan))
    _validate_refinement_request(
        reloaded_plan, request["source_sha256"], plan["preserved_contract"]
    )
    request_config = _rung_config(
        DEFAULT_RUNGS[0],
        request["target_q"],
        request["source_sha256"],
        request["initial_q"],
        mesh_intervals=len(plan["normalized_mesh_points"]) - 1,
    )
    request_config["mesh_refinement"] = _refinement_request_fields(plan)
    request_config["mesh_refinement"]["native_guess_time_grid_seconds"] = list(
        child_hs_grid(plan["normalized_mesh_points"], request["horizon_seconds"])
    )
    request_config["replay_control_interpolation"] = "piecewise_linear"
    request_config["actual_solver_settings"] = _actual_solver_settings(
        SimpleNamespace(
            get_transcription_scheme=lambda: "hermite-simpson",
            get_interpolate_control_mesh_interior_points=lambda: True,
            get_parallel=lambda: 0,
            get_num_mesh_intervals=lambda: 100,
        ),
        request_config["mesh_refinement"]["normalized_mesh_points"],
    )
    _validate_preparation_request_bindings(plan, request_config)
    request_config["actual_solver_settings"]["set_mesh_requested_normalized_points"][
        1
    ] += 0.001
    with pytest.raises(ValueError, match="differ from the bound request"):
        _validate_preparation_request_bindings(plan, request_config)
    request_config["actual_solver_settings"]["set_mesh_requested_normalized_points"] = (
        list(plan["normalized_mesh_points"])
    )
    request_config["mesh_refinement"]["native_guess_time_grid_seconds"][1] += 1e-12
    with pytest.raises(ValueError, match="native guess clock"):
        _validate_preparation_request_bindings(plan, request_config)
    for field, mutate in (
        (
            "normalized_mesh_points",
            lambda changed: changed.__setitem__(
                "normalized_mesh_points", [0.0, 0.05, 1.0]
            ),
        ),
        (
            "parent_hashes",
            lambda changed: changed["parent_hashes"].__setitem__(
                "source_osim", "0" * 64
            ),
        ),
        (
            "preserved_contract",
            lambda changed: changed["preserved_contract"].__setitem__("target_q", 99.0),
        ),
    ):
        changed = json.loads(json.dumps(plan))
        mutate(changed)
        assert changed["attempt_id"] == plan["attempt_id"], field
        with pytest.raises(ValueError, match="plan no longer matches"):
            _validate_refinement_request(
                changed, request["source_sha256"], plan["preserved_contract"]
            )
    assert plan["flagged_parent_cells"] == [0]
    assert plan["normalized_mesh_points"][:4] == (0.0, 0.025, 0.05, 0.1)
    assert len(plan["normalized_mesh_points"]) - 1 == 21
    assert plan["preserved_contract"]["initial_q"] == request["initial_q"]
    assert plan["preserved_contract"]["objective"] == "MocoControlGoal(effort)"
    assert plan["control_seed"]["interpolation"] == "piecewise_linear"
    assert "lower local activation" in plan["interpretation"]
    errors = channels["extensor"]["full_cell_local_vs_accumulated_allocation"][
        "local_scalar_minus_optimized_midpoint_endpoint"
    ]
    errors[7] = [2e-8, 0.0]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    interior_plan = build_refinement_plan(parent, report_path)
    assert interior_plan["flagged_parent_cells"] == [0, 7]
    assert 0.375 in interior_plan["normalized_mesh_points"]
    assert len(interior_plan["normalized_mesh_points"]) - 1 == 22
    errors[1:5] = [[2e-8, 0.0] for _ in range(4)]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="cell cap"):
        build_refinement_plan(parent, report_path)
    errors[1:5] = [[0.0, 0.0] for _ in range(4)]
    errors[7] = [0.0, 0.0]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    request_path.write_bytes(request_bytes + b" ")
    with pytest.raises(ValueError, match="parent inputs"):
        build_refinement_plan(parent, report_path)
    request_path.write_bytes(request_bytes)
    controls.write_bytes(b"changed controls")
    with pytest.raises(ValueError, match="parent inputs"):
        build_refinement_plan(parent, report_path)
    controls.write_bytes(b"controls")
    source.write_bytes(b"changed parent")
    with pytest.raises(ValueError, match="parent inputs"):
        build_refinement_plan(parent, report_path)


def test_refinement_mesh_and_piecewise_linear_control_contracts() -> None:
    from tests.opensim.native_moco_refinement import (
        child_hs_grid,
        make_refined_mesh,
        piecewise_linear_samples,
        validate_normalized_mesh,
    )

    mesh = validate_normalized_mesh([0.0, 0.025, 0.05, 0.1, 1.0])
    assert child_hs_grid(mesh, 0.01)[:4] == pytest.approx(
        (0.0, 0.000125, 0.00025, 0.000375)
    )
    _, interior = make_refined_mesh(20, [7])
    assert 0.375 in interior
    assert len(interior) - 1 == 21
    assert piecewise_linear_samples([0.0, 1.0], [0.0, 2.0], [0.0, 0.25, 1.0]) == (
        0.0,
        0.5,
        2.0,
    )
    for invalid in ([0.0, 1.0, 0.5], [0.0, float("nan"), 1.0], [0.1, 1.0]):
        with pytest.raises(ValueError):
            validate_normalized_mesh(invalid)


def test_refinement_grid_audit_reads_seed_grid_from_child_request() -> None:
    from tests.opensim.native_moco_ladder_worker import _build_mesh_grid_audit
    from tests.opensim.native_moco_refinement import child_hs_grid

    plan = {
        "attempt_id": "attempt",
        "parent_hashes": {"source_osim": "a" * 64},
        # The planner retains an immutable tuple; the configured request is
        # JSON-native and therefore uses a list. Their numeric sequence must
        # match exactly at the post-solve audit boundary.
        "normalized_mesh_points": (0.0, 0.5, 1.0),
        "parent_request": {"horizon_seconds": 0.01},
    }
    grid = child_hs_grid(plan["normalized_mesh_points"], 0.01)
    child_request = {
        "attempt_id": "attempt",
        "parent_hashes": plan["parent_hashes"],
        "normalized_mesh_points": list(plan["normalized_mesh_points"]),
        "native_guess_time_grid_seconds": list(grid),
    }
    audit = _build_mesh_grid_audit(plan, child_request, grid)
    assert audit["seed_time_grid_seconds"] == list(grid)
    assert audit["exact_solution_grid_match"] is True
    assert "native_guess_time_grid_seconds" not in plan
    changed_request = dict(child_request)
    changed_request["normalized_mesh_points"] = [0.0, 0.25, 1.0]
    with pytest.raises(ValueError, match="changed the retained refinement plan"):
        _build_mesh_grid_audit(plan, changed_request, grid)


def test_refinement_supervisor_stub_uses_exact_single_rung_protocol(
    tmp_path: Path,
) -> None:
    from tests.opensim.native_moco_refinement import build_refinement_plan
    from tests.opensim.native_moco_refinement_supervisor import run_stub

    repository = Path(__file__).resolve().parents[2]
    evidence_root = (
        repository
        / "docs/development/feedback_controls/experiments/native_moco_mesh20_e1548c5179"
    )
    plan = build_refinement_plan(
        evidence_root / "plain", evidence_root / "activation_residual_analysis.json"
    )
    plan_path = tmp_path / "exact-plan.json"
    plan_path.write_text(
        json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "stub-output"
    result = run_stub(plan_path, output, "sdk-python", repository)
    command = result["worker_command"]
    assert command[:5] == [
        "sdk-python",
        "-m",
        "tests.opensim.native_moco_ladder_worker",
        "--rung",
        "plain",
    ]
    assert command[command.index("--mesh-intervals") + 1] == "21"
    assert Path(command[command.index("--output") + 1]) == output / "plain"
    staged_plan = Path(command[command.index("--refinement-plan") + 1])
    assert staged_plan.read_bytes() == plan_path.read_bytes()
    assert result["sdk_launched"] is False
    assert result["status"] == "stub_preflight_pass_no_sdk_launch"
    assert not (output / "plain").exists()


def test_parent_refinement_guess_exports_state_table_and_restores_exact_initial(
    tmp_path: Path,
) -> None:
    from tests.opensim.native_moco_ladder_worker import _set_parent_solution_guess

    parent = tmp_path / "parent"
    parent.mkdir()
    solution_path = parent / "solution.sto"
    controls_path = parent / "time_only_controls.sto"
    solution_path.write_bytes(b"parent solution")
    controls_path.write_bytes(b"parent controls")
    state_name = "/forceset/flexor/activation"

    class FakeTable:
        def __init__(self, labels, values, times):
            self.labels = tuple(labels)
            self.values = values
            self.times = tuple(times)

        def getColumnLabels(self):
            return self.labels

        def getIndependentColumn(self):
            return self.times

        def getDependentColumn(self, name):
            return self.values[name]

        def getNumRows(self):
            return len(self.times)

    state_table = FakeTable(
        (state_name,), {state_name: (0.05000000000000001, 0.06)}, (0.0, 0.01)
    )
    control_table = FakeTable(
        ("/forceset/flexor",), {"/forceset/flexor": (0.1, 0.2)}, (0.0, 0.01)
    )

    class FakeTrajectory:
        def __init__(self, _path):
            self.exported_table = state_table

        def exportToStatesTable(self):
            return self.exported_table

        def getStateNames(self):
            return (state_name,)

    class FakeGuess:
        def __init__(self):
            self.states = {state_name: (0.05000000000000001, 0.055, 0.06)}
            self.controls = {}

        def insertStatesTrajectory(self, table, interpolate):
            if not isinstance(table, FakeTable):
                raise TypeError("insertStatesTrajectory expects TimeSeriesTable")
            assert interpolate is True
            self.inserted_state_table = table

        def getTime(self):
            return (0.0, 0.005, 0.01)

        def getNumTimes(self):
            return 3

        def getControlNames(self):
            return ("/forceset/flexor",)

        def getStateNames(self):
            return (state_name,)

        def getState(self, name):
            return self.states[name]

        def setState(self, name, values):
            self.states[name] = tuple(values)

        def setControl(self, name, values):
            self.controls[name] = tuple(values)

        def getControl(self, name):
            return self.controls[name]

    class FakeSolver:
        def createGuess(self):
            self.guess = FakeGuess()
            return self.guess

        def setGuess(self, guess):
            self.guess = guess

    class FakeOpenSim:
        MocoTrajectory = FakeTrajectory
        TimeSeriesTable = staticmethod(lambda _path: control_table)
        Vector = staticmethod(lambda size, initial: [initial] * size)

    refinement = {
        "parent_hashes": {
            "solution_sto": hashlib.sha256(solution_path.read_bytes()).hexdigest(),
            "time_only_controls_sto": hashlib.sha256(
                controls_path.read_bytes()
            ).hexdigest(),
        },
        "normalized_mesh_points": (0.0, 1.0),
        "parent_request": {
            "horizon_seconds": 0.01,
            "native_control_bounds": {"/forceset/flexor": [0.01, 1.0]},
        },
        "preserved_contract": {"cold_start_named_state": {state_name: 0.05}},
    }
    solver = FakeSolver()
    result = _set_parent_solution_guess(solver, parent, refinement, FakeOpenSim)
    assert result == (0.0, 0.005, 0.01)
    assert solver.guess.inserted_state_table is state_table
    assert solver.guess.states[state_name][0] == 0.05
    assert solver.guess.states[state_name][1:] == (0.055, 0.06)


def test_actual_parent_refinement_sdk_seed_uses_native_trajectory_api() -> None:
    import numpy as np

    osim = pytest.importorskip("opensim")
    from tests.opensim.native_moco_ladder_harness import _sha256
    from tests.opensim.native_moco_refinement import (
        build_refinement_plan,
        child_hs_grid,
    )
    from tests.opensim.native_moco_ladder_worker import _set_parent_solution_guess

    repository = Path(__file__).resolve().parents[2]
    parent = (
        repository
        / "docs/development/feedback_controls/experiments/native_moco_mesh20_e1548c5179/plain"
    )
    analysis = parent.parent / "activation_residual_analysis.json"
    plan = build_refinement_plan(parent, analysis)
    assert plan["flagged_parent_cells"] == [0]
    assert _sha256(parent / "solution.sto") == plan["parent_hashes"]["solution_sto"]
    solution = osim.MocoTrajectory(str(parent / "solution.sto"))
    state_names = tuple(str(value) for value in solution.getStateNames())
    control_names = tuple(str(value) for value in solution.getControlNames())
    multiplier_names = tuple(str(value) for value in solution.getMultiplierNames())
    parameter_names = tuple(str(value) for value in solution.getParameterNames())
    times = child_hs_grid(
        plan["normalized_mesh_points"], plan["parent_request"]["horizon_seconds"]
    )
    time_vector = osim.Vector(len(times), 0.0)
    for index, value in enumerate(times):
        time_vector[index] = value
    state_values = osim.Matrix(len(times), len(state_names), 0.0)
    control_values = osim.Matrix(len(times), len(control_names), 0.0)
    multiplier_values = osim.Matrix(len(times), len(multiplier_names), 0.0)
    parameter_values = osim.RowVector(len(parameter_names), 0.0)
    guess = osim.MocoTrajectory(
        time_vector,
        list(state_names),
        list(control_names),
        list(multiplier_names),
        list(parameter_names),
        state_values,
        control_values,
        multiplier_values,
        parameter_values,
    )

    class StubSolver:
        def createGuess(self):
            return guess

        def setGuess(self, value):
            self.guess = value

    solver = StubSolver()
    seeded_times = _set_parent_solution_guess(solver, parent, plan, osim)
    assert len(seeded_times) == 43
    assert (
        tuple(
            float(solver.guess.getTime()[index])
            for index in range(solver.guess.getNumTimes())
        )
        == times
    )
    assert tuple(str(value) for value in solver.guess.getStateNames()) == state_names
    assert (
        tuple(str(value) for value in solver.guess.getControlNames()) == control_names
    )
    for name, value in plan["preserved_contract"]["cold_start_named_state"].items():
        assert float(solver.guess.getState(name)[0]) == value


def test_committed_activation_analysis_reproduces_retained_parent_bindings(
    tmp_path: Path,
) -> None:
    from scripts.opensim.analyze_retained_activation_residual import analyze

    repository = Path(__file__).resolve().parents[2]
    experiment = (
        repository
        / "docs/development/feedback_controls/experiments/native_moco_mesh20_e1548c5179/plain"
    )
    audit = (
        experiment.parent
        / "diagnostic_retained_replay_fauldump_3e195b0e"
        / "native_audit"
        / "fresh_replay_diagnostic.json"
    )
    result = analyze(experiment, audit, tmp_path / "reproduced.json")
    assert result["status"] == "offline_diagnostic_only"
    assert (
        result["input_sha256"]["source_osim"]
        == hashlib.sha256((experiment / "source.osim").read_bytes()).hexdigest()
    )
    allocation = result["channels"]["flexor"][
        "full_cell_local_vs_accumulated_allocation"
    ]
    assert allocation["cell_count"] == 20
    assert allocation["local_scalar_minus_optimized_midpoint_endpoint"][0][
        0
    ] == pytest.approx(1.1531598344965976e-4, rel=1e-10)


@pytest.mark.parametrize("value", [True, 0, -1, 101, 1.5])
def test_mesh_interval_count_rejects_invalid_values(value: object) -> None:
    from tests.opensim.native_moco_ladder_harness import _validate_mesh_intervals

    with pytest.raises(ValueError, match="mesh interval count"):
        _validate_mesh_intervals(value)


def test_moving_path_spline_is_bounded_monotone_and_curved() -> None:
    from tests.opensim.native_moco_ladder_worker import _rung_config

    config = _rung_config(DEFAULT_RUNGS[2], 0.300005, "a" * 64, 0.3)
    knots = config["moving_path_simm_spline_knots"]
    values = config["moving_path_simm_spline_values"]
    slopes = [
        (values[index + 1] - values[index]) / (knots[index + 1] - knots[index])
        for index in range(len(knots) - 1)
    ]
    assert all(-0.02 <= value <= 0.02 for value in values)
    assert all(slope > 0 for slope in slopes)
    assert slopes[0] != slopes[1]


def test_staged_model_flush_uses_writable_descriptor(tmp_path: Path) -> None:
    from tests.opensim.native_moco_ladder_worker import _flush_file

    path = tmp_path / "staged.osim"
    path.write_bytes(b"native model bytes")
    _flush_file(path)


def test_native_loaded_mpp_curve_rejects_collinear_law(tmp_path: Path) -> None:
    import shutil
    import defusedxml.ElementTree as ET

    from tests.opensim.native_moco_ladder_harness import build_synthetic_rung
    from tests.opensim.native_moco_ladder_harness import _verify_native_mpp_curvature

    osim = pytest.importorskip("opensim")
    path, _ = build_synthetic_rung(DEFAULT_RUNGS[2], tmp_path / "curved")
    staged = tmp_path / "staged.osim"
    shutil.copyfile(path, staged)
    from tests.opensim.native_moco_ladder_worker import _flush_file

    _flush_file(staged)
    model = osim.Model(str(staged))
    _verify_native_mpp_curvature(model, model.initSystem(), osim)

    tree = ET.parse(staged)
    law = tree.find(".//MovingPathPoint[@name='guide']/x_location/SimmSpline/y")
    assert law is not None
    law.text = "-0.01 0 0.01"
    tree.write(staged, encoding="utf-8", xml_declaration=True)
    linear_model = osim.Model(str(staged))
    with pytest.raises(ValueError, match="curvature"):
        _verify_native_mpp_curvature(linear_model, linear_model.initSystem(), osim)


def test_synthetic_fiber_bounds_are_finite_and_strictly_positive() -> None:
    from tests.opensim.native_moco_ladder_worker import _synthetic_state_bounds

    initial = {
        "/forceset/flexor/fiber_length": 0.095,
        "/forceset/flexor/activation": 0.05,
        "/jointset/base/q/value": 0.3,
    }
    optimal = {"/forceset/flexor": 0.1}
    bounds = _synthetic_state_bounds(initial, optimal)
    low, high = bounds["/forceset/flexor/fiber_length"]
    assert 0.0 < low < initial["/forceset/flexor/fiber_length"] < high
    assert high < float("inf")
    assert bounds["/forceset/flexor/activation"] == (0.01, 0.15)


def test_native_control_bounds_reject_invalid_or_uncovered_channels() -> None:
    from tests.opensim.native_moco_ladder_worker import _native_control_bounds

    class Muscle:
        def __init__(self, name: str, lower: object, upper: object) -> None:
            self.name = name
            self.lower = lower
            self.upper = upper

        def getAbsolutePathString(self) -> str:
            return f"/forceset/{self.name}"

        def getMinControl(self) -> object:
            return self.lower

        def getMaxControl(self) -> object:
            return self.upper

    class Collection:
        def __init__(self, items: list[Muscle]) -> None:
            self.items = items

        def getSize(self) -> int:
            return len(self.items)

        def get(self, index: int) -> Muscle:
            return self.items[index]

    class Model:
        def __init__(self, items: list[Muscle]) -> None:
            self.items = items

        def getMuscles(self) -> Collection:
            return Collection(self.items)

    expected = {"/forceset/flexor", "/forceset/extensor"}
    valid = Model([Muscle("flexor", 0.02, 0.8), Muscle("extensor", 0.03, 0.9)])
    assert _native_control_bounds(valid, expected) == {
        "/forceset/flexor": (0.02, 0.8),
        "/forceset/extensor": (0.03, 0.9),
    }
    for bad in (
        Model([Muscle("flexor", float("nan"), 0.8), Muscle("extensor", 0.03, 0.9)]),
        Model([Muscle("flexor", 0.8, 0.2), Muscle("extensor", 0.03, 0.9)]),
        Model([Muscle("flexor", -0.01, 0.8), Muscle("extensor", 0.03, 0.9)]),
        Model([Muscle("flexor", 0.02, 0.8)]),
    ):
        with pytest.raises((TypeError, ValueError)):
            _native_control_bounds(bad, expected)


def test_forward_witness_excitations_must_fit_native_channel_bounds() -> None:
    import numpy as np

    from tests.opensim.native_moco_ladder_worker import (
        _validate_witness_control_bounds,
    )

    witness = {"controls": {"flexor": np.array([0.05, 0.05])}}
    _validate_witness_control_bounds(witness, {"/forceset/flexor": (0.01, 0.8)})
    with pytest.raises(ValueError, match="leaves native bounds"):
        _validate_witness_control_bounds(witness, {"/forceset/flexor": (0.06, 0.8)})


def test_native_source_control_bounds_are_applied_exactly(
    tmp_path: Path,
) -> None:
    import shutil
    import defusedxml.ElementTree as ET

    from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
        MocoInitialBindings,
        apply_moco_initial_bindings,
    )
    from tests.opensim.native_moco_ladder_harness import build_synthetic_rung
    from tests.opensim.native_moco_ladder_worker import _native_control_bounds

    osim = pytest.importorskip("opensim")
    source, _ = build_synthetic_rung(DEFAULT_RUNGS[0], tmp_path / "original")
    changed = tmp_path / "nondefault_bounds.osim"
    shutil.copyfile(source, changed)
    tree = ET.parse(changed)
    for muscle in tree.findall(".//Millard2012EquilibriumMuscle"):
        muscle.find("min_control").text = "0.02"
        muscle.find("max_control").text = "0.8"
    tree.write(changed, encoding="utf-8", xml_declaration=True)
    model = osim.Model(str(changed))
    paths = {"/forceset/flexor", "/forceset/extensor"}
    control_bounds = _native_control_bounds(model, paths)
    assert control_bounds == dict.fromkeys(paths, (0.02, 0.8))

    model.initSystem()
    state_names = model.getStateVariableNames()
    states = {str(state_names.get(i)): 0.0 for i in range(state_names.getSize())}
    bindings = MocoInitialBindings(
        dict.fromkeys(states, (-1000000.0, 1000000.0)),
        states,
        control_bounds,
    )

    class CapturingProblem:
        def __init__(self) -> None:
            self.controls: dict[str, tuple[float, float]] = {}

        def setStateInfo(self, *_args: object) -> None:
            pass

        def setControlInfo(self, name: str, bounds: object) -> None:
            self.controls[name] = (float(bounds.getLower()), float(bounds.getUpper()))

    problem = CapturingProblem()
    apply_moco_initial_bindings(problem, str(changed), bindings, osim)
    assert problem.controls == control_bounds


def test_forward_witness_has_exact_saved_clock_and_nonzero_endpoint(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
        declare_native_mechanics_execution_profile,
    )
    from tests.opensim.native_moco_ladder_harness import build_synthetic_rung
    from tests.opensim.native_moco_ladder_worker import (
        _actual_solver_settings,
        _build_feasible_witness,
        _build_witness_seeded_problem,
        _set_witness_guess,
        _rung_config,
        _synthetic_state_bounds,
        _witness_request_metadata,
        _validate_actual_solver_settings,
    )

    osim = pytest.importorskip("opensim")
    path, declaration = build_synthetic_rung(DEFAULT_RUNGS[0], tmp_path / "source")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    profile = declare_native_mechanics_execution_profile(path, source_hash)
    witness = _build_feasible_witness(
        path, declaration, profile, None, tmp_path / "witness", osim
    )
    metadata = _witness_request_metadata(declaration, profile, witness)
    config = _rung_config(
        DEFAULT_RUNGS[0], witness["target_q"], source_hash, witness["initial_q"]
    )
    config.update(metadata)
    round_trip = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    assert round_trip["cold_start_named_state"] == dict(declaration.named_state)
    assert round_trip["cold_start_state_units"] == metadata["cold_start_state_units"]
    assert round_trip["native_mechanics_profile"] == json.loads(
        json.dumps(metadata["native_mechanics_profile"])
    )
    assert round_trip["native_mechanics_profile_sha256"] == profile.profile_sha256
    assert round_trip["solution_serialization_max_ulps"] == 8
    assert metadata["cold_start_named_state"] == declaration.named_state
    assert set(metadata["cold_start_state_units"]) == set(declaration.named_state)
    assert metadata["native_mechanics_profile"]["source_sha256"] == source_hash
    assert metadata["witness_source_sha256"] == source_hash
    assert witness["times"].shape == (11,)
    assert witness["times"][0] == 0.0
    assert witness["times"][-1] == 0.01
    assert witness["source_sha256"] == source_hash
    assert witness["target_q"] - witness["initial_q"] > 2e-6
    assert all((values == 0.05).all() for values in witness["controls"].values())
    model = osim.Model(str(path))
    optimal = {
        f"/forceset/{model.getMuscles().get(index).getName()}": float(
            model.getMuscles().get(index).getOptimalFiberLength()
        )
        for index in range(model.getMuscles().getSize())
    }
    initial = {
        name: float(witness["states"][0, index])
        for index, name in enumerate(witness["state_names"])
    }
    bounds = _synthetic_state_bounds(initial, optimal)
    controls_bytes = witness["control_path"].read_bytes()
    witness["control_path"].write_bytes(controls_bytes + b"tamper")
    with pytest.raises(ValueError, match="changed"):
        _set_witness_guess(object(), witness, bounds, optimal, osim)
    witness["control_path"].write_bytes(controls_bytes)
    _, solver = _build_witness_seeded_problem(
        path, declaration, witness, witness["target_q"], osim
    )
    settings = _actual_solver_settings(solver)
    assert settings == {
        "mesh_intervals": 10,
        "transcription_scheme": "hermite-simpson",
        "interpolate_control_mesh_interior_points": True,
        "parallel": 0,
    }
    assert set(solver.getGuess().getStateNames()) == set(witness["state_names"])
    assert set(solver.getGuess().getControlNames()) == {
        "/forceset/flexor",
        "/forceset/extensor",
    }
    _, refined_solver = _build_witness_seeded_problem(
        path,
        declaration,
        witness,
        witness["target_q"],
        osim,
        mesh_intervals=20,
    )
    refined_settings = _actual_solver_settings(refined_solver)
    assert refined_settings["mesh_intervals"] == 20
    assert refined_settings["transcription_scheme"] == "hermite-simpson"
    assert refined_settings["interpolate_control_mesh_interior_points"] is True
    assert refined_settings["parallel"] == 0
    refined_request = _rung_config(
        DEFAULT_RUNGS[0],
        witness["target_q"],
        source_hash,
        witness["initial_q"],
        mesh_intervals=20,
    )
    refined_request["actual_solver_settings"] = refined_settings
    _validate_actual_solver_settings(refined_request)
    refined_request["actual_solver_settings"]["mesh_intervals"] = 10
    with pytest.raises(ValueError, match="differ from the bound request"):
        _validate_actual_solver_settings(refined_request)


def test_custom_mesh_does_not_misreport_uniform_interval_getter() -> None:
    from tests.opensim.native_moco_ladder_worker import (
        _actual_solver_settings,
        _configure_solver,
        _validate_actual_solver_settings,
    )

    points = [0.0, 0.25, 0.5, 0.75, 1.0]

    class FakeCustomMeshSolver:
        def __init__(self) -> None:
            self.mesh = None
            self.uniform_intervals = 100

        def setMesh(self, values: list[float]) -> None:
            self.mesh = list(values)

        def set_num_mesh_intervals(self, _value: int) -> None:
            raise AssertionError("custom mesh must not mutate uniform interval count")

        def get_num_mesh_intervals(self) -> int:
            return self.uniform_intervals

        def set_transcription_scheme(self, _value: str) -> None:
            pass

        def get_transcription_scheme(self) -> str:
            return "hermite-simpson"

        def set_interpolate_control_mesh_interior_points(self, _value: bool) -> None:
            pass

        def get_interpolate_control_mesh_interior_points(self) -> bool:
            return True

        def set_parallel(self, _value: int) -> None:
            pass

        def get_parallel(self) -> int:
            return 0

        def set_optim_max_iterations(self, _value: int) -> None:
            pass

        def set_optim_convergence_tolerance(self, _value: float) -> None:
            pass

        def set_optim_constraint_tolerance(self, _value: float) -> None:
            pass

    solver = FakeCustomMeshSolver()
    _configure_solver(solver, 4, normalized_mesh_points=points)
    assert solver.mesh == points
    with pytest.raises(ValueError, match="interval count differs"):
        _configure_solver(solver, 5, normalized_mesh_points=points)
    settings = _actual_solver_settings(solver, points)
    assert settings == {
        "transcription_scheme": "hermite-simpson",
        "interpolate_control_mesh_interior_points": True,
        "parallel": 0,
        "set_mesh_requested_normalized_points": points,
        "set_mesh_requested_interval_count": 4,
        "uniform_mesh_interval_getter_ignored": 100,
    }
    request = {
        "mesh_intervals": 4,
        "transcription_scheme": "hermite-simpson",
        "interpolate_control_mesh_interior_points": True,
        "parallel": 0,
        "mesh_refinement": {"normalized_mesh_points": points},
        "actual_solver_settings": settings,
    }
    _validate_actual_solver_settings(request)
    request["actual_solver_settings"]["set_mesh_requested_normalized_points"] = [
        0.0,
        0.5,
        1.0,
    ]
    with pytest.raises(ValueError, match="differ from the bound request"):
        _validate_actual_solver_settings(request)


def test_solver_settings_mismatch_persists_requested_and_actual_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.opensim import native_moco_ladder_worker as worker

    config = {
        "mesh_intervals": 10,
        "transcription_scheme": "hermite-simpson",
        "interpolate_control_mesh_interior_points": True,
        "parallel": 0,
    }
    actual = {
        "mesh_intervals": 10,
        "transcription_scheme": "hermite-simpson",
        "interpolate_control_mesh_interior_points": True,
        "parallel": 1,
    }
    monkeypatch.setattr(
        worker, "_build_witness_seeded_problem", lambda *_a, **_k: (object(), object())
    )
    monkeypatch.setattr(worker, "_actual_solver_settings", lambda *_a, **_k: actual)

    with pytest.raises(ValueError, match="differ from the bound request"):
        worker._configure_rung_problem(
            worker.DEFAULT_RUNGS[0],
            tmp_path / "source.osim",
            object(),
            {},
            0.300005,
            {},
            config,
            tmp_path,
            None,
            object(),
            {},
            10,
        )

    attempt = json.loads((tmp_path / "request.json").read_text(encoding="utf-8"))
    assert attempt["parallel"] == 0
    assert attempt["actual_solver_settings"] == actual
    assert attempt["solver_settings_validation_status"] == "pending"


def test_preparation_witness_evidence_uses_persisted_witness_keys(
    tmp_path: Path,
) -> None:
    from tests.opensim.native_moco_ladder_worker import (
        _preparation_witness_evidence,
        _sha256,
    )

    state_path = tmp_path / "witness_states.sto"
    control_path = tmp_path / "witness_time_only_controls.sto"
    state_path.write_bytes(b"states")
    control_path.write_bytes(b"controls")
    witness = {
        "state_path": state_path,
        "control_path": control_path,
        "states_sha256": _sha256(state_path),
        "controls_sha256": _sha256(control_path),
        "source_sha256": "a" * 64,
        "initial_q": 0.3,
        "target_q": 0.300005,
        "times": (0.0, 0.001, 0.01),
        "controls": {"flexor": (0.05, 0.05, 0.05)},
    }
    evidence = _preparation_witness_evidence(witness)
    assert evidence["target_q"] == 0.300005
    assert evidence["time_count"] == 3
    assert evidence["states_sha256"] == witness["states_sha256"]
    assert evidence["controls_sha256"] == witness["controls_sha256"]
    assert evidence["control_channels"] == ["flexor"]
    witness["state_path"] = tmp_path / "missing.sto"
    with pytest.raises(ValueError, match="witness files differ"):
        _preparation_witness_evidence(witness)


def test_refinement_staging_unpacks_native_cold_start_helper_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import dataclass

    import tests.opensim.native_moco_ladder_harness as ladder_harness
    from tests.opensim.native_moco_ladder_worker import (
        _sha256,
        _stage_rung_source,
    )

    @dataclass(frozen=True)
    class FakeDeclaration:
        model_path: Path
        source_sha256: str

    parent = tmp_path / "parent"
    parent.mkdir()
    parent_source = parent / "source.osim"
    parent_source.write_bytes(b"source-bound synthetic model")
    output = tmp_path / "child"
    output.mkdir()
    plan = {
        "parent_directory": str(parent),
        "parent_hashes": {"source_osim": _sha256(parent_source)},
    }

    def derive(path: Path, _rung: LadderRung) -> tuple[Path, FakeDeclaration]:
        return path, FakeDeclaration(path, _sha256(path))

    monkeypatch.setattr(ladder_harness, "_derive_synthetic_cold_start", derive)
    stages: dict = {}
    source, declaration, source_hash = _stage_rung_source(
        DEFAULT_RUNGS[0], output, stages, plan
    )
    assert declaration.model_path == source
    assert source == output / "source.osim"
    assert source.read_bytes() == parent_source.read_bytes()
    assert declaration.source_sha256 == source_hash == _sha256(source)


def test_ladder_stops_after_first_worker_failure(tmp_path: Path) -> None:
    invocations: list[str] = []

    def command(rung: LadderRung, _: Path) -> tuple[str, ...]:
        invocations.append(rung.name)
        return (sys.executable, "-c", "raise SystemExit(7)")

    result = run_ladder(DEFAULT_RUNGS[:2], command, tmp_path / "run")
    assert invocations == ["plain"]
    assert len(result) == 1
    assert result[0].status == "worker_failed"
    assert result[0].receipt_sha256 is None
    assert result[0].owned_tree_cleanup_verified is True


def test_ladder_bounds_worker_time_and_stops(tmp_path: Path) -> None:
    invocations: list[str] = []

    def command(rung: LadderRung, _: Path) -> tuple[str, ...]:
        invocations.append(rung.name)
        return (sys.executable, "-c", "import time; time.sleep(10)")

    result = run_ladder(
        DEFAULT_RUNGS[:2], command, tmp_path / "run", timeout_seconds=0.05
    )
    assert invocations == ["plain"]
    assert len(result) == 1
    assert result[0].status == "timeout"
    assert result[0].owned_tree_cleanup_verified is True
    assert result[0].elapsed_seconds < 3
    assert result[0].detail.endswith("wall time")


def test_sampled_memory_threshold_is_distinct_from_wall_timeout() -> None:
    assert _process_stop_reason(1.0, 1024, 30.0, 2048) is None
    assert _process_stop_reason(2.0, 2048, 30.0, 2048) == "sampled_rss_threshold"
    assert _process_stop_reason(30.0, 1024, 30.0, 2048) == "wall_timeout"


def test_sampled_memory_stop_has_distinct_result(monkeypatch, tmp_path: Path) -> None:
    def stop(*_args, **_kwargs):
        raise _WorkerTimeout("", "", "sampled_rss_threshold", 2048)

    monkeypatch.setattr(
        "tests.opensim.native_moco_ladder_harness._supervise_process_tree", stop
    )
    result = _run_one_rung(
        DEFAULT_RUNGS[0], tmp_path, (sys.executable,), timeout_seconds=30.0
    )
    assert result.status == "memory_limit"
    assert "soft 1 GiB" in result.detail
    assert result.peak_sampled_tree_rss_bytes == 2048


def test_replay_audit_rejects_miss_wrong_direction_and_nonfinite() -> None:
    import numpy as np

    from tests.opensim.native_moco_ladder_worker import _compare_replayed_states

    names = (
        "/jointset/base/q/value",
        "/jointset/base/q/speed",
        "/forceset/flexor/activation",
        "/forceset/flexor/fiber_length",
    )
    times = np.asarray([0.0, 0.01])
    solution_states = np.asarray([[0.3, 0.0, 0.05, 0.1], [0.300005, 0.0, 0.05, 0.1]])

    class Solution:
        def getTimeMat(self):
            return times

        def getStateMat(self, name):
            return solution_states[:, names.index(name)]

    passed = _compare_replayed_states(
        Solution(), solution_states.copy(), names, times, 0.300005, 0.3
    )
    assert passed["target_endpoint_within_tolerance"] is True
    assert set(passed["max_state_error_by_unit"]) == {
        "coordinates_rad",
        "speeds_rad_s",
        "activations",
        "fiber_lengths_m",
    }
    wrong_direction = solution_states.copy()
    wrong_direction[-1, 0] = 0.299995
    with pytest.raises(ValueError, match="state error|wrong direction|target"):
        _compare_replayed_states(
            Solution(), wrong_direction, names, times, 0.300005, 0.3
        )
    missed_replay = solution_states.copy()
    missed_replay[-1, 2] += 1e-5
    with pytest.raises(ValueError, match="state error"):
        _compare_replayed_states(Solution(), missed_replay, names, times, 0.300005, 0.3)
    nonfinite = solution_states.copy()
    nonfinite[-1, -1] = float("nan")
    with pytest.raises(ValueError, match="finiteness"):
        _compare_replayed_states(Solution(), nonfinite, names, times, 0.300005, 0.3)


def test_failed_replay_audit_is_persisted_before_acceptance_raises(
    tmp_path: Path,
) -> None:
    import numpy as np

    from tests.opensim.native_moco_ladder_worker import (
        _make_replay_state_audit,
        _record_replay_audit,
    )

    names = (
        "/jointset/base/q/value",
        "/jointset/base/q/speed",
        "/forceset/flexor/activation",
        "/forceset/flexor/fiber_length",
    )
    times = np.asarray([0.0, 0.005, 0.01])
    optimized_states = np.asarray(
        [
            [0.3, 0.0, 0.05, 0.1],
            [0.300002, 0.001, 0.04, 0.101],
            [0.300005, 0.002, 0.03, 0.102],
        ]
    )

    class Solution:
        def getTimeMat(self):
            return times

        def getStateMat(self, name):
            return optimized_states[:, names.index(name)]

    replay_states = optimized_states.copy()
    replay_states[2, 0] += 2e-5
    replay_states[1, 1] += 3e-7
    replay_states[2, 2] += 2e-5
    replay_states[2, 3] += 3e-6
    cold = dict(zip(names, optimized_states[0], strict=True))
    audit = _make_replay_state_audit(
        Solution(), replay_states, names, times, 0.300005, 0.3, cold
    )
    path = tmp_path / "fresh_replay_diagnostic.json"
    with pytest.raises(ValueError, match="failed acceptance"):
        _record_replay_audit(path, audit)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["state_names"] == list(names)
    assert persisted["times_seconds"] == times.tolist()
    assert persisted["replayed_state_matrix"] == replay_states.tolist()
    assert persisted["max_state_error_by_unit"]["speeds_rad_s"] == pytest.approx(3e-7)
    assert persisted["max_state_error_time_by_unit_seconds"]["speeds_rad_s"] == 0.005
    assert persisted["endpoint_residuals"][
        "replayed_target_residual_rad"
    ] == pytest.approx(2e-5)
    assert (
        persisted["initial_state_residuals_by_name"][names[0]][
            "replayed_minus_declared"
        ]
        == 0.0
    )
    assert len(persisted["replayed_state_matrix_sha256"]) == 64
    assert len(persisted["absolute_state_error_matrix_sha256"]) == 64


def test_diagnostic_replay_records_failed_status_without_promoting_it(
    tmp_path: Path,
) -> None:
    from tests.opensim.native_moco_ladder_worker import _record_replay_audit

    failed = {
        "status": "failed",
        "acceptance_failures": ["unit_error_exceeds:speeds_rad_s"],
    }
    path = tmp_path / "fresh_replay_diagnostic.json"
    _record_replay_audit(path, failed, require_pass=False)
    assert json.loads(path.read_text(encoding="utf-8")) == failed


def test_retained_replay_progress_is_flushed_before_native_imports(
    tmp_path: Path,
) -> None:
    from tests.opensim.native_moco_ladder_worker import (
        _start_replay_diagnostic,
        _write_diagnostic_stage,
    )

    supervisor_directory = tmp_path / "supervisor"
    supervisor_directory.mkdir()
    (supervisor_directory / "supervisor.stdout.log").write_text("")
    worker_output = supervisor_directory / "worker"
    _start_replay_diagnostic(worker_output)
    progress_path = worker_output / "diagnostic_progress.json"
    assert json.loads(progress_path.read_text(encoding="utf-8")) == {
        "stage": "worker_started"
    }
    _write_diagnostic_stage(worker_output, "opensim_imported")
    assert json.loads(progress_path.read_text(encoding="utf-8")) == {
        "stage": "opensim_imported"
    }

    with pytest.raises(FileExistsError):
        _start_replay_diagnostic(worker_output)


def test_periodic_stack_dump_is_cancelled_after_success(tmp_path: Path) -> None:
    from tests.opensim.native_moco_ladder_worker import (
        _run_with_periodic_replay_stack_dumps,
    )

    events = []

    def dump_later(timeout, *, repeat, file):
        events.append(("scheduled", timeout, repeat, Path(file.name).name))
        file.write("mock stack dump\n")
        file.flush()

    def cancel() -> None:
        events.append(("cancelled",))

    result = _run_with_periodic_replay_stack_dumps(
        tmp_path, lambda: "replayed", dump_later=dump_later, cancel_dump=cancel
    )

    assert result == "replayed"
    assert events == [
        ("scheduled", 10, True, "native_replay_stack_traces.log"),
        ("cancelled",),
    ]
    assert (tmp_path / "native_replay_stack_traces.log").read_text() == (
        "mock stack dump\n"
    )


def test_periodic_stack_dump_is_cancelled_after_replay_exception(
    tmp_path: Path,
) -> None:
    from tests.opensim.native_moco_ladder_worker import (
        _run_with_periodic_replay_stack_dumps,
    )

    events = []

    def dump_later(timeout, *, repeat, file):
        events.append((timeout, repeat))

    def cancel() -> None:
        events.append("cancelled")

    def fail() -> None:
        raise RuntimeError("expected replay failure")

    with pytest.raises(RuntimeError, match="expected replay failure"):
        _run_with_periodic_replay_stack_dumps(
            tmp_path, fail, dump_later=dump_later, cancel_dump=cancel
        )

    assert events == [(10, True), "cancelled"]


def test_retained_replay_supervisor_persists_wall_timeout_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from tests.opensim import native_moco_ladder_harness as harness

    source_run = tmp_path / "source_run"
    source_run.mkdir()
    for filename in (
        "source.osim",
        "request.json",
        "solution.sto",
        "time_only_controls.sto",
    ):
        (source_run / filename).write_text(filename)
    output_root = tmp_path / "diagnostic"
    observed = {}

    def timeout(command, cwd, timeout_seconds, memory_limit_bytes, *, env=None):
        observed.update(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            memory_limit_bytes=memory_limit_bytes,
            env=env,
        )
        raise harness._WorkerTimeout(
            "captured stdout", "captured stderr", "wall_timeout", 12345
        )

    monkeypatch.setattr(harness, "_supervise_process_tree", timeout)
    result = harness.run_retained_replay_diagnostic(
        source_run,
        output_root,
        worker_python="native-python",
        worker_environment={"CASADIPATH": "sdk-path", "OMP_NUM_THREADS": "1"},
    )

    assert observed["timeout_seconds"] == 30
    assert observed["memory_limit_bytes"] == 1024**3
    assert observed["env"]["CASADIPATH"] == "sdk-path"
    assert observed["command"][-1] == str(output_root / "native_audit")
    assert not (output_root / "native_audit").exists()
    assert (output_root / "supervisor.stdout.log").read_text() == "captured stdout"
    assert (output_root / "supervisor.stderr.log").read_text() == "captured stderr"
    manifest = json.loads((output_root / "invocation.json").read_text())
    assert manifest["worker_command"] == observed["command"]
    assert manifest["nonsecret_environment"] == {
        "CASADIPATH": "sdk-path",
        "OMP_NUM_THREADS": "1",
    }
    persisted = json.loads((output_root / "supervisor_result.json").read_text())
    assert persisted["status"] == "wall_timeout"
    assert persisted["stop_reason"] == "wall_timeout"
    assert persisted["peak_sampled_tree_rss_bytes"] == 12345
    assert persisted["owned_tree_cleanup_verified"] is True
    assert result == persisted


def test_solution_sto_roundtrip_allows_one_ulp_and_records_array_hashes(
    tmp_path: Path,
) -> None:
    import numpy as np

    from tests.opensim.native_moco_ladder_harness import (
        _validate_solution_serialization_audit,
    )
    from tests.opensim.native_moco_ladder_worker import (
        _audit_solution_roundtrip,
        _require_solution_roundtrip,
    )

    memory = {
        "state_names": ("/jointset/base/q/value",),
        "control_names": ("/forceset/flexor",),
        "time": np.asarray([0.0, 0.005, 0.01]),
        "states": np.asarray([[0.3], [0.305], [0.31]]),
        "controls": np.asarray([[0.1], [0.15], [0.2]]),
    }
    persisted = {
        name: value.copy() if hasattr(value, "copy") else value
        for name, value in memory.items()
    }
    persisted["states"][1, 0] = np.nextafter(persisted["states"][1, 0], np.inf)
    persisted["time"][1] = np.nextafter(persisted["time"][1], np.inf)

    audit = _audit_solution_roundtrip(memory, persisted)
    _require_solution_roundtrip(audit)

    assert audit["maximum_ulp_error"] == 1.0
    assert audit["arrays"]["states"]["max_absolute_error"] > 0.0
    assert (
        audit["arrays"]["states"]["in_memory_sha256"]
        != audit["arrays"]["states"]["persisted_sha256"]
    )
    assert audit["arrays"]["controls"]["max_absolute_error"] == 0.0
    request = {
        "solution_serialization_max_ulps": 8,
        "cold_start_named_state": {"/jointset/base/q/value": 0.3},
        "witness_excitation_by_channel": {"flexor": 0.05},
    }
    path = tmp_path / "solution_serialization_audit.json"
    path.write_text(json.dumps(audit, sort_keys=True))
    _validate_solution_serialization_audit(
        path, request, {"solution_serialization_audit": audit}
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "time_count",
        "time_value",
        "control_count",
        "control_value",
        "too_many_ulps",
        "nonfinite",
    ],
)
def test_solution_sto_roundtrip_rejects_changed_or_over_budget_arrays(
    mutation: str,
) -> None:
    import numpy as np

    from tests.opensim.native_moco_ladder_worker import (
        _audit_solution_roundtrip,
        _require_solution_roundtrip,
    )

    memory = {
        "state_names": ("q",),
        "control_names": ("u",),
        "time": np.asarray([0.0, 0.005, 0.01]),
        "states": np.asarray([[0.3], [0.305], [0.31]]),
        "controls": np.asarray([[0.1], [0.15], [0.2]]),
    }
    persisted = {
        name: value.copy() if hasattr(value, "copy") else value
        for name, value in memory.items()
    }
    if mutation == "time_count":
        persisted["time"] = persisted["time"][:-1]
    elif mutation == "time_value":
        value = persisted["time"][1]
        for _ in range(9):
            value = np.nextafter(value, np.inf)
        persisted["time"][1] = value
    elif mutation == "control_count":
        persisted["controls"] = persisted["controls"][:-1]
    elif mutation == "control_value":
        value = persisted["controls"][1, 0]
        for _ in range(9):
            value = np.nextafter(value, np.inf)
        persisted["controls"][1, 0] = value
    elif mutation == "nonfinite":
        persisted["controls"][1, 0] = float("nan")
    else:
        value = persisted["states"][1, 0]
        for _ in range(9):
            value = np.nextafter(value, np.inf)
        persisted["states"][1, 0] = value
    with pytest.raises(ValueError, match="round-trip|ULP|clock|shape|non-finite"):
        audit = _audit_solution_roundtrip(memory, persisted)
        _require_solution_roundtrip(audit)


def test_float64_ulp_distance_is_exact_across_binades_and_signs() -> None:
    import numpy as np

    from tests.opensim.native_moco_ladder_worker import _float64_ulp_distance

    cases = (
        (np.nextafter(1.0, -np.inf), np.inf),
        (1.0, -np.inf),
        (np.nextafter(-1.0, np.inf), -np.inf),
        (-1.0, np.inf),
    )
    for start, direction in cases:
        value = start
        for _ in range(8):
            value = np.nextafter(value, direction)
        assert (
            int(_float64_ulp_distance(np.asarray([start]), np.asarray([value]))[0]) == 8
        )

        value = np.nextafter(value, direction)
        assert (
            int(_float64_ulp_distance(np.asarray([start]), np.asarray([value]))[0]) == 9
        )


def test_float64_ulp_distance_collapses_signed_zero_and_counts_subnormals() -> None:
    import numpy as np

    from tests.opensim.native_moco_ladder_worker import _float64_ulp_distance

    smallest = np.nextafter(0.0, 1.0)
    actual = _float64_ulp_distance(
        np.asarray([-0.0, -smallest, 0.0]),
        np.asarray([0.0, 0.0, smallest]),
    )
    assert [int(value) for value in actual] == [0, 1, 1]


def test_float64_ulp_distance_roundtrip_gate_accepts_eight_and_rejects_nine() -> None:
    import numpy as np

    from tests.opensim.native_moco_ladder_worker import (
        _audit_solution_roundtrip,
        _require_solution_roundtrip,
    )

    memory = {
        "state_names": ("q",),
        "control_names": ("u",),
        "time": np.asarray([0.0, 0.005, 0.01]),
        "states": np.asarray([[1.0], [1.0], [1.0]]),
        "controls": np.asarray([[0.1], [0.1], [0.1]]),
    }
    persisted = {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in memory.items()
    }
    value = 1.0
    for _ in range(8):
        value = np.nextafter(value, np.inf)
    persisted["states"][1, 0] = value
    _require_solution_roundtrip(_audit_solution_roundtrip(memory, persisted))

    persisted["states"][1, 0] = np.nextafter(value, np.inf)
    with pytest.raises(ValueError, match="ULP"):
        _require_solution_roundtrip(_audit_solution_roundtrip(memory, persisted))


def test_success_provenance_checks_feature_and_runtime_evidence(tmp_path: Path) -> None:
    components = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in ("_common.pyd", "_simulation.pyd", "_moco.pyd", "osimMoco.dll")
    }
    runtime = {
        "schema": "opensim-bounded-runtime-fingerprint-v1",
        "version": "OpenSim test build",
        "components": components,
        "coverage_note": "bounded Python extension and Moco DLL fingerprint; "
        "not a closed dependency-runtime identity",
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(runtime, sort_keys=True, indent=2) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _validate_bounded_runtime_fingerprint(path, digest)
    with pytest.raises(ValueError, match="fingerprint"):
        _validate_bounded_runtime_fingerprint(path, "0" * 64)

    rung = DEFAULT_RUNGS[2]
    request = {"rung": rung.name, "features": list(rung.features)}
    receipt = {"features": list(rung.features)}
    _validate_request_rung_binding(request, receipt, rung)
    receipt["features"] = []
    with pytest.raises(ValueError, match="features"):
        _validate_request_rung_binding(request, receipt, rung)


def test_saved_time_only_controls_reject_persisted_sto_mutation(tmp_path: Path) -> None:
    import numpy as np

    from tests.opensim.native_moco_ladder_worker import (
        _load_time_only_controls,
        _write_time_only_controls,
    )

    osim = pytest.importorskip("opensim")
    path = tmp_path / "time_only_controls.sto"
    grid = np.asarray([0.0, 0.01])
    controls = {"flexor": np.asarray([0.1, 0.2]), "extensor": np.asarray([0.3, 0.4])}
    _write_time_only_controls(osim, grid, controls, path)
    saved_grid, saved_controls = _load_time_only_controls(
        osim, path, grid, ("flexor", "extensor"), controls
    )
    np.testing.assert_array_equal(saved_grid, grid)
    np.testing.assert_array_equal(saved_controls["flexor"], controls["flexor"])
    mutated = {**controls, "flexor": np.asarray([0.1, 1.2])}
    _write_time_only_controls(osim, grid, mutated, path)
    with pytest.raises(ValueError, match="out of bounds"):
        _load_time_only_controls(osim, path, grid, ("flexor", "extensor"), controls)


def test_saved_controls_must_match_persisted_solution_values(tmp_path: Path) -> None:
    import numpy as np

    from tests.opensim.native_moco_ladder_worker import (
        _load_time_only_controls,
        _write_time_only_controls,
    )

    osim = pytest.importorskip("opensim")
    grid = np.asarray([0.0, 0.01])
    expected = {
        "flexor": np.asarray([0.1, 0.2]),
        "extensor": np.asarray([0.3, 0.4]),
    }
    changed = {name: values.copy() for name, values in expected.items()}
    changed["flexor"][1] = 0.25
    path = tmp_path / "time_only_controls.sto"
    _write_time_only_controls(osim, grid, changed, path)
    with pytest.raises(ValueError, match="differ from persisted solution"):
        _load_time_only_controls(osim, path, grid, tuple(expected), expected)


def test_ladder_rejects_success_exit_without_native_evidence(tmp_path: Path) -> None:
    def command(_: LadderRung, __: Path) -> tuple[str, ...]:
        return (sys.executable, "-c", "pass")

    result = run_ladder(DEFAULT_RUNGS[:1], command, tmp_path / "run")
    assert len(result) == 1
    assert result[0].status == "invalid_evidence"
    assert result[0].receipt_sha256 is None


def test_ladder_timeout_is_hard_capped_at_thirty_seconds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="30"):
        run_ladder(
            DEFAULT_RUNGS[:1],
            lambda *_: (sys.executable,),
            tmp_path,
            timeout_seconds=31,
        )


@pytest.mark.parametrize(
    "invalid_timeout", [True, float("nan"), float("inf"), -float("inf")]
)
def test_ladder_rejects_nonfinite_or_boolean_timeout(
    tmp_path: Path, invalid_timeout: float
) -> None:
    with pytest.raises(ValueError, match="timeout"):
        run_ladder(
            DEFAULT_RUNGS[:1],
            lambda *_: (sys.executable,),
            tmp_path,
            timeout_seconds=invalid_timeout,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object containment")
def test_windows_timeout_kills_launcher_descendant_tree(tmp_path: Path) -> None:
    marker = tmp_path / "orphan-was-not-killed.txt"
    child = (
        "import time; from pathlib import Path; time.sleep(0.6); "
        f"Path({str(marker)!r}).write_text('orphan')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(5)"
    )
    result = run_ladder(
        DEFAULT_RUNGS[:1],
        lambda *_: (sys.executable, "-c", parent),
        tmp_path / "run",
        timeout_seconds=0.2,
    )
    assert result[0].status == "timeout"
    assert result[0].peak_sampled_tree_rss_bytes is not None
    time.sleep(0.8)
    assert not marker.exists()
