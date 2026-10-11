from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

from src.engines.physics_engines.opensim.python.tour_matching import native_moco_runner
from src.engines.physics_engines.opensim.python.tour_matching.native_diagnostic_contract import (
    DiagnosticSoftwareAdmission,
    PhysicalQualification,
)


pytestmark = pytest.mark.unit

GETTER_NAMES = native_moco_runner._SOLVER_PROGRESS_GETTERS


class _FakeSolver:
    def __init__(self) -> None:
        self.parallel = 7
        self.set_parallel_calls: list[int] = []

    def set_parallel(self, value: int) -> None:
        self.set_parallel_calls.append(value)
        self.parallel = value

    def __getattr__(self, name: str) -> Any:
        if name not in GETTER_NAMES.values():
            raise AttributeError(name)
        if name == "get_parallel":
            return lambda: self.parallel
        values = {
            "get_optim_solver": "ipopt",
            "get_optim_hessian_approximation": "limited-memory",
            "get_optim_sparsity_detection": "none",
            "get_optim_finite_difference_scheme": "central",
            "get_transcription_scheme": "hermite-simpson",
            "get_multibody_dynamics_mode": "explicit",
            "get_verbosity": 2,
            "get_num_mesh_intervals": 2,
            "get_optim_max_iterations": 5,
            "get_optim_convergence_tolerance": 1e-3,
            "get_optim_constraint_tolerance": 1e-3,
            "get_optim_ipopt_print_level": 0,
        }
        return lambda: values[name]


class _FakeStudy:
    def __init__(self, solver: _FakeSolver, *, fail: bool) -> None:
        self.solver = solver
        self.fail = fail
        self.solve_calls = 0

    def updSolver(self) -> _FakeSolver:
        return self.solver

    def solve(self) -> Any:
        self.solve_calls += 1
        assert self.solver.parallel == 0
        if self.fail:
            raise RuntimeError("injected optimizer failure")
        return SimpleNamespace(success=lambda: False, getStatus=lambda: "not-run")


@pytest.mark.parametrize("fail", [False, True])
def test_native_solve_progress_is_atomic_identity_bound_and_stage_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    solver = _FakeSolver()
    study = _FakeStudy(solver, fail=fail)
    expected_build_stages = [
        "study-created",
        "model-and-time-bounds-set",
        "initial-bindings-set",
        "goals-set",
        "solver-configured",
        "guess-created",
        "guess-trajectory-inserted",
        "guess-set",
    ]

    def build_study(*args: Any, progress_callback: Any, **kwargs: Any) -> _FakeStudy:
        for stage in expected_build_stages:
            progress_callback(stage)
        return study

    monkeypatch.setattr(native_moco_runner, "build_moco_study", build_study)
    monkeypatch.setattr(
        native_moco_runner, "_native_goal_weights", lambda _: {"m": 2.0}
    )
    monkeypatch.setitem(
        sys.modules,
        "opensim",
        SimpleNamespace(
            MocoCasADiSolver=SimpleNamespace(safeDownCast=lambda value: value)
        ),
    )
    request = SimpleNamespace(
        identity_sha256="request-id",
        model_sha256="source-id",
        model_path=Path("model.osim"),
        states_guess_path=Path("guess.sto"),
        config=SimpleNamespace(t_start_s=0.0, horizon_s=0.1),
        bindings=SimpleNamespace(control_bounds={}, state_bounds={}),
        marker_weights={"m": 2.0},
    )
    progress_path = tmp_path / "native_moco_progress.json"

    if fail:
        with pytest.raises(RuntimeError, match="injected optimizer failure"):
            native_moco_runner._native_solve_once(
                request,
                Path("registered.trc"),
                tmp_path,
                time.perf_counter(),
                time.process_time(),
                current_input_audit_elapsed_wall_seconds=0.25,
            )
    else:
        result = native_moco_runner._native_solve_once(
            request,
            Path("registered.trc"),
            tmp_path,
            time.perf_counter(),
            time.process_time(),
            current_input_audit_elapsed_wall_seconds=0.25,
        )
        assert result.success is False

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    expected_stages = [
        "building-study",
        *expected_build_stages,
        "study-built",
        "optimizing",
    ]
    if not fail:
        expected_stages.append("optimization-returned")
    assert [event["stage"] for event in payload["events"]] == expected_stages
    assert [event["sequence"] for event in payload["events"]] == list(
        range(len(expected_stages))
    )
    assert payload["request_sha256"] == "request-id"
    assert payload["source_sha256"] == "source-id"
    assert payload["stage"] == ("optimizing" if fail else "optimization-returned")
    assert payload["timer_origin"] == "solve_native_moco_after_current_input_audit"
    assert payload["current_input_audit_elapsed_wall_seconds"] == 0.25
    assert all(event["request_sha256"] == "request-id" for event in payload["events"])
    assert all(event["source_sha256"] == "source-id" for event in payload["events"])
    assert all(
        event["current_input_audit_elapsed_wall_seconds"] == 0.25
        for event in payload["events"]
    )
    assert all(event["monotonic_seconds"] >= 0.0 for event in payload["events"])
    assert all(event["elapsed_wall_seconds"] >= 0.0 for event in payload["events"])
    assert all(event["elapsed_cpu_seconds"] >= 0.0 for event in payload["events"])
    assert [event["monotonic_seconds"] for event in payload["events"]] == sorted(
        event["monotonic_seconds"] for event in payload["events"]
    )
    optimizing = next(e for e in payload["events"] if e["stage"] == "optimizing")
    properties = optimizing["solver_properties_after_set_parallel"]
    assert properties["parallel"]["status"] == "returned"
    assert properties["parallel"]["value"] == 0
    assert properties["num_mesh_intervals"]["status"] == "returned"
    assert properties["num_mesh_intervals"]["value"] == 2
    assert solver.set_parallel_calls == [0]
    assert study.solve_calls == 1


def test_solver_progress_getters_preserve_missing_error_and_nonfinite() -> None:
    class GetterCases:
        def get_value(self) -> float:
            return 3.0

        def get_missing_value(self) -> float:
            raise RuntimeError("getter unavailable")

        def get_nonfinite_value(self) -> float:
            return math.inf

        def get_object_value(self) -> object:
            return object()

    result = native_moco_runner._solver_progress_properties(
        GetterCases(),
        {
            "value": "get_value",
            "missing": "get_absent",
            "error": "get_missing_value",
            "nonfinite": "get_nonfinite_value",
            "object": "get_object_value",
        },
    )
    assert result["value"] == {
        "getter": "get_value",
        "status": "returned",
        "value": 3.0,
        "value_type": "float",
    }
    assert result["missing"]["status"] == "missing"
    assert result["error"]["status"] == "error"
    assert "value" not in result["error"]
    assert result["nonfinite"]["status"] == "nonfinite"
    assert "value" not in result["nonfinite"]
    assert result["object"]["status"] == "non_scalar"
    assert "value" not in result["object"]


def test_progress_atomic_write_leaves_previous_snapshot_on_serialization_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress.json"
    path.write_text('{"stage":"previous"}\n', encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Out of range float"):
        native_moco_runner._atomic_write_progress(path, {"not_json": math.nan})
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".progress-*.tmp")) == []


def test_solve_wrapper_separates_current_input_audit_from_legacy_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_id = "a" * 64
    source_id = "b" * 64
    assessment = native_moco_runner.NativeDiagnosticAssessment(
        request_sha256=request_id,
        source_sha256=source_id,
        findings=(),
        software_admission=DiagnosticSoftwareAdmission.ADMITTED,
        runtime_viability=native_moco_runner.RuntimeViability.VIABLE,
        physical_qualification=PhysicalQualification.NOT_ESTABLISHED,
        runtime_phase=native_moco_runner.RuntimePhase.SHORT_CAPABILITY_PROBE,
        runtime_window_seconds=(0.0, 0.001),
        native_runtime_sha256="c" * 64,
        runtime_evidence_sha256="d" * 64,
    )
    prepared = SimpleNamespace(diagnostic_assessment=assessment)
    request = SimpleNamespace(
        identity_sha256=request_id,
        config=SimpleNamespace(t_start_s=0.0, horizon_s=0.01),
    )
    audit_completed: list[float] = []
    solver_call: dict[str, Any] = {}

    def audit(*args: Any, **kwargs: Any) -> Path:
        time.sleep(0.02)
        audit_completed.append(time.perf_counter())
        return Path("registered.trc")

    def run_once(
        request_arg: Any,
        registered: Path,
        directory: Path,
        wall: float,
        cpu: float,
        *,
        current_input_audit_elapsed_wall_seconds: float,
    ) -> native_moco_runner.NativeMocoSolve:
        solver_call["wall"] = wall
        solver_call["cpu"] = cpu
        solver_call["audit_seconds"] = current_input_audit_elapsed_wall_seconds
        assert registered == Path("registered.trc")
        assert wall >= audit_completed[0]
        return native_moco_runner.NativeMocoSolve(
            False, "injected-no-solve", None, 0, None, None, None, None, 0.0, 0.0
        )

    monkeypatch.setattr(native_moco_runner, "_require_current_inputs", audit)
    monkeypatch.setattr(native_moco_runner, "_native_solve_once", run_once)
    result = native_moco_runner.solve_native_moco(request, prepared, tmp_path)

    assert solver_call["audit_seconds"] >= 0.015
    assert solver_call["wall"] >= audit_completed[0]
    assert result.success is False
    receipt = json.loads((tmp_path / "solve.json").read_text(encoding="utf-8"))
    assert receipt["wall_seconds"] == 0.0
    assert (
        receipt["diagnostic_assessment"]["runtime_phase"]
        == native_moco_runner.RuntimePhase.OPTIMIZATION.value
    )
