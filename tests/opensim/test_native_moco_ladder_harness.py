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


def test_forward_witness_has_exact_saved_clock_and_nonzero_endpoint(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
        declare_native_mechanics_execution_profile,
    )
    from tests.opensim.native_moco_ladder_harness import build_synthetic_rung
    from tests.opensim.native_moco_ladder_worker import (
        _build_feasible_witness,
        _build_witness_seeded_problem,
        _set_witness_guess,
        _synthetic_state_bounds,
        _witness_request_metadata,
    )

    osim = pytest.importorskip("opensim")
    path, declaration = build_synthetic_rung(DEFAULT_RUNGS[0], tmp_path / "source")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    profile = declare_native_mechanics_execution_profile(path, source_hash)
    witness = _build_feasible_witness(
        path, declaration, profile, None, tmp_path / "witness", osim
    )
    metadata = _witness_request_metadata(declaration, profile, witness)
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
    assert set(solver.getGuess().getStateNames()) == set(witness["state_names"])
    assert set(solver.getGuess().getControlNames()) == {
        "/forceset/flexor",
        "/forceset/extensor",
    }


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
        osim, path, grid, ("flexor", "extensor")
    )
    np.testing.assert_array_equal(saved_grid, grid)
    np.testing.assert_array_equal(saved_controls["flexor"], controls["flexor"])
    mutated = {**controls, "flexor": np.asarray([0.1, 1.2])}
    _write_time_only_controls(osim, grid, mutated, path)
    with pytest.raises(ValueError, match="out of bounds"):
        _load_time_only_controls(osim, path, grid, ("flexor", "extensor"))


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
