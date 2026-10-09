"""Maintained file-driven Moco preparation and independent native replay."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
    MocoTrackingConfig,
)
from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
    MocoInitialBindings,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner import (
    NativeMocoRequest,
    prepare_native_moco,
    solve_native_moco,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_replay import (
    export_native_moco_bundle,
    replay_native_moco_bundle,
    score_native_moco_replay,
)
from src.engines.physics_engines.opensim.python.tour_matching.cli import main
from src.engines.physics_engines.opensim.python.tour_matching.native_passive_readiness import (
    MusclePassiveLimits,
    PassiveReadinessPolicy,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CaptureRegistration,
)
from src.engines.physics_engines.opensim.python.tour_matching.trc import (
    read_trc,
    write_trc,
)
from src.engines.native_replay_contracts import native_replay_contract_types
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

from tests.opensim.test_moco_initial_bindings import _native_inputs

pytest_plugins = ("tests.opensim.test_native_muscle_replay",)
pytestmark = pytest.mark.unit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_request(request: NativeMocoRequest, path: Path) -> None:
    """Serialize independently authored JSON as a real CLI boundary fixture."""
    path.write_text(
        json.dumps(
            {
                "model_path": str(request.model_path),
                "trc_path": str(request.trc_path),
                "states_guess_path": str(request.states_guess_path),
                "model_sha256": request.model_sha256,
                "trc_sha256": request.trc_sha256,
                "states_guess_sha256": request.states_guess_sha256,
                "bindings": {
                    "state_bounds": dict(request.bindings.state_bounds),
                    "initial_state": dict(request.bindings.initial_state),
                    "control_bounds": dict(request.bindings.control_bounds),
                },
                "config": asdict(request.config),
                "marker_weights": dict(request.marker_weights),
                "marker_bindings": {
                    name: [frame, list(offset)]
                    for name, (frame, offset) in request.marker_bindings.items()
                },
                "registration": None
                if request.registration is None
                else {
                    "rotation": request.registration.rotation.tolist(),
                    "translation": request.registration.translation.tolist(),
                },
                "reference_frame_path": request.reference_frame_path,
                "passive_policy": None
                if request.passive_policy is None
                else asdict(request.passive_policy),
                "excluded_markers": dict(request.excluded_markers),
            }
        ),
        encoding="utf-8",
    )


def test_prepare_collects_unavailable_passive_policy_before_solve(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    path, trc, guess, binding = _native_inputs(muscle_fixture, tmp_path)
    request = NativeMocoRequest(
        model_path=path,
        trc_path=trc,
        states_guess_path=guess,
        model_sha256=_sha(path),
        trc_sha256=_sha(trc),
        states_guess_sha256=_sha(guess),
        bindings=binding,
        config=MocoTrackingConfig(horizon_s=0.01, allow_unused_references=False),
        marker_weights={"load_marker": 2.0},
        marker_bindings={"load_marker": ("/bodyset/load", (0.0, 0.0, 0.0))},
        registration=CaptureRegistration(np.eye(3), np.zeros(3)),
        reference_frame_path="/ground",
        passive_policy=None,
        excluded_markers={},
    )
    report = prepare_native_moco(request, tmp_path / "prepared")
    assert report.blockers == ("passive-policy-unavailable",)
    assert report.marker_count == 1
    assert report.observation_count == 3
    assert report.wall_seconds >= 0
    assert report.cpu_seconds >= 0
    assert report.source_sha256 == _sha(path)
    assert not (tmp_path / "prepared" / "solution.sto").exists()


def test_native_cli_preparation_uses_explicit_file_request(
    muscle_fixture: tuple[Path, dict[str, float]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, trc, guess, binding = _native_inputs(muscle_fixture, tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "model_path": str(path),
                "trc_path": str(trc),
                "states_guess_path": str(guess),
                "model_sha256": _sha(path),
                "trc_sha256": _sha(trc),
                "states_guess_sha256": _sha(guess),
                "bindings": {
                    "state_bounds": dict(binding.state_bounds),
                    "initial_state": dict(binding.initial_state),
                    "control_bounds": dict(binding.control_bounds),
                },
                "config": {
                    "horizon_s": 0.01,
                    "allow_unused_references": False,
                },
                "marker_weights": {"load_marker": 1.0},
                "marker_bindings": {"load_marker": ["/bodyset/load", [0.0, 0.0, 0.0]]},
                "registration": {
                    "rotation": np.eye(3).tolist(),
                    "translation": [0.0, 0.0, 0.0],
                },
                "reference_frame_path": "/ground",
                "passive_policy": None,
                "excluded_markers": {},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "native-cli"
    assert (
        main(
            [
                "moco-native",
                "--request",
                str(request_path),
                "--output-dir",
                str(output),
                "--prepare-only",
            ]
        )
        == 2
    )
    assert json.loads((output / "preparation.json").read_text())["blockers"] == [
        "passive-policy-unavailable"
    ]
    assert not (output / "solve.json").exists()
    relative = json.loads(request_path.read_text(encoding="utf-8"))
    for key in ("model_path", "trc_path", "states_guess_path"):
        relative[key] = Path(relative[key]).name
    request_path.write_text(json.dumps(relative), encoding="utf-8")
    relative_output = tmp_path / "relative-cli"
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    assert (
        main(
            [
                "moco-native",
                "--request",
                str(request_path),
                "--output-dir",
                str(relative_output),
                "--prepare-only",
            ]
        )
        == 2
    )
    assert json.loads((relative_output / "preparation.json").read_text())[
        "blockers"
    ] == ["passive-policy-unavailable"]
    unknown = json.loads(request_path.read_text(encoding="utf-8"))
    unknown["infer_missing_markers"] = True
    request_path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unknown"):
        main(
            ["moco-native", "--request", str(request_path), "--output-dir", str(output)]
        )


def test_prepare_rejects_stale_model_and_missing_marker_binding(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    path, trc, guess, binding = _native_inputs(muscle_fixture, tmp_path)
    request = NativeMocoRequest(
        model_path=path,
        trc_path=trc,
        states_guess_path=guess,
        model_sha256="0" * 64,
        trc_sha256=_sha(trc),
        states_guess_sha256=_sha(guess),
        bindings=binding,
        config=MocoTrackingConfig(horizon_s=0.01, allow_unused_references=False),
        marker_weights={"load_marker": 1.0},
        marker_bindings={},
        registration=CaptureRegistration(np.eye(3), np.zeros(3)),
        reference_frame_path="/ground",
        passive_policy=None,
        excluded_markers={},
    )
    report = prepare_native_moco(request, tmp_path / "stale")
    assert "model-sha256-mismatch" in report.blockers
    assert "marker-binding-coverage" in report.blockers
    assert not (tmp_path / "stale" / "solution.sto").exists()


def test_preparation_rejects_changed_weight_request_before_solver(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    path, trc, guess, binding = _native_inputs(muscle_fixture, tmp_path)
    request = NativeMocoRequest(
        path,
        trc,
        guess,
        _sha(path),
        _sha(trc),
        _sha(guess),
        binding,
        MocoTrackingConfig(horizon_s=0.01, allow_unused_references=False),
        {"load_marker": 2.0},
        {"load_marker": ("/bodyset/load", (0.0, 0.0, 0.0))},
        CaptureRegistration(np.eye(3), np.zeros(3)),
        "/ground",
        None,
        {},
    )
    prepared = prepare_native_moco(request, tmp_path / "changed")
    with pytest.raises(ValueError, match="marker placement"):
        replace(request, marker_bindings={"load_marker": ("load", (0.0, 0.0))})
    with pytest.raises(ValueError, match="request changed"):
        solve_native_moco(
            replace(request, marker_weights={"load_marker": 3.0}),
            prepared,
            tmp_path / "changed",
        )
    wrong_units = tmp_path / "wrong-units.trc"
    wrong_units.write_text(
        trc.read_text(encoding="utf-8").replace("\tm\t", "\tcm\t", 1),
        encoding="utf-8",
    )
    unit_request = replace(request, trc_path=wrong_units, trc_sha256=_sha(wrong_units))
    unit_report = prepare_native_moco(unit_request, tmp_path / "wrong-unit-run")
    assert "capture-trc-invalid" in unit_report.blockers
    original_capture = read_trc(trc)
    no_support = write_trc(
        TourCapture(
            original_capture.time_s,
            original_capture.labels,
            np.full_like(original_capture.points_m, np.nan),
            np.zeros_like(original_capture.valid),
        ),
        tmp_path / "no-support.trc",
        rate_hz=200.0,
    )
    no_support_request = replace(
        request, trc_path=no_support, trc_sha256=_sha(no_support)
    )
    no_support_report = prepare_native_moco(
        no_support_request, tmp_path / "no-support-run"
    )
    assert "missing-marker-observation-support" in no_support_report.blockers


def test_preparation_blocks_unused_references_and_nonmuscle_actuation(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    osim = pytest.importorskip("opensim")
    path, trc, guess, binding = _native_inputs(muscle_fixture, tmp_path)
    model = osim.Model(str(path))
    reserve = osim.CoordinateActuator("slide")
    reserve.setName("hidden_reserve")
    reserve.setOptimalForce(1.0)
    model.addForce(reserve)
    model.printToXML(str(path))
    request = NativeMocoRequest(
        path,
        trc,
        guess,
        _sha(path),
        _sha(trc),
        _sha(guess),
        binding,
        MocoTrackingConfig(horizon_s=0.01, allow_unused_references=True),
        {"load_marker": 1.0},
        {"load_marker": ("/bodyset/load", (0.0, 0.0, 0.0))},
        CaptureRegistration(np.eye(3), np.zeros(3)),
        "/ground",
        None,
        {},
    )
    report = prepare_native_moco(request, tmp_path / "unsafe")
    assert "unused-reference-policy" in report.blockers
    assert "nonmuscle-assistance-unqualified" in report.blockers
    assert "native-control-binding-coverage" in report.blockers
    assert "passive-policy-unavailable" in report.blockers
    with pytest.raises(ValueError, match="Preparation blockers"):
        solve_native_moco(request, report, tmp_path / "unsafe")
    shifted = replace(
        request,
        config=MocoTrackingConfig(
            t_start_s=0.001,
            horizon_s=0.01,
            allow_unused_references=False,
        ),
    )
    shifted_report = prepare_native_moco(shifted, tmp_path / "shifted")
    assert "nonzero-native-replay-origin" in shifted_report.blockers


def test_file_driven_native_moco_solve_reads_back_marker_weight(
    muscle_fixture: tuple[Path, dict[str, float]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    osim = pytest.importorskip("opensim")
    path, trc, guess, binding = _native_inputs(
        muscle_fixture, tmp_path, times=np.linspace(0.0, 0.01, 11)
    )
    model = osim.Model(str(path))
    model.initSystem()
    policy = PassiveReadinessPolicy(
        hashlib.sha256(model.dump().encode()).hexdigest(),
        "synthetic fixture only; no human physiology claim",
        (
            MusclePassiveLimits(
                "/forceset/flexor",
                (0.1, 10.0),
                1000.0,
                1000.0,
                False,
                False,
                "predeclared synthetic fixture envelope",
            ),
        ),
    )
    request = NativeMocoRequest(
        model_path=path,
        trc_path=trc,
        states_guess_path=guess,
        model_sha256=_sha(path),
        trc_sha256=_sha(trc),
        states_guess_sha256=_sha(guess),
        bindings=binding,
        config=MocoTrackingConfig(
            horizon_s=0.01,
            mesh_interval_s=0.005,
            optim_max_iterations=30,
            allow_unused_references=False,
        ),
        marker_weights={"load_marker": 2.0},
        marker_bindings={"load_marker": ("/bodyset/load", (0.0, 0.0, 0.0))},
        registration=CaptureRegistration(np.eye(3), np.zeros(3)),
        reference_frame_path="/ground",
        passive_policy=policy,
        excluded_markers={},
    )
    prepared = prepare_native_moco(request, tmp_path / "run")
    assert prepared.ready_for_software_solve, prepared.blockers
    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_moco_runner,
    )

    def failed_construction(*args: object, **kwargs: object) -> None:
        raise RuntimeError("native failure fixture")

    monkeypatch.setattr(native_moco_runner, "build_moco_study", failed_construction)
    failed = solve_native_moco(request, prepared, tmp_path / "run")
    assert not failed.success
    assert failed.status == "native-error:RuntimeError"
    assert json.loads((tmp_path / "run" / "solve.json").read_text())["success"] is False
    monkeypatch.undo()
    solved = solve_native_moco(request, prepared, tmp_path / "run")
    assert solved.success
    attempts = sorted((tmp_path / "run" / "solve_attempts").glob("*.json"))
    assert len(attempts) == 2
    assert json.loads(attempts[0].read_text())["success"] is False
    assert json.loads(attempts[1].read_text())["success"] is True
    assert solved.control_knot_count >= 3
    assert solved.marker_weight_sha256
    assert (tmp_path / "run" / "solution.sto").is_file()
    registered = tmp_path / "run" / "registered.trc"
    original_registered = registered.read_bytes()
    registered.write_bytes(original_registered + b"\n")
    with pytest.raises(ValueError, match="Registered marker reference changed"):
        export_native_moco_bundle(request, prepared, solved, tmp_path / "run")
    registered.write_bytes(original_registered)
    clock_file = tmp_path / "run" / "observation_clock.json"
    original_clock = clock_file.read_bytes()
    clock_file.write_text(json.dumps([0.0, 0.01]), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen observation"):
        export_native_moco_bundle(request, prepared, solved, tmp_path / "run")
    clock_file.write_bytes(original_clock)
    exported = export_native_moco_bundle(request, prepared, solved, tmp_path / "run")
    assert exported.original_knot_count == solved.control_knot_count
    assert exported.output_count > 11
    saved_knots = np.asarray(
        osim.MocoTrajectory(str(tmp_path / "run" / "solution.sto")).getTimeMat()
    ).reshape(-1)
    assert np.array_equal(
        exported.output_times[np.searchsorted(exported.output_times, saved_knots)],
        saved_knots,
    )
    from src.engines.physics_engines.opensim.python.tour_matching import trc

    monkeypatch.setattr(
        trc,
        "read_trc",
        lambda path: pytest.fail("independent replay accessed observations"),
    )
    native = replay_native_moco_bundle(exported, path, tmp_path / "run")
    assert np.array_equal(native.times, exported.output_times)
    assert native.states.shape[1] == len(binding.initial_state)
    monkeypatch.undo()
    scored = score_native_moco_replay(
        request, prepared, solved, exported, native, tmp_path / "run"
    )
    assert np.isfinite(scored.marker_rmse_m)
    assert scored.observed_samples == 11
    assert scored.full_state_max_abs >= 0
    request_file = tmp_path / "reviewed-native-request.json"
    _write_request(request, request_file)
    cli_output = tmp_path / "native-cli-full"
    assert (
        main(
            [
                "moco-native",
                "--request",
                str(request_file),
                "--output-dir",
                str(cli_output),
            ]
        )
        == 0
    )
    assert json.loads((cli_output / "score.json").read_text())["observed_samples"] == 11


@pytest.mark.parametrize("mesh_interval_s", [0.005, 0.0025])
def test_two_dof_ordered_multi_muscle_controls_survive_native_replay(
    muscle_fixture: tuple[Path, dict[str, float]],
    tmp_path: Path,
    mesh_interval_s: float,
) -> None:
    osim = pytest.importorskip("opensim")
    path, _ = muscle_fixture
    model = osim.Model(str(path))
    model.updMuscles().get(0).setName("zeta")
    load_b = osim.Body("load_b", 1.0, osim.Vec3(0), osim.Inertia(0.01))
    model.addBody(load_b)
    joint_b = osim.SliderJoint(
        "slider_b",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        load_b,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    joint_b.updCoordinate().setName("slide_b")
    joint_b.updCoordinate().setDefaultValue(0.315)
    model.addJoint(joint_b)
    law = model.getMuscles().get(0).getConcreteClassName()
    alpha = getattr(osim, law)("alpha", 10.0, 0.1, 0.2, 0.0)
    alpha.addNewPathPoint("origin_b", model.getGround(), osim.Vec3(0))
    alpha.addNewPathPoint("insertion_b", load_b, osim.Vec3(0))
    model.addForce(alpha)
    model.addMarker(osim.Marker("marker_b", load_b, osim.Vec3(0)))
    model.addMarker(
        osim.Marker("marker_a", model.getBodySet().get("load"), osim.Vec3(0))
    )
    model.finalizeConnections()
    state = model.initSystem()
    for index in range(model.getMuscles().getSize()):
        model.getMuscles().get(index).setActivation(state, 0.05)
    model.equilibrateMuscles(state)
    names = model.getStateVariableNames()
    native_initial = {
        names.get(i): float(model.getStateVariableValue(state, names.get(i)))
        for i in range(names.getSize())
    }
    model.printToXML(str(path))
    times = np.linspace(0.0, 0.01, 11)
    points = np.zeros((len(times), 2, 3))
    points[:, 0, 0] = native_initial["/jointset/slider_b/slide_b/value"]
    points[:, 1, 0] = native_initial["/jointset/slider/slide/value"]
    trc = write_trc(
        TourCapture(times, ("marker_b", "marker_a"), points, np.ones((11, 2), bool)),
        tmp_path / "two_muscle.trc",
        rate_hz=1000,
    )
    table = osim.TimeSeriesTable()
    labels = osim.StdVectorString()
    for name in reversed(tuple(native_initial)):
        labels.append(name)
    table.setColumnLabels(labels)
    for time in times:
        row = osim.RowVector(len(native_initial))
        for index, name in enumerate(reversed(tuple(native_initial))):
            row[index] = native_initial[name]
        table.appendRow(float(time), row)
    guess = tmp_path / "two_muscle_guess.sto"
    osim.STOFileAdapter.write(table, str(guess))
    bounds = {
        name: (
            (0.2, 0.4)
            if name.endswith("/value")
            else (
                (-2.0, 2.0)
                if name.endswith("/speed")
                else ((0.001, 1.0) if name.endswith("/activation") else (0.05, 0.2))
            )
        )
        for name in native_initial
    }
    controls = {"/forceset/alpha": (0.001, 1.0), "/forceset/zeta": (0.001, 1.0)}
    binding = MocoInitialBindings(
        bounds, dict(reversed(tuple(native_initial.items()))), controls
    )
    loaded = osim.Model(str(path))
    loaded.initSystem()
    policy = PassiveReadinessPolicy(
        hashlib.sha256(loaded.dump().encode()).hexdigest(),
        "synthetic two-body fixture only",
        tuple(
            MusclePassiveLimits(
                name,
                (0.1, 10.0),
                1000.0,
                1000.0,
                False,
                False,
                "predeclared synthetic fixture envelope",
            )
            for name in controls
        ),
    )
    request = NativeMocoRequest(
        path,
        trc,
        guess,
        _sha(path),
        _sha(trc),
        _sha(guess),
        binding,
        MocoTrackingConfig(
            horizon_s=0.01,
            mesh_interval_s=mesh_interval_s,
            optim_max_iterations=40,
            allow_unused_references=False,
        ),
        {"marker_b": 3.0, "marker_a": 1.0},
        {
            "marker_b": ("/bodyset/load_b", (0.0, 0.0, 0.0)),
            "marker_a": ("/bodyset/load", (0.0, 0.0, 0.0)),
        },
        CaptureRegistration(np.eye(3), np.zeros(3)),
        "/ground",
        policy,
        {},
    )
    directory = tmp_path / "two_muscle_run"
    incomplete = replace(
        request,
        bindings=MocoInitialBindings(
            binding.state_bounds,
            binding.initial_state,
            {"/forceset/alpha": (0.001, 1.0)},
        ),
    )
    incomplete_report = prepare_native_moco(incomplete, tmp_path / "incomplete")
    assert "native-control-binding-coverage" in incomplete_report.blockers
    prepared = prepare_native_moco(request, directory)
    assert prepared.ready_for_software_solve, prepared.blockers
    solved = solve_native_moco(request, prepared, directory)
    assert solved.success, solved.status
    exported = export_native_moco_bundle(request, prepared, solved, directory)
    native = replay_native_moco_bundle(exported, path, directory)
    bundle = native_replay_contract_types().load_experiment_replay_bundle(
        (directory / "native_replay_bundle.json").read_text(encoding="utf-8")
    )
    assert bundle.model.ordered_input_channel_ids == ("zeta", "alpha")
    assert native.states.shape[1] == 8
    assert native.applied_excitations.shape[1] == 2
    scored = score_native_moco_replay(
        request, prepared, solved, exported, native, directory
    )
    assert np.isfinite(scored.marker_rmse_m)
    with pytest.raises(ValueError, match="finite objective"):
        score_native_moco_replay(
            request,
            prepared,
            replace(solved, objective=None),
            exported,
            native,
            directory,
        )
