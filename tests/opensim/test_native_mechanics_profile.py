"""Exact-source SDK mechanics profiles retain the strict default replay guard."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
    DeclaredColdStart,
)

pytestmark = pytest.mark.unit


def _complex_source(tmp_path: Path) -> tuple[Path, DeclaredColdStart]:
    from tests.opensim.test_native_constrained_muscle import _source

    osim = pytest.importorskip("opensim")
    path, _ = _source(tmp_path, locked=False, custom=True, moving=True, wrapped=True)
    tree = ET.parse(path)
    joint = tree.find(".//CustomJoint[@name='base']")
    assert joint is not None
    coordinates = joint.find("coordinates")
    assert coordinates is not None
    coordinates.append(
        ET.fromstring(
            """<Coordinate name="q2">
              <motion_type>rotational</motion_type>
              <default_value>0.1</default_value>
              <default_speed_value>0</default_speed_value>
              <range>-0.5 0.5</range>
              <clamped>false</clamped>
              <locked>false</locked>
              <prescribed_function />
              <prescribed>false</prescribed>
            </Coordinate>"""
        )
    )
    axis = joint.find("./SpatialTransform/TransformAxis[@name='rotation2']")
    assert axis is not None
    coordinate_names = axis.find("coordinates")
    assert coordinate_names is not None
    coordinate_names.text = "q2"
    law = axis.find("Constant")
    assert law is not None
    axis.remove(law)
    axis.append(
        ET.fromstring(
            """<SimmSpline name="function">
              <x>-0.5 0.5</x>
              <y>0 0.05</y>
            </SimmSpline>"""
        )
    )

    moving = tree.find(".//MovingPathPoint[@name='guide']")
    assert moving is not None
    for axis_name, location_values in (
        ("x", "0.01 0.02"),
        ("y", "0.0 0.01"),
        ("z", "0.0 0.0"),
    ):
        coordinate = moving.find(f"socket_{axis_name}_coordinate")
        assert coordinate is not None
        coordinate.text = "/jointset/base/q2"
        location = moving.find(f"{axis_name}_location")
        assert location is not None
        old_function = next(iter(location))
        location.remove(old_function)
        location.append(
            ET.fromstring(
                f"""<SimmSpline name="function">
              <x>-0.5 0.5</x>
              <y>{location_values}</y>
            </SimmSpline>"""
            )
        )
    path_points = tree.find(
        ".//Millard2012EquilibriumMuscle[@name='flexor']//PathPointSet/objects"
    )
    assert path_points is not None
    insertion = path_points.find("PathPoint[@name='insertion']")
    assert insertion is not None
    conditional = ET.fromstring(
        """<ConditionalPathPoint name="insertion">
          <socket_parent_frame>/bodyset/base</socket_parent_frame>
          <location>0 0.15 0</location>
          <socket_coordinate>/jointset/base/q2</socket_coordinate>
          <range>-0.5 0.5</range>
        </ConditionalPathPoint>"""
    )
    path_points.remove(insertion)
    path_points.append(conditional)
    tree.write(path, encoding="utf-8", xml_declaration=True)

    model = osim.Model(str(path))
    state = model.initSystem()
    for index in range(model.getMuscles().getSize()):
        model.getMuscles().get(index).setActivation(state, 0.05)
    model.equilibrateMuscles(state)
    state_names = model.getStateVariableNames()
    initial = {
        str(state_names.get(index)): float(
            model.getStateVariableValue(state, state_names.get(index))
        )
        for index in range(state_names.getSize())
    }
    declaration = DeclaredColdStart(
        model_path=path,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        named_state=initial,
        time_seconds=0.0,
        lock_targets={},
        chart_bounds={
            "/jointset/base/q": (0.2, 0.4),
            "/jointset/base/q2": (-0.5, 0.5),
        },
        constraint_enforcement={"/constraintset/coupler": True},
        residual_tolerance=1e-9,
    )
    return path, declaration


def test_exact_native_mechanics_profile_enables_only_declared_sdk_components(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
        declare_native_mechanics_execution_profile,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_wrapped_path import (
        declare_wrapped_path_source,
    )
    from tests.opensim.test_native_constrained_muscle import _controls

    path, declaration = _complex_source(tmp_path)
    grid = np.array([0.0, 0.001, 0.002])
    controls = _controls(grid, 0.25)
    wrapped = declare_wrapped_path_source(path)
    assert len(wrapped.conditional_points) == 1
    profile = declare_native_mechanics_execution_profile(
        path, declaration.source_sha256
    )
    assert profile.source_sha256 == declaration.source_sha256
    assert {item.concrete_class for item in profile.components} == {
        "CustomJoint",
        "MovingPathPoint",
    }
    assert len(profile.components) == 2

    with pytest.raises(ValueError, match="moving path law is outside"):
        build_constrained_muscle_bundle(
            declaration,
            grid,
            controls,
            wrapped_path_declaration=wrapped,
        )

    bundle = build_constrained_muscle_bundle(
        declaration,
        grid,
        controls,
        wrapped_path_declaration=wrapped,
        native_mechanics_profile=profile,
    )
    replay = replay_constrained_muscle_bundle(
        bundle,
        declaration,
        wrapped_path_declaration=wrapped,
        native_mechanics_profile=profile,
    )
    assert replay.states.shape[0] == len(grid)
    assert np.isfinite(replay.states).all()
    assert np.isfinite(replay.muscle_forces_n).all()
    assert len(replay.constraint_audits) == len(grid)


def test_native_mechanics_profile_is_source_and_component_bound(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
        declare_native_mechanics_execution_profile,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_wrapped_path import (
        declare_wrapped_path_source,
    )
    from tests.opensim.test_native_constrained_muscle import _controls

    path, declaration = _complex_source(tmp_path)
    profile = declare_native_mechanics_execution_profile(
        path, declaration.source_sha256
    )
    wrapped = declare_wrapped_path_source(path)
    grid = np.array([0.0, 0.001, 0.002])
    controls = _controls(grid, 0.25)

    changed = path.read_bytes().replace(b"0.05</y>", b"0.06</y>", 1)
    path.write_bytes(changed)
    changed_declaration = replace(
        declaration,
        source_sha256=hashlib.sha256(changed).hexdigest(),
    )
    changed_wrapped = declare_wrapped_path_source(path)
    with pytest.raises(ValueError, match="native mechanics profile source identity"):
        build_constrained_muscle_bundle(
            changed_declaration,
            grid,
            controls,
            wrapped_path_declaration=changed_wrapped,
            native_mechanics_profile=profile,
        )


def test_native_mechanics_profile_crosses_moco_export_and_fresh_replay(
    tmp_path: Path,
) -> None:
    pytest.importorskip("opensim")
    if os.environ.get("UPSTREAMDRIFT_NATIVE_PROFILE_LIFECYCLE_CHILD") != "1":
        child_root = tmp_path / "bounded-profile-lifecycle"
        child_root.mkdir()
        child_work = child_root / "child-work"
        child_work.mkdir()
        log_path = tmp_path / "bounded-profile-lifecycle.log"
        environment = os.environ.copy()
        environment["UPSTREAMDRIFT_NATIVE_PROFILE_LIFECYCLE_CHILD"] = "1"
        inherited_paths = [entry for entry in sys.path if entry]
        prior_pythonpath = environment.get("PYTHONPATH")
        if prior_pythonpath:
            inherited_paths.append(prior_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(inherited_paths)
        child_code = "\n".join(
            (
                "from pathlib import Path",
                f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})",
                "from tests.opensim.test_native_mechanics_profile import test_native_mechanics_profile_crosses_moco_export_and_fresh_replay",
                f"test_native_mechanics_profile_crosses_moco_export_and_fresh_replay(Path({str(child_work)!r}))",
            )
        )
        command = [
            sys.executable,
            "-c",
            child_code,
        ]
        with log_path.open("wb") as log_stream:
            child = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                **({"start_new_session": True} if os.name != "nt" else {}),
            )
            try:
                return_code = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                        timeout=5,
                    )
                else:
                    import signal

                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
                progress_path = child_work / "profile-run" / "native_moco_progress.json"
                if not progress_path.is_file():
                    pytest.fail(
                        "bounded lifecycle child timed out without a native progress receipt"
                    )
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                receipt = {
                    "schema_version": "native-moco-lifecycle-timeout-v1",
                    "outcome": "timed-out",
                    "request_sha256": progress["request_sha256"],
                    "source_sha256": progress["source_sha256"],
                    "stage": progress["stage"],
                    "runtime_phase": "optimization",
                    "scope": "three-knot 2 ms complex native Moco fixture",
                    "solve_succeeded": False,
                    "replay_produced": False,
                    "timeout_seconds": 30,
                    "elapsed_wall_seconds": progress["elapsed_wall_seconds"],
                    "progress_receipt": str(progress_path),
                    "child_log": str(log_path),
                }
                (tmp_path / "native-moco-lifecycle-timeout.json").write_text(
                    json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                assert set(receipt) == {
                    "schema_version",
                    "outcome",
                    "request_sha256",
                    "source_sha256",
                    "stage",
                    "runtime_phase",
                    "scope",
                    "solve_succeeded",
                    "replay_produced",
                    "timeout_seconds",
                    "elapsed_wall_seconds",
                    "progress_receipt",
                    "child_log",
                }
                assert receipt["schema_version"] == "native-moco-lifecycle-timeout-v1"
                assert receipt["outcome"] == "timed-out"
                assert receipt["stage"] == "optimizing", receipt
                assert receipt["runtime_phase"] == "optimization"
                assert receipt["solve_succeeded"] is False
                assert receipt["replay_produced"] is False
                assert len(receipt["request_sha256"]) == 64
                assert len(receipt["source_sha256"]) == 64
                return
        output = log_path.read_text(encoding="utf-8", errors="replace")
        assert return_code == 0, output
        return

    from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
        MocoInitialBindings,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
        MocoTrackingConfig,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_diagnostic_contract import (
        DiagnosticCode,
        NativeDiagnosticPolicy,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_replay import (
        export_native_moco_bundle,
        replay_native_moco_bundle,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner import (
        NativeMocoRequest,
        prepare_native_moco,
        solve_native_moco,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_passive_readiness import (
        MusclePassiveLimits,
        PassiveReadinessPolicy,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
        declare_native_mechanics_execution_profile,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_wrapped_path import (
        declare_wrapped_path_source,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.registration import (
        CaptureRegistration,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.trc import (
        TourCapture,
        write_trc,
    )

    osim = pytest.importorskip("opensim")
    source_path, original_declaration = _complex_source(tmp_path)
    model = osim.Model(str(source_path))
    state = model.initSystem()
    muscles = model.getMuscles()
    for index in range(muscles.getSize()):
        muscles.get(index).setActivation(state, 0.05)
    model.equilibrateMuscles(state)
    state_names = model.getStateVariableNames()
    initial = {
        str(state_names.get(index)): float(
            model.getStateVariableValue(state, state_names.get(index))
        )
        for index in range(state_names.getSize())
    }
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    declaration = DeclaredColdStart(
        model_path=source_path,
        source_sha256=source_sha256,
        named_state=initial,
        time_seconds=0.0,
        lock_targets=original_declaration.lock_targets,
        chart_bounds=original_declaration.chart_bounds,
        constraint_enforcement=original_declaration.constraint_enforcement,
        residual_tolerance=original_declaration.residual_tolerance,
    )
    profile = declare_native_mechanics_execution_profile(source_path, source_sha256)
    wrapped = declare_wrapped_path_source(source_path)

    state_bounds: dict[str, tuple[float, float]] = {}
    for path, value in initial.items():
        if path.endswith("/value"):
            coordinate_path = path.removesuffix("/value")
            coordinate_name = coordinate_path.rsplit("/", 1)[-1]
            coordinate = model.getCoordinateSet().get(coordinate_name)
            interval = declaration.chart_bounds.get(coordinate_path)
            if interval is None:
                interval = (
                    float(coordinate.getRangeMin()),
                    float(coordinate.getRangeMax()),
                )
            if not np.isfinite(interval).all() or interval[0] == interval[1]:
                interval = (value - 0.5, value + 0.5)
            state_bounds[path] = interval
        elif path.endswith("/speed"):
            state_bounds[path] = (-2.0, 2.0)
        elif path.endswith("/activation"):
            state_bounds[path] = (0.001, 1.0)
        else:
            state_bounds[path] = (0.05, 0.2)
    controls = {
        str(model.getActuators().get(index).getAbsolutePathString()): (0.01, 0.8)
        for index in range(model.getActuators().getSize())
    }
    bindings = MocoInitialBindings(state_bounds, initial, controls)

    times = np.array([0.0, 0.001, 0.002])
    marker = model.getMarkerSet().get("base_marker")
    marker_position = marker.getLocationInGround(state)
    points = np.tile(
        np.asarray([marker_position.get(index) for index in range(3)]),
        (times.size, 1, 1),
    )
    capture = TourCapture(times, ("base_marker",), points, np.ones((3, 1), dtype=bool))
    trc_path = write_trc(capture, tmp_path / "profile-target.trc", rate_hz=1000.0)
    guess_table = osim.TimeSeriesTable()
    labels = osim.StdVectorString()
    for name in initial:
        labels.append(name)
    guess_table.setColumnLabels(labels)
    for time_value in times:
        row = osim.RowVector(len(initial))
        for index, value in enumerate(initial.values()):
            row[index] = value
        guess_table.appendRow(float(time_value), row)
    guess_path = tmp_path / "profile-guess.sto"
    osim.STOFileAdapter.write(guess_table, str(guess_path))
    policy = PassiveReadinessPolicy(
        hashlib.sha256(model.dump().encode()).hexdigest(),
        "bounded synthetic fixture policy; not human physiology evidence",
        tuple(
            MusclePassiveLimits(
                str(muscles.get(index).getAbsolutePathString()),
                (0.1, 10.0),
                10000.0,
                10000.0,
                False,
                False,
                "bounded synthetic fixture envelope",
            )
            for index in range(muscles.getSize())
        ),
    )
    request = NativeMocoRequest(
        model_path=source_path,
        trc_path=trc_path,
        states_guess_path=guess_path,
        model_sha256=source_sha256,
        trc_sha256=hashlib.sha256(trc_path.read_bytes()).hexdigest(),
        states_guess_sha256=hashlib.sha256(guess_path.read_bytes()).hexdigest(),
        bindings=bindings,
        config=MocoTrackingConfig(
            horizon_s=0.002,
            mesh_interval_s=0.001,
            optim_max_iterations=5,
            optim_convergence_tolerance=0.1,
            optim_constraint_tolerance=0.1,
            allow_unused_references=False,
        ),
        marker_weights={"base_marker": 1.0},
        marker_bindings={"base_marker": ("/bodyset/base", (0.0, -0.1, 0.0))},
        registration=CaptureRegistration(np.eye(3), np.zeros(3)),
        reference_frame_path="/ground",
        passive_policy=policy,
        excluded_markers={},
        constrained_cold_start=declaration,
        diagnostic_policy=NativeDiagnosticPolicy(
            source_sha256,
            (
                DiagnosticCode.WRAPPED_PATH_TRANSITION_UNQUALIFIED,
                DiagnosticCode.GUIDE_WORK_BUDGET_UNAVAILABLE,
                DiagnosticCode.GLOBAL_CHART_PROOF_UNAVAILABLE,
            ),
            "bounded native SDK capability test; no physical qualification",
        ),
        wrapped_path_declaration=wrapped,
        native_mechanics_profile=profile,
    )

    prepared = prepare_native_moco(request, tmp_path / "profile-run")
    if not prepared.ready_for_software_solve:
        pytest.fail(
            "native profile preparation rejected the exact fixture: "
            + ", ".join(prepared.blockers)
        )
    solved = solve_native_moco(request, prepared, tmp_path / "profile-run")
    if not solved.success:
        pytest.fail(f"bounded native profile Moco solve failed: {solved.status}")
    exported = export_native_moco_bundle(
        request, prepared, solved, tmp_path / "profile-run"
    )
    assert exported.native_mechanics_profile_sha256 == profile.profile_sha256
    replayed = replay_native_moco_bundle(
        exported,
        source_path,
        tmp_path / "profile-run",
        constrained_cold_start=declaration,
        wrapped_path_declaration=wrapped,
        native_mechanics_profile=profile,
    )
    assert np.array_equal(replayed.times, exported.output_times)
    assert np.isfinite(replayed.states).all()
