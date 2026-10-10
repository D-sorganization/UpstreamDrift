"""Constrained muscle replay needs an explicit cold-start policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
    DeclaredColdStart,
)

pytestmark = pytest.mark.unit


def _custom_base(path: Path) -> None:
    """Use the native XML loader for CustomJoint on this OpenSim4.6 binding."""
    tree = ET.parse(path)
    joint = tree.find(".//PinJoint[@name='base']")
    assert joint is not None
    joint.tag = "CustomJoint"
    spatial = ET.fromstring("<SpatialTransform />")
    joint.append(spatial)
    for index, (axis_name, axis_vector) in enumerate(
        (
            ("rotation1", "0 0 1"),
            ("rotation2", "1 0 0"),
            ("rotation3", "0 1 0"),
            ("translation1", "1 0 0"),
            ("translation2", "0 1 0"),
            ("translation3", "0 0 1"),
        )
    ):
        if index == 0:
            law = "<LinearFunction name='function'><coefficients>1 0</coefficients></LinearFunction>"
        else:
            law = "<Constant name='function'><value>0</value></Constant>"
        coordinates = "q" if index == 0 else ""
        spatial.append(
            ET.fromstring(
                f"<TransformAxis name='{axis_name}'><coordinates>{coordinates}</coordinates>"
                f"<axis>{axis_vector}</axis>{law}</TransformAxis>"
            )
        )
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _moving_flexor(path: Path) -> None:
    tree = ET.parse(path)
    flexor = tree.find(".//Millard2012EquilibriumMuscle[@name='flexor']")
    assert flexor is not None
    points = flexor.find(".//PathPointSet/objects")
    assert points is not None
    guide = ET.fromstring("""
        <MovingPathPoint name="guide">
          <socket_parent_frame>/bodyset/follower</socket_parent_frame>
          <socket_x_coordinate>/jointset/base/q</socket_x_coordinate>
          <socket_y_coordinate>/jointset/base/q</socket_y_coordinate>
          <socket_z_coordinate>/jointset/base/q</socket_z_coordinate>
          <x_location><LinearFunction><coefficients>0.01 0</coefficients></LinearFunction></x_location>
          <y_location><Constant><value>-0.1</value></Constant></y_location>
          <z_location><Constant><value>0</value></Constant></z_location>
        </MovingPathPoint>
    """)
    points.insert(1, guide)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _source(
    tmp_path: Path, *, locked: bool = True, custom: bool = False, moving: bool = False
) -> tuple[Path, DeclaredColdStart]:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    model.setGravity(osim.Vec3(0))
    for name, coordinate_name, default in (
        ("base", "q", 0.3),
        ("follower", "dependent", 0.6),
        ("anchor", "fixed", 0.2),
    ):
        body = osim.Body(name, 1.0, osim.Vec3(0), osim.Inertia(0.1))
        joint = osim.PinJoint(
            name,
            model.getGround(),
            osim.Vec3(0),
            osim.Vec3(0),
            body,
            osim.Vec3(0),
            osim.Vec3(0),
        )
        joint.updCoordinate().setName(coordinate_name)
        joint.updCoordinate().setDefaultValue(default)
        if name == "anchor" and locked:
            joint.updCoordinate().setDefaultLocked(True)
        model.addBody(body)
        model.addJoint(joint)
    coupler = osim.CoordinateCouplerConstraint()
    coupler.setName("coupler")
    names = osim.ArrayStr()
    names.append("q")
    coupler.setIndependentCoordinateNames(names)
    coupler.setDependentCoordinateName("dependent")
    coupler.setFunction(osim.LinearFunction(2.0, 0.0))
    model.addConstraint(coupler)
    for muscle_name, body_name in (("flexor", "base"), ("extensor", "follower")):
        muscle = osim.Millard2012EquilibriumMuscle(muscle_name, 10.0, 0.1, 0.2, 0.0)
        muscle.addNewPathPoint("origin", model.getGround(), osim.Vec3(0.0, -0.15, 0))
        muscle.addNewPathPoint(
            "insertion", model.getBodySet().get(body_name), osim.Vec3(0.0, 0.15, 0)
        )
        model.addForce(muscle)
    model.addMarker(
        osim.Marker(
            "base_marker", model.getBodySet().get("base"), osim.Vec3(0.0, -0.1, 0.0)
        )
    )
    model.finalizeConnections()
    state = model.initSystem()
    for index in range(model.getMuscles().getSize()):
        model.getMuscles().get(index).setActivation(state, 0.05)
    model.equilibrateMuscles(state)
    native_names = model.getStateVariableNames()
    initial = {
        native_names.get(i): float(
            model.getStateVariableValue(state, native_names.get(i))
        )
        for i in range(native_names.getSize())
    }
    path = tmp_path / "coupled_muscle.osim"
    model.printToXML(str(path))
    if custom:
        _custom_base(path)
    if moving:
        _moving_flexor(path)
    declaration = DeclaredColdStart(
        model_path=path,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        named_state=initial,
        time_seconds=0.0,
        lock_targets={"/jointset/anchor/fixed": 0.2} if locked else {},
        chart_bounds={"/jointset/base/q": (0.2, 0.4)},
        constraint_enforcement={"/constraintset/coupler": True},
        residual_tolerance=1e-9,
    )
    return path, declaration


def _controls(
    grid: np.ndarray[Any, np.dtype[np.float64]], flexor: float
) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    return {
        "flexor": np.full(grid.shape, flexor),
        "extensor": np.full(grid.shape, 0.15),
    }


def test_constrained_muscle_bundle_independently_replays_named_state(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )

    path, declaration = _source(tmp_path)
    grid = np.array([0.0, 0.001, 0.002])
    controls = {"flexor": np.array([0.1, 0.2, 0.2]), "extensor": np.full(3, 0.15)}
    bundle = build_constrained_muscle_bundle(declaration, grid, controls)
    first = replay_constrained_muscle_bundle(bundle, declaration)
    second = replay_constrained_muscle_bundle(bundle, declaration)
    assert bundle.input_history.input_kind.value == "muscle_excitation"
    assert bundle.model.variant_id == "declared-constrained-muscles"
    np.testing.assert_array_equal(first.states[0], second.states[0])
    np.testing.assert_allclose(first.states, second.states, rtol=0, atol=1e-10)
    assert first.states.shape[0] == len(grid)
    assert first.muscle_names == ("flexor", "extensor")
    assert first.applied_excitations[-1, 0] == pytest.approx(0.2, abs=1e-12)
    assert first.constraint_audits[0].position_errors == pytest.approx(
        (0.0, 0.0), abs=1e-9
    )


def test_custom_joint_coupler_and_two_muscles_replay_on_fresh_native_model(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )

    osim = pytest.importorskip("opensim")
    path, declaration = _source(tmp_path, locked=False, custom=True)
    assert (
        osim.Model(str(path)).getJointSet().get("base").getConcreteClassName()
        == "CustomJoint"
    )
    grid = np.array([0.0, 0.001, 0.002])
    bundle = build_constrained_muscle_bundle(declaration, grid, _controls(grid, 0.3))
    replay = replay_constrained_muscle_bundle(bundle, declaration)
    assert replay.states.shape == (3, len(declaration.named_state))
    assert len(replay.constraint_audits) == 3


def test_custom_joint_changed_transform_law_needs_new_policy(tmp_path: Path) -> None:
    from dataclasses import replace
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
    )

    path, declaration = _source(tmp_path, locked=False, custom=True)
    tree = ET.parse(path)
    law = tree.find(
        ".//CustomJoint[@name='base']/SpatialTransform/"
        "TransformAxis[@name='rotation1']/LinearFunction/coefficients"
    )
    assert law is not None
    law.text = "2 0"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    changed = replace(
        declaration, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    grid = np.array([0.0, 0.001])
    with pytest.raises(ValueError, match="CustomJoint rotation chart"):
        build_constrained_muscle_bundle(changed, grid, _controls(grid, 0.3))


def test_moving_path_with_coupler_replays_under_declared_law(tmp_path: Path) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )

    path, declaration = _source(tmp_path, locked=False, moving=True)
    grid = np.array([0.0, 0.001, 0.002])
    bundle = build_constrained_muscle_bundle(declaration, grid, _controls(grid, 0.3))
    replay = replay_constrained_muscle_bundle(bundle, declaration)
    assert replay.states.shape == (3, len(declaration.named_state))
    assert np.isfinite(replay.muscle_forces_n).all()
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(path))
    state = model.initSystem()
    guide = osim.MovingPathPoint.safeDownCast(
        model.getComponent("/forceset/flexor/path/guide")
    )
    base = guide.getLocation(state).get(0)
    model.getCoordinateSet().get("q").setValue(state, 0.31, False)
    model.realizePosition(state)
    assert guide.getLocation(state).get(0) - base == pytest.approx(0.0001, abs=1e-12)


def test_moving_path_non_linear_law_requires_separate_policy(tmp_path: Path) -> None:
    from dataclasses import replace
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
    )

    path, declaration = _source(tmp_path, locked=False, moving=True)
    tree = ET.parse(path)
    location = tree.find(".//MovingPathPoint[@name='guide']/x_location")
    assert location is not None
    location.clear()
    location.append(
        ET.fromstring("<SimmSpline><x>0 0.5 1</x><y>0 0.01 0.03</y></SimmSpline>")
    )
    tree.write(path, encoding="utf-8", xml_declaration=True)
    changed = replace(
        declaration, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    grid = np.array([0.0, 0.001])
    with pytest.raises(ValueError, match="moving path law"):
        build_constrained_muscle_bundle(changed, grid, _controls(grid, 0.3))


def test_constrained_replay_rejects_wrong_declared_lock_target(tmp_path: Path) -> None:
    from dataclasses import replace
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )

    _, declaration = _source(tmp_path)
    grid = np.array([0.0, 0.001])
    bundle = build_constrained_muscle_bundle(declaration, grid, _controls(grid, 0.1))
    changed = replace(declaration, lock_targets={"/jointset/anchor/fixed": 0.3})
    with pytest.raises(ValueError, match="policy|identity|target|state"):
        replay_constrained_muscle_bundle(bundle, changed)


def test_changed_future_excitation_preserves_prefix_then_changes_activation(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )

    _, declaration = _source(tmp_path)
    grid = np.array([0.0, 0.001, 0.002, 0.003])
    reference = build_constrained_muscle_bundle(declaration, grid, _controls(grid, 0.1))
    changed = build_constrained_muscle_bundle(
        declaration,
        grid,
        {"flexor": np.array([0.1, 0.1, 0.9, 0.9]), "extensor": np.full(4, 0.15)},
    )
    first = replay_constrained_muscle_bundle(reference, declaration)
    second = replay_constrained_muscle_bundle(changed, declaration)
    np.testing.assert_allclose(first.states[:2], second.states[:2], atol=1e-10, rtol=0)
    assert first.input_sha256 != second.input_sha256
    activation = first.state_names.index("/forceset/flexor/activation")
    assert second.states[-1, activation] > first.states[-1, activation]
    q = first.state_names.index("/jointset/base/q/value")
    assert second.states[-1, q] != pytest.approx(first.states[-1, q], abs=1e-12)
    assert all(
        all(
            abs(value) < declaration.residual_tolerance
            for value in audit.position_errors
        )
        for audit in second.constraint_audits
    )


def test_source_mutation_and_unreviewed_force_fail_closed(tmp_path: Path) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )

    osim = pytest.importorskip("opensim")
    path, declaration = _source(tmp_path)
    grid = np.array([0.0, 0.001])
    controls = _controls(grid, 0.1)
    bundle = build_constrained_muscle_bundle(declaration, grid, controls)
    original = path.read_bytes()
    path.write_bytes(original + b"\n<!-- source changed -->\n")
    with pytest.raises(ValueError, match="source"):
        replay_constrained_muscle_bundle(bundle, declaration)
    path.write_bytes(original)
    model = osim.Model(str(path))
    model.addForce(osim.CoordinateLimitForce("q", 1, 1, -1, 1, 0.1, 0.01))
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="source|force|component"):
        build_constrained_muscle_bundle(declaration, grid, controls)


def _moco_problem(
    path: Path, declaration: DeclaredColdStart, *, target_q: float | None = None
) -> Any:
    from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
        MocoInitialBindings,
        apply_moco_initial_bindings,
    )

    osim = pytest.importorskip("opensim")
    initial = declaration.named_state
    bounds = {
        name: (max(0.01, value - 0.1), min(1.0, value + 0.1))
        if name.endswith("/activation")
        else (value - 1.0, value + 1.0)
        for name, value in initial.items()
    }
    bindings = MocoInitialBindings(
        bounds,
        initial,
        {"/forceset/flexor": (0.0, 1.0), "/forceset/extensor": (0.0, 1.0)},
    )
    study = osim.MocoStudy()
    problem = study.updProblem()
    problem.setModelProcessor(osim.ModelProcessor(str(path)))
    problem.setTimeBounds(0.0, 0.01)
    apply_moco_initial_bindings(problem, str(path), bindings, osim)
    if target_q is not None:
        initial_q = initial["/jointset/base/q/value"]
        problem.setStateInfo(
            "/jointset/base/q/value",
            osim.MocoBounds(0.2, 0.4),
            osim.MocoInitialBounds(initial_q),
            osim.MocoFinalBounds(target_q),
        )
    effort = osim.MocoControlGoal("effort")
    problem.addGoal(effort)
    return study


def test_native_moco_rejects_locked_source_without_reviewed_transformation(
    tmp_path: Path,
) -> None:
    """The replay policy must not imply Moco can optimize locked sources."""
    path, declaration = _source(tmp_path)
    with pytest.raises(RuntimeError, match="Moco does not support locked coordinates"):
        _moco_problem(path, declaration).initCasADiSolver()


def test_native_moco_solves_coupled_unlocked_muscle_problem(
    tmp_path: Path,
) -> None:
    """Native optimizer feasibility, separate from anatomical qualification."""
    osim = pytest.importorskip("opensim")
    path, declaration = _source(tmp_path, locked=False)
    target_q = declaration.named_state["/jointset/base/q/value"] + 5e-6
    study = _moco_problem(path, declaration, target_q=target_q)
    solver = osim.MocoCasADiSolver.safeDownCast(study.initCasADiSolver())
    solver.set_num_mesh_intervals(10)
    solver.set_parallel(0)
    solver.set_optim_max_iterations(300)
    solver.set_optim_convergence_tolerance(1e-6)
    solver.set_optim_constraint_tolerance(1e-7)
    solution = study.solve()
    assert solution.success(), solution.getStatus()
    assert solution.getStateMat("/jointset/base/q/value")[-1] == pytest.approx(
        target_q, abs=1e-5
    )
    assert (
        abs(
            solution.getStateMat("/jointset/base/q/value")[-1]
            - declaration.named_state["/jointset/base/q/value"]
        )
        > 2e-6
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )

    grid = np.asarray(solution.getTimeMat(), dtype=float).reshape(-1)
    controls = {
        name: np.asarray(
            solution.getControlMat(f"/forceset/{name}"), dtype=float
        ).reshape(-1)
        for name in ("flexor", "extensor")
    }
    assert len(grid) > 2
    bundle = build_constrained_muscle_bundle(declaration, grid, controls)
    replay = replay_constrained_muscle_bundle(bundle, declaration)
    np.testing.assert_array_equal(replay.times, grid)
    assert np.isfinite(replay.states).all()
    assert replay.applied_excitations.shape == (len(grid), 2)
    optimized = np.column_stack(
        [
            np.asarray(solution.getStateMat(name), dtype=float).reshape(-1)
            for name in replay.state_names
        ]
    )
    differences = np.abs(optimized - replay.states)
    groups = {
        "coordinates_rad": tuple(
            i for i, name in enumerate(replay.state_names) if name.endswith("/value")
        ),
        "speeds_rad_s": tuple(
            i for i, name in enumerate(replay.state_names) if name.endswith("/speed")
        ),
        "activations": tuple(
            i
            for i, name in enumerate(replay.state_names)
            if name.endswith("/activation")
        ),
        "fiber_lengths_m": tuple(
            i
            for i, name in enumerate(replay.state_names)
            if name.endswith("/fiber_length")
        ),
    }
    assert all(groups.values())
    observed_errors = {
        group: float(np.max(differences[:, columns]))
        for group, columns in groups.items()
    }
    assert all(np.isfinite(value) for value in observed_errors.values())
    assert observed_errors["coordinates_rad"] < 2e-7
    assert observed_errors["speeds_rad_s"] < 1e-7
    assert observed_errors["activations"] < 1e-8
    assert observed_errors["fiber_lengths_m"] < 1e-7
    q_index = replay.state_names.index("/jointset/base/q/value")
    assert abs(replay.states[-1, q_index] - target_q) < 2e-7
    assert (
        max(
            abs(value)
            for audit in replay.constraint_audits
            for value in (*audit.position_errors, *audit.velocity_errors)
        )
        < declaration.residual_tolerance
    )


def test_marker_scoring_uses_replayed_coupled_state_without_reassembly(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        observe_constrained_markers,
        replay_constrained_muscle_bundle,
    )

    _, declaration = _source(tmp_path, locked=False)
    grid = np.array([0.0, 0.001, 0.002])
    replay = replay_constrained_muscle_bundle(
        build_constrained_muscle_bundle(declaration, grid, _controls(grid, 0.4)),
        declaration,
    )
    placements = {"base_marker": ("/bodyset/base", (0.0, -0.1, 0.0))}
    indices = np.array([0, 2], dtype=np.intp)
    first, digest = observe_constrained_markers(
        declaration, replay, placements, indices
    )
    second, repeated = observe_constrained_markers(
        declaration, replay, placements, indices
    )
    np.testing.assert_array_equal(first, second)
    assert digest == repeated
    assert not first.flags.writeable
    with pytest.raises(ValueError, match="frame"):
        observe_constrained_markers(
            declaration,
            replay,
            {"base_marker": ("/forceset/flexor", (0, 0, 0))},
            indices,
        )


@pytest.mark.parametrize("locked", [False, True])
def test_maintained_moco_preparation_separates_replay_from_solver_capability(
    tmp_path: Path, locked: bool
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
        MocoTrackingConfig,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner import (
        NativeMocoRequest,
        prepare_native_moco,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.registration import (
        CaptureRegistration,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.trc import write_trc
    from src.shared.python.motion_matching.tour_capture_contract import TourCapture

    osim = pytest.importorskip("opensim")
    path, declaration = _source(tmp_path, locked=locked)
    times = np.array([0.0, 0.001, 0.002])
    capture = TourCapture(
        times,
        ("base_marker",),
        np.zeros((len(times), 1, 3)),
        np.ones((len(times), 1), dtype=bool),
    )
    trc = write_trc(capture, tmp_path / "reference.trc", rate_hz=1000.0)
    table = osim.TimeSeriesTable()
    names = osim.StdVectorString()
    for name in declaration.named_state:
        names.append(name)
    table.setColumnLabels(names)
    for time in times:
        row = osim.RowVector(len(declaration.named_state))
        for index, value in enumerate(declaration.named_state.values()):
            row[index] = value
        table.appendRow(float(time), row)
    guess = tmp_path / "guess.sto"
    osim.STOFileAdapter.write(table, str(guess))
    from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
        MocoInitialBindings,
    )

    request = NativeMocoRequest(
        path,
        trc,
        guess,
        declaration.source_sha256,
        hashlib.sha256(trc.read_bytes()).hexdigest(),
        hashlib.sha256(guess.read_bytes()).hexdigest(),
        MocoInitialBindings(
            {
                name: (value - 1, value + 1)
                for name, value in declaration.named_state.items()
            },
            declaration.named_state,
            {"/forceset/flexor": (0.0, 1.0), "/forceset/extensor": (0.0, 1.0)},
        ),
        MocoTrackingConfig(
            horizon_s=0.002, mesh_interval_s=0.0004, allow_unused_references=False
        ),
        {"base_marker": 1.0},
        {"base_marker": ("/bodyset/base", (0.0, -0.1, 0.0))},
        CaptureRegistration(np.eye(3), np.zeros(3)),
        "/ground",
        None,
        {},
        declaration,
    )
    report = prepare_native_moco(request, tmp_path / "prepared")
    assert "independent-native-replay-unavailable" not in report.blockers
    assert "native-constraint-policy-unavailable" not in report.blockers
    assert ("native-moco-locked-coordinates-unsupported" in report.blockers) == locked
    assert "passive-policy-unavailable" in report.blockers

    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_request import (
        load_native_moco_request,
    )
    from tests.opensim.test_native_moco_runner import _write_request

    request_file = tmp_path / "constrained-request.json"
    _write_request(request, request_file)
    payload = json.loads(request_file.read_text(encoding="utf-8"))
    payload["constrained_cold_start"] = {
        "version": "declared-constrained-muscles/2.0.0",
        "lock_targets": dict(declaration.lock_targets),
        "chart_bounds": dict(declaration.chart_bounds),
        "linear_chart_bounds": {},
        "constraint_enforcement": dict(declaration.constraint_enforcement),
        "residual_tolerance": declaration.residual_tolerance,
    }
    request_file.write_text(json.dumps(payload), encoding="utf-8")
    restored = load_native_moco_request(request_file)
    assert restored.identity_sha256 == request.identity_sha256
    payload["constrained_cold_start"]["version"] = "declared-constrained-muscles/0.0.0"
    request_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown constrained muscle policy version"):
        load_native_moco_request(request_file)
