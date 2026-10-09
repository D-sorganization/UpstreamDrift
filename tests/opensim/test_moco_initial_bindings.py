"""Explicit Moco initialization constraints precede compliant-muscle guesses."""

from operator import setitem
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
    MocoInitialBindings,
)
from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
    MocoTrackingConfig,
    build_moco_study,
)
from src.engines.physics_engines.opensim.python.tour_matching.trc import write_trc
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytest_plugins = ("tests.opensim.test_native_muscle_replay",)
pytestmark = pytest.mark.unit


def test_bindings_freeze_caller_maps_and_preserve_fixed_bounds() -> None:
    bounds = {"/value": (0.0, 1.0)}
    initial = {"/value": 0.2}
    controls = {"/force": (0.0, 1.0)}
    binding = MocoInitialBindings(bounds, initial, controls)
    bounds["/value"] = (2.0, 3.0)
    initial["/value"] = 2.0
    controls.clear()
    assert binding.state_bounds["/value"] == (0.0, 1.0)
    assert binding.initial_state["/value"] == 0.2
    assert binding.control_bounds["/force"] == (0.0, 1.0)
    with pytest.raises(TypeError):
        setitem(binding.initial_state, "/value", 0.9)


@pytest.mark.parametrize(
    "bounds, initial, controls",
    [
        ({"/value": (0.0, 1.0)}, {}, {}),
        ({"/value": (0.0, 1.0)}, {"/unknown": 0.2}, {}),
        ({"/value": (np.nan, 1.0)}, {"/value": 0.2}, {}),
        ({"/value": (0.0, np.inf)}, {"/value": 0.2}, {}),
        ({"/value": (1.0, 0.0)}, {"/value": 0.2}, {}),
        ({"/value": (0.0, 1.0)}, {"/value": np.nan}, {}),
        ({"/value": (0.0, 1.0)}, {"/value": True}, {}),
        ({"/value": (False, 1.0)}, {"/value": 0.2}, {}),
        ({"/value": (0.0, 1.0)}, {"/value": 2.0}, {}),
        ({"relative": (0.0, 1.0)}, {"relative": 0.2}, {}),
        ({"/value": (0.0, 1.0)}, {"/value": 0.2}, {"/force": (0.0, np.inf)}),
    ],
)
def test_rejects_invalid_explicit_bindings(
    bounds: Any, initial: Any, controls: Any
) -> None:
    with pytest.raises((TypeError, ValueError)):
        MocoInitialBindings(bounds, initial, controls)


def _native_inputs(
    fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> tuple[Path, Path, Path, MocoInitialBindings]:
    osim = pytest.importorskip("opensim")
    path, initial = fixture
    model = osim.Model(str(path))
    model.addMarker(
        osim.Marker("load_marker", model.getBodySet().get("load"), osim.Vec3(0))
    )
    model.finalizeConnections()
    model.printToXML(str(path))
    times = np.array([0.0, 0.005, 0.01])
    points = np.zeros((3, 1, 3))
    points[:, 0, 0] = initial["/jointset/slider/slide/value"]
    capture = TourCapture(times, ("load_marker",), points, np.ones((3, 1), bool))
    trc = write_trc(capture, tmp_path / "target.trc", rate_hz=200)
    table = osim.TimeSeriesTable()
    labels = osim.StdVectorString()
    for name in initial:
        labels.append(name)
    table.setColumnLabels(labels)
    for time in times:
        row = osim.RowVector(len(initial))
        for index, value in enumerate(initial.values()):
            row[index] = value
        # A guess may disagree with an initial constraint; it cannot replace it.
        row[0] = 0.305
        table.appendRow(float(time), row)
    guess = tmp_path / "guess.sto"
    osim.STOFileAdapter.write(table, str(guess))
    bounds = {}
    for name in initial:
        if name.endswith("/value"):
            bounds[name] = (0.2, 0.4)
        elif name.endswith("/speed"):
            bounds[name] = (-2.0, 2.0)
        elif name.endswith("/activation"):
            bounds[name] = (0.001, 1.0)
        else:
            bounds[name] = (0.05, 0.2)
    binding = MocoInitialBindings(bounds, initial, {"/forceset/flexor": (0.001, 1.0)})
    return path, trc, guess, binding


def test_native_compliant_guess_has_frozen_complete_initial_constraints(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    path, trc, guess, binding = _native_inputs(muscle_fixture, tmp_path)
    study = build_moco_study(
        str(path),
        str(trc),
        str(guess),
        MocoTrackingConfig(horizon_s=0.01),
        initial_bindings=binding,
    )
    representation = study.updProblem().createRep()
    for name, expected in binding.initial_state.items():
        info = representation.getStateInfo(name)
        bounds = info.getInitialBounds()
        assert bounds.getLower() == expected
        assert bounds.getUpper() == expected
        general = info.getBounds()
        assert (general.getLower(), general.getUpper()) == binding.state_bounds[name]
    for name, expected_bounds in binding.control_bounds.items():
        bounds = representation.getControlInfo(name).getBounds()
        assert (bounds.getLower(), bounds.getUpper()) == expected_bounds


@pytest.mark.parametrize(
    "missing", ["state", "control", "unknown_state", "extra_control"]
)
def test_native_model_rejects_partial_binding_before_creating_guess(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path, missing: str
) -> None:
    path, trc, guess, binding = _native_inputs(muscle_fixture, tmp_path)
    states = dict(binding.state_bounds)
    initial = dict(binding.initial_state)
    controls = dict(binding.control_bounds)
    if missing in {"state", "unknown_state"}:
        name = next(name for name in states if name.endswith("/fiber_length"))
        interval = states.pop(name)
        value = initial.pop(name)
        if missing == "unknown_state":
            states["/forceset/unknown/fiber_length"] = interval
            initial["/forceset/unknown/fiber_length"] = value
    elif missing == "control":
        controls.clear()
    else:
        controls["/forceset/unknown"] = (0.0, 1.0)
    incomplete = MocoInitialBindings(states, initial, controls)
    with pytest.raises(ValueError, match="complete native"):
        build_moco_study(
            str(path),
            str(trc),
            str(guess),
            MocoTrackingConfig(horizon_s=0.01),
            initial_bindings=incomplete,
        )


def test_fixed_bounds_are_explicitly_supported() -> None:
    binding = MocoInitialBindings({"/value": (0.2, 0.2)}, {"/value": 0.2}, {})
    assert binding.state_bounds["/value"] == (0.2, 0.2)


def test_native_existing_controller_requires_separate_policy(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    osim = pytest.importorskip("opensim")
    path, trc, guess, binding = _native_inputs(muscle_fixture, tmp_path)
    model = osim.Model(str(path))
    controller = osim.PrescribedController()
    controller.addActuator(model.updMuscles().get("flexor"))
    controller.prescribeControlForActuator("flexor", osim.Constant(0.3))
    model.addController(controller)
    model.finalizeConnections()
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="controller"):
        build_moco_study(
            str(path),
            str(trc),
            str(guess),
            MocoTrackingConfig(horizon_s=0.01),
            initial_bindings=binding,
        )


def test_native_vector_actuator_requires_explicit_control_mapping(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    osim = pytest.importorskip("opensim")
    path, trc, guess, binding = _native_inputs(muscle_fixture, tmp_path)
    model = osim.Model(str(path))
    actuator = osim.BodyActuator()
    actuator.setName("vector_drive")
    actuator.setBodyName("/bodyset/load")
    model.addForce(actuator)
    model.finalizeConnections()
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="scalar actuators"):
        build_moco_study(
            str(path),
            str(trc),
            str(guess),
            MocoTrackingConfig(horizon_s=0.01),
            initial_bindings=binding,
        )
