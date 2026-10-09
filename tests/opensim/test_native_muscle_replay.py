"""Independent native muscle-state replay tests for F07/F08 (#11791)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.muscle_replay import (
    replay_muscle_excitations,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def muscle_fixture(tmp_path: Path) -> tuple[Path, dict[str, float]]:
    """Create an actual one-DOF compliant-tendon model, not a golf substitute."""
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    model.setName("synthetic_native_muscle_fixture")
    model.setGravity(osim.Vec3(0))
    body = osim.Body("load", 1.0, osim.Vec3(0), osim.Inertia(0.01))
    model.addBody(body)
    joint = osim.SliderJoint(
        "slider",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    coordinate = joint.updCoordinate()
    coordinate.setName("slide")
    coordinate.setDefaultValue(0.31)
    model.addJoint(joint)
    muscle = osim.Millard2012EquilibriumMuscle("flexor", 10.0, 0.1, 0.2, 0.0)
    muscle.addNewPathPoint("origin", model.getGround(), osim.Vec3(0))
    muscle.addNewPathPoint("insertion", body, osim.Vec3(0))
    model.addForce(muscle)
    model.finalizeConnections()
    state = model.initSystem()
    muscle.setActivation(state, 0.05)
    model.equilibrateMuscles(state)
    names = model.getStateVariableNames()
    initial = {
        names.get(i): float(model.getStateVariableValue(state, names.get(i)))
        for i in range(names.getSize())
    }
    assert any(name.endswith("/fiber_length") for name in initial)
    path = tmp_path / "synthetic_native_muscle.osim"
    model.printToXML(str(path))
    return path, initial


def test_native_replay_preserves_complete_nonzero_initial_state(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    path, initial = muscle_fixture
    times = np.linspace(0.0, 0.04, 9)
    excitations = {"flexor": np.full(times.shape, 0.4)}
    first = replay_muscle_excitations(path, initial, times, excitations)
    second = replay_muscle_excitations(path, initial, times, excitations)
    assert first.state_names == tuple(initial)
    np.testing.assert_array_equal(first.states[0], list(initial.values()))
    np.testing.assert_allclose(first.states, second.states, rtol=0, atol=1e-10)
    assert first.model_sha256 == second.model_sha256
    assert first.input_sha256 == second.input_sha256
    assert first.contact_geometry_count == 0
    assert first.mode == "native-muscle-excitation-replay"
    activation_index = first.state_names.index(
        next(name for name in initial if name.endswith("/activation"))
    )
    assert first.states[-1, activation_index] > first.states[0, activation_index]
    assert first.states[0, activation_index] != 0.4
    assert np.isfinite(first.muscle_forces_n).all()
    assert first.policy["input_boundary"] == "muscle_excitation"
    assert first.policy["interpolation"] == "linear"
    assert first.policy["state_resets"] is False


def test_excitation_changes_native_motion_not_just_recorded_controls(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    path, initial = muscle_fixture
    times = np.linspace(0.0, 0.04, 9)
    low = replay_muscle_excitations(path, initial, times, {"flexor": times * 0 + 0.05})
    high = replay_muscle_excitations(path, initial, times, {"flexor": times * 0 + 0.7})
    position = next(
        i for i, name in enumerate(low.state_names) if name.endswith("/value")
    )
    assert high.states[-1, position] < low.states[-1, position] - 1e-5
    assert high.input_sha256 != low.input_sha256


def test_missing_activation_cannot_be_replaced_with_default_state(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    path, initial = muscle_fixture
    activation = next(name for name in initial if name.endswith("/activation"))
    del initial[activation]
    with pytest.raises(ValueError, match="complete.*state"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.02]), {"flexor": np.array([0.1, 0.1])}
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01, 1.01])
def test_excitation_is_finite_and_bounded_without_clipping(
    muscle_fixture: tuple[Path, dict[str, float]],
    bad: float,
) -> None:
    path, initial = muscle_fixture
    with pytest.raises(ValueError, match="excitation"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.02]), {"flexor": np.array([0.1, bad])}
        )


def test_hidden_reserve_cannot_pass_muscle_only_replay(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    reserve = osim.CoordinateActuator("slide")
    reserve.setName("hidden_reserve")
    model.addForce(reserve)
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="non-muscle actuator"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.02]), {"flexor": np.array([0.1, 0.1])}
        )


def test_existing_controller_is_not_silently_executed(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    controller = osim.PrescribedController()
    controller.addActuator(model.getMuscles().get(0))
    controller.prescribeControlForActuator("flexor", osim.Constant(0.6))
    model.addController(controller)
    model.finalizeConnections()
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="existing controller"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.02]), {"flexor": np.array([0.1, 0.1])}
        )
