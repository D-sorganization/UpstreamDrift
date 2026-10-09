"""Independent native muscle-state replay tests for F07/F08 (#11791)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching import (
    replay_muscle_excitations,
)

pytestmark = pytest.mark.unit


@pytest.fixture(params=["Millard2012EquilibriumMuscle", "Thelen2003Muscle"])
def muscle_fixture(
    tmp_path: Path, request: pytest.FixtureRequest
) -> tuple[Path, dict[str, float]]:
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
    muscle = getattr(osim, request.param)("flexor", 10.0, 0.1, 0.2, 0.0)
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
    native_model = pytest.importorskip("opensim").Model(str(path))
    actual_law = native_model.getMuscles().get(0).getConcreteClassName()
    assert actual_law in first.policy["muscle_laws"]
    assert first.policy["muscle_class_policy"] == "exact-supported-concrete-law/1.0.0"


def test_unknown_derived_law_identity_is_rejected_before_native_init(
    muscle_fixture: tuple[Path, dict[str, float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proxy identity boundary test; this does not load a compiled C++ plugin."""
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    original_class_name = osim.Component.getConcreteClassName
    probe_model = osim.Model(str(path))
    muscle = probe_model.getMuscles().get(0)
    supported_name = original_class_name(muscle)
    law_type = getattr(osim, supported_name)
    assert law_type.safeDownCast(muscle) is not None

    def report_unknown_concrete_law(component: Any) -> str:
        actual = original_class_name(component)
        return "UnqualifiedDerived" + actual if actual == supported_name else actual

    def forbid_initialization(model: Any) -> None:
        raise AssertionError("unknown concrete law reached native initSystem")

    monkeypatch.setattr(
        osim.Component, "getConcreteClassName", report_unknown_concrete_law
    )
    monkeypatch.setattr(
        osim.Muscle, "getConcreteClassName", report_unknown_concrete_law
    )
    assert muscle.getConcreteClassName().startswith("UnqualifiedDerived")
    assert law_type.safeDownCast(muscle) is not None
    monkeypatch.setattr(osim.Model, "initSystem", forbid_initialization)
    with pytest.raises(ValueError, match="concrete muscle law"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])}
        )


def test_zero_fiber_length_is_not_an_admissible_physical_initial_state(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    path, initial = muscle_fixture
    initial[next(name for name in initial if name.endswith("/fiber_length"))] = 0
    with pytest.raises(ValueError, match="muscle state domain|fiber length.*positive"):
        replay_muscle_excitations(
            path, initial, np.array([0, 0.01]), {"flexor": np.array([0.4, 0.4])}
        )


@pytest.mark.parametrize("mode", ["activation_dynamics", "tendon_compliance"])
def test_ignored_muscle_dynamics_need_a_separate_state_policy(
    muscle_fixture: tuple[Path, dict[str, float]], mode: str
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    muscle = model.updMuscles().get(0)
    getattr(muscle, "set_ignore_" + mode)(True)
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="ignored.*dynamics.*policy"):
        replay_muscle_excitations(
            path, initial, np.array([0, 0.01]), {"flexor": np.array([0.4, 0.4])}
        )


def test_restored_native_force_matches_independent_physical_state(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    state = model.initSystem()
    for name, value in initial.items():
        model.setStateVariableValue(state, name, value)
    model.realizeDynamics(state)
    expected = model.getMuscles().get(0).getActuation(state)
    result = replay_muscle_excitations(
        path, initial, np.array([0, 0.01]), {"flexor": np.array([0.4, 0.4])}
    )
    np.testing.assert_allclose(result.muscle_forces_n[0, 0], expected, atol=1e-12)


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


def test_actual_native_excitations_follow_bounded_linear_input(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    path, initial = muscle_fixture
    times = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    excitation = np.array([0.05, 0.25, 0.45, 0.65, 0.85])
    result = replay_muscle_excitations(path, initial, times, {"flexor": excitation})
    np.testing.assert_allclose(result.applied_excitations[:, 0], excitation, atol=1e-12)
    refined_times = np.linspace(0.0, 0.04, 17)
    refined = replay_muscle_excitations(
        path,
        initial,
        refined_times,
        {"flexor": np.interp(refined_times, times, excitation)},
        accuracy=1e-10,
    )
    np.testing.assert_allclose(result.states[-1], refined.states[-1], atol=2e-7)
    assert result.input_sha256 != refined.input_sha256


@pytest.mark.parametrize("bad_times", [[0.0, 0.0], [0.1, 0.0], [0.0, np.nan]])
def test_invalid_native_time_grid_is_rejected(
    muscle_fixture: tuple[Path, dict[str, float]],
    bad_times: list[float],
) -> None:
    path, initial = muscle_fixture
    with pytest.raises(ValueError, match="times"):
        replay_muscle_excitations(
            path, initial, np.array(bad_times), {"flexor": np.array([0.1, 0.1])}
        )


def test_excitation_mapping_cannot_silently_omit_or_rename_muscle(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    path, initial = muscle_fixture
    with pytest.raises(ValueError, match="names"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.02]), {"other": np.array([0.1, 0.1])}
        )


def test_prescribed_coordinate_cannot_hide_motion_drive(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    coordinate = model.updCoordinateSet().get("slide")
    coordinate.setDefaultIsPrescribed(True)
    coordinate.setPrescribedFunction(osim.Constant(0.31))
    model.finalizeConnections()
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="prescribed coordinate"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.02]), {"flexor": np.array([0.1, 0.1])}
        )


def test_external_force_requires_a_separate_qualified_force_policy(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    force = osim.PrescribedForce()
    force.setName("hidden_external_drive")
    force.setBodyName("/bodyset/load")
    model.addForce(force)
    model.finalizeConnections()
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="non-muscle force"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.02]), {"flexor": np.array([0.1, 0.1])}
        )


def test_recursive_force_outside_legacy_force_set_cannot_hide_drive(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    force = osim.PrescribedForce()
    force.setName("nested_hidden_drive")
    force.setBodyName("/bodyset/load")
    force.setForceFunctions(osim.Constant(100), osim.Constant(0), osim.Constant(0))
    model.addComponent(force)
    model.finalizeConnections()
    assert model.getForceSet().getSize() == 1
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="non-muscle force"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])}
        )


def test_recursive_controller_outside_controller_set_cannot_hide_feedback(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    controller = osim.PrescribedController()
    controller.setName("nested_controller")
    controller.addActuator(model.getMuscles().get(0))
    controller.prescribeControlForActuator("flexor", osim.Constant(0.6))
    model.addComponent(controller)
    model.finalizeConnections()
    assert model.getControllerSet().getSize() == 0
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="existing controller"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])}
        )


def test_position_motion_cannot_bypass_prescribed_coordinate_guard(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    coordinate = model.updCoordinateSet().get("slide")
    motion = osim.PositionMotion()
    motion.setPositionForCoordinate(coordinate, osim.LinearFunction(1.0, 0.31))
    motion.setDefaultEnabled(True)
    model.addComponent(motion)
    model.finalizeConnections()
    assert not coordinate.getDefaultIsPrescribed()
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="prescribed motion"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])}
        )


def test_locked_coordinate_with_inconsistent_speed_cannot_be_repaired_silently(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    model.updCoordinateSet().get("slide").setDefaultLocked(True)
    model.finalizeConnections()
    model.printToXML(str(path))
    speed = next(name for name in initial if name.endswith("/speed"))
    initial[speed] = 0.2
    with pytest.raises(ValueError, match="locked coordinate"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])}
        )


@pytest.mark.parametrize("state_suffix", ["/activation", "/fiber_length"])
def test_initial_state_respects_model_specific_muscle_domain(
    muscle_fixture: tuple[Path, dict[str, float]],
    state_suffix: str,
) -> None:
    path, initial = muscle_fixture
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(path))
    model.initSystem()
    native = model.getMuscles().get(0)
    law = getattr(osim, native.getConcreteClassName()).safeDownCast(native)
    minimum = (
        law.getMinimumActivation()
        if state_suffix == "/activation"
        else law.getMinimumFiberLength()
    )
    name = next(name for name in initial if name.endswith(state_suffix))
    initial[name] = minimum - 1e-8
    with pytest.raises(ValueError, match="muscle state domain"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])}
        )


def test_native_activation_domain_violation_during_replay_is_not_clipped(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    path, initial = muscle_fixture
    activation = next(name for name in initial if name.endswith("/activation"))
    initial[activation] = 0.01
    with pytest.raises(ValueError, match="muscle state domain"):
        replay_muscle_excitations(
            path,
            initial,
            np.array([0.0, 0.005, 0.01]),
            {"flexor": np.array([0.0, 0.0, 0.0])},
        )


def test_muscle_outside_native_actuator_registry_is_not_silently_undriven(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    nested = osim.Millard2012EquilibriumMuscle(
        "unregistered_flexor", 10.0, 0.1, 0.2, 0.0
    )
    nested.addNewPathPoint("origin", model.getGround(), osim.Vec3(0))
    nested.addNewPathPoint("insertion", model.getBodySet().get("load"), osim.Vec3(0))
    model.addComponent(nested)
    model.finalizeConnections()
    state = model.initSystem()
    model.equilibrateMuscles(state)
    names = model.getStateVariableNames()
    initial = {
        names.get(i): float(model.getStateVariableValue(state, names.get(i)))
        for i in range(names.getSize())
    }
    assert model.getMuscles().getSize() == 1
    assert any("unregistered_flexor" in name for name in initial)
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="muscle registry"):
        replay_muscle_excitations(
            path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])}
        )
