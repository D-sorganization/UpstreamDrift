"""Independent native muscle-state replay tests for F07/F08 (#11791)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching import (
    replay_muscle_excitations,
)

pytestmark = pytest.mark.unit


@pytest.fixture(params=["smooth", "hunt-crossley"])
def contact_muscle_fixture(
    muscle_fixture: tuple[Path, dict[str, float]],
    request: pytest.FixtureRequest,
) -> tuple[Path, dict[str, float]]:
    """Add native compliant contact opposing the slider's muscle force."""
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    floor = osim.ContactHalfSpace(
        osim.Vec3(0.3, 0, 0), osim.Vec3(0, 0, np.pi), model.getGround(), "floor"
    )
    sphere = osim.ContactSphere(
        0.01, osim.Vec3(0), model.getBodySet().get("load"), "load_contact"
    )
    model.addContactGeometry(floor)
    model.addContactGeometry(sphere)
    if request.param == "smooth":
        contact = osim.SmoothSphereHalfSpaceForce()
        contact.connectSocket_sphere(sphere)
        contact.connectSocket_half_space(floor)
        contact.set_stiffness(1e7)
        contact.set_dissipation(1.0)
        contact.set_constant_contact_force(1e-12)
        contact.set_hertz_smoothing(5e4)
        contact.set_hunt_crossley_smoothing(50)
    else:
        contact = osim.HuntCrossleyForce()
        contact.addGeometry("floor")
        contact.addGeometry("load_contact")
        contact.setStiffness(1e7)
        contact.setDissipation(1.0)
    contact.setName("native_contact")
    model.addForce(contact)
    model.finalizeConnections()
    model.printToXML(str(path))
    return path, initial


def test_native_contact_is_explicit_and_recorded_without_external_drive(
    contact_muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    path, initial = contact_muscle_fixture
    times = np.linspace(0, 0.04, 17)
    kwargs = {"contact_force_paths": ("/forceset/native_contact",)}
    first = replay_muscle_excitations(
        path, initial, times, {"flexor": times * 0 + 0.4}, **kwargs
    )
    repeat = replay_muscle_excitations(
        path, initial, times, {"flexor": times * 0 + 0.4}, **kwargs
    )
    assert first.policy["force_policy"] == "muscle-gravity-and-listed-native-contact"
    assert first.policy["contact_frame"] == "world-z-up"
    assert len(first.contact_wrenches) == len(times)
    assert all(len(frame) == 1 for frame in first.contact_wrenches)
    assert first.contact_wrenches[-1][0].force_n[0] > 0
    # Fixture-specific smoothing leakage at initial geometric touch is small
    # relative to its 10 N maximum-isometric muscle force, never assumed zero.
    assert abs(first.contact_wrenches[0][0].force_n[0]) < 0.01
    assert first.contact_wrenches[-1][0].label == "contact:native_contact"
    assert first.contact_wrenches == repeat.contact_wrenches
    np.testing.assert_allclose(first.states, repeat.states, atol=1e-10, rtol=0)


def test_native_contact_remains_forbidden_without_explicit_policy(
    contact_muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    path, initial = contact_muscle_fixture
    with pytest.raises(ValueError, match="non-muscle force"):
        replay_muscle_excitations(
            path, initial, np.array([0, 0.02]), {"flexor": np.array([0.4, 0.4])}
        )


@pytest.mark.parametrize("gap_m", [0, 1e-5, 2e-5, 1e-4, 1e-3])
def test_fixture_contact_leakage_is_bounded_across_off_contact_gap_sweep(
    contact_muscle_fixture: tuple[Path, dict[str, float]],
    gap_m: float,
) -> None:
    path, initial = contact_muscle_fixture
    initial["/jointset/slider/slide/value"] += gap_m
    result = replay_muscle_excitations(
        path,
        initial,
        np.array([0, 1e-4]),
        {"flexor": np.array([0.4, 0.4])},
        contact_force_paths=("/forceset/native_contact",),
    )
    # Smooth off-contact leakage is nonmonotone near touch. Sweep the frozen
    # fixture's gaps; do not infer zero leakage from its exact-touch value.
    assert abs(result.contact_wrenches[0][0].force_n[0]) < 0.01


def test_moving_halfspace_requires_bilateral_contact_evidence_policy(
    contact_muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = contact_muscle_fixture
    model = osim.Model(str(path))
    plane = model.updContactGeometrySet().get("floor")
    plane.setFrame(model.getBodySet().get("load"))
    model.finalizeConnections()
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="ground-fixed half-space"):
        replay_muscle_excitations(
            path,
            initial,
            np.array([0, 0.02]),
            {"flexor": np.array([0.4, 0.4])},
            contact_force_paths=("/forceset/native_contact",),
        )


def test_hunt_crossley_extra_geometry_cannot_be_silently_omitted(
    contact_muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = contact_muscle_fixture
    model = osim.Model(str(path))
    force = osim.HuntCrossleyForce.safeDownCast(
        model.updForceSet().get("native_contact")
    )
    if force is None:
        pytest.skip("Extra-geometry parameter sets apply to HuntCrossley")
    extra = osim.ContactSphere(
        0.005, osim.Vec3(0), model.getBodySet().get("load"), "extra"
    )
    model.addContactGeometry(extra)
    force.addGeometry("extra")
    model.finalizeConnections()
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="exactly two supported geometries"):
        replay_muscle_excitations(
            path,
            initial,
            np.array([0, 0.02]),
            {"flexor": np.array([0.4, 0.4])},
            contact_force_paths=("/forceset/native_contact",),
        )


def test_contact_policy_rejects_nonexistent_force_paths(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    path, initial = muscle_fixture
    with pytest.raises(ValueError, match="contact.*paths"):
        replay_muscle_excitations(
            path,
            initial,
            np.array([0, 0.02]),
            {"flexor": np.array([0.4, 0.4])},
            contact_force_paths=("/forceset/missing",),
        )


def test_native_contact_changes_motion_and_survives_accuracy_refinement(
    contact_muscle_fixture: tuple[Path, dict[str, float]],
    tmp_path: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = contact_muscle_fixture
    times = np.linspace(0, 0.08, 33)
    controls = {"flexor": times * 0 + 0.4}
    coarse = replay_muscle_excitations(
        path,
        initial,
        times,
        controls,
        accuracy=1e-6,
        contact_force_paths=("/forceset/native_contact",),
    )
    fine = replay_muscle_excitations(
        path,
        initial,
        times,
        controls,
        accuracy=1e-9,
        contact_force_paths=("/forceset/native_contact",),
    )
    model = osim.Model(str(path))
    model.updForceSet().remove(1)
    no_contact = tmp_path / "synthetic_without_contact.osim"
    model.printToXML(str(no_contact))
    free = replay_muscle_excitations(no_contact, initial, times, controls)
    position = fine.state_names.index("/jointset/slider/slide/value")
    assert fine.states[-1, position] > free.states[-1, position] + 1e-4
    # A compliant law permits finite indentation. This bound is fixture-specific,
    # not a measured physiological/contact qualification threshold.
    assert np.min(fine.states[:, position]) > 0.309
    np.testing.assert_allclose(coarse.states, fine.states, atol=2e-5, rtol=0)
    assert coarse.input_sha256 != fine.input_sha256


@pytest.mark.parametrize("paths", [("/forceset/native_contact",) * 2, ("relative",)])
def test_contact_policy_requires_unique_absolute_paths(
    contact_muscle_fixture: tuple[Path, dict[str, float]],
    paths: tuple[str, ...],
) -> None:
    path, initial = contact_muscle_fixture
    with pytest.raises(ValueError, match="contact force paths"):
        replay_muscle_excitations(
            path,
            initial,
            np.array([0, 0.02]),
            {"flexor": np.array([0.4, 0.4])},
            contact_force_paths=paths,
        )


def test_allowlist_cannot_promote_arbitrary_external_force_to_contact(
    muscle_fixture: tuple[Path, dict[str, float]],
) -> None:
    osim = pytest.importorskip("opensim")
    path, initial = muscle_fixture
    model = osim.Model(str(path))
    force = osim.PrescribedForce()
    force.setName("external")
    force.setBodyName("load")
    model.addForce(force)
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="supported native contact"):
        replay_muscle_excitations(
            path,
            initial,
            np.array([0, 0.02]),
            {"flexor": np.array([0.4, 0.4])},
            contact_force_paths=("/forceset/external",),
        )


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
