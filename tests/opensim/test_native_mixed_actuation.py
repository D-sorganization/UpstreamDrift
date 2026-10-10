"""Assistance-explicit native actuation; fixtures do not qualify a golfer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def mixed_model(tmp_path: Path) -> tuple[Any, Any, Path]:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    model.setGravity(osim.Vec3(0))
    load = osim.Body("load", 1, osim.Vec3(0), osim.Inertia(0.1))
    arm = osim.Body("arm", 1, osim.Vec3(0), osim.Inertia(0.2))
    model.addBody(load)
    model.addBody(arm)
    slider = osim.SliderJoint(
        "slider",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        load,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    slider.updCoordinate().setName("pelvis_tx")
    slider.updCoordinate().setDefaultValue(0.31)
    model.addJoint(slider)
    pin = osim.PinJoint(
        "pin", load, osim.Vec3(0), osim.Vec3(0), arm, osim.Vec3(0), osim.Vec3(0)
    )
    pin.updCoordinate().setName("arm_flex_r")
    model.addJoint(pin)
    for name, origin in (("flexor", 0), ("extensor", 0.62)):
        muscle = osim.Millard2012EquilibriumMuscle(name, 10, 0.1, 0.2, 0)
        muscle.addNewPathPoint("origin", model.getGround(), osim.Vec3(origin, 0, 0))
        muscle.addNewPathPoint("insertion", load, osim.Vec3(0))
        model.addForce(muscle)
    for name, coordinate, gain in (
        ("reserve_pelvis_tx", "pelvis_tx", 7),
        ("upper_torque", "arm_flex_r", 13),
    ):
        actuator = osim.CoordinateActuator(coordinate)
        actuator.setName(name)
        actuator.setOptimalForce(gain)
        actuator.setMinControl(-0.5)
        actuator.setMaxControl(0.5)
        model.addForce(actuator)
    model.finalizeConnections()
    state = model.initSystem()
    model.equilibrateMuscles(state)
    path = tmp_path / "mixed.osim"
    model.printToXML(str(path))
    return model, state, path


def _profile() -> Any:
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_actuation import (
        MixedActuationProfile,
        MixedChannel,
        ActuationRole,
    )

    return MixedActuationProfile(
        (
            MixedChannel("/forceset/flexor", ActuationRole.MUSCLE, (0.01, 1)),
            MixedChannel("/forceset/extensor", ActuationRole.MUSCLE, (0.01, 1)),
            MixedChannel(
                "/forceset/reserve_pelvis_tx", ActuationRole.ROOT_RESIDUAL, (-0.4, 0.4)
            ),
            MixedChannel(
                "/forceset/upper_torque", ActuationRole.UPPER_ASSISTANCE, (-0.3, 0.3)
            ),
        )
    )


def test_native_mixed_profile_binds_actual_gain_units_and_root_role(
    mixed_model: tuple[Any, Any, Path],
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_actuation import (
        admit_native_mixed_profile,
    )

    model, state, _ = mixed_model
    admitted = admit_native_mixed_profile(model, state, _profile())
    root, upper = admitted.channels[-2:]
    assert root.coordinate_path == "/jointset/slider/pelvis_tx"
    assert root.output_unit == "N" and root.optimal_force == 7
    assert upper.output_unit == "N*m" and upper.optimal_force == 13
    assert root.role.value == "root-residual"
    assert root.control_bounds == (-0.4, 0.4)
    assert len(admitted.sha256) == 64


def test_pure_muscle_policy_still_rejects_mixed_native_model(
    mixed_model: tuple[Any, Any, Path],
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_bundle import (
        build_native_muscle_replay_bundle,
    )

    model, state, path = mixed_model
    names = model.getStateVariableNames()
    initial = {
        names.get(i): model.getStateVariableValue(state, names.get(i))
        for i in range(names.getSize())
    }
    with pytest.raises(ValueError, match="component|non-muscle"):
        build_native_muscle_replay_bundle(
            path,
            initial,
            np.array([0.0, 0.01]),
            {"flexor": np.array([0.1, 0.2]), "extensor": np.array([0.2, 0.1])},
        )


def test_root_role_cannot_follow_misleading_actuator_name(
    mixed_model: tuple[Any, Any, Path],
) -> None:
    from dataclasses import replace
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_actuation import (
        admit_native_mixed_profile,
        ActuationRole,
    )

    model, state, _ = mixed_model
    profile = _profile()
    wrong = replace(
        profile,
        channels=tuple(
            replace(channel, role=ActuationRole.LEG_RESERVE)
            if channel.path.endswith("reserve_pelvis_tx")
            else channel
            for channel in profile.channels
        ),
    )
    with pytest.raises(ValueError, match="root"):
        admit_native_mixed_profile(model, state, wrong)


def test_renamed_ground_coordinate_cannot_hide_root_assistance(
    mixed_model: tuple[Any, Any, Path],
) -> None:
    from dataclasses import replace
    import opensim as osim
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_actuation import (
        admit_native_mixed_profile,
        ActuationRole,
    )

    model, _, _ = mixed_model
    model.updCoordinateSet().get("pelvis_tx").setName("renamed_root_translation")
    actuator = osim.CoordinateActuator.safeDownCast(
        model.updForceSet().get("reserve_pelvis_tx")
    )
    actuator.setCoordinate(model.updCoordinateSet().get("renamed_root_translation"))
    model.finalizeConnections()
    state = model.initSystem()
    profile = _profile()
    wrong = replace(
        profile,
        channels=tuple(
            replace(channel, role=ActuationRole.LEG_RESERVE)
            if channel.path.endswith("reserve_pelvis_tx")
            else channel
            for channel in profile.channels
        ),
    )
    with pytest.raises(ValueError, match="root"):
        admit_native_mixed_profile(model, state, wrong)


def _replay_inputs(
    model: Any, state: Any
) -> tuple[dict[str, float], np.ndarray, dict[str, np.ndarray]]:
    names = model.getStateVariableNames()
    initial = {
        names.get(i): model.getStateVariableValue(state, names.get(i))
        for i in range(names.getSize())
    }
    times = np.linspace(0, 0.04, 9)
    controls = {
        "/forceset/flexor": np.linspace(0.05, 0.2, len(times)),
        "/forceset/extensor": np.linspace(0.15, 0.05, len(times)),
        "/forceset/reserve_pelvis_tx": np.full(len(times), -0.2),
        "/forceset/upper_torque": np.full(len(times), 0.1),
    }
    return initial, times, controls


@pytest.mark.parametrize("flag", ["setLocked", "setClamped"])
def test_mixed_admission_rejects_live_coordinate_restrictions(
    mixed_model: tuple[Any, Any, Path], flag: str
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_actuation import (
        admit_native_mixed_profile,
    )

    model, state, _ = mixed_model
    coordinate = model.getCoordinateSet().get("pelvis_tx")
    getattr(coordinate, flag)(state, True)
    with pytest.raises(ValueError, match="restriction"):
        admit_native_mixed_profile(model, state, _profile())


def test_native_mixed_t01_replay_preserves_controls_force_units_and_power(
    mixed_model: tuple[Any, Any, Path],
) -> None:
    from src.engines.native_replay_contracts import native_replay_contract_types
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_replay import (
        build_native_mixed_replay_bundle,
        replay_native_mixed_bundle,
    )

    model, state, path = mixed_model
    initial, times, controls = _replay_inputs(model, state)
    profile = _profile()
    bundle = build_native_mixed_replay_bundle(path, initial, times, controls, profile)
    wire = native_replay_contract_types()
    reloaded = wire.load_experiment_replay_bundle(
        wire.dumps_experiment_replay_bundle(bundle)
    )
    first = replay_native_mixed_bundle(reloaded, path, profile)
    repeat = replay_native_mixed_bundle(reloaded, path, profile)
    assert bundle.input_history.input_kind.value == "actuator_command"
    assert bundle.input_history.interpolation.value == "linear"
    np.testing.assert_array_equal(first.states[0], list(initial.values()))
    np.testing.assert_allclose(first.states, repeat.states, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        first.applied_controls,
        np.column_stack(list(controls.values())),
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        first.actuations[:, -2:],
        np.tile([-1.4, 1.3], (len(times), 1)),
        atol=1e-12,
        rtol=0,
    )
    speeds = [
        first.state_names.index("/jointset/slider/pelvis_tx/speed"),
        first.state_names.index("/jointset/pin/arm_flex_r/speed"),
    ]
    np.testing.assert_allclose(
        first.powers_w[:, -2:],
        first.actuations[:, -2:] * first.states[:, speeds],
        atol=1e-12,
        rtol=0,
    )
    assert np.max(np.abs(first.work_j[-1, -2:])) > 1e-5
    assert first.input_sha256 == bundle.applied_input_sha256
    assert first.initial_state_sha256 == bundle.integrity.initial_state_sha256
    assert first.policy_sha256 == bundle.integrity.execution_policy_sha256
    assert first.provider_sha256 == bundle.model.provider_sha256
    assert not first.states.flags.writeable


def test_changed_future_mixed_control_preserves_prefix_and_changes_motion(
    mixed_model: tuple[Any, Any, Path],
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_replay import (
        build_native_mixed_replay_bundle,
        replay_native_mixed_bundle,
    )

    model, state, path = mixed_model
    initial, times, controls = _replay_inputs(model, state)
    first = replay_native_mixed_bundle(
        build_native_mixed_replay_bundle(path, initial, times, controls, _profile()),
        path,
        _profile(),
    )
    controls["/forceset/upper_torque"][4:] = -0.2
    second = replay_native_mixed_bundle(
        build_native_mixed_replay_bundle(path, initial, times, controls, _profile()),
        path,
        _profile(),
    )
    np.testing.assert_allclose(first.states[:4], second.states[:4], atol=1e-12, rtol=0)
    assert np.max(np.abs(first.states[-1] - second.states[-1])) > 1e-4


def test_mixed_discrete_units_follow_native_owner(
    mixed_model: tuple[Any, Any, Path],
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_replay import (
        build_native_mixed_replay_bundle,
    )

    model, state, path = mixed_model
    initial, times, controls = _replay_inputs(model, state)
    bundle = build_native_mixed_replay_bundle(
        path, initial, times, controls, _profile()
    )
    units = {
        item.component_id: item.unit for item in bundle.model.state_schema.components
    }
    assert (
        units["registered-discrete:/forceset/upper_torque/override_actuation"] == "N*m"
    )
    assert (
        units["registered-discrete:/forceset/reserve_pelvis_tx/override_actuation"]
        == "N"
    )
    assert units["registered-discrete:/forceset/flexor/override_actuation"] == "N"


def test_fixed_zero_assistance_bounds_admit_and_execute(
    mixed_model: tuple[Any, Any, Path],
) -> None:
    from dataclasses import replace
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_actuation import (
        ActuationRole,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_replay import (
        build_native_mixed_replay_bundle,
        replay_native_mixed_bundle,
    )

    model, state, path = mixed_model
    initial, times, controls = _replay_inputs(model, state)
    profile = replace(
        _profile(),
        channels=tuple(
            replace(channel, control_bounds=(0, 0))
            if channel.role != ActuationRole.MUSCLE
            else channel
            for channel in _profile().channels
        ),
    )
    for channel in profile.channels:
        if channel.role != ActuationRole.MUSCLE:
            controls[channel.path] = np.zeros_like(times)
    result = replay_native_mixed_bundle(
        build_native_mixed_replay_bundle(path, initial, times, controls, profile),
        path,
        profile,
    )
    np.testing.assert_array_equal(result.actuations[:, -2:], 0)
    np.testing.assert_array_equal(result.work_j[:, -2:], 0)
    with pytest.raises(ValueError):
        result.states.setflags(write=True)


@pytest.mark.parametrize("mutation", ["bounds", "gain", "source", "roles", "missing"])
def test_mixed_replay_rejects_changed_execution_identity(
    mixed_model: tuple[Any, Any, Path], mutation: str
) -> None:
    from dataclasses import replace
    import opensim as osim
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_actuation import (
        ActuationRole,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_replay import (
        build_native_mixed_replay_bundle,
        replay_native_mixed_bundle,
    )

    model, state, path = mixed_model
    initial, times, controls = _replay_inputs(model, state)
    profile = _profile()
    bundle = build_native_mixed_replay_bundle(path, initial, times, controls, profile)
    if mutation == "bounds":
        profile = replace(
            profile,
            channels=tuple(
                replace(c, control_bounds=(-0.25, 0.25))
                if c.path.endswith("upper_torque")
                else c
                for c in profile.channels
            ),
        )
    elif mutation == "roles":
        profile = replace(
            profile,
            channels=tuple(
                replace(c, role=ActuationRole.LEG_RESERVE)
                if c.path.endswith("upper_torque")
                else c
                for c in profile.channels
            ),
        )
    elif mutation == "gain":
        osim.CoordinateActuator.safeDownCast(
            model.updForceSet().get("upper_torque")
        ).setOptimalForce(14)
        model.printToXML(str(path))
    elif mutation == "source":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        profile = replace(profile, channels=profile.channels[:-1])
    with pytest.raises(ValueError):
        replay_native_mixed_bundle(bundle, path, profile)


@pytest.mark.parametrize(
    "mutation", ["nonfinite", "out-of-bounds", "missing-channel", "incomplete-state"]
)
def test_mixed_bundle_rejects_invalid_physical_inputs(
    mixed_model: tuple[Any, Any, Path], mutation: str
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_replay import (
        build_native_mixed_replay_bundle,
    )

    model, state, path = mixed_model
    initial, times, controls = _replay_inputs(model, state)
    if mutation == "nonfinite":
        controls["/forceset/upper_torque"][0] = np.nan
    elif mutation == "out-of-bounds":
        controls["/forceset/upper_torque"][0] = 0.31
    elif mutation == "missing-channel":
        controls.pop("/forceset/upper_torque")
    else:
        initial.pop(next(iter(initial)))
    with pytest.raises(ValueError):
        build_native_mixed_replay_bundle(path, initial, times, controls, _profile())


def test_mixed_replay_rejects_actual_player_control_mismatch(
    mixed_model: tuple[Any, Any, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching import muscle_replay
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_replay import (
        build_native_mixed_replay_bundle,
        replay_native_mixed_bundle,
    )

    model, state, path = mixed_model
    initial, times, controls = _replay_inputs(model, state)
    bundle = build_native_mixed_replay_bundle(
        path, initial, times, controls, _profile()
    )
    original = muscle_replay._configure_input_player

    def wrong_player(
        model: Any, actuators: Any, names: Any, grid: Any, values: Any
    ) -> None:
        wrong = {name: array.copy() for name, array in values.items()}
        wrong["upper_torque"] *= 0.5
        original(model, actuators, names, grid, wrong)

    monkeypatch.setattr(muscle_replay, "_configure_input_player", wrong_player)
    with pytest.raises(RuntimeError, match="controls"):
        replay_native_mixed_bundle(bundle, path, _profile())


def test_scalar_core_rejects_mislabeled_initial_native_time(
    mixed_model: tuple[Any, Any, Path],
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_scalar_replay import (
        NativeScalarReplayPolicy,
        integrate_native_scalar_replay,
    )

    model, state, _ = mixed_model
    initial, times, controls = _replay_inputs(model, state)
    state.setTime(0.001)
    with pytest.raises(RuntimeError, match="time"):
        integrate_native_scalar_replay(
            model,
            state,
            tuple(initial),
            tuple(controls),
            times,
            {},
            NativeScalarReplayPolicy(1e-8),
        )


def test_mixed_identity_requires_actual_native_binary(
    mixed_model: tuple[Any, Any, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import opensim as osim
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_replay import (
        build_native_mixed_replay_bundle,
    )

    model, state, path = mixed_model
    initial, times, controls = _replay_inputs(model, state)
    original = Path.glob
    native_directory = Path(osim.__file__).parent

    def hide_binary(directory: Path, pattern: str) -> Any:
        if directory == native_directory and pattern == "*.pyd":
            return iter(())
        return original(directory, pattern)

    monkeypatch.setattr(Path, "glob", hide_binary)
    with pytest.raises(ValueError, match="binary identity"):
        build_native_mixed_replay_bundle(path, initial, times, controls, _profile())


def test_mixed_admission_rejects_live_prescription(
    mixed_model: tuple[Any, Any, Path],
) -> None:
    import opensim as osim
    from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_actuation import (
        admit_native_mixed_profile,
    )

    model, _, _ = mixed_model
    coordinate = model.updCoordinateSet().get("pelvis_tx")
    coordinate.setPrescribedFunction(osim.Constant(0.31))
    state = model.initSystem()
    coordinate.setIsPrescribed(state, True)
    with pytest.raises(ValueError, match="restriction"):
        admit_native_mixed_profile(model, state, _profile())
