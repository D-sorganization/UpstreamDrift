"""SDK-gated one-coordinate native observation fixture for the effort adapter."""

from __future__ import annotations

import importlib.metadata
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.requires_opensim]


def _antagonist_fixture(osim: Any) -> tuple[Any, Any]:
    """Build one hinge with two antagonists and an admissible path length."""
    model = osim.Model()
    model.setName("native_effort_adapter_antagonists")
    model.setGravity(osim.Vec3(0.0, 0.0, 0.0))
    body = osim.Body("load", 1.0, osim.Vec3(0.0), osim.Inertia(0.01))
    model.addBody(body)
    joint = osim.PinJoint(
        "hinge",
        model.getGround(),
        osim.Vec3(0.0),
        osim.Vec3(0.0),
        body,
        osim.Vec3(0.0),
        osim.Vec3(0.0),
    )
    coordinate = joint.updCoordinate()
    coordinate.setName("hinge_coord")
    coordinate.setDefaultValue(0.0)
    model.addJoint(joint)
    for name, y in (("flexor", 0.05), ("extensor", -0.05)):
        muscle = osim.Millard2012EquilibriumMuscle(name, 10.0, 0.1, 0.2, 0.1)
        muscle.addNewPathPoint("origin", model.getGround(), osim.Vec3(0.0, y, 0.0))
        muscle.addNewPathPoint("insertion", body, osim.Vec3(0.35, y, 0.0))
        model.addForce(muscle)
    model.finalizeConnections()
    return model, coordinate


def test_native_pin_fixture_has_complete_antagonist_observations(
    record_property: Callable[[str, str], None],
) -> None:
    osim = pytest.importorskip("opensim")
    record_property(
        "native_runtime_json",
        json.dumps(
            {
                "distribution_version": importlib.metadata.version("opensim"),
                "python": sys.executable,
                "native_get_version": str(osim.GetVersion()),
                "package_path": str(Path(osim.__file__).resolve()),
                "binding_type": "opensim.Millard2012EquilibriumMuscle",
            },
            sort_keys=True,
            allow_nan=False,
        ),
    )
    model, coordinate = _antagonist_fixture(osim)
    state = model.initSystem()
    for index in range(model.getMuscles().getSize()):
        model.getMuscles().get(index).setActivation(state, 0.1)
    model.equilibrateMuscles(state)
    model.realizeDynamics(state)

    assert model.getCoordinateSet().getSize() == 1
    assert model.getConstraintSet().getSize() == 0
    coordinate_path = str(coordinate.getAbsolutePathString())
    state_names = tuple(
        str(model.getStateVariableNames().get(index))
        for index in range(model.getStateVariableNames().getSize())
    )
    assert f"{coordinate_path}/value" in state_names
    assert f"{coordinate_path}/speed" in state_names
    state_values = {
        name: float(model.getStateVariableValue(state, name)) for name in state_names
    }
    record_property(
        "native_state_json",
        json.dumps(
            {
                "state_names": state_names,
                "state_values": state_values,
                "model_coordinate_order": tuple(
                    str(model.getCoordinateSet().get(index).getName())
                    for index in range(model.getCoordinateSet().getSize())
                ),
                "muscle_order": tuple(
                    str(model.getMuscles().get(index).getName())
                    for index in range(model.getMuscles().getSize())
                ),
            },
            sort_keys=True,
            allow_nan=False,
        ),
    )
    assert all(math.isfinite(value) for value in state_values.values())

    muscles = {
        str(model.getMuscles().get(index).getName()): model.getMuscles().get(index)
        for index in range(model.getMuscles().getSize())
    }
    assert tuple(muscles) == ("flexor", "extensor")
    moment_arms: list[float] = []
    passive_efforts: list[float] = []
    observations: dict[str, dict[str, float]] = {}
    for name in ("flexor", "extensor"):
        muscle = muscles[name]
        millard = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
        assert millard is not None
        path = str(muscle.getAbsolutePathString())
        assert f"{path}/activation" in state_names
        assert f"{path}/fiber_length" in state_names
        moment_arm = float(muscle.computeMomentArm(state, coordinate))
        passive_fiber_force = float(muscle.getPassiveFiberForce(state))
        passive_tendon_force = float(muscle.getPassiveFiberForceAlongTendon(state))
        passive_elastic_force = float(millard.getPassiveFiberElasticForce(state))
        passive_damping_force = float(millard.getPassiveFiberDampingForce(state))
        passive_elastic_tendon_force = float(
            millard.getPassiveFiberElasticForceAlongTendon(state)
        )
        passive_damping_tendon_force = float(
            millard.getPassiveFiberDampingForceAlongTendon(state)
        )
        pennation = float(muscle.getPennationAngle(state))
        activation = float(muscle.getActivation(state))
        fiber_length = float(muscle.getFiberLength(state))
        path_length = float(muscle.getLength(state))
        tendon_force = float(muscle.getTendonForce(state))
        fiber_force_along_tendon = float(muscle.getFiberForceAlongTendon(state))

        passive_effort = passive_tendon_force * moment_arm
        assert all(
            math.isfinite(value)
            for value in (
                moment_arm,
                passive_fiber_force,
                passive_elastic_force,
                passive_damping_force,
                passive_elastic_tendon_force,
                passive_damping_tendon_force,
                pennation,
                activation,
                fiber_length,
                path_length,
                tendon_force,
                fiber_force_along_tendon,
                passive_tendon_force,
                passive_effort,
            )
        )
        assert 0.0 <= activation <= 1.0
        assert fiber_length > 0.0
        assert path_length > 0.3
        assert math.isclose(
            fiber_force_along_tendon, tendon_force, rel_tol=0.0, abs_tol=1e-3
        )
        assert math.isclose(
            state_values[f"{path}/activation"], activation, rel_tol=0.0, abs_tol=1e-12
        )
        assert math.isclose(
            state_values[f"{path}/fiber_length"],
            fiber_length,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        observations[name] = {
            "activation": activation,
            "fiber_length_m": fiber_length,
            "path_length_m": path_length,
            "moment_arm_m": moment_arm,
            "pennation_rad": pennation,
            "passive_fiber_force_n": passive_fiber_force,
            "passive_fiber_elastic_force_n": passive_elastic_force,
            "passive_fiber_damping_force_n": passive_damping_force,
            "passive_fiber_elastic_force_along_tendon_n": passive_elastic_tendon_force,
            "passive_fiber_damping_force_along_tendon_n": passive_damping_tendon_force,
            "passive_tendon_force_n": passive_tendon_force,
            "passive_generalized_effort_nm": passive_effort,
            "tendon_force_n": tendon_force,
            "fiber_force_along_tendon_n": fiber_force_along_tendon,
        }
        record_property(
            f"native_muscle_{name}_json",
            json.dumps(observations[name], sort_keys=True, allow_nan=False),
        )
        passive_efforts.append(passive_effort)
        moment_arms.append(moment_arm)

    assert moment_arms[0] * moment_arms[1] < 0
    assert tuple(observations) == ("flexor", "extensor")
    record_property(
        "native_fixture_scope",
        "initSystem/equilibrateMuscles/realizeDynamics only; no Manager or integration",
    )
    assert all(math.isfinite(value) for value in passive_efforts)
    # Synthetic initialization/observation only. Adapter proposal-to-Manager
    # advance/accept and fresh raw-excitation replay remain untested; this does
    # not qualify muscle matching or any registered model.
