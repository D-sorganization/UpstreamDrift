"""Declared OpenSim cold starts must preserve hidden constraint targets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from dataclasses import replace
from typing import Any, cast
import pytest
import numpy as np

from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
    DeclaredColdStart,
    observe_declared_native_sample,
    reconstruct_declared_cold_start,
    replay_declared_constant_input,
    replay_declared_time_only_input,
)

pytestmark = pytest.mark.unit


def _locked_pin_source(path: Path) -> tuple[Path, dict[str, float]]:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    body = osim.Body("body", 1, osim.Vec3(0, -0.5, 0), osim.Inertia(1, 1, 1))
    joint = osim.PinJoint(
        "joint",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    coordinate = joint.updCoordinate()
    coordinate.setName("q")
    coordinate.setDefaultValue(0.3)
    coordinate.setDefaultLocked(True)
    model.addBody(body)
    model.addJoint(joint)
    model.finalizeConnections()
    model.printToXML(str(path))
    return path, {"/jointset/joint/q/value": 0.3, "/jointset/joint/q/speed": 0.0}


def _declaration(
    path: Path, named: dict[str, float], target: float
) -> DeclaredColdStart:
    return DeclaredColdStart(
        model_path=path,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        named_state=named,
        time_seconds=0.0,
        lock_targets={"/jointset/joint/q": target},
        chart_bounds={"/jointset/joint/q": (0.2, 0.4)},
        constraint_enforcement={},
        residual_tolerance=1e-10,
    )


def test_declared_target_reconstructs_independently_and_preserves_source(
    tmp_path: Path,
) -> None:
    path, named = _locked_pin_source(tmp_path / "locked.osim")
    original = path.read_bytes()
    declaration = _declaration(path, named, 0.3)
    with reconstruct_declared_cold_start(declaration) as first:
        assert first.audit.position_errors == pytest.approx((0.0,), abs=1e-10)
        assert first.audit.coordinates[0].locked
        expected = first.audit
    with reconstruct_declared_cold_start(
        declaration, expected_audit=expected
    ) as second:
        assert second.audit == expected
        assert second.model is not first.model
    assert path.read_bytes() == original


def test_identical_named_state_different_model_owned_target_is_rejected(
    tmp_path: Path,
) -> None:
    path, named = _locked_pin_source(tmp_path / "locked.osim")
    with reconstruct_declared_cold_start(_declaration(path, named, 0.3)) as good:
        expected = good.audit
    wrong = _declaration(path, named, 0.6)
    with pytest.raises(ValueError, match="chart|residual|target|observation|achieved"):
        with reconstruct_declared_cold_start(wrong, expected_audit=expected):
            pass


def test_missing_lock_recipe_and_changed_source_fail_closed(tmp_path: Path) -> None:
    path, named = _locked_pin_source(tmp_path / "locked.osim")
    declaration = _declaration(path, named, 0.3)
    missing = DeclaredColdStart(
        model_path=path,
        source_sha256=declaration.source_sha256,
        named_state=named,
        time_seconds=0.0,
        lock_targets={},
        chart_bounds=declaration.chart_bounds,
        constraint_enforcement={},
        residual_tolerance=1e-10,
    )
    with pytest.raises(ValueError, match="lock target"):
        with reconstruct_declared_cold_start(missing):
            pass
    path.write_bytes(path.read_bytes() + b"\n<!-- changed -->\n")
    with pytest.raises(ValueError, match="source"):
        with reconstruct_declared_cold_start(declaration):
            pass


def test_invalid_native_inputs_are_rejected(tmp_path: Path) -> None:
    path, named = _locked_pin_source(tmp_path / "locked.osim")
    with pytest.raises(ValueError, match="finite"):
        DeclaredColdStart(
            model_path=path,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            named_state={**named, "/jointset/joint/q/speed": float("nan")},
            time_seconds=0.0,
            lock_targets={"/jointset/joint/q": 0.3},
            chart_bounds={"/jointset/joint/q": (0.2, 0.4)},
            constraint_enforcement={},
            residual_tolerance=1e-10,
        )


def _coupled_source(path: Path) -> tuple[Path, dict[str, float], dict[str, bool]]:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    for joint_name, coordinate_name, default in (
        ("base", "q", 0.3),
        ("follower", "dependent", 0.6),
    ):
        body = osim.Body(joint_name, 1, osim.Vec3(0, -0.5, 0), osim.Inertia(1, 1, 1))
        joint = osim.PinJoint(
            joint_name,
            model.getGround(),
            osim.Vec3(0),
            osim.Vec3(0),
            body,
            osim.Vec3(0),
            osim.Vec3(0),
        )
        joint.updCoordinate().setName(coordinate_name)
        joint.updCoordinate().setDefaultValue(default)
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
    actuator = osim.CoordinateActuator("q")
    actuator.setName("mechanical_probe")
    actuator.setOptimalForce(1.0)
    model.addForce(actuator)
    model.finalizeConnections()
    model.printToXML(str(path))
    named = {
        "/jointset/base/q/value": 0.3,
        "/jointset/base/q/speed": 0.0,
        "/jointset/follower/dependent/value": 0.6,
        "/jointset/follower/dependent/speed": 0.0,
    }
    return path, named, {"/constraintset/coupler": True}


def _coupled_declaration(path: Path, named: dict[str, float]) -> DeclaredColdStart:
    return DeclaredColdStart(
        model_path=path,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        named_state=named,
        time_seconds=0.0,
        lock_targets={},
        chart_bounds={"/jointset/base/q": (0.2, 0.5)},
        constraint_enforcement={"/constraintset/coupler": True},
        residual_tolerance=1e-9,
    )


def _constant_input_rollout(declaration: DeclaredColdStart, command: float) -> float:
    driven = replace(
        declaration,
        constant_commands={"/forceset/mechanical_probe": command},
    )
    replay = replay_declared_constant_input(driven, (0.0, 0.005, 0.01))
    assert replay.time_seconds == (0.0, 0.005, 0.01)
    assert replay.actuator_paths == ("/forceset/mechanical_probe",)
    assert replay.applied_commands == ((command,),) * 3
    assert all(not audit.position_errors[0] for audit in replay.audits)
    return dict(replay.audits[-1].named_state)["/jointset/base/q/value"]


def test_coupled_fresh_rollout_matches_then_changed_input_diverges(
    tmp_path: Path,
) -> None:
    path, named, _ = _coupled_source(tmp_path / "coupled.osim")
    declaration = _coupled_declaration(path, named)
    same_a = _constant_input_rollout(declaration, 0.1)
    same_b = _constant_input_rollout(declaration, 0.1)
    changed = _constant_input_rollout(declaration, 0.3)
    assert same_a == pytest.approx(same_b, abs=1e-12)
    assert abs(changed - same_a) > 1e-7


def test_coupled_replay_rejects_wrong_enforcement_and_actuator_coverage(
    tmp_path: Path,
) -> None:
    path, named, _ = _coupled_source(tmp_path / "coupled.osim")
    declaration = _coupled_declaration(path, named)
    with pytest.raises(ValueError, match="enforcement"):
        with reconstruct_declared_cold_start(
            replace(
                declaration, constraint_enforcement={"/constraintset/coupler": False}
            )
        ):
            pass
    with pytest.raises(ValueError, match="coverage"):
        replay_declared_constant_input(
            replace(declaration, constant_commands={"/forceset/other": 0.1}),
            (0.0, 0.01),
        )


def test_coupled_replay_rejects_source_chart_exit(tmp_path: Path) -> None:
    path, named, _ = _coupled_source(tmp_path / "coupled.osim")
    declaration = replace(
        _coupled_declaration(path, named),
        constant_commands={"/forceset/mechanical_probe": 0.3},
        chart_bounds={"/jointset/base/q": (0.2999999999, 0.3000000001)},
    )
    with pytest.raises(ValueError, match="outside declared chart"):
        replay_declared_constant_input(declaration, (0.0, 0.005, 0.01))


def test_mechanical_diagnostic_rejects_unreviewed_native_force(
    tmp_path: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    path, named, _ = _coupled_source(tmp_path / "coupled.osim")
    model = osim.Model(str(path))
    limit = osim.CoordinateLimitForce()
    limit.setName("unreviewed_limit_force")
    limit.set_coordinate("q")
    model.addForce(limit)
    model.finalizeConnections()
    model.printToXML(str(path))
    declaration = replace(
        _coupled_declaration(path, named),
        constant_commands={"/forceset/mechanical_probe": 0.1},
    )
    with pytest.raises(ValueError, match="non-actuator force"):
        replay_declared_constant_input(declaration, (0.0, 0.01))


def test_scheduled_suffix_requires_exact_future_knot_and_channels(
    tmp_path: Path,
) -> None:
    path, named, _ = _coupled_source(tmp_path / "coupled.osim")
    declaration = replace(
        _coupled_declaration(path, named),
        constant_commands={"/forceset/mechanical_probe": 0.1},
    )
    with pytest.raises(ValueError, match="future exact channel"):
        replace(
            declaration,
            scheduled_command_step=(0.005, {"/forceset/other": 0.2}),
        )
    scheduled = replace(
        declaration,
        scheduled_command_step=(0.005, {"/forceset/mechanical_probe": 0.2}),
    )
    with pytest.raises(ValueError, match="interior replay knot"):
        replay_declared_time_only_input(scheduled, (0.0, 0.01))
    with pytest.raises(ValueError, match="constant replay"):
        replay_declared_constant_input(scheduled, (0.0, 0.005, 0.01))


def test_source_controller_replacement_identity_is_immutable(tmp_path: Path) -> None:
    path, named, _ = _coupled_source(tmp_path / "coupled.osim")
    with pytest.raises(TypeError, match="tuple"):
        replace(
            _coupled_declaration(path, named),
            constant_commands={"/forceset/mechanical_probe": 0.1},
            source_controller_replacement=cast(
                Any, ["/controllerset/source", "0" * 64, 0.1]
            ),
        )


def test_physical_humerus_chart_is_not_source_default() -> None:
    source_text = os.environ.get("UD_HUMERUS_V2_SOURCE")
    receipt_text = os.environ.get("UD_HUMERUS_V2_RECEIPT")
    if not source_text or not receipt_text:
        pytest.skip("owned physical-humerus source/receipt opt-in is absent")
    source = Path(source_text)
    receipt = json.loads(Path(receipt_text).read_text(encoding="utf-8"))
    initial = receipt["variants"]["129"]["rollout"]["initial_named"]
    model_hash = receipt["variants"]["129"]["artifact_sha256"]
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(source))
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    default_e = float(coordinates.get("shoulder_elv").getValue(state))
    assert default_e < 0.403598785852187
    constraints = {
        item.getAbsolutePathString(): bool(item.isEnforced(state))
        for item in tuple(model.getConstraintSet())
    }
    declaration = DeclaredColdStart(
        model_path=source,
        source_sha256=model_hash,
        named_state=initial,
        time_seconds=0.0,
        lock_targets={},
        chart_bounds={
            "/jointset/humerus_physical/shoulder_elv": (
                0.403598785852187,
                0.643598785852187,
            ),
            "/jointset/humerus_route/port_elevation": (
                0.403598785852187,
                0.643598785852187,
            ),
            "/jointset/clavicle_derived/clavicle_rhythm": (
                0.403598785852187,
                0.643598785852187,
            ),
        },
        constraint_enforcement=constraints,
        residual_tolerance=1e-8,
        allow_source_controllers_for_observation=True,
        linear_chart_bounds={
            "source_shoulder_rot": (
                {
                    "/jointset/humerus_route/elv_angle": 1.0,
                    "/jointset/humerus_physical/humerus_axial_relative": 1.0,
                },
                -1.57079633,
                2.0943951,
            ),
        },
    )
    with pytest.raises(ValueError, match="source controller"):
        with reconstruct_declared_cold_start(
            replace(declaration, allow_source_controllers_for_observation=False)
        ):
            pass
    retained = json.loads(
        (source.parent / "root-initialization-audit.json").read_text(encoding="utf-8")
    )
    with reconstruct_declared_cold_start(declaration) as prepared:
        assert prepared.source_controller_observed
        assert prepared.audit.position_errors == pytest.approx(
            (0.0,) * len(prepared.audit.position_errors), abs=1e-8
        )
        assert prepared.audit.constraint_jacobian_shape[0] >= 6
        assert prepared.audit.coordinates
        prepared.model.realizeVelocity(prepared.state)
        mass_native = osim.Matrix()
        matter = prepared.model.getMatterSubsystem()
        matter.calcM(prepared.state, mass_native)
        mass = np.asarray(mass_native.to_numpy())
        eigenvalues = np.linalg.eigvalsh(mass)
        expected = retained["prepared_rollout_state"]
        assert eigenvalues[0] == pytest.approx(
            expected["minimum_M_eigenvalue"], rel=1e-6
        )
        assert float(np.linalg.cond(mass)) == pytest.approx(
            expected["condition_M"], rel=1e-6
        )
        jacobian_native = osim.Matrix()
        matter.calcG(prepared.state, jacobian_native)
        jacobian = np.asarray(jacobian_native.to_numpy())
        assert np.linalg.matrix_rank(jacobian) == jacobian.shape[0]
    invalid = replace(
        declaration,
        linear_chart_bounds={
            "source_shoulder_rot": (
                {
                    "/jointset/humerus_route/elv_angle": 1.0,
                    "/jointset/humerus_physical/humerus_axial_relative": 1.0,
                },
                0.1,
                0.2,
            ),
        },
    )
    with pytest.raises(ValueError, match="source chart"):
        with reconstruct_declared_cold_start(invalid):
            pass

    original_controller = (
        "/controllerset/declared_constant_mechanical_probe",
        "ee9237a7bceacdb1052658699d83dd57e7f245f112d48db1316551903679435a",
        1.0,
    )
    derived = replace(
        declaration,
        allow_source_controllers_for_observation=False,
        source_controller_replacement=original_controller,
        constant_commands={"/forceset/unit_generalized_force": 1.0},
    )
    clock = (0.0, 0.001, 0.002)
    first = replay_declared_constant_input(derived, clock)
    repeated = replay_declared_constant_input(derived, clock)
    assert first.audits == repeated.audits
    assert first.applied_commands == ((1.0,),) * len(clock)
    assert first.loaded_model_sha256 != prepared.audit.loaded_model_sha256
    looser_chart = dict(derived.chart_bounds)
    bound = looser_chart["/jointset/humerus_physical/shoulder_elv"]
    looser_chart["/jointset/humerus_physical/shoulder_elv"] = (
        bound[0],
        bound[1] + 0.001,
    )
    same_signal_different_admission = replay_declared_constant_input(
        replace(derived, chart_bounds=looser_chart), clock
    )
    assert first.applied_commands == same_signal_different_admission.applied_commands
    assert first.admission_sha256 != same_signal_different_admission.admission_sha256
    assert len(first.adapter_source_sha256) == 64
    with reconstruct_declared_cold_start(declaration) as original:
        manager = osim.Manager(original.model)
        manager.setIntegratorMethod(osim.Manager.IntegratorMethod_RungeKuttaMerson)
        manager.setIntegratorAccuracy(1e-9)
        manager.setWriteToStorage(False)
        manager.initialize(original.state)
        original_audits = [original.audit]
        for time in clock[1:]:
            original_audits.append(
                observe_declared_native_sample(
                    original.model, manager.integrate(time), declaration
                )
            )
    for unchanged, replacement in zip(original_audits, first.audits, strict=True):
        assert tuple(name for name, _ in unchanged.named_state) == tuple(
            name for name, _ in replacement.named_state
        )
        assert tuple(value for _, value in unchanged.named_state) == pytest.approx(
            tuple(value for _, value in replacement.named_state), abs=1e-8
        )
        assert unchanged.position_errors == pytest.approx(replacement.position_errors)
        assert unchanged.velocity_errors == pytest.approx(replacement.velocity_errors)
        assert unchanged.acceleration_errors == pytest.approx(
            replacement.acceleration_errors
        )
        assert tuple(name for name, _ in unchanged.derivatives) == tuple(
            name for name, _ in replacement.derivatives
        )
        assert tuple(value for _, value in unchanged.derivatives) == pytest.approx(
            tuple(value for _, value in replacement.derivatives), abs=1e-8
        )
        assert unchanged.actuators[0].control == pytest.approx(
            replacement.actuators[0].control
        )
        assert unchanged.actuators[0].actuation == pytest.approx(
            replacement.actuators[0].actuation
        )
    with pytest.raises(ValueError, match="controller.*identity"):
        with reconstruct_declared_cold_start(
            replace(
                derived,
                source_controller_replacement=(original_controller[0], "0" * 64, 1.0),
            )
        ):
            pass
    changed = replay_declared_time_only_input(
        replace(
            derived,
            scheduled_command_step=(
                0.001,
                {"/forceset/unit_generalized_force": 1.5},
            ),
        ),
        clock,
    )
    assert changed.applied_commands == ((1.0,), (1.5,), (1.5,))
    assert dict(changed.audits[1].named_state) == pytest.approx(
        dict(first.audits[1].named_state), abs=1e-8
    )
    base_q = dict(first.audits[-1].named_state)[
        "/jointset/clavicle_derived/clavicle_rhythm/value"
    ]
    changed_q = dict(changed.audits[-1].named_state)[
        "/jointset/clavicle_derived/clavicle_rhythm/value"
    ]
    assert abs(changed_q - base_q) > 1e-10
