"""Native constrained-state observations are not complete restart certificates."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import subprocess
import sys
import re
from typing import Any

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_constraint_state import (
    audit_native_constraint_state,
    audit_source_constraint_state,
)

pytestmark = pytest.mark.unit


def _pin_model(*, locked: bool = False) -> tuple[Any, Any, Any]:
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
    coordinate.setDefaultLocked(locked)
    model.addBody(body)
    model.addJoint(joint)
    model.finalizeConnections()
    return model, model.initSystem(), coordinate


def test_native_lock_changes_identity_outside_registered_state_maps() -> None:
    model, state, coordinate = _pin_model()
    before = audit_native_constraint_state(model, state)
    coordinate.setLocked(state, True)
    after = audit_native_constraint_state(model, state)
    assert before.named_state == after.named_state
    assert before.registered_discrete == after.registered_discrete
    assert before.modeling_options == after.modeling_options
    assert before.loaded_model_sha256 == after.loaded_model_sha256
    assert before.coordinates[0].locked is False
    assert after.coordinates[0].locked is True
    assert before.constraint_jacobian_shape == (0, 1)
    assert after.constraint_jacobian_shape == (1, 1)
    assert before.derivatives != after.derivatives
    assert before.observation_sha256 != after.observation_sha256
    assert "native-lock-target-unverified" in after.blockers
    assert after.qualification == "not-qualified-for-native-restart"


def test_declared_lock_target_does_not_become_verified_native_target() -> None:
    model, state, _ = _pin_model(locked=True)
    path = "/jointset/joint/q"
    audit = audit_native_constraint_state(model, state, {path: 0.3})
    assert audit.coordinates[0].declared_lock_target == 0.3
    assert "native-lock-target-unverified" in audit.blockers
    mismatch = audit_native_constraint_state(model, state, {path: 0.4})
    assert "declared-lock-target-mismatch" in mismatch.blockers


def test_model_owned_lock_target_can_change_another_native_state() -> None:
    osim = pytest.importorskip("opensim")
    model, first, coordinate = _pin_model(locked=True)
    before = audit_native_constraint_state(model, first)
    second = osim.State(first)
    coordinate.setLocked(second, False)
    coordinate.setValue(second, 0.6, False)
    coordinate.setLocked(second, True)
    first.invalidateAllCacheAtOrAbove(osim.Stage(osim.Stage.Position))
    after = audit_native_constraint_state(model, first)
    assert before.named_state == after.named_state
    assert before.coordinates == after.coordinates
    assert before.loaded_model_sha256 == after.loaded_model_sha256
    assert before.position_errors == (0.0,)
    assert after.position_errors == pytest.approx((-0.3,))
    assert before.observation_sha256 != after.observation_sha256


def test_source_audit_owns_fresh_model_and_preserves_source_lock(
    tmp_path: Path,
) -> None:
    model, _, _ = _pin_model(locked=True)
    path = tmp_path / "locked.osim"
    model.printToXML(str(path))
    initial = {"/jointset/joint/q/value": 0.3, "/jointset/joint/q/speed": 0.0}
    first = audit_source_constraint_state(path, initial)
    second = audit_source_constraint_state(path, initial)
    assert first == second
    assert first.source_sha256
    assert first.coordinates[0].locked
    assert first.position_errors == (0.0,)
    assert "native-lock-target-unverified" in first.blockers


def test_source_audit_rejects_silently_ignored_locked_state_restore(
    tmp_path: Path,
) -> None:
    model, _, _ = _pin_model(locked=True)
    path = tmp_path / "locked.osim"
    model.printToXML(str(path))
    initial = {"/jointset/joint/q/value": 0.6, "/jointset/joint/q/speed": 0.0}
    with pytest.raises(ValueError, match="achieved|restore"):
        audit_source_constraint_state(path, initial)


@pytest.mark.parametrize("initial", [{}, {"unknown": 0.0}])
def test_source_audit_requires_complete_named_state(
    tmp_path: Path, initial: dict[str, float]
) -> None:
    model, _, _ = _pin_model()
    path = tmp_path / "pin.osim"
    model.printToXML(str(path))
    with pytest.raises(ValueError, match="complete|coverage"):
        audit_source_constraint_state(path, initial)


def test_native_coupler_enforcement_is_observed_separately() -> None:
    osim = pytest.importorskip("opensim")
    model, _, _ = _pin_model()
    body = osim.Body("second", 1, osim.Vec3(0), osim.Inertia(1, 1, 1))
    joint = osim.PinJoint(
        "second_joint",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    joint.updCoordinate().setName("dependent")
    joint.updCoordinate().setDefaultValue(0.6)
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
    model.finalizeConnections()
    state = model.initSystem()
    before = audit_native_constraint_state(model, state)
    coupler.setIsEnforced(state, False)
    after = audit_native_constraint_state(model, state)
    assert before.constraints[0].enforced is True
    assert after.constraints[0].enforced is False
    assert before.constraint_jacobian_shape == (1, 2)
    assert after.constraint_jacobian_shape == (0, 2)
    assert before.observation_sha256 != after.observation_sha256
    assert "native-constraint-state-unqualified" in after.blockers


def test_nonmuscle_assistance_remains_enabled_and_visible() -> None:
    osim = pytest.importorskip("opensim")
    model, _, _ = _pin_model()
    assistance = osim.CoordinateActuator("q")
    assistance.setName("reserve")
    assistance.setOptimalForce(10)
    model.addForce(assistance)
    model.finalizeConnections()
    state = model.initSystem()
    controls = osim.Vector(1, 0.2)
    model.setControls(state, controls)
    audit = audit_native_constraint_state(model, state)
    observed = audit.actuators[0]
    assert observed.path == "/forceset/reserve"
    assert observed.enabled and not observed.is_muscle
    assert observed.control == 0.2
    assert observed.actuation == pytest.approx(2.0)
    assert "nonmuscle-assistance-unqualified" in audit.blockers


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_target_rejected(value: object) -> None:
    model, state, _ = _pin_model(locked=True)
    with pytest.raises((TypeError, ValueError)):
        audit_native_constraint_state(model, state, {"/jointset/joint/q": value})


def test_unknown_target_rejected_and_result_cannot_claim_qualification() -> None:
    model, state, _ = _pin_model(locked=True)
    with pytest.raises(ValueError, match="unknown"):
        audit_native_constraint_state(model, state, {"/unknown": 0.3})
    result = audit_native_constraint_state(model, state)
    with pytest.raises(ValueError):
        replace(result, qualification="qualified")


@pytest.mark.parametrize("targets", [[], False, 1, "unknown"])
def test_target_container_requires_mapping_even_when_empty(targets: object) -> None:
    model, state, _ = _pin_model()
    with pytest.raises(TypeError, match="mapping"):
        audit_native_constraint_state(model, state, targets)


def test_missing_visual_mesh_native_log_does_not_break_owned_xml_cleanup(
    tmp_path: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    model, _, _ = _pin_model()
    body = model.updBodySet().get("body")
    body.attachGeometry(osim.Mesh("missing-visual-only.vtp"))
    model.finalizeConnections()
    source = tmp_path / "missing-visual.osim"
    model.printToXML(str(source))
    # Native XML upgrade logs before loading this older, otherwise original fixture.
    source.write_text(
        re.sub(r'Version="\d+"', 'Version="40000"', source.read_text(), count=1)
    )
    code = (
        "from pathlib import Path; "
        "from src.engines.physics_engines.opensim.python.tour_matching.native_constraint_state "
        "import audit_source_constraint_state; "
        "audit_source_constraint_state(Path(__import__('sys').argv[1]), "
        "{'/jointset/joint/q/value':0.3,'/jointset/joint/q/speed':0.0})"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(source)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
