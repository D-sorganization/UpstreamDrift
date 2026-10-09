"""Native assembly evidence for an original synthetic coupled wrist (#11834)."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.wrist_compatibility import (
    WristKinematicContract,
    observe_native_wrist,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def wrist_source(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Original small mechanism; no third-party anatomy or fitted parameters."""
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    model.setName("synthetic_coupled_wrist")
    radius = osim.Body("radius", 1, osim.Vec3(0), osim.Inertia(0.01))
    proximal = osim.Body("proximal_row", 0.1, osim.Vec3(0), osim.Inertia(0.001))
    hand = osim.Body("hand", 0.2, osim.Vec3(0), osim.Inertia(0.002))
    base_type = (
        osim.PinJoint
        if getattr(request, "param", None) == "free-base"
        else osim.WeldJoint
    )
    base = base_type(
        "base",
        model.getGround(),
        osim.Vec3(0.2, 0.3, 0),
        osim.Vec3(0, 0, 0.4),
        radius,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    if getattr(request, "param", None) == "free-base":
        base.updCoordinate().setName("unrequested_base_angle")
    first = osim.PinJoint(
        "radiocarpal",
        radius,
        osim.Vec3(0),
        osim.Vec3(0),
        proximal,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    if getattr(request, "param", None) == "coupled-custom":
        transform = osim.SpatialTransform()
        axis = transform.updTransformAxis(0)
        names = osim.ArrayStr()
        names.append("flexion")
        axis.setCoordinateNames(names)
        axis.setAxis(osim.Vec3(0, 0, 1))
        transform.updTransformAxis(2).setAxis(osim.Vec3(1, 0, 0))
        function = osim.SimmSpline()
        function.addPoint(-1, -0.5)
        function.addPoint(0, 0)
        function.addPoint(1, 0.5)
        axis.set_function(function)
        first = osim.CustomJoint(
            "radiocarpal",
            radius,
            osim.Vec3(0),
            osim.Vec3(0),
            proximal,
            osim.Vec3(0),
            osim.Vec3(0),
            transform,
        )
        first.finalizeFromProperties()
    second = osim.PinJoint(
        "wrist_hand",
        proximal,
        osim.Vec3(0),
        osim.Vec3(0),
        hand,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    for joint, name in ((first, "flexion"), (second, "hand_flexion")):
        coordinate = joint.updCoordinate()
        coordinate.setName(name)
        coordinate.setRangeMin(-1)
        coordinate.setRangeMax(1)
    for body in (radius, proximal, hand):
        model.addBody(body)
    for joint in (base, first, second):
        model.addJoint(joint)
    coupling = osim.CoordinateCouplerConstraint()
    coupling.setName("wrist_coupler")
    names = osim.ArrayStr()
    names.append("flexion")
    coupling.setIndependentCoordinateNames(names)
    coupling.setDependentCoordinateName("hand_flexion")
    coupling.setFunction(osim.LinearFunction(0.5, 0))
    model.addConstraint(coupling)
    wrap = osim.WrapCylinder()
    wrap.setName("wrist_wrap")
    wrap.set_radius(0.015)
    wrap.set_length(0.2)
    proximal.addWrapObject(wrap)
    muscle = osim.Millard2012EquilibriumMuscle("flexor", 100, 0.1, 0.1, 0)
    muscle.addNewPathPoint("origin", radius, osim.Vec3(0.04, 0.08, 0))
    muscle.addNewPathPoint("insertion", hand, osim.Vec3(-0.04, -0.08, 0))
    muscle.updGeometryPath().addPathWrap(wrap)
    model.addForce(muscle)
    model.finalizeConnections()
    path = tmp_path / "wrist.osim"
    model.printToXML(str(path))
    return path


@pytest.fixture
def contract() -> WristKinematicContract:
    return WristKinematicContract(
        independent_coordinates=("/jointset/radiocarpal/flexion",),
        dependent_coordinates=("/jointset/wrist_hand/hand_flexion",),
        muscle_paths=("/forceset/flexor",),
        required_component_paths=(
            "/constraintset/wrist_coupler",
            "/bodyset/proximal_row/wrapobjectset/wrist_wrap",
        ),
        anchor_frame="/bodyset/radius",
        hand_frame="/bodyset/hand",
        poses_rad=((0.2,), (-0.2,)),
        difference_steps_rad=(1e-3, 1e-4),
        coordinate_tolerance_rad=1e-7,
        translation_tolerance_m=1e-8,
        constraint_tolerance=1e-7,
        moment_arm_tolerance_m=1e-5,
    )


def _observe(path: Path, contract: WristKinematicContract):
    return observe_native_wrist(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), contract
    )


def test_actual_native_coupler_and_relative_hand_frame(wrist_source, contract):
    receipt = _observe(wrist_source, contract)
    assert receipt["scientific_status"] == "unqualified"
    assert receipt["source_file_unchanged"] is True
    assert receipt["prepared_parameters_unchanged"] is True
    assert len(receipt["poses"]) == 2
    first = receipt["poses"][0]
    assert first["coordinates_rad"][
        contract.independent_coordinates[0]
    ] == pytest.approx(0.2, abs=1e-7)
    assert first["coordinates_rad"][contract.dependent_coordinates[0]] == pytest.approx(
        0.1, abs=1e-7
    )
    matrix = first["hand_in_anchor_transform"]
    assert matrix[0][0] == pytest.approx(math.cos(0.3), abs=1e-7)
    assert matrix[1][0] == pytest.approx(math.sin(0.3), abs=1e-7)
    assert matrix[0][3] == pytest.approx(0, abs=1e-7)
    assert max(abs(x) for x in first["native_position_constraint_errors"]) < 1e-7
    assert len(first["moment_arm_checks"]) == 2
    for row in first["moment_arm_checks"]:
        assert row["actual_coordinate_delta_rad"] == pytest.approx(
            2 * row["requested_step_rad"], abs=1e-7
        )
        assert math.isfinite(row["finite_difference_m"])
    assert (
        receipt["required_components"][0]["concrete_class"]
        == "CoordinateCouplerConstraint"
    )


@pytest.mark.parametrize("accuracy", [0, 1, float("nan")])
def test_invalid_assembly_accuracy_rejected_before_loading(
    contract, tmp_path, accuracy
):
    with pytest.raises(ValueError, match="assembly accuracy"):
        observe_native_wrist(
            tmp_path / "absent.osim",
            "0" * 64,
            replace(contract, assembly_accuracy=accuracy),
        )


def test_locked_coordinate_cannot_silently_achieve_motion(wrist_source, contract):
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(wrist_source))
    model.updCoordinateSet().get("flexion").setDefaultLocked(True)
    model.printToXML(str(wrist_source))
    with pytest.raises(ValueError, match="locked"):
        _observe(wrist_source, contract)


def test_dependent_coordinate_cannot_be_declared_independent(wrist_source, contract):
    invalid = replace(
        contract,
        independent_coordinates=contract.dependent_coordinates,
        dependent_coordinates=contract.independent_coordinates,
    )
    with pytest.raises(ValueError, match="dependent"):
        _observe(wrist_source, invalid)


def test_source_hash_is_verified_before_native_loading(wrist_source, contract):
    with pytest.raises(ValueError, match="[Hh]ash|[Ss][Hh][Aa]|checkpoint"):
        observe_native_wrist(wrist_source, "0" * 64, contract)


def test_missing_required_wrap_is_not_ignored(wrist_source, contract):
    invalid = replace(contract, required_component_paths=("/bodyset/missing/wrap",))
    with pytest.raises(ValueError, match="component"):
        _observe(wrist_source, invalid)


@pytest.mark.parametrize("step", [0, -0.1, float("nan")])
def test_invalid_difference_steps_fail_before_native_load(contract, tmp_path, step):
    with pytest.raises(ValueError, match="step"):
        observe_native_wrist(
            tmp_path / "absent.osim",
            "0" * 64,
            replace(contract, difference_steps_rad=(step,)),
        )


def test_perturbation_outside_native_range_fails(wrist_source, contract):
    with pytest.raises(ValueError, match="range"):
        _observe(wrist_source, replace(contract, poses_rad=((0.9999,),)))


def test_wrap_geometry_changes_actual_native_path_length(wrist_source, contract):
    osim = pytest.importorskip("opensim")
    before = _observe(wrist_source, contract)
    model = osim.Model(str(wrist_source))
    wrap = osim.WrapCylinder.safeDownCast(
        model.getComponent(contract.required_component_paths[1])
    )
    wrap.set_radius(0.025)
    model.printToXML(str(wrist_source))
    after = _observe(wrist_source, contract)
    assert after["source_sha256"] != before["source_sha256"]
    assert (
        after["required_components"][1]["native_serialization_sha256"]
        != before["required_components"][1]["native_serialization_sha256"]
    )
    assert (
        abs(
            after["poses"][0]["muscle_lengths_m"]["/forceset/flexor"]
            - before["poses"][0]["muscle_lengths_m"]["/forceset/flexor"]
        )
        > 1e-4
    )


def test_native_coupler_function_changes_actual_hand_transform(wrist_source, contract):
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(wrist_source))
    coupling = osim.CoordinateCouplerConstraint.safeDownCast(
        model.getComponent(contract.required_component_paths[0])
    )
    coupling.setFunction(osim.LinearFunction(0.8, 0))
    model.printToXML(str(wrist_source))
    receipt = _observe(wrist_source, contract)
    pose = receipt["poses"][0]
    assert pose["coordinates_rad"][contract.dependent_coordinates[0]] == pytest.approx(
        0.16, abs=1e-7
    )
    assert pose["hand_in_anchor_transform"][0][0] == pytest.approx(
        math.cos(0.36), abs=1e-7
    )


def test_derivative_disagreement_is_retained_not_promoted(wrist_source, contract):
    receipt = _observe(wrist_source, replace(contract, moment_arm_tolerance_m=1e-20))
    rows = [r for pose in receipt["poses"] for r in pose["moment_arm_checks"]]
    assert any(not row["within_declared_tolerance"] for row in rows)
    assert receipt["scientific_status"] == "unqualified"


def test_disabled_coupler_is_rejected(wrist_source, contract):
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(wrist_source))
    coupling = osim.CoordinateCouplerConstraint.safeDownCast(
        model.getComponent(contract.required_component_paths[0])
    )
    coupling.set_isEnforced(False)
    model.printToXML(str(wrist_source))
    with pytest.raises(ValueError, match="dependent|disabled"):
        _observe(wrist_source, contract)


@pytest.mark.parametrize(
    "field",
    [
        "coordinate_tolerance_rad",
        "translation_tolerance_m",
        "constraint_tolerance",
        "moment_arm_tolerance_m",
    ],
)
def test_invalid_numerical_tolerance_rejected_before_loading(contract, tmp_path, field):
    with pytest.raises(ValueError, match="positive and finite"):
        observe_native_wrist(
            tmp_path / "absent.osim",
            "0" * 64,
            replace(contract, **{field: float("nan")}),
        )


def test_difference_step_must_exceed_assembly_coordinate_tolerance(contract, tmp_path):
    with pytest.raises(ValueError, match="step.*tolerance"):
        observe_native_wrist(
            tmp_path / "absent.osim",
            "0" * 64,
            replace(contract, coordinate_tolerance_rad=1e-3),
        )


def test_prescribed_coordinate_is_not_an_independent_probe(wrist_source, contract):
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(wrist_source))
    coordinate = model.updCoordinateSet().get("flexion")
    coordinate.setPrescribedFunction(osim.Constant(0))
    coordinate.setDefaultIsPrescribed(True)
    model.printToXML(str(wrist_source))
    with pytest.raises(ValueError, match="prescribed"):
        _observe(wrist_source, contract)


@pytest.mark.parametrize("wrist_source", ["coupled-custom"], indirect=True)
def test_native_coupled_motion_requires_explicit_radian_declaration(
    wrist_source, contract
):
    with pytest.raises(ValueError, match="coupled.*radian"):
        _observe(wrist_source, contract)
    declared = replace(
        contract, declared_coupled_radian_coordinates=contract.independent_coordinates
    )
    receipt = _observe(wrist_source, declared)
    assert (
        receipt["native_coordinate_motion_types"][contract.independent_coordinates[0]]
        == 3
    )
    assert receipt["poses"][0]["hand_in_anchor_transform"][0][0] == pytest.approx(
        math.cos(0.2), abs=1e-7
    )
    assert all(
        row["within_declared_tolerance"]
        for pose in receipt["poses"]
        for row in pose["moment_arm_checks"]
    )
    assert receipt["scientific_status"] == "unqualified"


@pytest.mark.parametrize("wrist_source", ["free-base"], indirect=True)
def test_non_target_native_assembly_drift_is_rejected(
    wrist_source, contract, monkeypatch
):
    osim = pytest.importorskip("opensim")
    assemble = osim.Model.assemble

    def drift(model, state):
        assemble(model, state)
        model.updCoordinateSet().get("unrequested_base_angle").setValue(
            state, 0.01, False
        )

    monkeypatch.setattr(osim.Model, "assemble", drift)
    with pytest.raises(ValueError, match="unrequested independent coordinate moved"):
        _observe(wrist_source, contract)


def test_requested_native_assembly_value_is_checked(
    wrist_source, contract, monkeypatch
):
    osim = pytest.importorskip("opensim")
    assemble = osim.Model.assemble

    def wrong_target(model, state):
        assemble(model, state)
        model.updCoordinateSet().get("flexion").setValue(state, 0, False)
        model.updCoordinateSet().get("hand_flexion").setValue(state, 0, False)

    monkeypatch.setattr(osim.Model, "assemble", wrong_target)
    with pytest.raises(ValueError, match="not achieved"):
        _observe(wrist_source, contract)


def test_explicit_assembly_accuracy_binds_changed_numerical_policy(
    wrist_source, contract
):
    before = wrist_source.read_bytes()
    original = _observe(wrist_source, contract)
    refined = _observe(wrist_source, replace(contract, assembly_accuracy=1e-14))
    assert wrist_source.read_bytes() == before
    assert refined["assembly_policy"]["requested_accuracy"] == 1e-14
    assert refined["assembly_policy"]["effective_accuracy"] == 1e-14
    assert (
        refined["assembly_policy"]["source_accuracy"]
        == original["assembly_policy"]["effective_accuracy"]
    )
    assert (
        refined["loaded_serialization_sha256"]
        != original["loaded_serialization_sha256"]
    )
    assert refined["scientific_status"] == "unqualified"
