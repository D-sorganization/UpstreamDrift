"""Observed model structure must never become anatomical qualification (#11819)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

from defusedxml.common import DTDForbidden
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.model_admission import (
    audit_muscle_model_asset,
)

pytestmark = pytest.mark.unit


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def serialized_candidate(tmp_path: Path) -> Path:
    """Synthetic structural counterexample; no external model bytes are copied."""
    muscles = "".join(
        f'<{law} name="M{i}"><GeometryPath><PathPointSet><objects>'
        '<PathPoint name="origin"><socket_parent_frame>/bodyset/torso'
        "</socket_parent_frame></PathPoint></objects></PathPointSet>"
        f"</GeometryPath></{law}>"
        for i, law in enumerate(
            ["Millard2012EquilibriumMuscle"] * 98 + ["Thelen2003Muscle"] * 422
        )
    )
    path = tmp_path / "counterexample.osim"
    path.write_text(
        '<OpenSimDocument><Model name="synthetic_520">'
        '<BodySet><objects><Body name="hand_r"><attached_geometry>'
        '<Mesh name="missing"><mesh_file>unavailable.vtp</mesh_file></Mesh>'
        "</attached_geometry></Body></objects></BodySet>"
        '<JointSet><objects><PinJoint name="wrist"><coordinates>'
        '<Coordinate name="q"><locked>true</locked></Coordinate>'
        "</coordinates></PinJoint></objects></JointSet><ForceSet><objects>"
        + muscles
        + '<CoordinateActuator name="reserve"/><PointToPointActuator name="rib"/>'
        "</objects></ForceSet><ConstraintSet><objects>"
        '<CoordinateCouplerConstraint name="coupling"/>'
        "</objects></ConstraintSet></Model></OpenSimDocument>",
        encoding="utf-8",
    )
    return path


def test_serialized_counterexample_preserves_unknowns(
    serialized_candidate: Path,
) -> None:
    result = audit_muscle_model_asset(
        serialized_candidate, _digest(serialized_candidate), run_native=False
    )
    assert result.status == "serialized-only"
    assert result.native is None
    assert len(result.serialized["muscle_like_declarations"]) == 520
    assert (
        result.serialized["muscle_like_declarations"][0]["ignore_tendon_compliance"]
        is None
    )
    assert result.serialized["coordinates"][0]["locked"] == "true"
    assert {
        row["class"] for row in result.serialized["actuator_like_declarations"]
    } == {"CoordinateActuator", "PointToPointActuator"}
    assert result.resources[0]["status"] == "missing"
    assert "anatomical-coverage-and-capacity" in result.required_evidence
    assert result.as_dict()["scientific_status"] == "unqualified"


def test_hash_mismatch_precedes_runtime_loading(serialized_candidate: Path) -> None:
    with patch(
        "src.engines.physics_engines.opensim.python.tour_matching.model_admission.import_module"
    ) as loader:
        with pytest.raises(ValueError, match="hash mismatch"):
            audit_muscle_model_asset(serialized_candidate, "0" * 64)
    loader.assert_not_called()


def test_unavailable_runtime_retains_serialized_facts(
    serialized_candidate: Path,
) -> None:
    with patch(
        "src.engines.physics_engines.opensim.python.tour_matching.model_admission.import_module",
        side_effect=ModuleNotFoundError("no native runtime", name="opensim"),
    ):
        result = audit_muscle_model_asset(
            serialized_candidate, _digest(serialized_candidate)
        )
    assert result.status == "native-unavailable"
    assert result.native is None
    assert len(result.serialized["muscle_like_declarations"]) == 520
    assert "native-initialization" in result.required_evidence


def test_source_only_retains_unresolved_role_declarations(
    serialized_candidate: Path,
) -> None:
    declarations = {"right_wrist": ("/bodyset/hand_r",)}
    result = audit_muscle_model_asset(
        serialized_candidate,
        _digest(serialized_candidate),
        run_native=False,
        region_frames=declarations,
    )
    assert result.declared_regions == declarations
    assert result.native is None


def test_dtd_is_forbidden_even_when_namespace_interpretation_is_disabled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dtd.osim"
    path.write_text(
        '<!DOCTYPE Model [<!ENTITY name "injected">]><Model name="&name;"/>',
        encoding="utf-8",
    )
    with pytest.raises(DTDForbidden):
        audit_muscle_model_asset(path, _digest(path), run_native=False)


@pytest.fixture
def native_asset(tmp_path: Path) -> Path:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    model.setName("synthetic_asset_admission")
    body = osim.Body("hand", 1.0, osim.Vec3(0), osim.Inertia(0.01))
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
    coordinate.setName("q")
    coordinate.setDefaultValue(0.31)
    coordinate.setDefaultLocked(True)
    model.addJoint(joint)
    muscle = osim.Millard2012EquilibriumMuscle("ambiguous_label", 10, 0.1, 0.2, 0)
    muscle.addNewPathPoint("a", model.getGround(), osim.Vec3(0))
    muscle.addNewPathPoint("b", body, osim.Vec3(0))
    muscle.set_ignore_tendon_compliance(True)
    model.addForce(muscle)
    reserve = osim.CoordinateActuator("q")
    reserve.setName("do_not_infer_role")
    model.addForce(reserve)
    # A force outside ForceSet must still appear in recursive native inventory.
    hidden = osim.PrescribedForce()
    hidden.setName("nested_drive")
    hidden.connectSocket_frame(body)
    model.addComponent(hidden)
    model.finalizeConnections()
    model.initSystem()
    path = tmp_path / "native_asset.osim"
    model.printToXML(str(path))
    return path


def test_native_inventory_records_structure_without_anatomical_claim(
    native_asset: Path,
) -> None:
    result = audit_muscle_model_asset(
        native_asset,
        _digest(native_asset),
        region_frames={"right_wrist": ("/bodyset/hand",), "torso": ("/ground",)},
        coordinate_roles={"right_wrist": ("/jointset/slider/q",)},
    )
    assert result.status == "native-initialized-structural-only"
    native = result.native
    assert native is not None
    assert native["runtime_version"].startswith("4.")
    assert len(native["loaded_identity_sha256"]) == 64
    assert native["muscles"][0]["ignore_tendon_compliance"] is True
    assert native["muscles"][0]["path_base_frames"] == ("/ground", "/bodyset/hand")
    assert native["coordinates"][0]["locked"] is True
    assert native["coordinates"][0]["declared_roles"] == ("right_wrist",)
    assert native["regions"][0]["anatomy_status"] == "unverified"
    assert native["regions"][0]["attached_muscles"] == ("/forceset/ambiguous_label",)
    assert any(row["path"] == "/nested_drive" for row in native["forces"])
    assert native["nonmuscle_actuators"][0]["role_status"] == "unverified"
    assert "anatomical-coverage-and-capacity" in result.required_evidence
    assert "source-license-and-ancestry" in result.required_evidence
    assert result.as_dict()["scientific_status"] == "unqualified"


def test_native_identity_changes_with_effective_model_property(
    native_asset: Path,
) -> None:
    first = audit_muscle_model_asset(native_asset, _digest(native_asset))
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(native_asset))
    model.updMuscles().get(0).set_ignore_tendon_compliance(False)
    model.printToXML(str(native_asset))
    second = audit_muscle_model_asset(native_asset, _digest(native_asset))
    assert first.native is not None and second.native is not None
    assert first.source_sha256 != second.source_sha256
    assert (
        first.native["loaded_identity_sha256"]
        != second.native["loaded_identity_sha256"]
    )
    assert second.native["muscles"][0]["ignore_tendon_compliance"] is False


def test_native_identity_is_repeatable_and_independent_of_role_labels(
    native_asset: Path,
) -> None:
    first = audit_muscle_model_asset(native_asset, _digest(native_asset))
    second = audit_muscle_model_asset(
        native_asset,
        _digest(native_asset),
        region_frames={"declared_hand": ("/bodyset/hand",)},
    )
    assert first.native is not None and second.native is not None
    assert (
        first.native["loaded_identity_sha256"]
        == second.native["loaded_identity_sha256"]
    )


def test_recursive_muscles_outside_native_registry_are_explicit(
    native_asset: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(native_asset))
    extra = osim.Millard2012EquilibriumMuscle("outside_set", 10, 0.1, 0.2, 0)
    extra.addNewPathPoint("a", model.getGround(), osim.Vec3(0))
    extra.addNewPathPoint("b", model.getBodySet().get("hand"), osim.Vec3(0))
    model.addComponent(extra)
    model.finalizeConnections()
    model.printToXML(str(native_asset))
    result = audit_muscle_model_asset(native_asset, _digest(native_asset))
    assert result.native is not None, result.diagnostic
    assert result.native["registered_muscle_paths"] == ("/forceset/ambiguous_label",)
    assert result.native["unregistered_muscle_paths"] == ("/outside_set",)


def test_native_subtypes_and_auxiliary_components_are_observed(
    native_asset: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(native_asset))
    hand = model.getBodySet().get("hand")
    thelen = osim.Thelen2003Muscle("another_label", 10, 0.1, 0.2, 0)
    thelen.addNewPathPoint("a", model.getGround(), osim.Vec3(0))
    thelen.addNewPathPoint("b", hand, osim.Vec3(0))
    model.addForce(thelen)
    rib = osim.PointToPointActuator()
    rib.setName("unreviewed_internal_actuation")
    rib.set_bodyA("hand")
    rib.set_bodyB("hand")
    rib.setPointB(osim.Vec3(0, 0.1, 0))
    model.addForce(rib)
    floor = osim.ContactHalfSpace(
        osim.Vec3(0), osim.Vec3(0), model.getGround(), "floor"
    )
    sphere = osim.ContactSphere(0.01, osim.Vec3(0), hand, "sphere")
    model.addContactGeometry(floor)
    model.addContactGeometry(sphere)
    contact = osim.HuntCrossleyForce()
    contact.setName("native_contact")
    contact.addGeometry("floor")
    contact.addGeometry("sphere")
    model.addForce(contact)
    controller = osim.PrescribedController()
    controller.setName("empty_controller")
    model.addController(controller)
    model.finalizeConnections()
    model.printToXML(str(native_asset))
    result = audit_muscle_model_asset(native_asset, _digest(native_asset))
    assert result.native is not None, result.diagnostic
    native = result.native
    assert {muscle["class"] for muscle in native["muscles"]} == {
        "Millard2012EquilibriumMuscle",
        "Thelen2003Muscle",
    }
    assert all(
        isinstance(muscle["ignore_tendon_compliance"], bool)
        for muscle in native["muscles"]
    )
    assert {actuator["class"] for actuator in native["nonmuscle_actuators"]} == {
        "CoordinateActuator",
        "PointToPointActuator",
    }
    assert {geometry["class"] for geometry in native["contact_geometry"]} == {
        "ContactHalfSpace",
        "ContactSphere",
    }
    assert any(force["class"] == "HuntCrossleyForce" for force in native["forces"])
    assert native["controllers"][0]["class"] == "PrescribedController"


def test_native_constraints_do_not_become_independent_dof_counts(
    native_asset: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(native_asset))
    body = osim.Body("other", 1, osim.Vec3(0), osim.Inertia(0.01))
    model.addBody(body)
    joint = osim.SliderJoint(
        "slave",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    joint.updCoordinate().setName("q_slave")
    model.addJoint(joint)
    coupling = osim.CoordinateCouplerConstraint()
    coupling.setName("coupled")
    coupling.setDependentCoordinateName("q_slave")
    independent = osim.ArrayStr()
    independent.append("q")
    coupling.setIndependentCoordinateNames(independent)
    coupling.setFunction(osim.LinearFunction(1.0, 0.0))
    model.addConstraint(coupling)
    model.finalizeConnections()
    model.printToXML(str(native_asset))
    result = audit_muscle_model_asset(native_asset, _digest(native_asset))
    assert result.native is not None, result.diagnostic
    assert result.native["constraints"][0]["class"] == "CoordinateCouplerConstraint"
    assert result.native["num_q"] == 2
    assert (
        result.native["independent_dof_status"] == "unverified-constraints-not-reduced"
    )


def test_resource_candidates_are_hashed_without_claiming_native_use(
    serialized_candidate: Path,
) -> None:
    geometry = serialized_candidate.parent / "Geometry"
    geometry.mkdir()
    mesh = geometry / "unavailable.vtp"
    mesh.write_bytes(b"synthetic mesh bytes; not a valid native mesh")
    result = audit_muscle_model_asset(
        serialized_candidate, _digest(serialized_candidate), run_native=False
    )
    assert result.resources[0]["status"] == "local-candidate-unverified-use"
    assert result.resources[0]["sha256"] == _digest(mesh)
    assert "external-dependency-closure" in result.required_evidence


@pytest.mark.parametrize(
    "mapping", [{"wrist": ()}, {"wrist": ("hand",)}, {"": ("/ground",)}]
)
def test_invalid_role_declarations_are_rejected(
    serialized_candidate: Path, mapping: dict
) -> None:
    with pytest.raises(ValueError, match="declared"):
        audit_muscle_model_asset(
            serialized_candidate,
            _digest(serialized_candidate),
            region_frames=mapping,
            run_native=False,
        )


def test_unknown_coordinate_role_cannot_be_reported_as_unlocked(
    native_asset: Path,
) -> None:
    with pytest.raises(ValueError, match="unknown coordinate role"):
        audit_muscle_model_asset(
            native_asset,
            _digest(native_asset),
            coordinate_roles={"wrist": ("/missing",)},
        )


def test_unknown_declared_frames_cannot_look_like_zero_coverage(
    native_asset: Path,
) -> None:
    with pytest.raises(ValueError, match="unknown region frame"):
        audit_muscle_model_asset(
            native_asset, _digest(native_asset), region_frames={"wrist": ("/missing",)}
        )


def test_native_load_failure_is_not_initialization_evidence(tmp_path: Path) -> None:
    pytest.importorskip("opensim")
    path = tmp_path / "bad_socket.osim"
    path.write_text(
        '<OpenSimDocument Version="40500"><Model name="bad">'
        '<ForceSet><objects><CoordinateActuator name="bad">'
        "<coordinate>missing</coordinate></CoordinateActuator></objects>"
        "</ForceSet></Model></OpenSimDocument>",
        encoding="utf-8",
    )
    result = audit_muscle_model_asset(path, _digest(path))
    assert result.status == "native-load-failed"
    assert result.native is None
    assert result.diagnostic
