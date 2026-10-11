"""Native reference observations require declared state and correspondence."""

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_reference_conventions import (
    NativeReferenceCorrespondence,
    ReferenceStateDeclaration,
    audit_source_reference,
    compare_native_references,
)

pytestmark = pytest.mark.unit


def _source(tmp_path: Path, coordinate: float = 0.2, *, dgf: bool = False) -> Path:
    osim = pytest.importorskip("opensim")
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = osim.Model()
    model.setGravity(osim.Vec3(0))
    body = osim.Body("body", 1, osim.Vec3(0), osim.Inertia(0.1))
    joint = osim.SliderJoint("joint", model.getGround(), body)
    joint.updCoordinate().setDefaultValue(coordinate)
    model.addBody(body)
    model.addJoint(joint)
    if dgf:
        muscle = osim.DeGrooteFregly2016Muscle()
        muscle.setName("muscle")
        muscle.set_max_isometric_force(1000)
        muscle.set_optimal_fiber_length(0.1)
        muscle.set_tendon_slack_length(0.1)
        muscle.set_pennation_angle_at_optimal(0)
        muscle.set_ignore_passive_fiber_force(True)
        muscle.set_ignore_tendon_compliance(True)
        muscle.set_fiber_damping(0.02)
        cylinder = osim.WrapCylinder()
        cylinder.setName("test_cylinder")
        cylinder.set_radius(0.01)
        cylinder.set_length(0.1)
        cylinder.set_translation(osim.Vec3(0.1, 0, 0))
        model.getGround().addWrapObject(cylinder)
        muscle.updGeometryPath().addPathWrap(cylinder)
    else:
        muscle = osim.Thelen2003Muscle("muscle", 1000, 0.1, 0.1, 0)
    muscle.addNewPathPoint("origin", model.getGround(), osim.Vec3(0))
    muscle.addNewPathPoint("insertion", body, osim.Vec3(0))
    model.addForce(muscle)
    model.finalizeConnections()
    path = tmp_path / "reference.osim"
    model.printToXML(str(path))
    return path


def _declaration(path: Path) -> ReferenceStateDeclaration:
    return ReferenceStateDeclaration(
        reference_id="synthetic-slider-default",
        provenance="test fixture source default; no assembly or equilibrium",
        mode="post-init-system",
        expected_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _audit(path: Path):
    return audit_source_reference(path, _declaration(path), "/ground")


def test_native_reference_records_analytic_frame_path_and_law_without_mutation(
    tmp_path: Path,
) -> None:
    path = _source(tmp_path)
    source = path.read_bytes()
    first = _audit(path)
    second = _audit(path)
    assert first == second
    assert path.read_bytes() == source
    assert first.source_sha256 == hashlib.sha256(source).hexdigest()
    assert first.reference.mode == "post-init-system"
    assert first.qualification == "not-qualified-for-anatomy"
    frame = next(item for item in first.frames if item.path == "/bodyset/body")
    assert frame.position_in_reference_m == pytest.approx((0.2, 0.0, 0.0))
    for actual, expected in zip(
        frame.rotation_in_reference,
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    joint = first.joints[0]
    assert (joint.parent_frame_path, joint.child_frame_path) == (
        "/ground",
        "/bodyset/body",
    )
    muscle = first.muscles[0]
    assert muscle.path_length_m == pytest.approx(0.2)
    assert muscle.optimal_fiber_length_m == pytest.approx(0.1)
    assert muscle.tendon_slack_length_m == pytest.approx(0.1)
    assert muscle.maximum_isometric_force_n == pytest.approx(1000)
    assert muscle.concrete_class == "Thelen2003Muscle"
    assert muscle.ignore_passive_fiber_force is None
    assert len(muscle.path_points) == 2
    assert muscle.path_points[1].ground_m == pytest.approx((0.2, 0.0, 0.0))
    assert len(muscle.current_route) >= 2
    assert first.native_state.qualification == "not-qualified-for-native-restart"


def test_source_or_reference_state_change_rejects_stale_declaration(
    tmp_path: Path,
) -> None:
    path = _source(tmp_path)
    declaration = _declaration(path)
    audit = audit_source_reference(path, declaration, "/ground")
    assert audit.observation_sha256
    path.write_text(path.read_text().replace("0.2", "0.3", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="source.*SHA|source.*identity"):
        audit_source_reference(path, declaration, "/ground")
    with pytest.raises(ValueError, match="complete|named state"):
        audit_source_reference(
            path,
            replace(
                _declaration(path),
                mode="complete-named-state",
                named_state=(("/jointset/joint/translation/value", 0.3),),
            ),
            "/ground",
        )
    with pytest.raises(ValueError, match="clock|time"):
        audit_source_reference(
            path,
            replace(_declaration(path), expected_native_time_seconds=0.1),
            "/ground",
        )


def test_native_dgf_options_and_declared_wrap_are_observed_without_law_change(
    tmp_path: Path,
) -> None:
    path = _source(tmp_path, dgf=True)
    before = path.read_bytes()
    audit = _audit(path)
    muscle = audit.muscles[0]
    assert muscle.concrete_class == "DeGrooteFregly2016Muscle"
    assert muscle.ignore_passive_fiber_force is True
    assert muscle.ignore_tendon_compliance is True
    assert muscle.fiber_damping == pytest.approx(0.02)
    assert len(muscle.wraps) == 1
    assert muscle.wraps[0].object_name == "test_cylinder"
    assert muscle.wraps[0].object_class == "WrapCylinder"
    assert any(
        point.concrete_class == "PathWrapPoint" for point in muscle.current_route
    )
    assert any(
        (point.wrap_length_m or 0) > 0 and point.wrap_curve_native_m
        for point in muscle.current_route
    )
    assert path.read_bytes() == before


def test_explicit_named_reference_changes_state_identity_without_assembly(
    tmp_path: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    path = _source(tmp_path)
    model = osim.Model(str(path))
    state = model.initSystem()
    names = model.getStateVariableNames()
    values = {
        names.get(i): model.getStateVariableValue(state, names.get(i))
        for i in range(names.getSize())
    }
    position = next(name for name in values if name.endswith("/value"))
    values[position] = 0.3
    declaration = replace(
        _declaration(path),
        mode="complete-named-state",
        named_state=tuple(sorted(values.items())),
    )
    changed = audit_source_reference(path, declaration, "/ground")
    default = _audit(path)
    assert changed.observation_sha256 != default.observation_sha256
    frame = next(item for item in changed.frames if item.path == "/bodyset/body")
    assert frame.position_in_reference_m == pytest.approx((0.3, 0.0, 0.0))
    assert model.getStateVariableValue(state, position) == pytest.approx(0.2)


def test_post_init_system_reference_exposes_native_assembly_of_coupled_defaults(
    tmp_path: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    model.setGravity(osim.Vec3(0))
    for name, default in (("independent", 0.2), ("dependent", 0.9)):
        body = osim.Body(name, 1, osim.Vec3(0), osim.Inertia(0.1))
        joint = osim.SliderJoint(name + "_joint", model.getGround(), body)
        joint.updCoordinate().setName(name)
        joint.updCoordinate().setDefaultValue(default)
        model.addBody(body)
        model.addJoint(joint)
    coupler = osim.CoordinateCouplerConstraint()
    coupler.setName("twice")
    independent = osim.ArrayStr()
    independent.append("independent")
    coupler.setIndependentCoordinateNames(independent)
    coupler.setDependentCoordinateName("dependent")
    coupler.setFunction(osim.LinearFunction(2, 0))
    model.addConstraint(coupler)
    model.finalizeConnections()
    path = tmp_path / "coupled.osim"
    model.printToXML(str(path))
    observation = _audit(path)
    coordinates = {
        c.path.rsplit("/", 1)[-1]: c for c in observation.native_state.coordinates
    }
    assert observation.native_assembly_accuracy == pytest.approx(
        model.get_assembly_accuracy()
    )
    assert coordinates["independent"].source_default_value == pytest.approx(0.2)
    assert coordinates["dependent"].source_default_value == pytest.approx(0.9)
    assert coordinates["dependent"].value == pytest.approx(
        2 * coordinates["independent"].value
    )
    assert any(
        c.value != pytest.approx(c.source_default_value) for c in coordinates.values()
    )
    assert observation.couplers[0].dependent_coordinate_path.endswith("/dependent")
    assert (
        observation.couplers[0].independent_coordinate_paths[0].endswith("/independent")
    )


def test_moving_path_point_reads_native_location_at_each_declared_state(
    tmp_path: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    model.setGravity(osim.Vec3(0))
    body = osim.Body("body", 1, osim.Vec3(0), osim.Inertia(0.1))
    joint = osim.SliderJoint("joint", model.getGround(), body)
    joint.updCoordinate().setName("q")
    joint.updCoordinate().setDefaultValue(0.2)
    model.addBody(body)
    model.addJoint(joint)
    muscle = osim.Thelen2003Muscle("muscle", 1000, 0.1, 0.1, 0)
    muscle.addNewPathPoint("start", model.getGround(), osim.Vec3(0))
    moving = osim.MovingPathPoint()
    moving.setName("moving")
    moving.set_x_location(osim.LinearFunction(0.5, 0))
    moving.set_y_location(osim.Constant(0))
    moving.set_z_location(osim.Constant(0))
    muscle.updGeometryPath().updPathPointSet().cloneAndAppend(moving)
    owned = osim.MovingPathPoint.safeDownCast(
        muscle.updGeometryPath().updPathPointSet().get(1)
    )
    owned.connectSocket_parent_frame(body)
    for axis in "xyz":
        getattr(owned, f"connectSocket_{axis}_coordinate")(joint.updCoordinate())
    conditional = osim.ConditionalPathPoint()
    conditional.setName("conditional")
    conditional.setRangeMin(0.0)
    conditional.setRangeMax(0.25)
    conditional.set_location(osim.Vec3(0.15, 0, 0))
    muscle.updGeometryPath().updPathPointSet().cloneAndAppend(conditional)
    conditional_owned = osim.ConditionalPathPoint.safeDownCast(
        muscle.updGeometryPath().updPathPointSet().get(2)
    )
    conditional_owned.connectSocket_parent_frame(body)
    conditional_owned.connectSocket_coordinate(joint.updCoordinate())
    muscle.addNewPathPoint("end", body, osim.Vec3(0.2, 0, 0))
    model.addForce(muscle)
    model.finalizeConnections()
    path = tmp_path / "moving.osim"
    model.printToXML(str(path))
    default = _audit(path)
    loaded = osim.Model(str(path))
    state = loaded.initSystem()
    names = loaded.getStateVariableNames()
    values = {
        names.get(i): loaded.getStateVariableValue(state, names.get(i))
        for i in range(names.getSize())
    }
    values[next(name for name in values if name.endswith("/q/value"))] = 0.3
    changed = audit_source_reference(
        path,
        replace(
            _declaration(path),
            mode="complete-named-state",
            named_state=tuple(sorted(values.items())),
        ),
        "/ground",
    )
    first_point = default.muscles[0].path_points[1]
    second_point = changed.muscles[0].path_points[1]
    assert first_point.concrete_class == "MovingPathPoint"
    assert first_point.active and second_point.active
    assert first_point.local_m[0] == pytest.approx(0.1)
    assert second_point.local_m[0] == pytest.approx(0.15)
    assert first_point.ground_m[0] == pytest.approx(0.3)
    assert second_point.ground_m[0] == pytest.approx(0.45)
    assert default.muscles[0].path_points[2].concrete_class == "ConditionalPathPoint"
    assert default.muscles[0].path_points[2].active is True
    assert changed.muscles[0].path_points[2].active is False
    assert any(
        point.path.endswith("/conditional")
        for point in default.muscles[0].current_route
    )
    assert all(
        not point.path.endswith("/conditional")
        for point in changed.muscles[0].current_route
    )


def test_comparison_requires_explicit_identity_and_correspondence(
    tmp_path: Path,
) -> None:
    left_path = _source(tmp_path / "left", 0.2)
    right_path = _source(tmp_path / "right", 0.3)
    left, right = _audit(left_path), _audit(right_path)
    mapping = NativeReferenceCorrespondence(
        left_observation_sha256=left.observation_sha256,
        right_observation_sha256=right.observation_sha256,
        frame_pairs=(("/bodyset/body", "/bodyset/body"),),
        joint_pairs=(("/jointset/joint", "/jointset/joint"),),
        coordinate_pairs=(
            (
                left.native_state.coordinates[0].path,
                right.native_state.coordinates[0].path,
            ),
        ),
        muscle_pairs=(("/forceset/muscle", "/forceset/muscle"),),
        reference_rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        reference_translation_m=(0.0, 0.0, 0.0),
        provenance="synthetic identity registration; not an anatomy assertion",
    )
    compared = compare_native_references(left, right, mapping)
    assert compared.qualification == "not-qualified-for-anatomy"
    assert compared.frame_offsets_m[0].delta_m == pytest.approx((0.1, 0.0, 0.0))
    assert compared.muscle_length_deltas_m[0].delta_m == pytest.approx(0.1)
    with pytest.raises(ValueError, match="identity"):
        compare_native_references(
            left, right, replace(mapping, left_observation_sha256="0" * 64)
        )
    with pytest.raises(ValueError, match="unknown|correspondence"):
        compare_native_references(
            left,
            right,
            replace(mapping, muscle_pairs=(("/forceset/other", "/forceset/muscle"),)),
        )
    with pytest.raises(ValueError, match="correspondence|pair"):
        replace(
            mapping,
            frame_pairs=(),
            joint_pairs=(),
            coordinate_pairs=(),
            muscle_pairs=(),
        )
    with pytest.raises(ValueError, match="rotation"):
        replace(
            mapping,
            reference_rotation=((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )


def test_frame_orientation_change_is_not_hidden_by_equal_frame_origins(
    tmp_path: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    paths = []
    for name, yaw in (("left", 0.0), ("right", 0.5)):
        model = osim.Model()
        model.setGravity(osim.Vec3(0))
        body = osim.Body("body", 1, osim.Vec3(0), osim.Inertia(0.1))
        joint = osim.WeldJoint(
            "joint",
            model.getGround(),
            osim.Vec3(0),
            osim.Vec3(0),
            body,
            osim.Vec3(0),
            osim.Vec3(0, 0, yaw),
        )
        model.addBody(body)
        model.addJoint(joint)
        model.finalizeConnections()
        path = tmp_path / f"{name}.osim"
        model.printToXML(str(path))
        paths.append(path)
    left, right = (_audit(path) for path in paths)
    mapping = NativeReferenceCorrespondence(
        left.observation_sha256,
        right.observation_sha256,
        (("/bodyset/body", "/bodyset/body"),),
        (),
        (),
        (),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (0.0, 0.0, 0.0),
        "synthetic frame-only registration",
    )
    result = compare_native_references(left, right, mapping)
    assert result.frame_offsets_m[0].delta_m == pytest.approx((0.0, 0.0, 0.0))
    assert result.frame_orientation_deltas_rad[0].angle_rad == pytest.approx(0.5)
