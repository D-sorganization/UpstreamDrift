"""Source-bound wrapped-path declarations remain explicit and narrow."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
    DeclaredColdStart,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_reference_conventions import (
    ReferenceStateDeclaration,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_wrapped_path import (
    WrapObjectIdentity,
    audit_native_wrapped_path_source,
    audit_wrapped_path_source,
    declare_wrapped_path_source,
    validate_wrapped_path_declaration,
)

pytestmark = pytest.mark.unit


def _wrapped_source(tmp_path: Path, shape: str = "WrapCylinder") -> Path:
    from tests.opensim.test_native_reference_conventions import _source

    path = _source(tmp_path, dgf=True)
    if shape == "WrapCylinder":
        return path
    tree = ET.parse(path)
    cylinder = tree.find(".//WrapCylinder[@name='test_cylinder']")
    assert cylinder is not None
    cylinder.tag = shape
    length = cylinder.find("length")
    if length is not None:
        cylinder.remove(length)
    if shape == "WrapEllipsoid":
        radius = cylinder.find("radius")
        if radius is not None:
            cylinder.remove(radius)
        dimensions = ET.fromstring("<dimensions>0.01 0.015 0.02</dimensions>")
        cylinder.append(dimensions)
    elif shape == "WrapSphere":
        translation = cylinder.find("translation")
        assert translation is not None
        translation.text = "0.1 0.005 0"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _reference(path: Path) -> ReferenceStateDeclaration:
    return ReferenceStateDeclaration(
        reference_id="wrapped-path-default",
        provenance="source-derived native post-init fixture state",
        mode="post-init-system",
        expected_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@pytest.mark.parametrize("shape", ["WrapCylinder", "WrapEllipsoid", "WrapSphere"])
def test_typed_declaration_closes_supported_wrap_shape_and_native_route(
    tmp_path: Path, shape: str
) -> None:
    path = _wrapped_source(tmp_path, shape)
    source = path.read_bytes()
    declaration = declare_wrapped_path_source(path)

    assert declaration.schema_version == "native-wrapped-path-v1"
    assert declaration.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(declaration.wrap_objects) == 1
    assert declaration.wrap_objects[0].concrete_class == shape
    assert len(declaration.path_wraps) == 1
    assert (
        declaration.path_wraps[0].wrap_object_path == declaration.wrap_objects[0].path
    )
    observation = audit_wrapped_path_source(path, declaration, _reference(path))
    muscle = observation.reference_observation.muscles[0]
    assert muscle.wraps[0].object_class == shape
    assert muscle.wraps[0].object_path == declaration.wrap_objects[0].path
    assert any(
        point.concrete_class == "PathWrapPoint" for point in muscle.current_route
    )
    assert path.read_bytes() == source


def test_wrapped_declaration_rejects_changed_source_hash(tmp_path: Path) -> None:
    path = _wrapped_source(tmp_path)
    declaration = declare_wrapped_path_source(path)
    path.write_bytes(path.read_bytes().replace(b"0.01", b"0.02", 1))

    with pytest.raises(ValueError, match="source.*identity|source.*SHA"):
        audit_wrapped_path_source(path, declaration, _reference(path))


def test_wrapped_declaration_rejects_link_name_that_disagrees_with_native_target(
    tmp_path: Path,
) -> None:
    path = _wrapped_source(tmp_path)
    declaration = declare_wrapped_path_source(path)
    mismatched = replace(
        declaration,
        path_wraps=(replace(declaration.path_wraps[0], wrap_object_name="other"),),
    )
    import opensim as osim

    model = osim.Model(str(path))
    model.finalizeConnections()
    with pytest.raises(ValueError, match="differs from native source"):
        validate_wrapped_path_declaration(model, mismatched, declaration.source_sha256)


def test_wrapped_declaration_rejects_pathwrap_target_outside_closed_objects(
    tmp_path: Path,
) -> None:
    path = _wrapped_source(tmp_path)
    declaration = declare_wrapped_path_source(path)
    with pytest.raises(ValueError, match="target is missing"):
        replace(
            declaration,
            path_wraps=(
                replace(
                    declaration.path_wraps[0],
                    wrap_object_path="/bodyset/body/wrapobject/missing",
                ),
            ),
        )


def test_wrap_component_identity_requires_hex_sha256(tmp_path: Path) -> None:
    declaration = declare_wrapped_path_source(_wrapped_source(tmp_path))
    wrap = declaration.wrap_objects[0]
    with pytest.raises(ValueError, match="serialized SHA-256"):
        WrapObjectIdentity(
            path=wrap.path,
            concrete_class=wrap.concrete_class,
            parent_frame_path=wrap.parent_frame_path,
            translation_m=wrap.translation_m,
            xyz_body_rotation_rad=wrap.xyz_body_rotation_rad,
            active=wrap.active,
            quadrant=wrap.quadrant,
            shape_parameters=wrap.shape_parameters,
            serialized_sha256="g" * 64,
        )


def test_native_contract_adapter_reports_required_declaration_as_execution_block(
    tmp_path: Path,
) -> None:
    path = _wrapped_source(tmp_path)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    audit = audit_native_wrapped_path_source(path, source_sha256, None)

    assert audit.declaration_sha256 is None
    assert audit.findings
    assert all(
        finding.code.value == "wrapped_path_declaration_required"
        for finding in audit.findings
    )
    assert all(finding.execution_effect.name == "BLOCK" for finding in audit.findings)
    assert all(
        finding.qualification_effect.name == "NONE" for finding in audit.findings
    )


def test_native_contract_adapter_reports_observation_and_qualification_separately(
    tmp_path: Path,
) -> None:
    path = _wrapped_source(tmp_path)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    declaration = declare_wrapped_path_source(path)

    audit = audit_native_wrapped_path_source(
        path, source_sha256, declaration, _reference(path)
    )

    by_code = {finding.code.value: finding for finding in audit.findings}
    assert by_code["wrapped_path_source_closed"].execution_effect.name == "PROCEED"
    assert (
        by_code["wrapped_path_native_route_observed"].qualification_effect.name
        == "OBSERVATION"
    )
    assert (
        by_code["wrapped_path_transition_unqualified"].execution_effect.name
        == "REQUIRES_OPT_IN"
    )
    assert (
        by_code["wrapped_path_transition_unqualified"].qualification_effect.name
        == "UNQUALIFIED"
    )


def test_native_contract_adapter_blocks_mismatched_source_identity(
    tmp_path: Path,
) -> None:
    path = _wrapped_source(tmp_path)
    declared_hash = hashlib.sha256(b"different model bytes").hexdigest()

    audit = audit_native_wrapped_path_source(path, declared_hash, None)

    assert len(audit.findings) == 1
    finding = audit.findings[0]
    assert finding.code.value == "wrapped_path_source_closure_invalid"
    assert finding.execution_effect.name == "BLOCK"
    assert finding.qualification_effect.name == "NONE"


def test_native_contract_adapter_blocks_native_route_audit_failure(
    tmp_path: Path,
) -> None:
    path = _wrapped_source(tmp_path)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    declaration = declare_wrapped_path_source(path)
    mismatched_reference = replace(_reference(path), expected_source_sha256="0" * 64)

    audit = audit_native_wrapped_path_source(
        path, source_sha256, declaration, mismatched_reference
    )

    route_failure = next(
        finding
        for finding in audit.findings
        if finding.code.value == "wrapped_path_native_route_invalid"
    )
    assert route_failure.execution_effect.name == "BLOCK"
    assert route_failure.qualification_effect.name == "NONE"


def test_wrapped_path_defaults_still_reject_without_opt_in_declaration(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
    )
    from tests.opensim.test_native_constrained_muscle import _controls, _source

    _path, declaration = _source(tmp_path)
    wrapped_dir = tmp_path / "wrapped"
    wrapped_dir.mkdir()
    wrapped = _wrapped_source(wrapped_dir)
    # The legacy source fixture is still rejected before any native execution.
    # A wrapped source cannot enter the default API either.
    wrapped_declaration = replace(
        declaration,
        model_path=wrapped,
        source_sha256=hashlib.sha256(wrapped.read_bytes()).hexdigest(),
    )
    grid = np.array([0.0, 0.001])
    with pytest.raises(ValueError, match="unreviewed constrained muscle component"):
        build_constrained_muscle_bundle(wrapped_declaration, grid, _controls(grid, 0.2))


def test_declared_wrapped_path_replays_only_with_exact_opt_in_identity(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )
    from tests.opensim.test_native_constrained_muscle import _controls, _source

    path, cold_start = _source(tmp_path, locked=False, wrapped=True)
    declaration = declare_wrapped_path_source(path)
    grid = np.array([0.0, 0.001, 0.002])
    controls = _controls(grid, 0.25)

    bundle = build_constrained_muscle_bundle(
        cold_start, grid, controls, wrapped_path_declaration=declaration
    )
    replay = replay_constrained_muscle_bundle(
        bundle, cold_start, wrapped_path_declaration=declaration
    )

    assert replay.states.shape == (3, len(cold_start.named_state))
    assert np.isfinite(replay.muscle_forces_n).all()
    changed = path.read_bytes().replace(b"hybrid", b"midpoint", 1)
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="source identity"):
        build_constrained_muscle_bundle(
            cold_start, grid, controls, wrapped_path_declaration=declaration
        )


def test_declaration_records_conditional_points_and_rejects_bad_active_range(
    tmp_path: Path,
) -> None:
    path = _wrapped_source(tmp_path)
    tree = ET.parse(path)
    points = tree.find(".//PathPointSet/objects")
    assert points is not None
    insertion = points.find("PathPoint[@name='insertion']")
    assert insertion is not None
    conditional = ET.fromstring(
        """<ConditionalPathPoint name="insertion">
          <socket_parent_frame>/bodyset/body</socket_parent_frame>
          <location>0 0.1 0</location>
          <socket_coordinate>/jointset/joint/joint_coord_0</socket_coordinate>
          <range>0.2 0.4</range>
        </ConditionalPathPoint>"""
    )
    points.remove(insertion)
    points.append(conditional)
    tree.write(path, encoding="utf-8", xml_declaration=True)

    declaration = declare_wrapped_path_source(path)

    assert len(declaration.conditional_points) == 1
    assert declaration.conditional_points[0].active_range == (0.2, 0.4)

    conditional.find("range").text = "0.4 0.2"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match="active range"):
        declare_wrapped_path_source(path)


def test_native_reference_observes_conditional_point_on_both_sides(
    tmp_path: Path,
) -> None:
    osim = pytest.importorskip("opensim")
    path = _wrapped_source(tmp_path)
    tree = ET.parse(path)
    points = tree.find(".//PathPointSet/objects")
    assert points is not None
    insertion = points.find("PathPoint[@name='insertion']")
    assert insertion is not None
    conditional = ET.fromstring(
        """<ConditionalPathPoint name="insertion">
          <socket_parent_frame>/bodyset/body</socket_parent_frame>
          <location>0 0 0</location>
          <socket_coordinate>/jointset/joint/joint_coord_0</socket_coordinate>
          <range>0.1 0.3</range>
        </ConditionalPathPoint>"""
    )
    points.remove(insertion)
    points.append(conditional)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    declaration = declare_wrapped_path_source(path)
    model = osim.Model(str(path))
    state = model.initSystem()
    names = model.getStateVariableNames()
    named = {
        names.get(i): float(model.getStateVariableValue(state, names.get(i)))
        for i in range(names.getSize())
    }
    coordinate_path = "/jointset/joint/joint_coord_0/value"

    active = audit_wrapped_path_source(path, declaration, _reference(path))
    named[coordinate_path] = 0.5
    inactive_reference = ReferenceStateDeclaration(
        reference_id="wrapped-path-conditional-inactive",
        provenance="same source, one named slider value changed for state observation",
        mode="complete-named-state",
        expected_source_sha256=declaration.source_sha256,
        named_state=tuple(sorted(named.items())),
    )
    inactive = audit_wrapped_path_source(path, declaration, inactive_reference)
    active_point = next(
        point
        for point in active.reference_observation.muscles[0].path_points
        if point.path.endswith("/insertion")
    )
    inactive_point = next(
        point
        for point in inactive.reference_observation.muscles[0].path_points
        if point.path.endswith("/insertion")
    )
    assert active_point.active
    assert not inactive_point.active
    assert active_point.path in {
        point.path for point in active.reference_observation.muscles[0].current_route
    }
    assert inactive_point.path not in {
        point.path for point in inactive.reference_observation.muscles[0].current_route
    }
