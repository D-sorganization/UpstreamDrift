"""Exact-source SDK mechanics profiles retain the strict default replay guard."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
    DeclaredColdStart,
)

pytestmark = pytest.mark.unit


def _complex_source(tmp_path: Path) -> tuple[Path, DeclaredColdStart]:
    from tests.opensim.test_native_constrained_muscle import _source

    osim = pytest.importorskip("opensim")
    path, _ = _source(tmp_path, locked=False, custom=True, moving=True, wrapped=True)
    tree = ET.parse(path)
    joint = tree.find(".//CustomJoint[@name='base']")
    assert joint is not None
    coordinates = joint.find("coordinates")
    assert coordinates is not None
    coordinates.append(
        ET.fromstring(
            """<Coordinate name="q2">
              <motion_type>rotational</motion_type>
              <default_value>0.1</default_value>
              <default_speed_value>0</default_speed_value>
              <range>-0.5 0.5</range>
              <clamped>false</clamped>
              <locked>false</locked>
              <prescribed_function />
              <prescribed>false</prescribed>
            </Coordinate>"""
        )
    )
    axis = joint.find("./SpatialTransform/TransformAxis[@name='rotation2']")
    assert axis is not None
    coordinate_names = axis.find("coordinates")
    assert coordinate_names is not None
    coordinate_names.text = "q2"
    law = axis.find("Constant")
    assert law is not None
    axis.remove(law)
    axis.append(
        ET.fromstring(
            """<SimmSpline name="function">
              <x>-0.5 0.5</x>
              <y>0 0.05</y>
            </SimmSpline>"""
        )
    )

    moving = tree.find(".//MovingPathPoint[@name='guide']")
    assert moving is not None
    for axis_name, location_values in (
        ("x", "0.01 0.02"),
        ("y", "0.0 0.01"),
        ("z", "0.0 0.0"),
    ):
        coordinate = moving.find(f"socket_{axis_name}_coordinate")
        assert coordinate is not None
        coordinate.text = "/jointset/base/q2"
        location = moving.find(f"{axis_name}_location")
        assert location is not None
        old_function = next(iter(location))
        location.remove(old_function)
        location.append(
            ET.fromstring(
                f"""<SimmSpline name="function">
              <x>-0.5 0.5</x>
              <y>{location_values}</y>
            </SimmSpline>"""
            )
        )
    path_points = tree.find(
        ".//Millard2012EquilibriumMuscle[@name='flexor']//PathPointSet/objects"
    )
    assert path_points is not None
    insertion = path_points.find("PathPoint[@name='insertion']")
    assert insertion is not None
    conditional = ET.fromstring(
        """<ConditionalPathPoint name="insertion">
          <socket_parent_frame>/bodyset/base</socket_parent_frame>
          <location>0 0.15 0</location>
          <socket_coordinate>/jointset/base/q2</socket_coordinate>
          <range>-0.5 0.5</range>
        </ConditionalPathPoint>"""
    )
    path_points.remove(insertion)
    path_points.append(conditional)
    tree.write(path, encoding="utf-8", xml_declaration=True)

    model = osim.Model(str(path))
    state = model.initSystem()
    for index in range(model.getMuscles().getSize()):
        model.getMuscles().get(index).setActivation(state, 0.05)
    model.equilibrateMuscles(state)
    state_names = model.getStateVariableNames()
    initial = {
        str(state_names.get(index)): float(
            model.getStateVariableValue(state, state_names.get(index))
        )
        for index in range(state_names.getSize())
    }
    declaration = DeclaredColdStart(
        model_path=path,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        named_state=initial,
        time_seconds=0.0,
        lock_targets={},
        chart_bounds={
            "/jointset/base/q": (0.2, 0.4),
            "/jointset/base/q2": (-0.5, 0.5),
        },
        constraint_enforcement={"/constraintset/coupler": True},
        residual_tolerance=1e-9,
    )
    return path, declaration


def test_exact_native_mechanics_profile_enables_only_declared_sdk_components(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
        declare_native_mechanics_execution_profile,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_wrapped_path import (
        declare_wrapped_path_source,
    )
    from tests.opensim.test_native_constrained_muscle import _controls

    path, declaration = _complex_source(tmp_path)
    grid = np.array([0.0, 0.001, 0.002])
    controls = _controls(grid, 0.25)
    wrapped = declare_wrapped_path_source(path)
    assert len(wrapped.conditional_points) == 1
    profile = declare_native_mechanics_execution_profile(
        path, declaration.source_sha256
    )
    assert profile.source_sha256 == declaration.source_sha256
    assert {item.concrete_class for item in profile.components} == {
        "CustomJoint",
        "MovingPathPoint",
    }
    assert len(profile.components) == 2

    with pytest.raises(ValueError, match="moving path law is outside"):
        build_constrained_muscle_bundle(
            declaration,
            grid,
            controls,
            wrapped_path_declaration=wrapped,
        )

    bundle = build_constrained_muscle_bundle(
        declaration,
        grid,
        controls,
        wrapped_path_declaration=wrapped,
        native_mechanics_profile=profile,
    )
    replay = replay_constrained_muscle_bundle(
        bundle,
        declaration,
        wrapped_path_declaration=wrapped,
        native_mechanics_profile=profile,
    )
    assert replay.states.shape[0] == len(grid)
    assert np.isfinite(replay.states).all()
    assert np.isfinite(replay.muscle_forces_n).all()
    assert len(replay.constraint_audits) == len(grid)


def test_native_mechanics_profile_is_source_and_component_bound(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
        declare_native_mechanics_execution_profile,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_wrapped_path import (
        declare_wrapped_path_source,
    )
    from tests.opensim.test_native_constrained_muscle import _controls

    path, declaration = _complex_source(tmp_path)
    profile = declare_native_mechanics_execution_profile(
        path, declaration.source_sha256
    )
    wrapped = declare_wrapped_path_source(path)
    grid = np.array([0.0, 0.001, 0.002])
    controls = _controls(grid, 0.25)

    changed = path.read_bytes().replace(b"0.05</y>", b"0.06</y>", 1)
    path.write_bytes(changed)
    changed_declaration = replace(
        declaration,
        source_sha256=hashlib.sha256(changed).hexdigest(),
    )
    changed_wrapped = declare_wrapped_path_source(path)
    with pytest.raises(ValueError, match="native mechanics profile source identity"):
        build_constrained_muscle_bundle(
            changed_declaration,
            grid,
            controls,
            wrapped_path_declaration=changed_wrapped,
            native_mechanics_profile=profile,
        )
