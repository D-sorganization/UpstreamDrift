"""Exact-source native contact declarations for constrained muscle replay."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _contact_source(tmp_path: Path) -> tuple[Path, Any, Any, Any]:
    from tests.opensim.test_native_constrained_muscle import _source

    osim = pytest.importorskip("opensim")
    path, declaration = _source(
        tmp_path, locked=False, custom=True, moving=True, wrapped=True
    )
    model = osim.Model(str(path))
    floor = osim.ContactHalfSpace(
        osim.Vec3(0.3, 0, 0),
        osim.Vec3(0, 0, np.pi),
        model.getGround(),
        "floor",
    )
    sphere = osim.ContactSphere(
        0.01, osim.Vec3(0), model.getBodySet().get("base"), "base_contact"
    )
    model.addContactGeometry(floor)
    model.addContactGeometry(sphere)
    contact = osim.SmoothSphereHalfSpaceForce()
    contact.connectSocket_sphere(sphere)
    contact.connectSocket_half_space(floor)
    contact.set_stiffness(1e5)
    contact.set_dissipation(1.0)
    contact.set_constant_contact_force(1e-12)
    contact.set_hertz_smoothing(5e4)
    contact.set_hunt_crossley_smoothing(50)
    contact.setName("base_contact_force")
    model.addForce(contact)
    model.finalizeConnections()
    model.printToXML(str(path))

    model = osim.Model(str(path))
    state = model.initSystem()
    muscles = model.getMuscles()
    for index in range(muscles.getSize()):
        muscles.get(index).setActivation(state, 0.05)
    model.equilibrateMuscles(state)
    state_names = model.getStateVariableNames()
    named = {
        str(state_names.get(index)): float(
            model.getStateVariableValue(state, state_names.get(index))
        )
        for index in range(state_names.getSize())
    }
    declaration = replace(
        declaration,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        named_state=named,
    )
    return path, declaration, model, osim


def test_exact_source_contact_declaration_replays_and_samples_native_wrench(
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
    from src.engines.physics_engines.opensim.python.tour_matching.native_contact_policy import (
        declare_native_contact_execution,
    )
    from tests.opensim.test_native_constrained_muscle import _controls

    path, cold_start, _model, _osim = _contact_source(tmp_path)
    contacts = declare_native_contact_execution(path, cold_start.source_sha256)
    mechanics = declare_native_mechanics_execution_profile(
        path, cold_start.source_sha256
    )
    wrapped = declare_wrapped_path_source(path)
    grid = np.array([0.0, 0.001, 0.002])
    controls = _controls(grid, 0.25)

    bundle = build_constrained_muscle_bundle(
        cold_start,
        grid,
        controls,
        wrapped_path_declaration=wrapped,
        native_mechanics_profile=mechanics,
        contact_declaration=contacts,
    )
    replay = replay_constrained_muscle_bundle(
        bundle,
        cold_start,
        wrapped_path_declaration=wrapped,
        native_mechanics_profile=mechanics,
        contact_declaration=contacts,
    )
    assert len(replay.contact_wrenches) == len(grid)
    assert all(len(frame) == 1 for frame in replay.contact_wrenches)
    wrench = replay.contact_wrenches[-1][0]
    assert wrench.label == "contact:base_contact_force"
    assert wrench.force_n is not None and wrench.torque_nm is not None
    assert np.isfinite((*wrench.force_n, *wrench.torque_nm, *wrench.point_m)).all()


def test_contact_is_forbidden_without_exact_source_declaration(tmp_path: Path) -> None:
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

    path, cold_start, _model, _osim = _contact_source(tmp_path)
    grid = np.array([0.0, 0.001])
    mechanics = declare_native_mechanics_execution_profile(
        path, cold_start.source_sha256
    )
    wrapped = declare_wrapped_path_source(path)
    with pytest.raises(ValueError, match="unreviewed constrained muscle component"):
        build_constrained_muscle_bundle(
            cold_start,
            grid,
            _controls(grid, 0.25),
            wrapped_path_declaration=wrapped,
            native_mechanics_profile=mechanics,
        )


def test_contact_source_mutation_invalidates_declaration(tmp_path: Path) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_contact_policy import (
        declare_native_contact_execution,
        validate_native_contact_execution,
    )

    path, cold_start, model, _osim = _contact_source(tmp_path)
    declaration = declare_native_contact_execution(path, cold_start.source_sha256)
    contact = _osim.SmoothSphereHalfSpaceForce.safeDownCast(
        model.updForceSet().get("base_contact_force")
    )
    assert contact is not None
    contact.set_stiffness(3e5)
    model.printToXML(str(path))
    mutated = _osim.Model(str(path))
    mutated.finalizeConnections()
    with pytest.raises(ValueError, match="contact force closure"):
        validate_native_contact_execution(
            mutated, declaration, cold_start.source_sha256
        )
