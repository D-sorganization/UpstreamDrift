"""Exact-source native contact declarations for constrained muscle replay."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _contact_source(
    tmp_path: Path, contact_kind: str = "smooth"
) -> tuple[Path, Any, Any, Any]:
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
    if contact_kind == "smooth":
        contact = osim.SmoothSphereHalfSpaceForce()
        contact.connectSocket_sphere(sphere)
        contact.connectSocket_half_space(floor)
        contact.set_stiffness(1e5)
        contact.set_dissipation(1.0)
        contact.set_constant_contact_force(1e-12)
        contact.set_hertz_smoothing(5e4)
        contact.set_hunt_crossley_smoothing(50)
    elif contact_kind == "hunt_crossley":
        contact = osim.HuntCrossleyForce()
        contact.addGeometry("floor")
        contact.addGeometry("base_contact")
        contact.setStiffness(1e5)
        contact.setDissipation(1.0)
    else:
        raise ValueError("unknown test contact kind")
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


@pytest.mark.parametrize("contact_kind", ["smooth", "hunt_crossley"])
def test_contact_is_forbidden_without_exact_source_declaration(
    tmp_path: Path, contact_kind: str
) -> None:
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

    path, cold_start, _model, _osim = _contact_source(tmp_path, contact_kind)
    grid = np.array([0.0, 0.001])
    mechanics = declare_native_mechanics_execution_profile(
        path, cold_start.source_sha256
    )
    wrapped = declare_wrapped_path_source(path)
    with pytest.raises(
        ValueError,
        match="unreviewed constrained muscle component|native source XML must be self-contained",
    ):
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


def test_contact_provider_implementation_identity_is_part_of_policy(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
        _admission_identity,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_contact_policy import (
        declare_native_contact_execution,
        native_contact_provider_sha256,
    )

    path, cold_start, _model, _osim = _contact_source(tmp_path)
    declaration = declare_native_contact_execution(path, cold_start.source_sha256)
    provider_sha = native_contact_provider_sha256()
    first = _admission_identity(
        cold_start,
        "a" * 64,
        contact_declaration_sha256=declaration.declaration_sha256,
        native_contact_provider_sha256=provider_sha,
    )
    changed = _admission_identity(
        cold_start,
        "a" * 64,
        contact_declaration_sha256=declaration.declaration_sha256,
        native_contact_provider_sha256="b" * 64,
    )
    assert provider_sha != "b" * 64
    assert first != changed


def test_contact_declaration_rejects_wrong_source_identity(tmp_path: Path) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_contact_policy import (
        declare_native_contact_execution,
        validate_native_contact_execution,
    )

    path, cold_start, model, _osim = _contact_source(tmp_path)
    declaration = declare_native_contact_execution(path, cold_start.source_sha256)
    with pytest.raises(ValueError, match="source identity differs"):
        validate_native_contact_execution(model, declaration, "0" * 64)


@pytest.mark.parametrize(
    ("contact_kind", "expected_class"),
    [
        ("smooth", "SmoothSphereHalfSpaceForce"),
        ("hunt_crossley", "HuntCrossleyForce"),
    ],
)
def test_supported_contact_laws_bind_force_and_geometry_closure(
    tmp_path: Path, contact_kind: str, expected_class: str
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_contact_policy import (
        declare_native_contact_execution,
        validate_native_contact_execution,
    )

    path, cold_start, model, _osim = _contact_source(tmp_path, contact_kind)
    declaration = declare_native_contact_execution(path, cold_start.source_sha256)
    assert len(declaration.forces) == 1
    assert declaration.forces[0].concrete_class == expected_class
    assert {item.concrete_class for item in declaration.geometries} == {
        "ContactHalfSpace",
        "ContactSphere",
    }
    validate_native_contact_execution(model, declaration, cold_start.source_sha256)


def test_hunt_crossley_request_admits_only_exact_runner_contact_closure(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
        MocoInitialBindings,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
        MocoTrackingConfig,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_contact_policy import (
        declare_native_contact_execution,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_diagnostic_contract import (
        DiagnosticCode,
        NativeDiagnosticPolicy,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
        declare_native_mechanics_execution_profile,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_preparation import (
        _audit_loaded_model_components,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_request import (
        NativeMocoRequest,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_wrapped_path import (
        declare_wrapped_path_source,
    )

    path, cold_start, model, _osim = _contact_source(tmp_path, "hunt_crossley")
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    cold_start = replace(cold_start, source_sha256=source_sha256)
    state_bounds = {
        name: (
            (0.0, 1.0)
            if name.endswith(("/activation", "/fiber_length"))
            else (-10.0, 10.0)
        )
        for name in cold_start.named_state
    }
    bindings = MocoInitialBindings(
        state_bounds,
        cold_start.named_state,
        {"/forceset/flexor": (0.001, 1.0), "/forceset/extensor": (0.001, 1.0)},
    )
    declaration = declare_native_contact_execution(path, source_sha256)
    request = NativeMocoRequest(
        model_path=path,
        trc_path=path,
        states_guess_path=path,
        model_sha256=source_sha256,
        trc_sha256="a" * 64,
        states_guess_sha256="b" * 64,
        bindings=bindings,
        config=MocoTrackingConfig(horizon_s=0.01),
        marker_weights={"base_marker": 1.0},
        marker_bindings={"base_marker": ("/bodyset/base", (0.0, 0.0, 0.0))},
        registration=None,
        reference_frame_path="/ground",
        passive_policy=None,
        excluded_markers={},
        constrained_cold_start=cold_start,
        diagnostic_policy=NativeDiagnosticPolicy(
            source_sha256,
            (DiagnosticCode.PASSIVE_POLICY_UNAVAILABLE,),
            "explicit native contact diagnostic test",
        ),
        wrapped_path_declaration=declare_wrapped_path_source(path),
        native_mechanics_profile=declare_native_mechanics_execution_profile(
            path, source_sha256
        ),
        contact_declaration=declaration,
    )
    blockers = _audit_loaded_model_components(model, request)
    assert "native-contact-declaration-invalid" not in blockers
    assert "native-external-force-uninventoried" not in blockers

    changed_force = replace(declaration.forces[0], serialized_sha256="e" * 64)
    changed = replace(declaration, forces=(changed_force,))
    changed_request = replace(request, contact_declaration=changed)
    changed_blockers = _audit_loaded_model_components(model, changed_request)
    assert "native-contact-declaration-invalid" in changed_blockers
