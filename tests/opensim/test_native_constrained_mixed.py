"""Declared coupled mixed replay preserves source and applied-input semantics."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.opensim.test_native_constrained_muscle import _source
from tests.opensim.test_native_mixed_actuation import mixed_model as mixed_model
from tests.opensim.test_native_mixed_moco import mixed_request as mixed_request

from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_actuation import (
    ActuationRole,
    MixedActuationProfile,
    MixedChannel,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_mixed_replay import (
    build_native_mixed_replay_bundle,
    replay_native_mixed_bundle,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def coupled_mixed(tmp_path: Path) -> tuple[Any, Any]:
    osim = pytest.importorskip("opensim")
    path, declaration = _source(tmp_path, custom=True)
    model = osim.Model(str(path))
    actuator = osim.CoordinateActuator("q")
    actuator.setName("root_torque")
    actuator.setOptimalForce(7.0)
    actuator.setMinControl(-0.4)
    actuator.setMaxControl(0.4)
    model.addForce(actuator)
    model.finalizeConnections()
    model.printToXML(str(path))
    declaration = replace(
        declaration, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    profile = MixedActuationProfile(
        (
            MixedChannel("/forceset/flexor", ActuationRole.MUSCLE, (0.01, 1.0)),
            MixedChannel("/forceset/extensor", ActuationRole.MUSCLE, (0.01, 1.0)),
            MixedChannel(
                "/forceset/root_torque", ActuationRole.ROOT_RESIDUAL, (-0.4, 0.4)
            ),
        )
    )
    return declaration, profile


def _controls(grid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "/forceset/flexor": np.full(grid.shape, 0.1),
        "/forceset/extensor": np.full(grid.shape, 0.15),
        "/forceset/root_torque": np.full(grid.shape, 0.2),
    }


def _bundle(declaration: Any, profile: Any, controls: Any = None) -> Any:
    grid = np.array([0.0, 0.005, 0.01, 0.015])
    return build_native_mixed_replay_bundle(
        declaration.model_path,
        declaration.named_state,
        grid,
        _controls(grid) if controls is None else controls,
        profile,
        constrained_cold_start=declaration,
    )


def test_coupled_mixed_fresh_replay_preserves_constraints_and_units(
    coupled_mixed: tuple[Any, Any],
) -> None:
    declaration, profile = coupled_mixed
    bundle = _bundle(declaration, profile)
    result = replay_native_mixed_bundle(
        bundle, declaration.model_path, profile, constrained_cold_start=declaration
    )
    assert result.states.shape[0] == 4
    assert len(result.constraint_audits) == 4
    assert result.profile.channels[-1].output_unit == "N*m"
    assert np.allclose(result.actuations[:, -1], 1.4, rtol=0, atol=1e-12)
    assert np.allclose(result.applied_controls[:, -1], 0.2, rtol=0, atol=1e-12)
    for audit in result.constraint_audits:
        assert max(map(abs, audit.position_errors), default=0) <= 1e-9
        assert max(map(abs, audit.velocity_errors), default=0) <= 1e-9
        assert all(item.enforced for item in audit.constraints)
    assert not result.states.flags.writeable


def test_coupled_mixed_changed_future_has_identical_prefix(
    coupled_mixed: tuple[Any, Any],
) -> None:
    declaration, profile = coupled_mixed
    baseline = _bundle(declaration, profile)
    controls = _controls(np.array([0.0, 0.005, 0.01, 0.015]))
    controls["/forceset/root_torque"][2:] = -0.3
    changed = _bundle(declaration, profile, controls)
    first = replay_native_mixed_bundle(
        baseline, declaration.model_path, profile, constrained_cold_start=declaration
    )
    second = replay_native_mixed_bundle(
        changed, declaration.model_path, profile, constrained_cold_start=declaration
    )
    np.testing.assert_array_equal(first.states[:2], second.states[:2])
    assert not np.allclose(first.states[-1], second.states[-1], rtol=0, atol=1e-10)
    assert baseline.applied_input_sha256 != changed.applied_input_sha256


def test_coupled_source_still_rejects_without_declaration(
    coupled_mixed: tuple[Any, Any],
) -> None:
    declaration, profile = coupled_mixed
    with pytest.raises(ValueError):
        build_native_mixed_replay_bundle(
            declaration.model_path,
            declaration.named_state,
            np.array([0.0, 0.01]),
            _controls(np.array([0.0, 0.01])),
            profile,
        )


def test_coupled_mixed_rejects_changed_declared_policy(
    coupled_mixed: tuple[Any, Any],
) -> None:
    declaration, profile = coupled_mixed
    bundle = _bundle(declaration, profile)
    changed = replace(declaration, residual_tolerance=1e-8)
    with pytest.raises(ValueError, match="identity|policy"):
        replay_native_mixed_bundle(
            bundle, changed.model_path, profile, constrained_cold_start=changed
        )


@pytest.mark.parametrize("kind", ["value", "speed"])
def test_native_manager_projection_cannot_hide_changed_seed(
    coupled_mixed: tuple[Any, Any], kind: str
) -> None:
    declaration, profile = coupled_mixed
    name = "/jointset/follower/dependent/" + kind
    changed = replace(
        declaration,
        named_state={
            **declaration.named_state,
            name: declaration.named_state[name] + 5e-10,
        },
    )
    bundle = _bundle(changed, profile)
    with pytest.raises(RuntimeError, match="initial.*state|seed"):
        replay_native_mixed_bundle(
            bundle, changed.model_path, profile, constrained_cold_start=changed
        )


@pytest.mark.parametrize(
    "field", ["constant_commands", "allow_source_controllers_for_observation"]
)
def test_coupled_mixed_rejects_competing_input_policy(
    coupled_mixed: tuple[Any, Any], field: str
) -> None:
    declaration, profile = coupled_mixed
    value = {"/forceset/root_torque": 0.2} if field == "constant_commands" else True
    with pytest.raises(ValueError):
        _bundle(replace(declaration, **{field: value}), profile)


def test_maintained_moco_admits_explicit_coupled_mixed_request(
    mixed_request: Any, tmp_path: Path
) -> None:
    import opensim as osim
    from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
        DeclaredColdStart,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner import (
        prepare_native_moco,
        solve_native_moco,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_replay import (
        export_native_moco_bundle,
        replay_native_moco_bundle,
        score_native_moco_replay,
    )

    model = osim.Model(str(mixed_request.model_path))
    coupler = osim.CoordinateCouplerConstraint()
    coupler.setName("coupler")
    names = osim.ArrayStr()
    names.append("pelvis_tx")
    coupler.setIndependentCoordinateNames(names)
    coupler.setDependentCoordinateName("arm_flex_r")
    coupler.setFunction(osim.LinearFunction(1.0, -0.31))
    model.addConstraint(coupler)
    model.finalizeConnections()
    model.initSystem()
    model.printToXML(str(mixed_request.model_path))
    digest = hashlib.sha256(mixed_request.model_path.read_bytes()).hexdigest()
    declaration = DeclaredColdStart(
        mixed_request.model_path,
        digest,
        mixed_request.bindings.initial_state,
        0.0,
        {},
        {
            path: mixed_request.bindings.state_bounds[path + "/value"]
            for path in ("/jointset/slider/pelvis_tx", "/jointset/pin/arm_flex_r")
        },
        {"/constraintset/coupler": True},
        1e-9,
    )
    request = replace(
        mixed_request,
        model_sha256=digest,
        config=replace(mixed_request.config, marker_weight=1e4),
        constrained_cold_start=declaration,
        passive_policy=replace(
            mixed_request.passive_policy,
            loaded_model_sha256=hashlib.sha256(model.dump().encode()).hexdigest(),
        ),
    )
    directory = tmp_path / "prepared"
    prepared = prepare_native_moco(request, directory)
    assert prepared.ready_for_software_solve, prepared.blockers
    narrowed = replace(
        declaration, chart_bounds={"/jointset/slider/pelvis_tx": (0.30, 0.32)}
    )
    blocked = prepare_native_moco(
        replace(request, constrained_cold_start=narrowed), tmp_path / "outside-chart"
    )
    assert "native-moco-chart-bounds-unqualified" in blocked.blockers
    solved = solve_native_moco(request, prepared, directory)
    assert solved.success, solved.status
    exported = export_native_moco_bundle(request, prepared, solved, directory)
    native = replay_native_moco_bundle(
        exported,
        request.model_path,
        directory,
        constrained_cold_start=declaration,
        mixed_actuation=request.mixed_actuation,
    )
    assert len(native.constraint_audits) == len(native.times)
    assert np.any(np.abs(native.actuations[:, -2:]) > 1e-6)
    assert (
        abs(
            native.states[
                -1, native.state_names.index("/jointset/slider/pelvis_tx/value")
            ]
            - 0.31
        )
        > 1e-4
    )
    score = score_native_moco_replay(
        request, prepared, solved, exported, native, directory
    )
    assert score.marker_rmse_m < 2e-4
    evidence = np.load(directory / "native_replay.npz", allow_pickle=False)
    assert evidence["constraint_observation_sha256"].shape == native.times.shape
    assert np.max(evidence["constraint_position_error_max"]) <= 1e-9
