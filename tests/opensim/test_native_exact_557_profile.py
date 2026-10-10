"""Exact-source diagnostic admission keeps the broad source out of v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_exact_557_profile import (
    EXACT_SOURCE_SHA256,
    inspect_visual_references,
    validate_exact_source_bytes,
)

pytestmark = pytest.mark.unit
_PREPARATION_SHA256 = "21f1095267f1b0d476c39df655118f134da2e7292aada7a4b9d0c45e9b9eca9b"


def _installed_source() -> tuple[Path, Path]:
    source = os.environ.get("OPEN_SIM_EXACT_557_SOURCE")
    preparation = os.environ.get("OPEN_SIM_EXACT_557_PREPARATION")
    if source is None or preparation is None:
        pytest.skip("external exact model and prepared-state receipt are unavailable")
    return Path(source), Path(preparation)


def _native_declaration() -> tuple[
    Any, np.ndarray[Any, Any], dict[str, np.ndarray[Any, Any]]
]:
    import opensim as osim

    from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
        DeclaredColdStart,
    )

    source, preparation = _installed_source()
    assert hashlib.sha256(preparation.read_bytes()).hexdigest() == _PREPARATION_SHA256
    model = osim.Model(str(source))
    state = model.initSystem()
    prepared = json.loads(preparation.read_text(encoding="utf-8"))["named_state"]
    names = model.getStateVariableNames()
    complete = {names.get(i): prepared[names.get(i)] for i in range(names.getSize())}
    assert len(complete) == 1226
    coordinates = model.getCoordinateSet()
    bounds = {}
    adjusted = []
    for index in range(coordinates.getSize()):
        coordinate = coordinates.get(index)
        if not coordinate.getDefaultClamped():
            continue
        path = coordinate.getAbsolutePathString()
        lower, upper = coordinate.getRangeMin(), coordinate.getRangeMax()
        bounds[path] = (lower, upper)
        key = path + "/value"
        if abs(complete[key] - lower) < 1e-9:
            complete[key] = lower + 0.001 * (upper - lower)
            adjusted.append(path)
        elif abs(complete[key] - upper) < 1e-9:
            complete[key] = upper - 0.001 * (upper - lower)
            adjusted.append(path)
    assert len(bounds) == 40 and len(adjusted) == 6
    constraints = model.getConstraintSet()
    enforcement = {
        constraints.get(i).getAbsolutePathString(): bool(
            constraints.get(i).isEnforced(state)
        )
        for i in range(constraints.getSize())
    }
    declaration = DeclaredColdStart(
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        complete,
        0.0,
        {},
        bounds,
        enforcement,
        1e-8,
    )
    grid = np.array([0.0, 0.0001, 0.0002])
    muscles = model.getMuscles()
    controls = {
        muscles.get(i).getName(): np.full(3, 0.05) for i in range(muscles.getSize())
    }
    return declaration, grid, controls


def test_only_attached_visual_mesh_files_are_exempt_from_source_closure() -> None:
    visual = b"""<OpenSimDocument><Model><Body><attached_geometry>
    <Mesh><mesh_file>body.vtp</mesh_file></Mesh>
    </attached_geometry></Body></Model></OpenSimDocument>"""
    assert inspect_visual_references(visual) == ("body.vtp",)
    dynamic = visual.replace(b"<Mesh>", b"<ExternalForce><Mesh>").replace(
        b"</Mesh>", b"</Mesh></ExternalForce>"
    )
    with pytest.raises(ValueError, match="visual|resource"):
        inspect_visual_references(dynamic)
    with pytest.raises(ValueError, match="visual|resource"):
        inspect_visual_references(visual.replace(b"mesh_file", b"data_file"))


def test_exact_source_rejects_rehashed_semantic_mutation() -> None:
    source, _ = _installed_source()
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXACT_SOURCE_SHA256
    assert len(inspect_visual_references(raw)) == 106
    validate_exact_source_bytes(raw)
    with pytest.raises(ValueError, match="exact source"):
        validate_exact_source_bytes(raw.replace(b"<gravity>", b"<!--x--><gravity>", 1))


def test_exact_557_saved_excitation_replays_on_two_fresh_native_models() -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )

    declaration, grid, controls = _native_declaration()
    with pytest.raises(ValueError, match="external source resources"):
        build_constrained_muscle_bundle(declaration, grid, controls)
    bundle = build_constrained_muscle_bundle(
        declaration, grid, controls, exact_557=True
    )
    first = replay_constrained_muscle_bundle(bundle, declaration, exact_557=True)
    second = replay_constrained_muscle_bundle(bundle, declaration, exact_557=True)
    assert bundle.model.variant_id == "exact-buet-hamner-557-muscle-diagnostic"
    assert len(first.muscle_names) == 557 and first.states.shape == (3, 1226)
    assert len(first.constraint_audits) == 3
    np.testing.assert_allclose(first.states, second.states, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        first.applied_excitations, second.applied_excitations, rtol=0, atol=1e-12
    )


def test_exact_profile_rejects_rehashed_model_and_wrong_chart(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
    )

    declaration, grid, controls = _native_declaration()
    mutant = tmp_path / "mutated.osim"
    mutant.write_bytes(declaration.model_path.read_bytes() + b"\n<!-- changed -->")
    changed = replace(
        declaration,
        model_path=mutant,
        source_sha256=hashlib.sha256(mutant.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="exact source"):
        build_constrained_muscle_bundle(changed, grid, controls, exact_557=True)
    missing_chart = dict(declaration.chart_bounds)
    missing_chart.pop("/jointset/sc3/SC_z")
    with pytest.raises(ValueError, match="clamped coordinate"):
        build_constrained_muscle_bundle(
            replace(declaration, chart_bounds=missing_chart),
            grid,
            controls,
            exact_557=True,
        )


def test_changed_future_excitation_preserves_prefix_then_changes_native_activation() -> (
    None
):
    from src.engines.physics_engines.opensim.python.tour_matching.native_constrained_muscle import (
        build_constrained_muscle_bundle,
        replay_constrained_muscle_bundle,
    )

    declaration, grid, controls = _native_declaration()
    first_name = next(iter(controls))
    changed = {name: values.copy() for name, values in controls.items()}
    changed[first_name][-1] = 0.15
    original_bundle = build_constrained_muscle_bundle(
        declaration, grid, controls, exact_557=True
    )
    changed_bundle = build_constrained_muscle_bundle(
        declaration, grid, changed, exact_557=True
    )
    assert original_bundle.applied_input_sha256 != changed_bundle.applied_input_sha256
    original = replay_constrained_muscle_bundle(
        original_bundle, declaration, exact_557=True
    )
    future = replay_constrained_muscle_bundle(
        changed_bundle, declaration, exact_557=True
    )
    np.testing.assert_allclose(
        original.states[:2], future.states[:2], rtol=0, atol=1e-10
    )
    activation = original.state_names.index(f"/forceset/{first_name}/activation")
    assert abs(original.states[-1, activation] - future.states[-1, activation]) > 1e-8
