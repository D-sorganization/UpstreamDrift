"""Native T01 muscle replay admission; synthetic fixtures are not swing evidence."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.helpers.opensim_support import require_opensim

pytest_plugins = ("tests.opensim.test_native_muscle_replay",)

pytestmark = pytest.mark.unit
NativeFixture = tuple[Path, dict[str, float]]


def _api() -> tuple[Callable[..., Any], Callable[..., Any]]:
    from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_bundle import (
        build_native_muscle_replay_bundle,
        replay_native_muscle_bundle,
    )

    return build_native_muscle_replay_bundle, replay_native_muscle_bundle


def _bundle(fixture: NativeFixture) -> Any:
    build, _ = _api()
    path, initial = fixture
    return build(path, initial, np.linspace(0, 0.04, 9), {"flexor": np.full(9, 0.4)})


def _add_visual_mesh(
    fixture: NativeFixture, monkeypatch: pytest.MonkeyPatch
) -> NativeFixture:
    """Attach one real native Mesh backed by a small source-local VTP file."""
    osim = require_opensim()

    path, initial = fixture
    monkeypatch.chdir(path.parent)
    mesh_path = path.parent / "visual.vtp"
    mesh_path.write_text(
        '<?xml version="1.0"?>'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">'
        '<PolyData><Piece NumberOfPoints="3" NumberOfVerts="0" NumberOfLines="0" '
        'NumberOfStrips="0" NumberOfPolys="1"><Points><DataArray type="Float32" '
        'NumberOfComponents="3" format="ascii">0 0 0 1 0 0 0 1 0</DataArray></Points>'
        '<Polys><DataArray type="Int32" Name="connectivity" format="ascii">0 1 2'
        '</DataArray><DataArray type="Int32" Name="offsets" format="ascii">3'
        "</DataArray></Polys></Piece></PolyData></VTKFile>",
        encoding="utf-8",
    )
    model = osim.Model(str(path))
    mesh = osim.Mesh("visual.vtp")
    mesh.setName("synthetic_visual")
    model.getBodySet().get("load").attachGeometry(mesh)
    model.finalizeConnections()
    model.printToXML(str(path))
    return path, initial


def _rebuild(bundle: Any, **changes: Any) -> Any:
    """Create valid wire integrity so admission tests reach native semantics."""
    from src.engines.native_replay_contracts import native_replay_contract_types

    contracts = native_replay_contract_types()
    history = changes.get("history", bundle.input_history)
    state = changes.get("state", bundle.initial_state)
    return contracts.build_experiment_replay_bundle(
        bundle.experiment_id,
        changes.get("model", bundle.model),
        changes.get("capabilities", bundle.capabilities),
        tuple((v.component_id, v.values) for v in state),
        history.channels,
        history.input_kind,
        history.interpolation,
        history.time_seconds,
        history.values,
        changes.get("policy", bundle.policy),
    )


def test_native_bundle_replays_complete_state_and_actual_excitation(
    muscle_fixture: NativeFixture,
) -> None:
    path, initial = muscle_fixture
    bundle = _bundle(muscle_fixture)
    assert not hasattr(bundle, "resources")
    _, replay = _api()
    first, second = replay(bundle, path), replay(bundle, path)
    np.testing.assert_array_equal(first.states[0], list(initial.values()))
    np.testing.assert_allclose(first.states, second.states, atol=1e-10, rtol=0)
    np.testing.assert_allclose(first.applied_excitations, 0.4, atol=1e-12, rtol=0)
    assert not first.states.flags.writeable
    assert bundle.input_history.input_kind.value == "muscle_excitation"
    assert bundle.input_history.interpolation.value == "linear"
    assert not bundle.policy.state_feedback_access
    assert not bundle.policy.state_reset_allowed


def test_native_modeling_options_are_explicit_initial_state(
    muscle_fixture: NativeFixture,
) -> None:
    bundle = _bundle(muscle_fixture)
    values = {v.component_id: v.values for v in bundle.initial_state}
    assert values["/forceset/flexor/native-options"] == (0.0, 1.0, 0.0, 0.0)


def test_native_schema_preserves_physical_units(muscle_fixture: NativeFixture) -> None:
    bundle = _bundle(muscle_fixture)
    units = {
        item.component_id: item.unit for item in bundle.model.state_schema.components
    }
    assert units["/jointset/slider/slide/value"] == "m"
    assert units["/jointset/slider/slide/speed"] == "m/s"
    assert units["/forceset/flexor/activation"] == "1"
    assert units["/forceset/flexor/fiber_length"] == "m"
    assert units["registered-discrete:/forceset/flexor/override_actuation"] == "N"


def test_missing_compliant_fiber_state_is_rejected(
    muscle_fixture: NativeFixture,
) -> None:
    path, initial = muscle_fixture
    initial = {k: v for k, v in initial.items() if not k.endswith("fiber_length")}
    build, _ = _api()
    with pytest.raises(ValueError, match="complete.*state"):
        build(path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])})


def test_modified_runtime_options_cannot_be_silently_defaulted(
    muscle_fixture: NativeFixture,
) -> None:
    bundle = _bundle(muscle_fixture)
    _, replay = _api()
    values = tuple(
        replace(v, values=(1.0, 1.0, 0.0, 0.0))
        if v.component_id.endswith("native-options")
        else v
        for v in bundle.initial_state
    )
    with pytest.raises(ValueError):
        replay(_rebuild(bundle, state=values), muscle_fixture[0])


def test_required_native_capability_cannot_be_replaced(
    muscle_fixture: NativeFixture,
) -> None:
    bundle = _bundle(muscle_fixture)
    other = replace(bundle.capabilities[0], capability_id="unrelated-capability")
    _, replay = _api()
    with pytest.raises(ValueError, match="capability|identity"):
        replay(_rebuild(bundle, capabilities=(other,)), muscle_fixture[0])


@pytest.mark.parametrize("field", ["provider", "interpolation", "integrator"])
def test_valid_wire_bundle_rejects_changed_execution_semantics(
    muscle_fixture: NativeFixture,
    field: str,
) -> None:
    bundle = _bundle(muscle_fixture)
    if field == "provider":
        changed = _rebuild(
            bundle, model=replace(bundle.model, provider_sha256="0" * 64)
        )
    elif field == "integrator":
        changed = _rebuild(
            bundle, policy=replace(bundle.policy, integration_method="other-integrator")
        )
    else:
        from src.engines.native_replay_contracts import native_replay_contract_types

        contracts = native_replay_contract_types()
        changed = _rebuild(
            bundle,
            history=replace(
                bundle.input_history,
                interpolation=contracts.InputInterpolation.ZERO_ORDER_HOLD,
            ),
        )
    _, replay = _api()
    with pytest.raises(ValueError, match="identity"):
        replay(changed, muscle_fixture[0])


def test_external_resource_reference_is_rejected_before_native_load(
    muscle_fixture: NativeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    osim = require_opensim()

    path, initial = muscle_fixture
    raw = path.read_bytes()
    changed = raw.replace(b"<BodySet ", b'<BodySet file="unqualified.xml" ')
    assert changed != raw
    path.write_bytes(changed)

    def forbidden_load(*args: Any) -> None:
        raise AssertionError("native loader must not see external source references")

    monkeypatch.setattr(osim, "Model", forbidden_load)
    build, _ = _api()
    with pytest.raises(ValueError, match="external|source"):
        build(path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])})


def test_source_bound_visual_mesh_has_an_exercised_native_consumer(
    muscle_fixture: NativeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    build, replay = _api()
    times = np.linspace(0, 0.02, 5)
    controls = {"flexor": np.full(5, 0.4)}
    baseline_bundle = build(muscle_fixture[0], muscle_fixture[1], times, controls)
    baseline = replay(baseline_bundle, muscle_fixture[0])
    path, initial = _add_visual_mesh(muscle_fixture, monkeypatch)
    bundle = build(path, initial, times, controls)
    assert bundle.__class__.__name__ == "NativeMuscleResourceEnvelope"
    assert len(bundle.resources) == 1
    result = replay(bundle, path)
    np.testing.assert_allclose(result.states, baseline.states, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        result.muscle_forces_n, baseline.muscle_forces_n, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(result.applied_excitations, 0.4, atol=1e-12, rtol=0)


def test_visual_resource_bundle_round_trip_and_changed_asset_rejection(
    muscle_fixture: NativeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_bundle import (
        dumps_native_muscle_resource_bundle,
        loads_native_muscle_resource_bundle,
    )

    path, initial = _add_visual_mesh(muscle_fixture, monkeypatch)
    build, replay = _api()
    bundle = build(
        path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.2, 0.2])}
    )
    decoded = loads_native_muscle_resource_bundle(
        dumps_native_muscle_resource_bundle(bundle)
    )
    assert decoded == bundle
    (path.parent / "visual.vtp").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="digest|closure|resource"):
        replay(decoded, path)


def test_changed_source_model_is_rejected(muscle_fixture: NativeFixture) -> None:
    bundle = _bundle(muscle_fixture)
    path, _ = muscle_fixture
    path.write_bytes(
        path.read_bytes().replace(b"synthetic_native_muscle_fixture", b"changed_model")
    )
    _, replay = _api()
    with pytest.raises(ValueError):
        replay(bundle, path)


def test_execution_uses_owned_frozen_source_bytes(
    muscle_fixture: NativeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching import muscle_replay

    bundle = _bundle(muscle_fixture)
    path, _ = muscle_fixture
    _, replay = _api()
    reference = replay(bundle, path)
    original = muscle_replay.replay_muscle_excitations

    def mutate_caller_source(frozen_path: Path, *args: Any, **kwargs: Any) -> Any:
        assert frozen_path != path
        path.write_bytes(
            path.read_bytes().replace(
                b"synthetic_native_muscle_fixture", b"changed_during_execution"
            )
        )
        return original(frozen_path, *args, **kwargs)

    monkeypatch.setattr(
        muscle_replay, "replay_muscle_excitations", mutate_caller_source
    )
    result = replay(bundle, path)
    assert result.model_sha256 == bundle.model.source_model_sha256
    np.testing.assert_array_equal(result.states, reference.states)


def test_unreviewed_native_component_is_rejected(muscle_fixture: NativeFixture) -> None:
    osim = require_opensim()

    path, initial = muscle_fixture
    model = osim.Model(str(path))
    model.addForce(osim.CoordinateLimitForce("slide", 1, 1, -1, 1, 0.1, 0.01))
    model.printToXML(str(path))
    build, _ = _api()
    with pytest.raises(ValueError, match="policy|component|force"):
        build(path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])})
