"""Native T01 muscle replay admission; synthetic fixtures are not swing evidence."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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


def test_native_bundle_replays_complete_state_and_actual_excitation(
    muscle_fixture: NativeFixture,
) -> None:
    path, initial = muscle_fixture
    bundle = _bundle(muscle_fixture)
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
        replace(v, values=(1.0, 0.0, 1.0, 0.0, 0.0))
        if v.component_id.endswith("native-options")
        else v
        for v in bundle.initial_state
    )
    with pytest.raises(ValueError):
        replay(replace(bundle, initial_state=values), muscle_fixture[0])


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
    import opensim as osim

    path, initial = muscle_fixture
    model = osim.Model(str(path))
    model.addForce(osim.CoordinateLimitForce("slide", 1, 1, -1, 1, 0.1, 0.01))
    model.printToXML(str(path))
    build, _ = _api()
    with pytest.raises(ValueError, match="policy|component|force"):
        build(path, initial, np.array([0.0, 0.01]), {"flexor": np.array([0.1, 0.1])})
