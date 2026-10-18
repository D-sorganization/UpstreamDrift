"""Persistence helper for the force-colour scale and overlay toggles (#13146)."""

from __future__ import annotations

import json
import logging

import pytest

from src.shared.python.body_part_viz.force_color_settings import (
    SETTINGS_APP,
    SETTINGS_ORG,
    dump_force_color_state,
    load_force_color_state,
    parse_force_color_state,
    save_force_color_state,
)
from src.shared.python.body_part_viz.force_colors import ForceColorScale

pytestmark = pytest.mark.unit


class FakeSettings:
    """Minimal QSettings stand-in (value/setValue)."""

    def __init__(self, data: dict | None = None) -> None:
        self.data = dict(data or {})

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, value):  # noqa: N802 - Qt naming
        self.data[key] = value


def test_org_and_app_are_upstreamdrift() -> None:
    assert SETTINGS_ORG == "UpstreamDrift"
    assert SETTINGS_APP == "UpstreamDrift"


def test_round_trip_scale_and_toggles() -> None:
    scale = ForceColorScale(
        enabled=True,
        tension_limit_n=500.0,
        opacity=0.4,
        blend_mode="tint",
        tension_color="#00ff00",
    )
    toggles = {"show_torques": True, "show_forces": False}
    store = FakeSettings()
    save_force_color_state(store, "mujoco", scale, toggles)
    loaded_scale, loaded_toggles = load_force_color_state(store, "mujoco")
    assert loaded_scale == scale
    assert loaded_toggles == toggles


def test_engines_are_isolated() -> None:
    store = FakeSettings()
    save_force_color_state(store, "mujoco", ForceColorScale(enabled=True), {})
    scale, _ = load_force_color_state(store, "drake")
    assert scale == ForceColorScale()


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[1, 2]",
        json.dumps({"scale": {"opacity": 7}}),
        json.dumps({"scale": {"bogus": 1}}),
        json.dumps({"scale": "x"}),
        json.dumps(42),
        None,
        b"\xff",
    ],
)
def test_corrupt_values_fall_back_to_defaults_with_warning(text, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        scale, toggles = parse_force_color_state(text)
    assert scale == ForceColorScale()
    assert toggles == {}
    if text is not None:
        assert any("force" in r.message.lower() for r in caplog.records)


def test_legacy_scale_without_opacity_keys_loads() -> None:
    legacy = json.dumps(
        {"scale": {"enabled": True, "tension_limit_n": 300.0}, "toggles": {}}
    )
    scale, _ = parse_force_color_state(legacy)
    assert scale.enabled and scale.tension_limit_n == 300.0
    assert scale.opacity == 1.0 and scale.blend_mode == "replace"


def test_non_bool_toggles_are_dropped() -> None:
    text = json.dumps({"scale": {}, "toggles": {"a": True, "b": "yes", "c": 1}})
    _, toggles = parse_force_color_state(text)
    assert toggles == {"a": True}


def test_dump_validates_inputs() -> None:
    with pytest.raises(TypeError):
        dump_force_color_state({"enabled": True}, {})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        dump_force_color_state(ForceColorScale(), {"a": "x"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        save_force_color_state(FakeSettings(), "", ForceColorScale(), {})
