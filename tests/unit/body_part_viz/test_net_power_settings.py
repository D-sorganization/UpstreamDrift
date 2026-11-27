"""Net-power settings persistence (#15399)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.shared.python.body_part_viz.net_power_settings import (
    load_net_power_state,
    save_net_power_state,
)
from src.shared.python.body_part_viz.segment_net_power import (
    default_net_power_color_scale,
)

pytestmark = pytest.mark.unit


class _FakeStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def value(self, key: str, default=None):
        return self._data.get(key, default)

    def setValue(self, key: str, value) -> None:  # noqa: N802
        self._data[key] = value


def test_net_power_settings_round_trip():
    store = _FakeStore()
    scale = replace(default_net_power_color_scale(), enabled=True, opacity=0.5)
    save_net_power_state(
        store,
        "mujoco",
        scale,
        {"show_interface_power_glyphs": True},
    )
    loaded, toggles = load_net_power_state(store, "mujoco")
    assert loaded == scale
    assert toggles["show_interface_power_glyphs"] is True
