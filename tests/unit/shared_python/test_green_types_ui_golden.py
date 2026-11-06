"""``ui/src/utils/greenTypes.golden.json`` matches the shared core (#14464).

``greenTypes.test.ts`` holds the React side to the same file.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.api.routes.putting_green import GreenShapeRequest, list_green_presets
from src.shared.python.putting_dynamics.green_spec import GREEN_PRESETS, green_preset

pytestmark = pytest.mark.unit

_GOLDEN = Path(__file__).resolve().parents[3] / "ui/src/utils/greenTypes.golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def test_types_are_the_desktop_green_types(golden: dict) -> None:
    gui = pytest.importorskip("src.tools.putting_green_gui._scene_builder")
    assert golden["types"] == [
        {"key": key, "label": gui.green_label(key)} for key in (None, *GREEN_PRESETS)
    ]


def test_presets_are_the_green_presets_route(golden: dict) -> None:
    served = asyncio.run(list_green_presets.__wrapped__())
    assert golden["presets"] == {k: v.model_dump() for k, v in served.items()}


@pytest.mark.parametrize("key", sorted(GREEN_PRESETS))
def test_a_preset_request_rolls_the_core_preset(golden: dict, key: str) -> None:
    request = GreenShapeRequest(**{**golden["presets"][key], "preset": key})
    assert request.spec() == green_preset(key)
