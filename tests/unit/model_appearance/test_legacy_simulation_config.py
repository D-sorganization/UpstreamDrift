"""Legacy ``simulation_config.json`` colours: explicit one-time import + deprecation (HRG-B2)."""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.appearance_selection import (
    hex_to_rgb,
    overrides_from_legacy_config,
    resolve_appearance,
)
from src.shared.python.model_appearance.schema import document_from_dict

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
MUJOCO = ROOT / "src/engines/physics_engines/mujoco"


def test_legacy_colours_map_to_overrides() -> None:
    cfg = {
        "colors": {
            "skin": [0.8, 0.6, 0.4, 1.0],
            "shirt": [1.0, 0.5, 0.75, 1.0],
            "pants": [0.4, 0.2, 0.0, 1.0],
            "shoes": [0.1] * 4,
        }
    }
    o = overrides_from_legacy_config(cfg)
    assert (o.skin_tone, o.top_color, o.bottom_color) == (
        "#cc9966",
        "#ff80bf",
        "#663300",
    )
    assert o.hair is None and o.clothing is None  # legacy colours never carried hair
    doc = document_from_dict(resolve_appearance(o, None))
    mats = library.library_materials(doc)
    assert mats[doc.skin_tone].rgba[:3] == pytest.approx(hex_to_rgb("#cc9966"))


def test_config_without_colours_gives_empty_overrides() -> None:
    assert overrides_from_legacy_config({}).is_empty
    assert overrides_from_legacy_config({"colors": {}}).is_empty


def test_bad_legacy_colour_fails_closed() -> None:
    with pytest.raises(ValueError, match="skin"):
        overrides_from_legacy_config({"colors": {"skin": [2.0, 0.0, 0.0, 1.0]}})
    with pytest.raises(ValueError, match="skin"):
        overrides_from_legacy_config({"colors": {"skin": "tan"}})


@pytest.mark.parametrize(
    "path",
    [
        "simulation_config.json",
        "docker/simulation_config.json",
        "docker/src/simulation_config.json",
    ],
)
def test_every_committed_legacy_file_converts(path: str) -> None:
    cfg = json.loads((MUJOCO / path).read_text(encoding="utf-8"))
    document_from_dict(resolve_appearance(overrides_from_legacy_config(cfg), None))


def test_sim_reader_warns_that_colours_are_deprecated() -> None:
    notice = _load_notice()
    with pytest.warns(DeprecationWarning, match=r"appearance\.json"):
        assert notice.warn_if_legacy_colors({"colors": {"skin": [0.8, 0.6, 0.4, 1.0]}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not notice.warn_if_legacy_colors({"height_m": 1.8})
        assert not notice.warn_if_legacy_colors(None)


def _load_notice():
    spec = importlib.util.spec_from_file_location(
        "legacy_notice", MUJOCO / "docker/src/humanoid_golf/legacy_notice.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_legacy_readers_call_the_notice() -> None:
    sim = (MUJOCO / "docker/src/humanoid_golf/sim.py").read_text(encoding="utf-8")
    gui = (MUJOCO / "docker/gui/deepmind_control_suite_MuJoCo_GUI.py").read_text(
        encoding="utf-8"
    )
    assert "warn_if_legacy_colors(config)" in sim
    assert "DeprecationWarning" in gui
