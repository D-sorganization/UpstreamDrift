"""CMB-12 (#11663): the tutorial's glTF step matches what the code produces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.humanoid_character_builder.spec_export import (
    compile_character,
    export_character,
)
from src.shared.python.model_appearance.skinned_body import (
    build_skinned_body,
    skinned_body_gltf,
    spec_motion_bone_poses,
)

pytestmark = pytest.mark.unit

TUTORIAL = (
    Path(__file__).resolve().parents[3]
    / "docs/user_guide/character_builder_tutorial.md"
)

PRESET_ID = "tour_average_male"
OVERRIDES = {"mass_kg": 90.0}
TURN_FRAMES = [{"TorsoInput": 0.8 * i / 30} for i in range(31)]


@pytest.fixture(scope="module")
def compiled():
    return compile_character(PRESET_ID, OVERRIDES)


@pytest.fixture(scope="module")
def tutorial_text() -> str:
    return TUTORIAL.read_text(encoding="utf-8")


def test_static_export_matches_tutorial(compiled, tutorial_text: str) -> None:
    text, media_type, filename = export_character(compiled, "gltf")
    doc = json.loads(text)
    claims = {
        filename: "tour_average_male_2bba2beb.gltf",
        str(len(text)): "334632",
        media_type: "model/gltf+json",
        str(len(doc["skins"][0]["joints"])): "24",
        str(doc["accessors"][0]["count"]): "3416",
        doc["extras"]["spec_sha256"][:8]: "2bba2beb",
    }
    for measured, documented in claims.items():
        assert measured == documented
        assert documented in tutorial_text
    assert "animations" not in doc or not doc["animations"]


def test_static_export_is_deterministic(compiled) -> None:
    again = compile_character(PRESET_ID, OVERRIDES)
    assert export_character(compiled, "gltf")[0] == export_character(again, "gltf")[0]


def test_turn_animation_matches_tutorial(compiled, tutorial_text: str) -> None:
    body = build_skinned_body(compiled.document, spec_sha256=compiled.sha256)
    motion = spec_motion_bone_poses(body, compiled.document, TURN_FRAMES)
    doc = skinned_body_gltf(body, motion, fps=60.0)
    (anim,) = doc["animations"]
    times = doc["accessors"][anim["samplers"][0]["input"]]
    assert motion.shape == (31, 24, 4, 4)
    assert len(anim["channels"]) == 48
    assert times["count"] == 31 and times["max"] == [0.5]
    assert {s["interpolation"] for s in anim["samplers"]} == {"LINEAR"}
    for documented in ("(31, 24, 4, 4)", "48 channels", "0.5 s", "LINEAR"):
        assert documented in tutorial_text
    assert '{"TorsoInput": 0.8 * i / 30} for i in range(31)' in tutorial_text
