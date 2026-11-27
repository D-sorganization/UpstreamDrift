"""Golf-shoe look (#15280): light upper, dark sole, clear of the trousers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance.body_meshes import body_meshes

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
DOC = document_from_dict({"schema_version": "appearance-v1"})
MIN_SHOE_TROUSER_CONTRAST = 3.0  # WCAG-style luminance ratio
MIN_UPPER_SOLE_CONTRAST = 3.0


def _luminance(name: str) -> float:
    rgb = library.MATERIALS[name].rgba[:3]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_default_shoe_differs_from_trousers() -> None:
    outfit = library.CLOTHING[DOC.clothing]
    shoe, pants = outfit["foot"], outfit[library.PANTS_PART]
    assert _contrast(shoe, pants) >= MIN_SHOE_TROUSER_CONTRAST


def test_sole_material_is_darker_than_the_upper() -> None:
    upper = library.CLOTHING[DOC.clothing]["foot"]
    assert _contrast(upper, library.SOLE_MATERIAL) >= MIN_UPPER_SOLE_CONTRAST


def test_every_foot_gets_a_sole_shell_in_the_sole_material() -> None:
    meshes = body_meshes(json.loads(SPEC.read_text(encoding="utf-8")), DOC)
    soles = [m for m in meshes if m.label.startswith("sole")]
    assert {m.body for m in soles} >= {"calcn_r", "calcn_l", "toes_r", "toes_l"}
    assert {m.material_name for m in soles} == {library.SOLE_MATERIAL}


def test_leg_tapers_meet_at_the_knee_and_ankle() -> None:
    # The knee join is measured on the drawn sections in test_leg_shape (#15299).
    assert library.limb_distal_scale("shin") < library.limb_distal_scale("thigh") < 1
    assert library.limb_distal_scale("forearm") == library.distal_scale("forearm")
