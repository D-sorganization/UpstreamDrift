"""The default golfer comes from ONE serialized contract (HRG-B1, #13163)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.shared.python.model_appearance import defaults, library
from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    document_from_dict,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "src/config/appearance_defaults.json"


def _luma(rgb: tuple[float, ...]) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def test_default_golfer_is_light_skin_with_mid_brown_hair() -> None:
    doc = defaults.default_appearance()
    assert isinstance(doc, AppearanceDocument)
    assert doc.skin_tone == "skin_2"
    assert library.headwear_material_name(doc) == "hair_brown"
    r, g, b = library.MATERIALS["hair_brown"].base_color[:3]
    assert r > g > b, "brown hair must be warm, not neutral"
    assert 0.25 <= _luma((r, g, b)) <= 0.40, "mid brown must not render near-black"


def test_minimal_document_and_every_backend_share_the_contract() -> None:
    from src.tools.native_viewer_export.backends import _body, _meshcat_scene

    expected = defaults.default_appearance()
    assert document_from_dict({"schema_version": "appearance-v1"}) == expected
    assert _body.default_appearance() == expected
    assert _meshcat_scene.default_appearance() == expected


def test_palette_is_descriptive_ordered_and_never_fitzpatrick() -> None:
    raw = CONTRACT.read_text(encoding="utf-8")
    assert "fitzpatrick" not in raw.lower()
    palette = defaults.skin_palette()
    assert [s["id"] for s in palette] == [f"skin_{i}" for i in range(1, 9)]
    lumas = [_luma(tuple(s["srgb"])) for s in palette]
    assert lumas == sorted(lumas, reverse=True), "palette runs light to dark"
    for swatch in palette:
        assert swatch["description"].strip()
        assert swatch["hex"] == "#" + "".join(
            f"{round(c * 255):02x}" for c in swatch["srgb"]
        )
    assert tuple(s["id"] for s in palette) == library.SKIN_TONES


def test_legacy_skin_names_alias_the_new_swatches_exactly() -> None:
    for legacy, target in defaults.skin_aliases().items():
        assert library.MATERIALS[legacy].base_color == (
            library.MATERIALS[target].base_color
        )
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "skin_tone": "skin_dark"}
    )
    assert doc.skin_tone == "skin_dark"


def test_hair_palette_covers_the_requested_colours() -> None:
    ids = [h["id"] for h in defaults.hair_palette()]
    assert ids == [
        "hair_black",
        "hair_dark_brown",
        "hair_brown",
        "hair_light_brown",
        "hair_auburn",
        "hair_blond",
        "hair_grey",
    ]
    assert all(i in library.MATERIALS for i in ids)


def test_typescript_module_is_fresh_and_ui_has_no_hardcoded_defaults() -> None:
    from scripts import generate_appearance_defaults_ts as gen

    assert gen.TARGET.read_text(encoding="utf-8") == gen.render(), (
        "run: python3 -m scripts.generate_appearance_defaults_ts"
    )
    ts = gen.TARGET.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert f"'{contract['default']['skin_tone']}'" in ts
    for hardcoded in (
        ROOT / "ui/src/components/visualization/headModelGeometry.ts",
        ROOT / "ui/src/components/character/AppearancePanel.tsx",
        ROOT / "ui/src/components/character/appearanceTypes.ts",
    ):
        text = hardcoded.read_text(encoding="utf-8")
        assert not re.search(r"skin_medium|#cc99(66|75)|#332114", text), hardcoded.name
