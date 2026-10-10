"""CMB-7b (#11658): character appearance API routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.character_appearance import router
from src.api.routes.character_builder import router as character_builder_router
from src.shared.python.model_appearance import document_from_dict
from src.shared.python.model_appearance.library import (
    CLOTHING,
    CLUB_FINISHES,
    MATERIALS,
    SKIN_TONES,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.include_router(character_builder_router)
    return TestClient(app)


def test_library_lists_every_named_choice(client: TestClient) -> None:
    body = client.get("/character-builder/appearance/library").json()
    assert set(body["skin_tones"]) == set(SKIN_TONES)
    assert set(body["clothing"]) == set(CLOTHING)
    assert set(body["club_finishes"]) == set(CLUB_FINISHES)
    assert set(body["headwear"]) == {"none", "hair", "cap"}
    assert set(body["materials"]) == set(MATERIALS)
    for material in body["materials"].values():
        assert {"base_color", "roughness", "metallic"} <= set(material)
    # Every referenced name (skin tones, clothing parts, finishes, headwear
    # defaults, ground materials) must resolve to a listed material.
    referenced = (
        set(body["skin_tones"])
        | {m for parts in body["clothing"].values() for m in parts.values()}
        | set(body["club_finishes"])
        | set(body["headwear_default_material"].values())
        | set(body["ground_materials"])
    )
    assert referenced <= set(body["materials"])


def test_build_with_defaults_validates(client: TestClient) -> None:
    r = client.post("/character-builder/appearance", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["spec_sha256"] is None
    doc = document_from_dict(body["appearance"])
    assert doc.skin_tone == "skin_medium"


def test_build_applies_picks(client: TestClient) -> None:
    r = client.post(
        "/character-builder/appearance",
        json={
            "skin_tone": "skin_dark",
            "clothing": "golf_polo_trousers",
            "club_finish": "chrome",
            "headwear": "cap",
            "headwear_material": "cap_white",
            "ground_material": "studio_floor",
            "name": "my_character",
        },
    )
    assert r.status_code == 200
    doc = document_from_dict(r.json()["appearance"])
    assert doc.skin_tone == "skin_dark"
    assert doc.clothing == "golf_polo_trousers"
    assert doc.club_finish == "chrome"
    assert doc.head.headwear == "cap"
    assert doc.head.headwear_material == "cap_white"
    assert doc.environment.ground_material == "studio_floor"
    assert doc.name == "my_character"


def test_unknown_skin_tone_is_422(client: TestClient) -> None:
    r = client.post("/character-builder/appearance", json={"skin_tone": "skin_green"})
    assert r.status_code == 422
    assert "skin_green" in r.json()["detail"]


def test_unknown_clothing_is_422(client: TestClient) -> None:
    r = client.post("/character-builder/appearance", json={"clothing": "space_suit"})
    assert r.status_code == 422


def test_unknown_headwear_material_is_422(client: TestClient) -> None:
    r = client.post(
        "/character-builder/appearance",
        json={"headwear": "cap", "headwear_material": "not_a_material"},
    )
    assert r.status_code == 422


def test_character_binding_matches_build_spec_sha(client: TestClient) -> None:
    payload = {"character": {"preset": "junior"}}
    build = client.post("/character-builder/build", json={"preset": "junior"})
    appearance = client.post("/character-builder/appearance", json=payload)
    assert appearance.status_code == 200
    assert appearance.json()["spec_sha256"] == build.json()["spec_sha256"]
    assert appearance.json()["appearance"]["spec_sha256"] == build.json()["spec_sha256"]


def test_character_binding_unknown_preset_is_404(client: TestClient) -> None:
    r = client.post(
        "/character-builder/appearance", json={"character": {"preset": "nobody"}}
    )
    assert r.status_code == 404


def test_extra_field_is_422(client: TestClient) -> None:
    r = client.post("/character-builder/appearance", json={"bogus": "value"})
    assert r.status_code == 422


@pytest.mark.parametrize("name", ['bad"name', "a/b", "line\r\nX-Injected: 1", ""])
def test_unsafe_name_is_422(client: TestClient, name: str) -> None:
    r = client.post("/character-builder/appearance/export", json={"name": name})
    assert r.status_code == 422


def test_export_returns_attachment_json(client: TestClient) -> None:
    r = client.post("/character-builder/appearance/export", json={"name": "export_me"})
    assert r.status_code == 200
    assert "application/json" in r.headers["content-type"]
    assert "attachment" in r.headers["content-disposition"]
    assert "export_me" in r.headers["content-disposition"]
    assert ".appearance.json" in r.headers["content-disposition"]
    doc = document_from_dict(r.json())
    assert doc.name == "export_me"


def test_export_filename_includes_spec_hash_when_bound(client: TestClient) -> None:
    r = client.post(
        "/character-builder/appearance/export",
        json={"character": {"preset": "junior"}},
    )
    assert r.status_code == 200
    disposition = r.headers["content-disposition"]
    assert "junior_" in disposition
    doc = document_from_dict(r.json())
    assert doc.spec_sha256 is not None
    assert doc.spec_sha256[:8] in disposition
