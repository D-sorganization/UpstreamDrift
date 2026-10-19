"""POST /character-builder/grip: grip picks to a revision summary (HRG-GR4, #13205)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.character_appearance import router
from src.api.routes.character_builder import router as character_builder_router

pytestmark = pytest.mark.unit

FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[2] / "fixtures/grip/grip_roundtrip.json"
    ).read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.include_router(character_builder_router)
    return TestClient(app)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
def test_route_matches_the_shared_fixture(client: TestClient, case: dict) -> None:
    body = client.post("/character-builder/grip", json=case["picks"]).json()
    expected = case["expected"]
    assert body["physics_spec_sha256"] == expected["character_physics_spec_sha256"]
    assert body["refit_required"] == expected["refit_required"]
    if expected["grip"] is not None:
        assert body["grip"] == expected["grip"]
    parent = FIXTURE["parent"]["character_physics_spec_sha256"]
    assert body["parent_physics_spec_sha256"] == parent
    assert (body["physics_spec_sha256"] == parent) is (not expected["refit_required"])


def test_route_refuses_an_unknown_pick_and_an_impossible_one(
    client: TestClient,
) -> None:
    assert client.post("/character-builder/grip", json={"spin": 1}).status_code == 422
    r = client.post("/character-builder/grip", json={"lead_shaft_rotation": -60})
    # -60 deg is the lead alpha bound (lead V azimuth -40 deg, HRG-GS-G4); the
    # trail hand then sits outside the trail V offset range: still impossible.
    assert r.status_code == 422 and "trail_v_offset_deg" in r.json()["detail"]


def test_appearance_export_carries_only_picture_picks(client: TestClient) -> None:
    ok = client.post(
        "/character-builder/appearance",
        json={"grip_style": "interlock", "grip_thumb_offset_deg": 20},
    ).json()
    assert ok["appearance"]["grip"] == {
        "style": "interlock",
        "lead_thumb_offset_deg": 20.0,
    }
    # a physical value is not an appearance field: refused, never written
    bad = client.post("/character-builder/appearance", json={"hand_spacing": 0.1})
    assert bad.status_code == 422
