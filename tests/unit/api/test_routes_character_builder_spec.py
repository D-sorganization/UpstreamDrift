"""CMB-3 (#11654): spec-native character builder endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.character_builder import router

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_presets_listing(client: TestClient) -> None:
    body = client.get("/character-builder/presets").json()
    ids = {p["id"] for p in body["presets"]}
    assert {"tour_average_male", "junior", "anthro_driver"} <= ids
    first = body["presets"][0]
    assert {"parameters", "limitations", "provenance"} <= set(first)


def test_build_from_preset(client: TestClient) -> None:
    r = client.post("/character-builder/build", json={"preset": "junior"})
    assert r.status_code == 200
    body = r.json()
    assert body["preset"] == "junior"
    assert body["parameters"]["stature_m"] == 1.52
    assert body["schema_version"] == "full-body-v1"
    assert len(body["spec_sha256"]) == 64
    assert 30.0 < body["total_mass_kg"] < 120.0


def test_build_is_deterministic(client: TestClient) -> None:
    payload = {"preset": "senior", "mass_kg": 85.0}
    a = client.post("/character-builder/build", json=payload).json()
    b = client.post("/character-builder/build", json=payload).json()
    assert a["spec_sha256"] == b["spec_sha256"]
    assert a["parameters"]["mass_kg"] == 85.0


def test_build_from_explicit_parameters(client: TestClient) -> None:
    r = client.post(
        "/character-builder/build",
        json={"stature_m": 1.9, "mass_kg": 95.0, "club": "iron7"},
    )
    assert r.status_code == 200
    assert r.json()["club"] == "iron7"
    assert r.json()["preset"] is None


def test_preview_lists_bodies_and_joints(client: TestClient) -> None:
    body = client.post("/character-builder/preview", json={"preset": "junior"}).json()
    names = {b["name"] for b in body["bodies"]}
    assert any(n.startswith("femur_") for n in names)
    assert len(body["joints"]) > 20
    assert body["stature_m"] == 1.52


def test_unknown_preset_is_404(client: TestClient) -> None:
    r = client.post("/character-builder/build", json={"preset": "nobody"})
    assert r.status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"stature_m": 0.5},
        {"mass_kg": 900.0},
        {"club": "putter"},
        {"wingspan": 2.0},
    ],
)
def test_invalid_requests_are_422(client: TestClient, payload: dict) -> None:
    assert client.post("/character-builder/build", json=payload).status_code == 422


def test_export_spec_round_trips(client: TestClient) -> None:
    r = client.post("/character-builder/export/spec", json={"preset": "junior"})
    assert r.status_code == 200
    assert "application/json" in r.headers["content-type"]
    assert "junior_" in r.headers["content-disposition"]
    assert json.loads(r.text)["subject"]["stature_m"] == 1.52


@pytest.mark.parametrize(
    "fmt,marker",
    [("urdf", "<robot"), ("mjcf", "<mujoco"), ("osim", "<OpenSimDocument")],
)
def test_export_engine_formats(client: TestClient, fmt: str, marker: str) -> None:
    r = client.post(f"/character-builder/export/{fmt}", json={"preset": "junior"})
    assert r.status_code == 200
    assert marker in r.text
    assert "attachment" in r.headers["content-disposition"]


def test_export_unknown_format_is_404(client: TestClient) -> None:
    r = client.post("/character-builder/export/fbx", json={"preset": "junior"})
    assert r.status_code == 404


def test_export_is_byte_identical_across_calls(client: TestClient) -> None:
    a = client.post("/character-builder/export/spec", json={"preset": "senior"})
    b = client.post("/character-builder/export/spec", json={"preset": "senior"})
    assert a.content == b.content


def test_missing_reference_assets_is_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.api.services import character_builder_service as service

    def boom(*_a: object, **_k: object) -> None:
        raise FileNotFoundError("native_geometry_spec_9967.json")

    monkeypatch.setattr(service, "compile_full_body_spec", boom)
    r = client.post("/character-builder/build", json={"preset": "junior"})
    assert r.status_code == 503
