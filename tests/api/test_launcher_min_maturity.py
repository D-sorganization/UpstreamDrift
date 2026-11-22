"""``GET /api/launcher/tiles?min_maturity=`` filters via the shared helper (#15150)."""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
    from src.api.server import app
except ImportError:
    pytest.skip("API server deps not available", allow_module_level=True)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_default_lists_every_visible_tile(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MIN_MATURITY", raising=False)
    everything = client.get("/api/launcher/tiles?min_maturity=deprecated").json()
    assert client.get("/api/launcher/tiles").json() == everything


def test_min_maturity_filters(client: TestClient) -> None:
    allowed = {"beta", "stable"}
    response = client.get("/api/launcher/tiles?min_maturity=beta")
    assert response.status_code == 200
    tiles = response.json()
    assert tiles and {t["maturity"] for t in tiles} <= allowed
    assert len(tiles) < len(client.get("/api/launcher/tiles").json())


def test_setting_supplies_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIN_MATURITY", "stable")
    tiles = client.get("/api/launcher/tiles").json()
    assert {t["maturity"] for t in tiles} == {"stable"}


def test_unknown_level_is_422(client: TestClient) -> None:
    response = client.get("/api/launcher/tiles?min_maturity=bogus")
    assert response.status_code == 422
    assert "min_maturity" in response.json()["detail"]
