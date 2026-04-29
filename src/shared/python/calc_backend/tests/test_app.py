"""Tests for calc_backend app startup, health, and endpoint listing."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

# ──────────────────────────────────────────────────────────────────────────────
# App-level smoke tests
# ──────────────────────────────────────────────────────────────────────────────


class TestAppStartup:
    def test_health_check(self, client: TestClient) -> Any:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_list_endpoints(self, client: TestClient) -> Any:
        r = client.get("/api/calc/endpoints")
        assert r.status_code == 200
        body = r.json()
        assert "calculators" in body
        calc_list = body["calculators"]
        assert any("/api/calc/flare" in s for s in calc_list)
        assert any("/api/calc/pressure-drop" in s for s in calc_list)

    def test_openapi_schema_reachable(self, client: TestClient) -> Any:
        r = client.get("/openapi.json")
        assert r.status_code == 200
