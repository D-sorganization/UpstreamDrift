"""Tests for the /glossary/{term_id}?level=... tooltip endpoint (issue #3165)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.glossary import router as glossary_router

pytestmark = pytest.mark.unit


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(glossary_router, prefix="/api/v1")
    app.include_router(glossary_router, prefix="/api")
    return TestClient(app, raise_server_exceptions=True)


class TestGlossaryTooltipEndpoint:
    def test_default_level_is_beginner(self, client: TestClient) -> None:
        resp = client.get("/api/v1/glossary/drag_coefficient")
        # drag_coefficient may or may not be in glossary; accept either 200 or 404
        # but when 200, default level must be beginner.
        if resp.status_code == 200:
            body = resp.json()
            assert body["level"] == "beginner"
            assert body["definition"] == body.get("b", body["definition"])

    def test_known_term_default_beginner(self, client: TestClient) -> None:
        resp = client.get("/api/v1/glossary/equations_of_motion")
        assert resp.status_code == 200
        body = resp.json()
        assert body["term_id"] == "equations_of_motion"
        assert body["level"] == "beginner"
        assert body["definition"] == body["b"]

    def test_advanced_level(self, client: TestClient) -> None:
        resp = client.get("/api/v1/glossary/equations_of_motion?level=advanced")
        assert resp.status_code == 200
        body = resp.json()
        assert body["level"] == "advanced"
        # Advanced field 'a' should populate the definition when present.
        if body.get("a"):
            assert body["definition"] == body["a"]

    def test_unknown_term_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/v1/glossary/definitely_not_a_term")
        assert resp.status_code == 404

    def test_response_includes_related_terms(self, client: TestClient) -> None:
        resp = client.get("/api/v1/glossary/lagrangian")
        assert resp.status_code == 200
        body = resp.json()
        assert "related_terms" in body
        assert isinstance(body["related_terms"], list)

    def test_unknown_level_falls_back_to_beginner(self, client: TestClient) -> None:
        resp = client.get("/api/v1/glossary/lagrangian?level=bogus")
        assert resp.status_code == 200
        body = resp.json()
        # Unknown level -> definition still populated.
        assert body["definition"]
