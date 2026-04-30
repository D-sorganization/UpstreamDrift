"""Tests for the GET /glossary/{term_id} endpoint.

Verifies that:
- Known terms return 200 with the correct entry fields
- Unknown terms return 404
- The glossary route is registered in local_server.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.glossary import router as glossary_router

_LOCAL_SERVER_SRC = (
    Path(__file__).parent.parent / "src" / "api" / "local_server.py"
).read_text(encoding="utf-8")


@pytest.fixture()
def client() -> TestClient:
    """Return a TestClient for a minimal app with only the glossary router."""
    app = FastAPI()
    app.include_router(glossary_router, prefix="/api/v1", tags=["Glossary"])
    app.include_router(glossary_router, prefix="/api", tags=["Glossary"])
    return TestClient(app, raise_server_exceptions=True)


@pytest.mark.unit
class TestGlossaryEndpoint:
    """Tests for /api/v1/glossary/{term_id}."""

    def test_known_term_returns_200(self, client: TestClient) -> None:
        """A glossary term that exists returns HTTP 200."""
        resp = client.get("/api/v1/glossary/equations_of_motion")
        assert resp.status_code == 200

    def test_known_term_response_has_required_fields(self, client: TestClient) -> None:
        """Response body includes key, term, cat, and at least one definition field."""
        resp = client.get("/api/v1/glossary/equations_of_motion")
        data = resp.json()
        assert data["key"] == "equations_of_motion"
        assert "term" in data
        assert "cat" in data
        # At least one of b/i/a must be present
        assert any(field in data for field in ("b", "i", "a"))

    def test_unknown_term_returns_404(self, client: TestClient) -> None:
        """A term not in the glossary returns HTTP 404."""
        resp = client.get("/api/v1/glossary/nonexistent_term_xyz")
        assert resp.status_code == 404

    def test_404_response_has_detail(self, client: TestClient) -> None:
        """The 404 response contains a detail message."""
        resp = client.get("/api/v1/glossary/nonexistent_term_xyz")
        data = resp.json()
        assert "detail" in data

    def test_legacy_route_works(self, client: TestClient) -> None:
        """The legacy /api/glossary/{term_id} route also works."""
        resp = client.get("/api/glossary/lagrangian")
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "lagrangian"

    def test_multiple_known_terms(self, client: TestClient) -> None:
        """Several known glossary terms all return 200."""
        known_terms = ["lagrangian", "hamiltonian", "newton_euler"]
        for term in known_terms:
            resp = client.get(f"/api/v1/glossary/{term}")
            assert resp.status_code == 200, f"Expected 200 for term '{term}'"


@pytest.mark.unit
class TestGlossaryRouteRegistered:
    """Verify the glossary router is referenced in local_server.py source."""

    def test_local_server_imports_glossary(self) -> None:
        """local_server.py imports the glossary router module."""
        assert "glossary" in _LOCAL_SERVER_SRC, (
            "local_server.py must import the glossary route module"
        )

    def test_local_server_includes_glossary_router(self) -> None:
        """local_server.py calls app.include_router for glossary.router."""
        assert "glossary.router" in _LOCAL_SERVER_SRC, (
            "local_server.py must call app.include_router(glossary.router, ...) "
            "to expose the /glossary/{term_id} endpoint"
        )
