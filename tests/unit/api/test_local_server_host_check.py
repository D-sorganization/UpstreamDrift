"""The local desktop server only answers requests addressed to an allowed host.

The hosted app in ``src/api/server.py`` already validates the Host header.
The local app is built separately and applies the same policy here.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")
local_server = pytest.importorskip("src.api.local_server")

from fastapi.testclient import TestClient  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[object]:
    monkeypatch.setenv("GOLF_UI_DIST", str(tmp_path / "ui" / "dist"))
    monkeypatch.delenv("ALLOWED_HOSTS", raising=False)
    yield local_server.create_local_app()


@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.1:8000", "http://localhost:8000", "http://testserver"],
)
def test_allowed_host_is_served(app: object, base_url: str) -> None:
    with TestClient(app, base_url=base_url) as client:
        assert client.get("/api/health").status_code == 200


@pytest.mark.parametrize(
    "host",
    ["example.invalid", "example.invalid:8000", "127.0.0.1.example.invalid"],
)
def test_other_host_is_refused(app: object, host: str) -> None:
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        response = client.get("/api/health", headers={"host": host})
    assert response.status_code == 400


def test_refusal_applies_to_every_route(app: object) -> None:
    """The check is app-wide middleware, not a per-route dependency."""
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        for path in ("/api/health", "/api/v1/health", "/api/launcher/manifest"):
            response = client.get(path, headers={"host": "example.invalid"})
            assert response.status_code == 400, path


def test_local_app_uses_the_shared_allowed_host_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """One policy for both apps: the ALLOWED_HOSTS setting is honoured."""
    monkeypatch.setenv("GOLF_UI_DIST", str(tmp_path / "ui" / "dist"))
    monkeypatch.setenv("ALLOWED_HOSTS", "desk.example.invalid")
    configured = local_server.create_local_app()
    with TestClient(configured, base_url="http://desk.example.invalid") as client:
        assert client.get("/api/health").status_code == 200
    with TestClient(configured, base_url="http://127.0.0.1:8000") as client:
        assert client.get("/api/health").status_code == 400
