"""Coverage for src/shared/python/cors.py."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.shared.python.cors import DEFAULT_ORIGINS, add_cors_middleware  # noqa: E402


def _make_app(**kwargs: object) -> fastapi.FastAPI:
    app = fastapi.FastAPI()
    add_cors_middleware(app, **kwargs)  # type: ignore[arg-type]

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"ok": "yes"}

    return app


def test_default_origins_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    app = _make_app()
    client = TestClient(app)
    r = client.options(
        "/ping",
        headers={
            "Origin": DEFAULT_ORIGINS[0],
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == DEFAULT_ORIGINS[0]


def test_explicit_origins_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    app = _make_app(origins=["http://example.com"])
    client = TestClient(app)
    r = client.options(
        "/ping",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-origin") == "http://example.com"


def test_env_var_takes_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://from-env.com, http://other.com ,")
    # origins arg should be ignored when env var is set
    app = _make_app(origins=["http://ignored.com"])
    client = TestClient(app)
    r = client.options(
        "/ping",
        headers={
            "Origin": "http://from-env.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-origin") == "http://from-env.com"


def test_disallowed_origin_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    app = _make_app(origins=["http://allowed.com"])
    client = TestClient(app)
    r = client.options(
        "/ping",
        headers={
            "Origin": "http://blocked.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # CORS middleware does not echo origin back when origin isn't allowed
    assert r.headers.get("access-control-allow-origin") != "http://blocked.com"


def test_default_origins_contains_localhost() -> None:
    assert any("localhost" in o for o in DEFAULT_ORIGINS)
