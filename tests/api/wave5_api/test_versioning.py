"""Tests for src/api/versioning.py."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import versioning

pytestmark = pytest.mark.unit


def test_get_app_version_returns_string() -> None:
    versioning.get_app_version.cache_clear()
    v = versioning.get_app_version()
    assert isinstance(v, str)
    assert v  # non-empty
    versioning.get_app_version.cache_clear()


def test_get_app_version_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> str:
        from importlib.metadata import PackageNotFoundError

        raise PackageNotFoundError

    monkeypatch.setattr(versioning, "version", _raise)
    monkeypatch.setattr(
        versioning.Path, "open", lambda self, *a, **kw: (_ for _ in ()).throw(OSError())
    )
    versioning.get_app_version.cache_clear()
    assert versioning.get_app_version() == "0.0.0"
    versioning.get_app_version.cache_clear()


def test_make_versioned_router_prefix() -> None:
    r = versioning.make_versioned_router("v1")
    assert r.prefix == "/v1"
    assert r.deprecated is False


def test_make_versioned_router_v2() -> None:
    r = versioning.make_versioned_router("v2")
    assert r.prefix == "/v2"


def test_make_versioned_router_invalid_value() -> None:
    with pytest.raises(ValueError):
        versioning.make_versioned_router("1")
    with pytest.raises(ValueError):
        versioning.make_versioned_router("vX")
    with pytest.raises(ValueError):
        versioning.make_versioned_router("v")


def test_make_versioned_router_invalid_type() -> None:
    with pytest.raises(TypeError):
        versioning.make_versioned_router(1)  # type: ignore[arg-type]


def test_make_versioned_router_invalid_sunset_type() -> None:
    with pytest.raises(TypeError):
        versioning.make_versioned_router("v1", deprecated=True, sunset=123)  # type: ignore[arg-type]


def test_make_versioned_router_empty_sunset() -> None:
    with pytest.raises(ValueError):
        versioning.make_versioned_router("v1", deprecated=True, sunset="   ")


def test_deprecated_router_injects_headers() -> None:
    app = FastAPI()
    r = versioning.make_versioned_router(
        "v1", deprecated=True, sunset="Wed, 11 Nov 2026 23:59:59 GMT"
    )

    @r.get("/ping")
    def _ping() -> dict[str, str]:
        return {"ok": "yes"}

    app.include_router(r)
    client = TestClient(app)
    resp = client.get("/v1/ping")
    assert resp.status_code == 200
    assert resp.headers.get("Deprecation") == "true"
    assert resp.headers.get("Sunset") == "Wed, 11 Nov 2026 23:59:59 GMT"


def test_non_deprecated_router_omits_headers() -> None:
    app = FastAPI()
    r = versioning.make_versioned_router("v1")

    @r.get("/ping")
    def _ping() -> dict[str, str]:
        return {"ok": "yes"}

    app.include_router(r)
    resp = TestClient(app).get("/v1/ping")
    assert resp.status_code == 200
    assert "Deprecation" not in resp.headers
    assert "Sunset" not in resp.headers
