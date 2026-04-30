"""Unit tests for ``src.api.versioning.make_versioned_router``.

Covers the URI-path versioning policy described in
``docs/adr/api-versioning.md``:

- Version-segment validation (``^v\\d+$``).
- Optional RFC 9745 ``Deprecation`` header injection.
- Optional RFC 8594 ``Sunset`` header injection.
- Mounting on a FastAPI app and exercising a registered endpoint.

Tests gracefully skip the FastAPI-dependent assertions when FastAPI is not
installed; the pure-Python validation paths still execute via the importable
helpers.
"""

from __future__ import annotations

import pytest


def test_module_imports_without_errors() -> None:
    """The versioning module must import even without FastAPI installed."""
    from src.api import versioning

    assert hasattr(versioning, "make_versioned_router")
    assert hasattr(versioning, "get_app_version")


def test_invalid_version_strings_raise_value_error() -> None:
    """Strings that do not match ``^v\\d+$`` are rejected with ValueError."""
    from src.api.versioning import make_versioned_router

    for bad in ("1", "version1", "", "V1", "v", "v1.0", "v1a", "/v1"):
        with pytest.raises(ValueError):
            make_versioned_router(bad)


def test_non_string_version_raises_type_error() -> None:
    """Non-string version values raise TypeError, not ValueError."""
    from src.api.versioning import make_versioned_router

    for bad in (None, 1, 1.0, ["v1"], {"v": 1}):
        with pytest.raises(TypeError):
            make_versioned_router(bad)  # type: ignore[arg-type]


def test_empty_sunset_string_raises_value_error() -> None:
    """An empty/whitespace sunset value is a programming error."""
    pytest.importorskip("fastapi")
    from src.api.versioning import make_versioned_router

    with pytest.raises(ValueError):
        make_versioned_router("v1", deprecated=True, sunset="   ")


def test_make_versioned_router_v1_prefix() -> None:
    """``make_versioned_router('v1')`` returns an APIRouter prefixed /v1."""
    pytest.importorskip("fastapi")
    from fastapi import APIRouter
    from src.api.versioning import make_versioned_router

    router = make_versioned_router("v1")
    assert isinstance(router, APIRouter)
    assert router.prefix == "/v1"
    assert router.deprecated is False


def test_make_versioned_router_v2_deprecated_with_sunset() -> None:
    """Deprecated router records the flag and registers a dependency hook."""
    pytest.importorskip("fastapi")
    from fastapi import APIRouter
    from src.api.versioning import make_versioned_router

    router = make_versioned_router(
        "v2",
        deprecated=True,
        sunset="Wed, 11 Nov 2026 23:59:59 GMT",
    )
    assert isinstance(router, APIRouter)
    assert router.prefix == "/v2"
    assert router.deprecated is True
    # Exactly one router-level dependency: the deprecation header injector.
    assert len(router.dependencies) == 1


def test_mounted_router_returns_deprecation_headers() -> None:
    """Mounting a deprecated router yields ``Deprecation``/``Sunset``."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.versioning import make_versioned_router

    sunset = "Wed, 11 Nov 2026 23:59:59 GMT"
    router = make_versioned_router("v2", deprecated=True, sunset=sunset)

    @router.get("/ping")
    def _ping() -> dict[str, str]:
        return {"status": "ok"}

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)
    response = client.get("/v2/ping")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers.get("Deprecation") == "true"
    assert response.headers.get("Sunset") == sunset


def test_mounted_non_deprecated_router_omits_headers() -> None:
    """A non-deprecated router must not advertise deprecation/sunset."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.versioning import make_versioned_router

    router = make_versioned_router("v1")

    @router.get("/ping")
    def _ping() -> dict[str, str]:
        return {"status": "ok"}

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)
    response = client.get("/v1/ping")

    assert response.status_code == 200
    assert "Deprecation" not in response.headers
    assert "Sunset" not in response.headers
