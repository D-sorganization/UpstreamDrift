"""Tests for src/shared/python/cors.py — FastAPI CORS factory."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402

from cors import DEFAULT_ORIGINS, add_cors_middleware  # noqa: E402


def _origins_from(app: FastAPI) -> list[str]:
    """Pull the resolved allow_origins from the most recently added middleware."""
    # FastAPI stores middleware specs in app.user_middleware.
    mw = app.user_middleware[-1]
    return list(mw.kwargs["allow_origins"])


class TestAddCorsMiddleware:
    def test_default_origins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        app = FastAPI()
        add_cors_middleware(app)
        assert _origins_from(app) == DEFAULT_ORIGINS

    def test_explicit_origins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        app = FastAPI()
        add_cors_middleware(app, origins=["https://example.com"])
        assert _origins_from(app) == ["https://example.com"]

    def test_env_overrides_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com")
        app = FastAPI()
        add_cors_middleware(app, origins=["https://ignored.com"])
        assert _origins_from(app) == ["https://a.com", "https://b.com"]

    def test_env_strips_empty_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", " ,, https://c.com , ")
        app = FastAPI()
        add_cors_middleware(app)
        assert _origins_from(app) == ["https://c.com"]

    def test_default_methods_and_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        app = FastAPI()
        add_cors_middleware(app)
        mw = app.user_middleware[-1]
        assert mw.kwargs["allow_methods"] == ["*"]
        assert mw.kwargs["allow_headers"] == ["*"]
        assert mw.kwargs["allow_credentials"] is True

    def test_custom_methods_and_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        app = FastAPI()
        add_cors_middleware(
            app,
            allow_methods=["GET", "POST"],
            allow_headers=["X-Custom"],
            allow_credentials=False,
        )
        mw = app.user_middleware[-1]
        assert mw.kwargs["allow_methods"] == ["GET", "POST"]
        assert mw.kwargs["allow_headers"] == ["X-Custom"]
        assert mw.kwargs["allow_credentials"] is False

    def test_default_origins_module_constant(self) -> None:
        # Spot-check default origins look like localhost dev URLs.
        for origin in DEFAULT_ORIGINS:
            assert "localhost" in origin or "127.0.0.1" in origin
