"""Tests for src.shared.python.cors (Issues #1949, #1744)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from src.shared.python.cors import DEFAULT_ORIGINS, add_cors_middleware


class TestDefaultOrigins:
    def test_contains_localhost(self) -> None:
        assert any("localhost" in o for o in DEFAULT_ORIGINS)

    def test_cors_all_are_strings(self) -> None:
        assert all(isinstance(o, str) for o in DEFAULT_ORIGINS)

    def test_cors_non_empty(self) -> None:
        assert len(DEFAULT_ORIGINS) > 0


class TestAddCorsMiddleware:
    def _make_app(self) -> MagicMock:
        app = MagicMock()
        return app

    def test_adds_middleware(self) -> None:
        app = self._make_app()
        add_cors_middleware(app)
        app.add_middleware.assert_called_once()

    def test_uses_default_origins_when_none_set(self) -> None:
        app = self._make_app()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORS_ORIGINS", None)
            add_cors_middleware(app)
        _, kwargs = app.add_middleware.call_args
        assert kwargs["allow_origins"] == DEFAULT_ORIGINS

    def test_env_var_overrides_defaults(self) -> None:
        app = self._make_app()
        custom = "https://example.com,https://app.example.com"
        with patch.dict(os.environ, {"CORS_ORIGINS": custom}):
            add_cors_middleware(app)
        _, kwargs = app.add_middleware.call_args
        assert kwargs["allow_origins"] == [
            "https://example.com",
            "https://app.example.com",
        ]

    def test_caller_origins_used_when_no_env(self) -> None:
        app = self._make_app()
        caller_origins = ["https://myapp.com"]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORS_ORIGINS", None)
            add_cors_middleware(app, origins=caller_origins)
        _, kwargs = app.add_middleware.call_args
        assert kwargs["allow_origins"] == caller_origins

    def test_env_overrides_caller_origins(self) -> None:
        app = self._make_app()
        with patch.dict(os.environ, {"CORS_ORIGINS": "https://env.example.com"}):
            add_cors_middleware(app, origins=["https://caller.example.com"])
        _, kwargs = app.add_middleware.call_args
        assert kwargs["allow_origins"] == ["https://env.example.com"]

    def test_default_allow_credentials(self) -> None:
        app = self._make_app()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORS_ORIGINS", None)
            add_cors_middleware(app)
        _, kwargs = app.add_middleware.call_args
        assert kwargs["allow_credentials"] is True

    def test_custom_allow_credentials(self) -> None:
        app = self._make_app()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORS_ORIGINS", None)
            add_cors_middleware(app, allow_credentials=False)
        _, kwargs = app.add_middleware.call_args
        assert kwargs["allow_credentials"] is False

    def test_default_allow_methods_wildcard(self) -> None:
        app = self._make_app()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORS_ORIGINS", None)
            add_cors_middleware(app)
        _, kwargs = app.add_middleware.call_args
        assert kwargs["allow_methods"] == ["*"]

    def test_custom_methods(self) -> None:
        app = self._make_app()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORS_ORIGINS", None)
            add_cors_middleware(app, allow_methods=["GET", "POST"])
        _, kwargs = app.add_middleware.call_args
        assert kwargs["allow_methods"] == ["GET", "POST"]

    def test_env_origins_strips_whitespace(self) -> None:
        app = self._make_app()
        with patch.dict(
            os.environ, {"CORS_ORIGINS": " https://a.com , https://b.com "}
        ):
            add_cors_middleware(app)
        _, kwargs = app.add_middleware.call_args
        assert kwargs["allow_origins"] == ["https://a.com", "https://b.com"]
