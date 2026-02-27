"""Shared CORS configuration factory for FastAPI applications.

Centralises the CORS middleware setup that was previously copy-pasted across
four separate FastAPI apps. Each app can still override the defaults by
passing keyword arguments.

Usage::

    from cors import add_cors_middleware

    app = FastAPI(title="My App")
    add_cors_middleware(app)
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Default local-development origins used when CORS_ORIGINS env var is unset.
DEFAULT_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


def add_cors_middleware(
    app: FastAPI,
    *,
    origins: list[str] | None = None,
    allow_credentials: bool = True,
    allow_methods: list[str] | None = None,
    allow_headers: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Attach CORS middleware to *app* with sensible defaults.

    Origins are resolved in priority order:
    1. ``CORS_ORIGINS`` environment variable (comma-separated).
    2. The *origins* argument (caller override).
    3. ``DEFAULT_ORIGINS`` (localhost dev servers).

    Args:
        app: The FastAPI application instance.
        origins: Explicit list of allowed origins; overridden by env var.
        allow_credentials: Whether to allow credentials. Defaults to True.
        allow_methods: Allowed HTTP methods. Defaults to ``["*"]``.
        allow_headers: Allowed HTTP headers. Defaults to ``["*"]``.
        **kwargs: Extra keyword arguments forwarded to ``CORSMiddleware``.
    """
    env_origins = os.environ.get("CORS_ORIGINS")
    resolved_origins: list[str]
    if env_origins:
        resolved_origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    elif origins is not None:
        resolved_origins = origins
    else:
        resolved_origins = DEFAULT_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods or ["*"],
        allow_headers=allow_headers or ["*"],
        **kwargs,
    )
