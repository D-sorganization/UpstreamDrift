"""Shared fixtures for the launch-mode functional QA gate (#8966).

The gate exercises the four launch modes routed by
``launch_upstream_drift.py`` (web, ``--classic``, ``--api-only``,
``--engine``) far enough to prove each reaches a ready state, without
spawning subprocesses or opening real windows.

Environment is forced headless before any Qt import: ``QT_QPA_PLATFORM``
is set to ``offscreen`` and onboarding dialogs are disabled, mirroring
what ``launch_upstream_drift.py`` does on a display-less host.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from typing import Any

import pytest

# Must happen before the first PyQt6 import anywhere in this package.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("UPSTREAMDRIFT_DISABLE_ONBOARDING", "1")
os.environ.setdefault("GOLF_SUITE_MODE", "local")

#: Cold-start ceiling for building the FastAPI app (issue #8966; ties to
#: #8934/#8938 — deliberately generous first, ratchet down later).
WEB_APP_CONSTRUCTION_BUDGET_S: float = 60.0


@pytest.fixture(scope="session")
def local_app_bundle() -> tuple[Any, float]:
    """Construct the local FastAPI app once and record construction time.

    Returns:
        Tuple of (FastAPI app, construction seconds).
    """
    from src.api.local_server import create_local_app

    start = time.monotonic()
    app = create_local_app()
    elapsed = time.monotonic() - start
    return app, elapsed


@pytest.fixture(scope="session")
def web_client(local_app_bundle: tuple[Any, float]) -> Generator[Any, None, None]:
    """Session-scoped in-process TestClient against the local app.

    This is the same app object served by the default web mode and by
    ``--api-only`` (both call ``src.api.local_server.main`` which builds
    the app via ``create_local_app``), so one client covers both modes.
    """
    from fastapi.testclient import TestClient

    app, _ = local_app_bundle
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def qapp() -> Generator[Any, None, None]:
    """Session-scoped offscreen ``QApplication`` (never calls ``exec``)."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
