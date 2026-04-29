"""Tests for shared API version resolution."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from src.api import versioning

pytestmark = pytest.mark.unit


def test_get_app_version_matches_pyproject() -> None:
    """Fallback version source should match pyproject version."""
    versioning.get_app_version.cache_clear()
    assert versioning.get_app_version() == "2.1.0"


def test_get_app_version_prefers_package_metadata(monkeypatch) -> None:
    """Installed metadata version should take precedence when available."""

    def fake_version(name: str) -> str:
        if name == "upstream-drift":
            return "9.9.9"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(versioning, "version", fake_version)
    versioning.get_app_version.cache_clear()
    assert versioning.get_app_version() == "9.9.9"


def test_api_servers_use_shared_version_source() -> None:
    """Both server entrypoints should use the shared resolver."""
    repo_root = Path(__file__).resolve().parents[3]
    server_source = (repo_root / "src" / "api" / "server.py").read_text(
        encoding="utf-8"
    )
    local_server_source = (repo_root / "src" / "api" / "local_server.py").read_text(
        encoding="utf-8"
    )

    assert "version=get_app_version()" in server_source
    assert "version=get_app_version()" in local_server_source
