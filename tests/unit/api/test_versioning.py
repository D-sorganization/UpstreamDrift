"""Tests for shared API version resolution."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

from src.api import versioning


def test_get_app_version_matches_pyproject() -> None:
    """Fallback version source should match pyproject version."""
    versioning.get_app_version.cache_clear()
    assert versioning.get_app_version() == "2.1.1"


def test_get_app_version_prefers_package_metadata(monkeypatch) -> None:
    """Installed metadata version should take precedence when available."""

    def fake_version(name: str) -> str:
        if name == "upstream-drift":
            return "9.9.9"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(versioning, "version", fake_version)
    versioning.get_app_version.cache_clear()
    assert versioning.get_app_version() == "9.9.9"


import pytest


def test_make_versioned_router_success() -> None:
    router = versioning.make_versioned_router("v1")
    assert router.prefix == "/v1"
    assert not router.deprecated


def test_make_versioned_router_deprecated() -> None:
    router = versioning.make_versioned_router("v2", deprecated=True)
    assert router.prefix == "/v2"
    assert router.deprecated
    assert len(router.dependencies) == 1


def test_make_versioned_router_deprecated_sunset() -> None:
    router = versioning.make_versioned_router(
        "v3", deprecated=True, sunset="Mon, 01 Jan 2026 00:00:00 GMT"
    )
    assert router.prefix == "/v3"
    assert router.deprecated


def test_validate_version() -> None:
    assert versioning._validate_version("v1") == "v1"

    with pytest.raises(TypeError, match="version must be a string"):
        versioning._validate_version(1)

    with pytest.raises(ValueError, match="version must match"):
        versioning._validate_version("1")


def test_format_sunset() -> None:
    assert versioning._format_sunset(None) is None
    assert (
        versioning._format_sunset(" Mon, 01 Jan 2026 00:00:00 GMT ")
        == "Mon, 01 Jan 2026 00:00:00 GMT"
    )

    with pytest.raises(TypeError, match="sunset must be a string"):
        versioning._format_sunset(123)

    with pytest.raises(ValueError, match="sunset must be a non-empty"):
        versioning._format_sunset("   ")


def test_get_app_version_pyproject_fallback(monkeypatch, tmp_path) -> None:
    def fake_version(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(versioning, "version", fake_version)

    # create fake pyproject.toml
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text('[project]\nversion = "1.2.3"')

    # Mock Path to point to this tmp dir
    class MockPath:
        def __init__(self, *args, **kwargs):
            self.p = tmp_path

        def resolve(self):
            return self

        @property
        def parents(self):
            return [self, self, self, self]

        def __truediv__(self, other):
            return pyproject_file if other == "pyproject.toml" else tmp_path

    monkeypatch.setattr(versioning, "Path", MockPath)
    versioning.get_app_version.cache_clear()
    assert versioning.get_app_version() == "1.2.3"


def test_get_app_version_all_fail(monkeypatch) -> None:
    def fake_version(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(versioning, "version", fake_version)

    class MockPath:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self):
            return self

        @property
        def parents(self):
            return [self, self, self, self]

        def __truediv__(self, other):
            return self

        def open(self, *args, **kwargs):
            raise FileNotFoundError

    monkeypatch.setattr(versioning, "Path", MockPath)
    versioning.get_app_version.cache_clear()
    assert versioning.get_app_version() == "0.0.0"


def test_deprecation_headers() -> None:
    router = versioning.make_versioned_router(
        "v1", deprecated=True, sunset="Mon, 01 Jan 2026 00:00:00 GMT"
    )
    dep = router.dependencies[0].dependency

    class MockResponse:
        def __init__(self):
            self.headers = {}

    response = MockResponse()
    dep(response)
    assert response.headers["Deprecation"] == "true"
    assert response.headers["Sunset"] == "Mon, 01 Jan 2026 00:00:00 GMT"
