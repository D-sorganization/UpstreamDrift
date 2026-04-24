"""Tests for version metadata consistency across all surfaces (issue #2453).

The canonical version lives in src/api/_version.py and all other surfaces
(server.py FastAPI metadata, local_server.py, and the root endpoint in core.py)
import from there.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).parents[3]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _pyproject_version() -> str:
    """Read the package version from pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(_PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return str(data["project"]["version"])


class TestVersionModuleExists:
    """src.api._version provides the canonical version."""

    def test_can_import_version_module(self) -> None:
        from src.api._version import __version__  # noqa: F401

    def test_version_is_non_empty_string(self) -> None:
        from src.api._version import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_matches_pyproject_toml(self) -> None:
        from src.api._version import __version__

        expected = _pyproject_version()
        assert __version__ == expected, (
            f"src.api._version.__version__ ({__version__!r}) must match "
            f"pyproject.toml version ({expected!r})"
        )


class TestVersionSurfacesAligned:
    """All version surfaces read from the single canonical source."""

    def test_server_py_uses_canonical_version(self) -> None:
        """server.py FastAPI version should not hardcode a stale string."""
        server_src = (_REPO_ROOT / "src" / "api" / "server.py").read_text(
            encoding="utf-8"
        )
        assert "_version" in server_src or "__version__" in server_src, (
            "server.py must import __version__ from src.api._version "
            "instead of hardcoding a version string"
        )

    def test_server_py_does_not_hardcode_old_version(self) -> None:
        """3.0.0 (old hardcoded value) is not in server.py."""
        server_src = (_REPO_ROOT / "src" / "api" / "server.py").read_text(
            encoding="utf-8"
        )
        assert '"3.0.0"' not in server_src, (
            "server.py must not hardcode version '3.0.0'; use __version__ from _version.py"
        )

    def test_local_server_py_uses_canonical_version(self) -> None:
        """local_server.py FastAPI version should not hardcode a stale string."""
        local_src = (_REPO_ROOT / "src" / "api" / "local_server.py").read_text(
            encoding="utf-8"
        )
        assert "_version" in local_src or "__version__" in local_src, (
            "local_server.py must import __version__ from src.api._version"
        )

    def test_local_server_py_does_not_hardcode_old_version(self) -> None:
        """2.0.0 (old hardcoded value) is not in local_server.py."""
        local_src = (_REPO_ROOT / "src" / "api" / "local_server.py").read_text(
            encoding="utf-8"
        )
        assert '"2.0.0"' not in local_src, (
            "local_server.py must not hardcode version '2.0.0'; use __version__"
        )

    def test_core_route_uses_canonical_version(self) -> None:
        """The root endpoint in core.py must not return a hardcoded version."""
        core_src = (_REPO_ROOT / "src" / "api" / "routes" / "core.py").read_text(
            encoding="utf-8"
        )
        assert "_version" in core_src or "__version__" in core_src, (
            "core.py root endpoint must return __version__ from _version.py, "
            "not a hardcoded string"
        )

    def test_core_route_does_not_hardcode_old_version(self) -> None:
        """1.0.0 (old hardcoded value) is not in core.py."""
        core_src = (_REPO_ROOT / "src" / "api" / "routes" / "core.py").read_text(
            encoding="utf-8"
        )
        assert '"1.0.0"' not in core_src, (
            "core.py must not hardcode version '1.0.0'; use __version__ from _version.py"
        )
