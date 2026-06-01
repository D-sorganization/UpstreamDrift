"""Unit tests for shared provenance probes (#6939)."""

from __future__ import annotations

import subprocess
from types import ModuleType

import pytest

from src.shared.python.motion_matching import provenance


def _raise(dist: str):
    from importlib.metadata import PackageNotFoundError

    raise PackageNotFoundError(dist)


def test_engine_package_version_prefers_module_dunder() -> None:
    mod = ModuleType("fake_engine")
    mod.__version__ = "9.9.9"  # type: ignore[attr-defined]
    assert provenance.engine_package_version(mod, "nonexistent-dist") == "9.9.9"


def test_engine_package_version_skips_empty_dunder_then_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = ModuleType("fake_engine")
    mod.__version__ = ""  # type: ignore[attr-defined]
    monkeypatch.setattr(
        provenance,
        "_metadata_version",
        lambda dist: "1.2.3" if dist == "real-dist" else _raise(dist),
    )
    assert (
        provenance.engine_package_version(mod, "missing-dist", "real-dist") == "1.2.3"
    )


def test_engine_package_version_unknown_when_nothing_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_metadata_version", lambda dist: _raise(dist))
    assert provenance.engine_package_version(None, "a", "b") == "unknown"


def test_engine_package_version_handles_none_module() -> None:
    # A None module simply skips the __version__ probe.
    assert provenance.engine_package_version(None) == "unknown"


def test_git_commit_short_returns_string() -> None:
    # Runs inside the repo: should be a short hex SHA, never raise.
    out = provenance.git_commit_short()
    assert isinstance(out, str) and out


def test_git_commit_short_unknown_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*a, **k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "check_output", _boom)
    assert provenance.git_commit_short() == "unknown"
