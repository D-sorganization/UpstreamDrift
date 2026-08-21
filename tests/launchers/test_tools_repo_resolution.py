"""Tests for the canonical Tools repository resolution facade (issue #8858)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.launchers import tools_repo_path as resolver
from src.shared.python.config.tools_vendor_authority import ToolsVendorAuthority

pytestmark = pytest.mark.unit

_PIN = "aa" * 20


def _make_workspace(tmp_path: Path, *, vendor: bool, sibling: bool) -> Path:
    """Build a workspace with an UpstreamDrift checkout and optional layouts."""
    repo_root = tmp_path / "Repositories" / "UpstreamDrift"
    repo_root.mkdir(parents=True)
    if vendor:
        (repo_root / "vendor" / "ud-tools" / "src").mkdir(parents=True)
    if sibling:
        (tmp_path / "Repositories" / "Tools" / "src").mkdir(parents=True)
    return repo_root


def _stub_authority(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path, *, available: bool
) -> None:
    vendor_root = repo_root / "vendor" / "ud-tools"
    result = ToolsVendorAuthority(
        root=vendor_root,
        expected_sha=_PIN,
        available=available,
        reason=None if available else f"Tools pin stale (expected {_PIN}, found ??)",
    )
    monkeypatch.setattr(
        resolver, "inspect_tools_vendor_authority", lambda _repo_root: result
    )


def test_env_override_wins_over_vendor_and_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _make_workspace(tmp_path, vendor=True, sibling=True)
    _stub_authority(monkeypatch, repo_root, available=True)
    explicit = tmp_path / "Explicit_Tools"
    (explicit / "src").mkdir(parents=True)

    resolution = resolver.resolve_tools_repo(repo_root, str(explicit))

    assert resolution is not None
    assert resolution.path == explicit.resolve()
    assert resolution.source == "env"
    assert resolution.pinned is False


def test_invalid_env_override_fails_closed(tmp_path: Path) -> None:
    repo_root = _make_workspace(tmp_path, vendor=False, sibling=True)

    with pytest.raises(RuntimeError, match="TOOLS_REPO_PATH"):
        resolver.resolve_tools_repo(repo_root, str(tmp_path / "does-not-exist"))


def test_validated_vendor_beats_sibling_and_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _make_workspace(tmp_path, vendor=True, sibling=True)
    _stub_authority(monkeypatch, repo_root, available=True)

    resolution = resolver.resolve_tools_repo(repo_root, None)

    assert resolution is not None
    assert resolution.path == repo_root / "vendor" / "ud-tools"
    assert resolution.source == "vendor"
    assert resolution.pinned is True


def test_unvalidated_vendor_falls_back_to_sibling_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    repo_root = _make_workspace(tmp_path, vendor=True, sibling=True)
    _stub_authority(monkeypatch, repo_root, available=False)
    sibling = tmp_path / "Repositories" / "Tools"

    with caplog.at_level(logging.WARNING, logger=resolver.__name__):
        resolution = resolver.resolve_tools_repo(repo_root, None)

    assert resolution is not None
    assert resolution.path == sibling
    assert resolution.source == "sibling"
    assert resolution.pinned is False
    warnings = " | ".join(record.getMessage() for record in caplog.records)
    assert "Tools pin stale" in warnings


def test_sibling_fallback_logs_unpinned_warning_naming_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo_root = _make_workspace(tmp_path, vendor=False, sibling=True)
    sibling = tmp_path / "Repositories" / "Tools"

    with caplog.at_level(logging.WARNING, logger=resolver.__name__):
        resolution = resolver.resolve_tools_repo(repo_root, None)

    assert resolution is not None
    assert resolution.path == sibling
    assert resolution.source == "sibling"
    assert resolution.pinned is False
    warnings = " | ".join(record.getMessage() for record in caplog.records)
    assert "UNPINNED" in warnings
    assert str(sibling) in warnings


def test_no_source_returns_none(tmp_path: Path) -> None:
    repo_root = _make_workspace(tmp_path, vendor=False, sibling=False)

    assert resolver.resolve_tools_repo(repo_root, None) is None


def test_repo_root_type_is_enforced() -> None:
    with pytest.raises(TypeError):
        resolver.resolve_tools_repo("not-a-path", None)  # type: ignore[arg-type]


def test_resolution_path_always_contains_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Postcondition: every resolved path exposes a src/ directory."""
    repo_root = _make_workspace(tmp_path, vendor=True, sibling=True)
    _stub_authority(monkeypatch, repo_root, available=True)

    for env_value in (None, None):
        resolution = resolver.resolve_tools_repo(repo_root, env_value)
        assert resolution is not None
        assert (resolution.path / "src").is_dir()


def test_resolve_tools_source_root_uses_shared_sibling_walk(tmp_path: Path) -> None:
    repo_root = _make_workspace(tmp_path, vendor=False, sibling=True)

    source_root = resolver.resolve_tools_source_root(repo_root, None)

    assert source_root == tmp_path / "Repositories" / "Tools" / "src"
