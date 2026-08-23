"""Adapter-level tests for canonical Tools discovery (issue #8858)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from src.launchers import external_tools_adapter as adapter  # noqa: E402
from src.launchers.tools_repo_path import ToolsRepoResolution  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_adapter_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(adapter, "_TOOLS_REPO", None)
    monkeypatch.setattr(adapter, "_TOOLS_RESOLUTION", None)
    yield


def test_find_tools_repo_delegates_to_canonical_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter must not re-probe the filesystem itself (LoD)."""
    vendored = tmp_path / "UpstreamDrift" / "vendor" / "ud-tools"
    (vendored / "src").mkdir(parents=True)
    resolution = ToolsRepoResolution(path=vendored, source="vendor", pinned=True)
    seen: list[tuple[Path, str | None]] = []

    def fake_resolve(repo_root: Path, env_value: str | None):
        seen.append((repo_root, env_value))
        return resolution

    monkeypatch.setattr(adapter, "resolve_tools_repo", fake_resolve)

    assert adapter._find_tools_repo() == vendored
    # Cached: the resolver runs once.
    assert adapter._find_tools_repo() == vendored
    assert len(seen) == 1
    assert seen[0][0] == adapter._REPO_ROOT


def test_find_tools_repo_returns_none_when_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter, "resolve_tools_repo", lambda *_args: None)

    assert adapter._find_tools_repo() is None


def test_invalid_env_override_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_invalid(_repo_root: Path, _env_value: str | None):
        raise RuntimeError("TOOLS_REPO_PATH must point to a Tools checkout")

    monkeypatch.setattr(adapter, "resolve_tools_repo", raise_invalid)

    assert adapter._find_tools_repo() is None
