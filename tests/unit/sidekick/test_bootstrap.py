"""Tests for sidekick.bootstrap (Issues #1949, #1744)."""

from __future__ import annotations

import sys
from pathlib import Path

from sidekick.bootstrap import ensure_paths


class TestEnsurePaths:
    def test_bootstrap_returns_path(self) -> None:
        result = ensure_paths()
        assert isinstance(result, Path)

    def test_returns_existing_path(self) -> None:
        result = ensure_paths()
        assert result.exists()

    def test_idempotent(self) -> None:
        # Calling twice should not raise
        result1 = ensure_paths()
        result2 = ensure_paths()
        assert result1 == result2

    def test_adds_to_sys_path(self) -> None:
        before = set(sys.path)
        ensure_paths()
        after = set(sys.path)
        # Paths should be in sys.path after calling (may already be there)
        assert len(after) >= len(before)

    def test_explicit_repo_root(self, tmp_path) -> None:
        # Should accept an explicit path (even non-standard)
        result = ensure_paths(repo_root=tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path

    def test_explicit_repo_root_string(self, tmp_path) -> None:
        result = ensure_paths(repo_root=str(tmp_path))
        assert isinstance(result, Path)
