"""Tests for upstream_drift_tools.utils.paths (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.upstream_drift_tools.utils.paths import get_repo_root


class TestGetRepoRoot:
    def test_returns_path_object(self) -> None:
        root = get_repo_root()
        assert isinstance(root, Path)

    def test_returned_path_is_absolute(self) -> None:
        root = get_repo_root()
        assert root.is_absolute()

    def test_returned_path_exists(self) -> None:
        root = get_repo_root()
        assert root.exists()

    def test_returned_path_has_pyproject_toml(self) -> None:
        root = get_repo_root()
        # The repo root should have one of the marker files
        has_marker = any(
            (root / m).exists() for m in (".git", "pyproject.toml", "tools.json")
        )
        assert has_marker

    def test_explicit_start_path_from_repo_subdir(self) -> None:
        # Starting from a known subdir should still find the root
        root = get_repo_root()
        subdir = root / "src"
        if subdir.exists():
            found = get_repo_root(subdir)
            assert found == root

    def test_no_root_found_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Patch markers to a sentinel that does not exist anywhere on disk so
        # upward traversal cannot accidentally match a runner-level file (e.g.
        # /tmp/pyproject.toml created by editable installs on self-hosted CI).
        import src.shared.python.upstream_drift_tools.utils.paths as paths_mod

        monkeypatch.setattr(
            paths_mod, "_REPO_ROOT_MARKERS", (".THIS_MARKER_DOES_NOT_EXIST_ANYWHERE",)
        )
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            get_repo_root(deep)

    def test_start_path_accepts_string(self) -> None:
        root = get_repo_root()
        root_via_str = get_repo_root(str(root))
        assert root_via_str == root
