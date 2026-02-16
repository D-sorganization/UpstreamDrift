"""Tests for src.shared.python.data_io.path_utils module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.data_io.path_utils import (
    ensure_directory,
    get_data_dir,
    get_docs_dir,
    get_engines_dir,
    get_output_dir,
    get_relative_path,
    get_repo_root,
    get_shared_dir,
    get_src_root,
)


class TestGetRepoRoot:
    """Tests for get_repo_root function."""

    def test_returns_path(self) -> None:
        root = get_repo_root()
        assert isinstance(root, Path)

    def test_repo_root_exists(self) -> None:
        root = get_repo_root()
        assert root.exists()
        assert root.is_dir()

    def test_repo_root_contains_src(self) -> None:
        root = get_repo_root()
        assert (root / "src").exists()


class TestGetSrcRoot:
    """Tests for get_src_root function."""

    def test_returns_path(self) -> None:
        src = get_src_root()
        assert isinstance(src, Path)

    def test_is_subdirectory_of_repo(self) -> None:
        repo = get_repo_root()
        src = get_src_root()
        assert str(src).startswith(str(repo))


class TestGetDirectories:
    """Tests for directory getter functions."""

    def test_get_data_dir(self) -> None:
        d = get_data_dir()
        assert isinstance(d, Path)

    def test_get_output_dir(self) -> None:
        d = get_output_dir()
        assert isinstance(d, Path)

    def test_get_docs_dir(self) -> None:
        d = get_docs_dir()
        assert isinstance(d, Path)

    def test_get_engines_dir(self) -> None:
        d = get_engines_dir()
        assert isinstance(d, Path)

    def test_get_shared_dir(self) -> None:
        d = get_shared_dir()
        assert isinstance(d, Path)


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_creates_new_dir(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new_sub"
        result = ensure_directory(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_existing_dir_is_noop(self, tmp_path: Path) -> None:
        result = ensure_directory(tmp_path)
        assert result == tmp_path

    def test_nested_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        result = ensure_directory(nested)
        assert result.exists()


class TestGetRelativePath:
    """Tests for get_relative_path function."""

    def test_relative_from_repo(self) -> None:
        root = get_repo_root()
        target = root / "src" / "engines"
        rel = get_relative_path(target)
        # Should not start with repo root
        assert not str(rel).startswith(str(root))

    def test_relative_path_is_shorter(self) -> None:
        root = get_repo_root()
        target = root / "src"
        rel = get_relative_path(target)
        assert len(str(rel)) < len(str(target))
