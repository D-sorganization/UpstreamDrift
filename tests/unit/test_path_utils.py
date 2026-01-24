"""Unit tests for path utilities."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.shared.python.path_utils import (
    ensure_directory,
    find_file_in_parents,
    get_data_dir,
    get_docs_dir,
    get_drake_python_root,
    get_engines_dir,
    get_mujoco_python_root,
    get_output_dir,
    get_pinocchio_python_root,
    get_relative_path,
    get_repo_root,
    get_shared_dir,
    get_shared_python_root,
    get_simscape_model_path,
    get_src_root,
    get_tests_root,
    setup_import_paths,
)


class TestGetRepoRoot:
    """Tests for get_repo_root function."""

    def test_returns_path(self) -> None:
        """Test that get_repo_root returns a Path object."""
        result = get_repo_root()
        assert isinstance(result, Path)

    def test_path_exists(self) -> None:
        """Test that the returned path exists."""
        result = get_repo_root()
        assert result.exists()

    def test_contains_src_directory(self) -> None:
        """Test that repo root contains src directory."""
        result = get_repo_root()
        assert (result / "src").exists() or (result / "pyproject.toml").exists()


class TestGetSrcRoot:
    """Tests for get_src_root function."""

    def test_returns_path(self) -> None:
        """Test that get_src_root returns a Path object."""
        result = get_src_root()
        assert isinstance(result, Path)

    def test_is_child_of_repo_root(self) -> None:
        """Test that src root is under repo root."""
        src_root = get_src_root()
        repo_root = get_repo_root()
        assert str(src_root).startswith(str(repo_root))


class TestGetTestsRoot:
    """Tests for get_tests_root function."""

    def test_returns_path(self) -> None:
        """Test that get_tests_root returns a Path object."""
        result = get_tests_root()
        assert isinstance(result, Path)

    def test_path_name(self) -> None:
        """Test that tests root has correct name."""
        result = get_tests_root()
        assert result.name == "tests"


class TestGetDataDir:
    """Tests for get_data_dir function."""

    def test_returns_path(self) -> None:
        """Test that get_data_dir returns a Path object."""
        result = get_data_dir()
        assert isinstance(result, Path)

    def test_path_name(self) -> None:
        """Test that data dir has correct name."""
        result = get_data_dir()
        assert result.name == "data"


class TestGetOutputDir:
    """Tests for get_output_dir function."""

    def test_returns_path(self) -> None:
        """Test that get_output_dir returns a Path object."""
        result = get_output_dir()
        assert isinstance(result, Path)

    def test_path_name(self) -> None:
        """Test that output dir has correct name."""
        result = get_output_dir()
        assert result.name == "output"

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Test that output dir is created if it doesn't exist."""
        with patch("src.shared.python.path_utils.get_repo_root", return_value=tmp_path):
            # Clear the cache to force recalculation
            import src.shared.python.path_utils as path_utils

            old_root = path_utils._REPO_ROOT
            path_utils._REPO_ROOT = tmp_path
            try:
                result = get_output_dir()
                assert result.exists()
            finally:
                path_utils._REPO_ROOT = old_root


class TestGetDocsDir:
    """Tests for get_docs_dir function."""

    def test_returns_path(self) -> None:
        """Test that get_docs_dir returns a Path object."""
        result = get_docs_dir()
        assert isinstance(result, Path)

    def test_path_name(self) -> None:
        """Test that docs dir has correct name."""
        result = get_docs_dir()
        assert result.name == "docs"


class TestGetEnginesDir:
    """Tests for get_engines_dir function."""

    def test_returns_path(self) -> None:
        """Test that get_engines_dir returns a Path object."""
        result = get_engines_dir()
        assert isinstance(result, Path)

    def test_path_name(self) -> None:
        """Test that engines dir has correct name."""
        result = get_engines_dir()
        assert result.name == "engines"


class TestGetSharedDir:
    """Tests for get_shared_dir function."""

    def test_returns_path(self) -> None:
        """Test that get_shared_dir returns a Path object."""
        result = get_shared_dir()
        assert isinstance(result, Path)

    def test_path_name(self) -> None:
        """Test that shared dir has correct name."""
        result = get_shared_dir()
        assert result.name == "shared"


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_creates_new_directory(self, tmp_path: Path) -> None:
        """Test that new directory is created."""
        new_dir = tmp_path / "new_subdir"
        result = ensure_directory(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_handles_existing_directory(self, tmp_path: Path) -> None:
        """Test that existing directory is handled."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        result = ensure_directory(existing_dir)
        assert result.exists()
        assert result == existing_dir

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        """Test that nested directories are created."""
        nested_dir = tmp_path / "a" / "b" / "c"
        result = ensure_directory(nested_dir)
        assert result.exists()
        assert result.is_dir()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Test that string paths are accepted."""
        new_dir = str(tmp_path / "string_path")
        result = ensure_directory(new_dir)
        assert result.exists()


class TestGetRelativePath:
    """Tests for get_relative_path function."""

    def test_relative_to_repo_root(self) -> None:
        """Test getting path relative to repo root."""
        repo_root = get_repo_root()
        full_path = repo_root / "src" / "shared" / "python" / "path_utils.py"
        result = get_relative_path(full_path)
        assert "src" in str(result)
        assert ".." not in str(result)

    def test_with_custom_base(self, tmp_path: Path) -> None:
        """Test getting path relative to custom base."""
        base = tmp_path / "base"
        base.mkdir()
        target = base / "subdir" / "file.txt"
        result = get_relative_path(target, base)
        assert str(result) == str(Path("subdir") / "file.txt")

    def test_path_not_relative_to_base(self, tmp_path: Path) -> None:
        """Test handling path not relative to base."""
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        result = get_relative_path(outside, base)
        # Should return the resolved path when not relative
        assert result.is_absolute()


class TestFindFileInParents:
    """Tests for find_file_in_parents function."""

    def test_finds_existing_file(self, tmp_path: Path) -> None:
        """Test finding an existing file in parents."""
        # Create nested structure
        parent = tmp_path / "parent"
        parent.mkdir()
        child = parent / "child"
        child.mkdir()

        # Create target file in parent
        target = parent / "target.txt"
        target.touch()

        result = find_file_in_parents("target.txt", start_path=child)
        assert result is not None
        assert result == target

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        """Test returning None when file not found."""
        result = find_file_in_parents("nonexistent_file_xyz.txt", start_path=tmp_path)
        assert result is None

    def test_respects_max_levels(self, tmp_path: Path) -> None:
        """Test respecting max_levels parameter."""
        # Create deep nested structure
        deep = tmp_path
        for i in range(10):
            deep = deep / f"level{i}"
            deep.mkdir()

        # Create target at root
        target = tmp_path / "target.txt"
        target.touch()

        # Should not find with low max_levels
        result = find_file_in_parents("target.txt", start_path=deep, max_levels=2)
        assert result is None


class TestGetSharedPythonRoot:
    """Tests for get_shared_python_root function."""

    def test_returns_path(self) -> None:
        """Test that returns a Path object."""
        result = get_shared_python_root()
        assert isinstance(result, Path)

    def test_ends_with_correct_path(self) -> None:
        """Test that path ends with shared/python."""
        result = get_shared_python_root()
        assert result.name == "python"
        assert result.parent.name == "shared"


class TestGetMujocoPythonRoot:
    """Tests for get_mujoco_python_root function."""

    def test_returns_path(self) -> None:
        """Test that returns a Path object."""
        result = get_mujoco_python_root()
        assert isinstance(result, Path)

    def test_contains_mujoco(self) -> None:
        """Test that path contains mujoco."""
        result = get_mujoco_python_root()
        assert "mujoco" in str(result)


class TestGetPinocchioPythonRoot:
    """Tests for get_pinocchio_python_root function."""

    def test_returns_path(self) -> None:
        """Test that returns a Path object."""
        result = get_pinocchio_python_root()
        assert isinstance(result, Path)

    def test_contains_pinocchio(self) -> None:
        """Test that path contains pinocchio."""
        result = get_pinocchio_python_root()
        assert "pinocchio" in str(result)


class TestGetDrakePythonRoot:
    """Tests for get_drake_python_root function."""

    def test_returns_path(self) -> None:
        """Test that returns a Path object."""
        result = get_drake_python_root()
        assert isinstance(result, Path)

    def test_contains_drake(self) -> None:
        """Test that path contains drake."""
        result = get_drake_python_root()
        assert "drake" in str(result)


class TestGetSimscapeModelPath:
    """Tests for get_simscape_model_path function."""

    def test_default_model(self) -> None:
        """Test with default model name."""
        result = get_simscape_model_path()
        assert isinstance(result, Path)
        assert "3D_Golf_Model" in str(result)
        assert result.name == "src"

    def test_custom_model(self) -> None:
        """Test with custom model name."""
        result = get_simscape_model_path("Custom_Model")
        assert isinstance(result, Path)
        assert "Custom_Model" in str(result)

    def test_path_structure(self) -> None:
        """Test path has correct structure."""
        result = get_simscape_model_path()
        assert "Simscape_Multibody_Models" in str(result)
        assert "python" in str(result)


class TestSetupImportPaths:
    """Tests for setup_import_paths function."""

    def test_adds_paths_to_sys_path(self) -> None:
        """Test that paths are added to sys.path."""
        import sys

        original_path = sys.path.copy()
        setup_import_paths()

        # Should have added some paths
        # Restore doesn't matter for this test as paths remain
        src_root = str(get_src_root())
        assert src_root in sys.path

    def test_with_additional_paths(self, tmp_path: Path) -> None:
        """Test adding additional custom paths."""
        import sys

        custom_path = tmp_path / "custom"
        custom_path.mkdir()

        setup_import_paths(additional_paths=[custom_path])

        assert str(custom_path) in sys.path

    def test_does_not_duplicate_paths(self) -> None:
        """Test that paths are not duplicated."""
        import sys

        src_root = str(get_src_root())

        # Call twice
        setup_import_paths()
        setup_import_paths()

        # Should only appear once
        count = sys.path.count(src_root)
        assert count == 1
