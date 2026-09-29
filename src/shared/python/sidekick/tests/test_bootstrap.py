"""Tests for upstream_drift_tools.bootstrap (ensure_paths).

Full coverage of both branches:
- Auto-detect repo root from get_repo_root()
- Explicit repo_root path provided
- Already-present paths not inserted twice
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


class TestEnsurePaths:
    def test_returns_repo_root_auto_detected(self, tmp_path: Path) -> None:
        """Auto-detect branch: calls get_repo_root() when no root given."""
        from sidekick.bootstrap import ensure_paths

        # get_repo_root is imported locally inside ensure_paths;
        # patch it at the source module.
        with patch(
            "upstream_drift_tools.utils.paths.get_repo_root",
            return_value=tmp_path,
        ):
            result = ensure_paths()
        assert result == tmp_path

    def test_explicit_repo_root(self, tmp_path: Path) -> None:
        """Explicit repo_root branch: resolves given Path."""
        from sidekick.bootstrap import ensure_paths

        result = ensure_paths(repo_root=tmp_path)
        assert result == tmp_path

    def test_explicit_repo_root_as_string(self, tmp_path: Path) -> None:
        """Accepts string as repo_root."""
        from sidekick.bootstrap import ensure_paths

        result = ensure_paths(repo_root=str(tmp_path))
        assert result == tmp_path

    def test_does_not_add_nonexistent_paths(self, tmp_path: Path) -> None:
        """Nonexistent paths are not inserted into sys.path."""
        from sidekick.bootstrap import ensure_paths

        # tmp_path/src/shared/python etc. do not exist
        before = set(sys.path)
        ensure_paths(repo_root=tmp_path)
        after = set(sys.path)
        # No new entries should have been added (none of the standard paths exist)
        new_entries = after - before
        for entry in new_entries:
            check_path = Path(entry)
            assert check_path.exists(), f"Non-existent path added: {entry}"

    def test_does_not_insert_duplicate(self, tmp_path: Path) -> None:
        """Already-present paths don't get inserted again."""
        from sidekick.bootstrap import ensure_paths

        src_root = tmp_path / "src"
        src_root.mkdir(parents=True)
        path_str = str(src_root)

        # Ensure it's already in sys.path
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

        count_before = sys.path.count(path_str)
        ensure_paths(repo_root=tmp_path)
        count_after = sys.path.count(path_str)

        assert count_after == count_before  # Not duplicated

    def test_adds_existing_paths(self, tmp_path: Path) -> None:
        """Creates and registers the standard paths when they exist."""
        from sidekick.bootstrap import ensure_paths

        # Set up a synthetic repo structure
        src_root = tmp_path / "src"
        python_src = tmp_path / "src" / "python" / "src"
        shared_python = tmp_path / "src" / "shared" / "python"
        src_root.mkdir(parents=True)
        python_src.mkdir(parents=True)
        shared_python.mkdir(parents=True)

        # Strip synthetic path from sys.path to test insertion
        expected_paths = {str(src_root), str(python_src)}
        forbidden_path = str(shared_python)
        original = sys.path.copy()
        try:
            # Remove if present
            for path_str in {*expected_paths, forbidden_path}:
                while path_str in sys.path:
                    sys.path.remove(path_str)

            ensure_paths(repo_root=tmp_path)
            assert expected_paths <= set(sys.path)
            assert forbidden_path not in sys.path
        finally:
            # Restore original sys.path
            sys.path[:] = original
