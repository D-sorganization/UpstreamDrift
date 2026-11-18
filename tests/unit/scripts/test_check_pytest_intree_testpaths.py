from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_pytest_intree_testpaths as check


def test_is_covered_accepts_nested_test_file() -> None:
    assert check._is_covered(
        Path("src/shared/python/sidekick/tests/test_paths.py"),
        [Path("tests"), Path("src/shared/python/sidekick/tests")],
    )


def test_is_covered_rejects_unlisted_in_tree_test_file() -> None:
    assert not check._is_covered(
        Path("src/shared/python/new_feature/tests/test_new_feature.py"),
        [Path("tests"), Path("src/shared/python/sidekick/tests")],
    )


@pytest.mark.unit
def test_find_missing_testpaths_detects_absent_directory(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    missing = check.find_missing_testpaths(
        tmp_path,
        [Path("tests"), Path("src/shared/python/deleted_pkg/tests")],
    )
    assert missing == ["src/shared/python/deleted_pkg/tests"]


@pytest.mark.unit
def test_configured_pyproject_testpaths_exist_on_disk() -> None:
    assert check.find_missing_testpaths(check.ROOT, check._configured_testpaths()) == []
