from __future__ import annotations

from pathlib import Path

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
