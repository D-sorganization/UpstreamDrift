"""Policy tests for the in-tree test exclusion contract (issue #7126).

These lock in the documented policy that ``src/**/tests`` and root-level
``tests/test_*.py`` files are intentionally excluded from the default pytest
lane, that the exclusion is enforced by ``check_test_layout`` (so new in-tree
tests cannot be added silently), and that the named diagnostic command stays
wired to the single source-of-truth allowlist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_test_layout import (
    LEGACY_ROOT_TEST_FILES,
    LEGACY_SRC_TEST_DIRS,
    audit_test_layout,
)
from scripts.collect_in_tree_tests import in_tree_dirs_with_python_tests

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POLICY_DOC = _REPO_ROOT / "docs" / "development" / "in_tree_test_policy.md"
_MAKEFILE = _REPO_ROOT / "Makefile"


def test_policy_doc_exists() -> None:
    assert _POLICY_DOC.exists(), (
        "docs/development/in_tree_test_policy.md must document the in-tree "
        "test exclusion policy (issue #7126)"
    )


def test_policy_doc_points_at_source_of_truth() -> None:
    text = _POLICY_DOC.read_text(encoding="utf-8")
    # The doc must name the allowlist source-of-truth and the named command so
    # the two cannot drift apart.
    assert "scripts/check_test_layout.py" in text
    assert "LEGACY_SRC_TEST_DIRS" in text
    assert "make test-in-tree" in text
    assert "#7126" in text


def test_named_command_is_wired_in_makefile() -> None:
    text = _MAKEFILE.read_text(encoding="utf-8")
    assert "test-in-tree:" in text
    assert "scripts/collect_in_tree_tests.py" in text


def test_collector_targets_are_a_subset_of_the_allowlist() -> None:
    """Every directory the diagnostic inspects must be a grandfathered entry."""
    targets = in_tree_dirs_with_python_tests(_REPO_ROOT)
    allowlisted = {(_REPO_ROOT / rel).resolve() for rel in LEGACY_SRC_TEST_DIRS}
    for target in targets:
        assert target.resolve() in allowlisted, target


def test_guard_blocks_new_in_tree_src_tests(tmp_path: Path) -> None:
    """A brand-new src/**/tests directory (not allowlisted) must be rejected.

    This is the regression that prevents new uncollected in-tree tests from
    being added by accident (issue #7126 acceptance criterion 3).
    """
    new_dir = tmp_path / "src" / "brand_new_pkg" / "tests"
    new_dir.mkdir(parents=True)
    (new_dir / "test_thing.py").write_text("def test_x():\n    pass\n")

    findings = audit_test_layout(tmp_path)

    reasons = {(f.path.as_posix(), f.reason) for f in findings}
    assert any(
        "tests directory under src" in reason and path == "src/brand_new_pkg/tests"
        for path, reason in reasons
    ), reasons


def test_allowlists_are_frozensets() -> None:
    """The allowlists are immutable contracts, not mutable scratch state."""
    assert isinstance(LEGACY_SRC_TEST_DIRS, frozenset)
    assert isinstance(LEGACY_ROOT_TEST_FILES, frozenset)
