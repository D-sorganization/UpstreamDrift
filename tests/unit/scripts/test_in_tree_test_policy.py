from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_in_tree_policy_doc_points_to_guard_and_named_command() -> None:
    text = (ROOT / "docs/development/in_tree_test_policy.md").read_text(
        encoding="utf-8"
    )

    assert "scripts/check_pytest_intree_testpaths.py" in text
    assert "make test-in-tree" in text
    assert "Issue #7126" in text


def test_makefile_wires_test_in_tree_target() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "test-in-tree:" in text
    assert "scripts/check_pytest_intree_testpaths.py" in text
