"""Tests for the repository document title-capitalization gate."""

from pathlib import Path

import pytest

from scripts.check_document_title_case import (
    _added_lines_from_diff,
    expected_title,
    findings_for_text,
)

pytestmark = pytest.mark.unit


def test_expected_title_preserves_minor_words_and_technical_tokens() -> None:
    assert (
        expected_title("agreement with the literature")
        == "Agreement With the Literature"
    )
    assert (
        expected_title("state-of-the-art control in SO(3)")
        == "State-of-the-Art Control in SO(3)"
    )


def test_quarto_and_latex_structural_titles_are_checked() -> None:
    quarto = '---\ntitle: "a guide to drift"\n---\n\n## why timing matters\n'
    latex = r"\title{a guide to drift}" + "\n" + r"\section{why timing matters}"

    assert [item.expected for item in findings_for_text(Path("paper.qmd"), quarto)] == [
        "A Guide to Drift",
        "Why Timing Matters",
    ]
    assert [item.expected for item in findings_for_text(Path("paper.tex"), latex)] == [
        "A Guide to Drift",
        "Why Timing Matters",
    ]


def test_zero_context_diff_parser_returns_only_added_line_numbers() -> None:
    diff = "@@ -3,2 +3,3 @@\n-old\n+new\n+added\n context\n@@ -10 +11 @@\n-x\n+y\n"

    assert _added_lines_from_diff(diff) == {3, 4, 5, 11}
