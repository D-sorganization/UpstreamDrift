"""Unit tests for ui/scripts/theme_tokens_codemod.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.scripts.theme_tokens_codemod import (
    apply_token_replacements,
    count_palette_classes,
    process_file,
)

pytestmark = pytest.mark.unit


def test_apply_token_replacements_basic() -> None:
    sample = '<div className="bg-gray-900 text-gray-400 border border-gray-700">'
    transformed, stats = apply_token_replacements(sample)
    assert (
        transformed
        == '<div className="bg-canvas text-token-muted border border-token-border">'
    )
    assert stats["bg-gray-900 -> bg-canvas"] == 1
    assert stats["text-gray-400 -> text-token-muted"] == 1
    assert stats["border-gray-700 -> border-token-border"] == 1


def test_apply_token_replacements_with_variants() -> None:
    sample = '<button className="bg-blue-600 hover:bg-blue-700 hover:bg-gray-800 text-gray-200 border-gray-600 hover:bg-gray-600 hover:bg-blue-500 focus:ring-blue-500">'
    transformed, stats = apply_token_replacements(sample)
    assert (
        transformed
        == '<button className="bg-primary hover:bg-primary-hover hover:bg-surface text-token-text border-token-border hover:bg-surface hover:bg-primary-hover focus:ring-primary">'
    )
    assert stats["bg-blue-600 -> bg-primary"] == 1
    assert stats["hover:bg-blue-700 -> hover:bg-primary-hover"] == 1
    assert stats["bg-gray-800 -> bg-surface"] == 1
    assert stats["text-gray-200 -> text-token-text"] == 1
    assert stats["border-gray-600 -> border-token-border"] == 1
    assert stats["hover:bg-gray-600 -> hover:bg-surface"] == 1
    assert stats["hover:bg-blue-500 -> hover:bg-primary-hover"] == 1
    assert stats["focus:ring-blue-500 -> focus:ring-primary"] == 1


def test_apply_token_replacements_no_partial_word_match() -> None:
    # Ensure classes like custom-bg-gray-900 or bg-gray-900-extra are not corruptly replaced
    sample = '<div className="not-bg-gray-900 bg-gray-900">'
    transformed, stats = apply_token_replacements(sample)
    assert transformed == '<div className="not-bg-gray-900 bg-canvas">'
    assert stats["bg-gray-900 -> bg-canvas"] == 1


def test_count_palette_classes() -> None:
    sample = """
    <div className="bg-gray-900 text-gray-400 border-gray-700 bg-blue-600 hover:border-blue-500">
      <span className="text-token-muted bg-canvas">Untracked</span>
    </div>
    """
    counts = count_palette_classes(sample)
    assert counts["gray"] == 3
    assert counts["blue"] == 2


def test_process_file_dry_run_and_write(tmp_path: Path) -> None:
    test_file = tmp_path / "Component.tsx"
    test_file.write_text('<div className="bg-gray-900">Hello</div>', encoding="utf-8")

    # Dry run
    changed, stats = process_file(test_file, write=False)
    assert changed is True
    assert stats["bg-gray-900 -> bg-canvas"] == 1
    assert (
        test_file.read_text(encoding="utf-8")
        == '<div className="bg-gray-900">Hello</div>'
    )

    # Write run
    changed, stats = process_file(test_file, write=True)
    assert changed is True
    assert (
        test_file.read_text(encoding="utf-8")
        == '<div className="bg-canvas">Hello</div>'
    )
