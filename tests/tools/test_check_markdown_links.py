"""Tests for check_markdown_links.py"""

from pathlib import Path

from src.shared.python.contracts import PreconditionError
from src.tools.check_markdown_links import (
    extract_links_from_markdown,
    resolve_and_verify_link,
)


def test_extract_links_from_markdown():
    content = """
    Here is a [link](file.md).
    External [link](https://example.com).
    Anchor [link](#anchor-only).
    Another [link](dir/file2.md#anchor).
    """
    links = extract_links_from_markdown(content)
    assert links == ["file.md", "dir/file2.md#anchor"]


def test_resolve_and_verify_link_success(tmp_path):
    f = tmp_path / "file.md"
    f.touch()

    # Should return None when link exists
    error = resolve_and_verify_link("file.md", tmp_path)
    assert error is None


def test_resolve_and_verify_link_failure(tmp_path):
    error = resolve_and_verify_link("missing.md", tmp_path)
    assert error is not None
    assert "Broken link" in error
