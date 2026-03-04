"""Tests for check_markdown_links.py — full coverage including DbC, check_links, and edge cases."""

from pathlib import Path

import pytest

from src.shared.python.contracts import PreconditionError
from src.tools.check_markdown_links import (
    check_links,
    extract_links_from_markdown,
    resolve_and_verify_link,
)

# ─── extract_links_from_markdown ───────────────────────────────


def test_extract_links_from_markdown():
    content = """
    Here is a [link](file.md).
    External [link](https://example.com).
    Anchor [link](#anchor-only).
    Another [link](dir/file2.md#anchor).
    """
    links = extract_links_from_markdown(content)
    assert links == ["file.md", "dir/file2.md#anchor"]


def test_extract_links_skips_http():
    content = "[ext](http://example.com) and [local](local.md)"
    links = extract_links_from_markdown(content)
    assert "http://example.com" not in links
    assert "local.md" in links


def test_extract_links_skips_https():
    content = "[secure](https://secure.com)"
    links = extract_links_from_markdown(content)
    assert links == []


def test_extract_links_skips_mailto():
    content = "[email](mailto:test@test.com)"
    links = extract_links_from_markdown(content)
    assert links == []


def test_extract_links_skips_fragment_only():
    content = "[anchor](#section)"
    links = extract_links_from_markdown(content)
    assert links == []


def test_extract_links_empty_content():
    links = extract_links_from_markdown("")
    assert links == []


def test_extract_links_no_links():
    content = "This is plain text with **bold** and _italic_."
    links = extract_links_from_markdown(content)
    assert links == []


def test_extract_links_dbc_non_string():
    with pytest.raises(PreconditionError):
        extract_links_from_markdown(None)  # type: ignore[arg-type]


# ─── resolve_and_verify_link ───────────────────────────────────


def test_resolve_and_verify_link_success(tmp_path):
    f = tmp_path / "file.md"
    f.touch()
    error = resolve_and_verify_link("file.md", tmp_path)
    assert error is None


def test_resolve_and_verify_link_failure(tmp_path):
    error = resolve_and_verify_link("missing.md", tmp_path)
    assert error is not None
    assert "Broken link" in error


def test_resolve_and_verify_link_anchor_stripped(tmp_path):
    f = tmp_path / "page.md"
    f.touch()
    error = resolve_and_verify_link("page.md#section", tmp_path)
    assert error is None


def test_resolve_and_verify_link_anchor_only():
    """Fragment-only links return None (valid)."""
    error = resolve_and_verify_link("#section", Path("/some/dir"))
    assert error is None


def test_resolve_and_verify_link_url_encoded(tmp_path):
    f = tmp_path / "my file.md"
    f.touch()
    error = resolve_and_verify_link("my%20file.md", tmp_path)
    assert error is None


def test_resolve_and_verify_link_dbc_non_string(tmp_path):
    with pytest.raises(PreconditionError):
        resolve_and_verify_link(123, tmp_path)  # type: ignore[arg-type]


def test_resolve_and_verify_link_dbc_non_path():
    with pytest.raises(PreconditionError):
        resolve_and_verify_link("file.md", "/string/not/path")  # type: ignore[arg-type]


# ─── check_links ───────────────────────────────────────────────


def test_check_links_no_markdown(tmp_path):
    errors = check_links(tmp_path)
    assert errors == []


def test_check_links_all_good(tmp_path):
    readme = tmp_path / "README.md"
    linked = tmp_path / "other.md"
    linked.touch()
    readme.write_text("[other](other.md)\n")
    errors = check_links(tmp_path)
    assert errors == []


def test_check_links_broken_link(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("[missing](nonexistent.md)\n")
    errors = check_links(tmp_path)
    assert any("Broken link" in e for e in errors)


def test_check_links_ignores_external_links(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("[site](https://example.com)\n")
    errors = check_links(tmp_path)
    assert errors == []


def test_check_links_skips_node_modules(tmp_path):
    node_mods = tmp_path / "node_modules"
    node_mods.mkdir()
    bad = node_mods / "README.md"
    bad.write_text("[broken](definitely_missing.md)\n")
    errors = check_links(tmp_path)
    assert len(errors) == 0


def test_check_links_dbc_non_path():
    with pytest.raises(PreconditionError):
        check_links("/some/string/path")  # type: ignore[arg-type]


def test_check_links_dbc_missing_dir(tmp_path):
    with pytest.raises(PreconditionError):
        check_links(tmp_path / "nonexistent")
