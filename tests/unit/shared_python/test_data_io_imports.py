"""Tests for data_io module public API imports.

Addresses issue #5478: data_io/__init__.py must export add_provenance_header
without crashing at import time.
"""

import importlib

import pytest


def test_data_io_module_imports_cleanly() -> None:
    """The data_io package must import without AttributeError or ImportError."""
    mod = importlib.import_module("src.shared.python.data_io")
    assert mod is not None


def test_add_provenance_header_exists() -> None:
    """add_provenance_header must be exported from data_io and be callable."""
    from src.shared.python.data_io import add_provenance_header

    assert callable(add_provenance_header)


def test_add_provenance_header_returns_string() -> None:
    """add_provenance_header(content, metadata) must return a string."""
    from src.shared.python.data_io import add_provenance_header

    result = add_provenance_header("col1,col2\n1,2\n", {"run_id": "test"})
    assert isinstance(result, str)


def test_add_provenance_header_prepends_comments() -> None:
    """add_provenance_header must prepend '#' comment lines to content."""
    from src.shared.python.data_io import add_provenance_header

    original = "col1,col2\n1,2\n"
    result = add_provenance_header(original, {"run_id": "test42"})
    lines = result.splitlines()
    # At least the first line must be a comment
    assert lines[0].startswith("#")
    # Original content must appear somewhere in the result
    assert "col1,col2" in result


def test_add_provenance_header_embeds_metadata() -> None:
    """add_provenance_header must embed metadata key/value pairs in the header."""
    from src.shared.python.data_io import add_provenance_header

    result = add_provenance_header("data\n", {"experiment": "swing_001"})
    assert "swing_001" in result


def test_add_provenance_header_invalid_content_raises() -> None:
    """add_provenance_header must raise TypeError when content is not a string (DbC)."""
    from src.shared.python.data_io import add_provenance_header

    with pytest.raises(TypeError):
        add_provenance_header(123, {})  # type: ignore[arg-type]


def test_add_provenance_header_invalid_metadata_raises() -> None:
    """add_provenance_header must raise TypeError when metadata is not a dict (DbC)."""
    from src.shared.python.data_io import add_provenance_header

    with pytest.raises(TypeError):
        add_provenance_header("data\n", "not_a_dict")  # type: ignore[arg-type]


def test_add_provenance_header_file_still_exported() -> None:
    """add_provenance_header_file must still be available (backward compat)."""
    from src.shared.python.data_io import add_provenance_header_file

    assert callable(add_provenance_header_file)


def test_provenance_info_exported() -> None:
    """ProvenanceInfo must be exported from data_io."""
    from src.shared.python.data_io import ProvenanceInfo

    assert ProvenanceInfo is not None
