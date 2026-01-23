"""Unit tests for shared security utilities."""

from pathlib import Path

import pytest

from src.shared.python.security_utils import validate_path


def test_validate_path_with_tmp_path(tmp_path: Path) -> None:
    """Test path validation using temporary directories."""
    root = tmp_path / "root"
    root.mkdir()

    # Create a safe file
    safe_file = root / "safe.txt"
    safe_file.touch()

    # Create a file outside
    outside_file = tmp_path / "outside.txt"
    outside_file.touch()

    # Test allowed
    result = validate_path(safe_file, [root])
    assert result == safe_file.resolve()

    # Test disallowed strict
    with pytest.raises(ValueError, match="Path traversal blocked"):
        validate_path(outside_file, [root], strict=True)

    # Test disallowed non-strict
    result_outside = validate_path(outside_file, [root], strict=False)
    assert result_outside == outside_file.resolve()


def test_validate_path_traversal(tmp_path: Path) -> None:
    """Test that path traversal attempts are caught."""
    root = tmp_path / "root"
    root.mkdir()

    # subdir
    subdir = root / "subdir"
    subdir.mkdir()

    # traversal attempt: root/subdir/../../outside.txt
    # This resolves to root/../outside.txt -> tmp_path/outside.txt (which is outside root)

    outside_file = tmp_path / "outside.txt"
    outside_file.touch()

    traversal_path = subdir / ".." / ".." / "outside.txt"

    with pytest.raises(ValueError, match="Path traversal blocked"):
        validate_path(traversal_path, [root], strict=True)
