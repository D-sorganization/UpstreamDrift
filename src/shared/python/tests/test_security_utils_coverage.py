"""Tests for shared.python.security_utils coverage."""

from pathlib import Path
from unittest.mock import patch

import pytest

from shared.python.security_utils import validate_path


def test_validate_path_success(tmp_path: Path) -> None:
    """Test successful path validation."""
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    target_file = allowed_root / "test.txt"
    target_file.touch()

    # Test exact file
    result = validate_path(target_file, [allowed_root])
    assert result == target_file.resolve()

    # Test directory
    result = validate_path(allowed_root, [allowed_root])
    assert result == allowed_root.resolve()


def test_validate_path_traversal(tmp_path: Path) -> None:
    """Test detection of path traversal attacks."""
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    forbidden_root = tmp_path / "forbidden"
    forbidden_root.mkdir()
    target_file = forbidden_root / "secret.txt"
    target_file.touch()

    # Test going up directories
    traversal_path = allowed_root / ".." / "forbidden" / "secret.txt"

    with pytest.raises(ValueError, match="Path traversal blocked"):
        validate_path(traversal_path, [allowed_root])


def test_validate_path_outside_root(tmp_path: Path) -> None:
    """Test path completely outside allowed roots."""
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_path = tmp_path / "outside.txt"
    outside_path.touch()

    with pytest.raises(ValueError, match="Path traversal blocked"):
        validate_path(outside_path, [allowed_root])


def test_validate_path_non_strict(tmp_path: Path) -> None:
    """Test non-strict mode."""
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_path = tmp_path / "outside.txt"
    outside_path.touch()

    # Should not raise in non-strict mode, but returns path "as is" or resolved?
    # The code says: returns resolved_path, but waits, logic says:
    # if not is_allowed: if strict: raise... return resolved_path
    # So it returns resolved path even if not allowed in non-strict mode.

    # However, there is a catch in the code:
    # try: resolved_path = Path(path).resolve() ... except ... if strict raise else return Path(path)
    # This handles invalid format.

    # Let's test non-strict traversal
    result = validate_path(outside_path, [allowed_root], strict=False)
    assert result == outside_path.resolve()


def test_validate_path_invalid_format() -> None:
    """Test handling of invalid path formats."""
    # It's hard to make Path().resolve() fail on modern OS with just a string,
    # unless it is really malformed or permissions issue.
    # We can mock Path.resolve to raise an exception.

    with pytest.raises(ValueError, match="Invalid path format"):
        with patch.object(Path, "resolve", side_effect=Exception("Disk error")):
            validate_path("some/path", [Path(".")])

    # Non-strict should return the input as Path
    with patch.object(Path, "resolve", side_effect=Exception("Disk error")):
        result = validate_path("some/path", [Path(".")], strict=False)
        assert isinstance(result, Path)
        # Compare as Path for cross-platform compatibility (Windows uses backslashes)
        assert result == Path("some/path")
