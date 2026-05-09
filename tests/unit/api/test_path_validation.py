"""Tests for path_validation - API path validation utilities.

These tests verify the path validation functions using
Design by Contract principles to prevent path traversal attacks.
"""

from pathlib import Path

import pytest
from fastapi import HTTPException


class TestValidateModelPathContract:
    """Design by Contract tests for validate_model_path function.

    Preconditions:
    - model_path must be a string representing a relative path
    - model_path must not contain '..' (parent directory references)
    - model_path must not be an absolute path

    Postconditions:
    - Returns a string path that exists within allowed directories
    - Raises HTTPException with 400 for invalid paths
    - Raises HTTPException with 404 for non-existent files
    """

    def test_returns_string(self, tmp_path) -> None:
        """Postcondition: Returns a string when path is valid."""
        import src.api.utils.path_validation as pv
        from src.api.utils.path_validation import validate_model_path

        # Create a test file in a mocked allowed directory
        test_file = tmp_path / "test_model.xml"
        test_file.write_text("<model/>")

        original_dirs = pv.ALLOWED_MODEL_DIRS
        pv.ALLOWED_MODEL_DIRS = [tmp_path]
        try:
            result = validate_model_path("test_model.xml")
            assert isinstance(result, str)
        finally:
            pv.ALLOWED_MODEL_DIRS = original_dirs

    def test_rejects_absolute_path(self) -> None:
        """Precondition: Absolute paths must be rejected."""
        import sys

        from src.api.utils.path_validation import validate_model_path

        # Use platform-appropriate absolute path
        if sys.platform == "win32":
            absolute_path = "C:\\absolute\\path\\to\\model.xml"
        else:
            absolute_path = "/absolute/path/to/model.xml"

        with pytest.raises(HTTPException) as exc_info:
            validate_model_path(absolute_path)

        assert exc_info.value.status_code == 400
        assert "absolute" in exc_info.value.detail.lower()

    def test_rejects_parent_directory_traversal(self) -> None:
        """Precondition: Parent directory references must be rejected."""
        from src.api.utils.path_validation import validate_model_path

        with pytest.raises(HTTPException) as exc_info:
            validate_model_path("../../../etc/passwd")

        assert exc_info.value.status_code == 400
        assert "parent directory" in exc_info.value.detail.lower()

    def test_rejects_invalid_path_type(self) -> None:
        """Precondition: Invalid path types must be rejected."""
        from src.api.utils.path_validation import validate_model_path

        with pytest.raises(HTTPException) as exc_info:
            validate_model_path(None)  # type: ignore

        assert exc_info.value.status_code == 400


class TestAllowedModelDirs:
    """Tests for ALLOWED_MODEL_DIRS configuration."""

    def test_allowed_dirs_are_resolved_paths(self) -> None:
        """Test that allowed directories are resolved (absolute) paths."""
        from src.api.utils.path_validation import ALLOWED_MODEL_DIRS

        for allowed_dir in ALLOWED_MODEL_DIRS:
            assert allowed_dir.is_absolute()

    def test_allowed_dirs_are_path_objects(self) -> None:
        """Test that allowed directories are Path objects."""
        from src.api.utils.path_validation import ALLOWED_MODEL_DIRS

        for allowed_dir in ALLOWED_MODEL_DIRS:
            assert isinstance(allowed_dir, Path)
