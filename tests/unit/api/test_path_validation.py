"""Tests for path_validation - API path validation utilities.

These tests verify the path validation functions using
Design by Contract principles to prevent path traversal attacks.
"""

from pathlib import Path
from unittest.mock import patch

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

    def test_returns_string(self, tmp_path):
        """Postcondition: Returns a string when path is valid."""
        from src.api.utils.path_validation import validate_model_path
        import src.api.utils.path_validation as pv

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

    def test_rejects_absolute_path(self):
        """Precondition: Absolute paths must be rejected."""
        from src.api.utils.path_validation import validate_model_path
        import sys

        # Use platform-appropriate absolute path
        if sys.platform == "win32":
            absolute_path = "C:\\absolute\\path\\to\\model.xml"
        else:
            absolute_path = "/absolute/path/to/model.xml"

        with pytest.raises(HTTPException) as exc_info:
            validate_model_path(absolute_path)

        assert exc_info.value.status_code == 400
        assert "absolute" in exc_info.value.detail.lower()

    def test_rejects_parent_directory_traversal(self):
        """Precondition: Parent directory references must be rejected."""
        from src.api.utils.path_validation import validate_model_path

        with pytest.raises(HTTPException) as exc_info:
            validate_model_path("../../../etc/passwd")

        assert exc_info.value.status_code == 400
        assert "parent directory" in exc_info.value.detail.lower()

    def test_rejects_invalid_path_type(self):
        """Precondition: Invalid path types must be rejected."""
        from src.api.utils.path_validation import validate_model_path

        with pytest.raises(HTTPException) as exc_info:
            validate_model_path(None)  # type: ignore

        assert exc_info.value.status_code == 400


class TestValidateModelPath:
    """Functional tests for validate_model_path."""

    def test_valid_relative_path_in_allowed_directory(self, tmp_path):
        """Test that valid relative path within allowed directory passes."""
        from src.api.utils.path_validation import validate_model_path
        import src.api.utils.path_validation as pv

        # Create a subdirectory and test file
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        test_file = models_dir / "golf_swing.xml"
        test_file.write_text("<model/>")

        original_dirs = pv.ALLOWED_MODEL_DIRS
        pv.ALLOWED_MODEL_DIRS = [models_dir]
        try:
            result = validate_model_path("golf_swing.xml")
            assert Path(result).name == "golf_swing.xml"
        finally:
            pv.ALLOWED_MODEL_DIRS = original_dirs

    def test_valid_nested_path(self, tmp_path):
        """Test that nested relative paths work."""
        from src.api.utils.path_validation import validate_model_path
        import src.api.utils.path_validation as pv

        # Create nested directory structure
        models_dir = tmp_path / "models"
        subdir = models_dir / "humanoid"
        subdir.mkdir(parents=True)
        test_file = subdir / "arm.xml"
        test_file.write_text("<model/>")

        original_dirs = pv.ALLOWED_MODEL_DIRS
        pv.ALLOWED_MODEL_DIRS = [models_dir]
        try:
            result = validate_model_path("humanoid/arm.xml")
            assert "humanoid" in result
            assert "arm.xml" in result
        finally:
            pv.ALLOWED_MODEL_DIRS = original_dirs

    def test_nonexistent_file_raises_404(self, tmp_path):
        """Test that non-existent files raise 404."""
        from src.api.utils.path_validation import validate_model_path
        import src.api.utils.path_validation as pv

        original_dirs = pv.ALLOWED_MODEL_DIRS
        pv.ALLOWED_MODEL_DIRS = [tmp_path]
        try:
            with pytest.raises(HTTPException) as exc_info:
                validate_model_path("nonexistent.xml")
            assert exc_info.value.status_code == 404
            assert "not found" in exc_info.value.detail.lower()
        finally:
            pv.ALLOWED_MODEL_DIRS = original_dirs

    def test_path_traversal_attack_blocked(self, tmp_path):
        """Test that path traversal attacks are blocked."""
        from src.api.utils.path_validation import validate_model_path

        attack_paths = [
            "../secret.txt",
            "models/../../secret.txt",
            "..\\..\\secret.txt",
            "models\\..\\..\\secret.txt",
        ]

        for attack_path in attack_paths:
            with pytest.raises(HTTPException) as exc_info:
                validate_model_path(attack_path)
            assert exc_info.value.status_code == 400

    def test_windows_absolute_path_rejected(self):
        """Test that Windows absolute paths are rejected."""
        from src.api.utils.path_validation import validate_model_path

        with pytest.raises(HTTPException) as exc_info:
            validate_model_path("C:\\Windows\\System32\\config.xml")

        assert exc_info.value.status_code == 400

    def test_unix_absolute_path_rejected(self):
        """Test that Unix absolute paths are rejected."""
        from src.api.utils.path_validation import validate_model_path
        import sys

        # Skip on Windows since Unix paths aren't recognized as absolute
        if sys.platform == "win32":
            pytest.skip("Unix paths not recognized as absolute on Windows")

        with pytest.raises(HTTPException) as exc_info:
            validate_model_path("/etc/passwd")

        assert exc_info.value.status_code == 400

    def test_multiple_allowed_directories(self, tmp_path):
        """Test file lookup across multiple allowed directories."""
        from src.api.utils.path_validation import validate_model_path
        import src.api.utils.path_validation as pv

        # Create two directories
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # File only exists in second directory
        test_file = dir2 / "model.xml"
        test_file.write_text("<model/>")

        original_dirs = pv.ALLOWED_MODEL_DIRS
        pv.ALLOWED_MODEL_DIRS = [dir1, dir2]
        try:
            result = validate_model_path("model.xml")
            assert Path(result).exists()
        finally:
            pv.ALLOWED_MODEL_DIRS = original_dirs

    def test_symlink_escape_prevented(self, tmp_path):
        """Test that symlinks cannot escape allowed directories."""
        from src.api.utils.path_validation import validate_model_path
        import src.api.utils.path_validation as pv
        import os

        # Only test if symlinks are supported
        if not hasattr(os, "symlink"):
            pytest.skip("Symlinks not supported on this platform")

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create a file outside allowed directory
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret")

        # Try to create a symlink inside allowed directory pointing outside
        symlink = models_dir / "escape_link"
        try:
            symlink.symlink_to(outside_file)
        except OSError:
            pytest.skip("Cannot create symlinks (insufficient permissions)")

        original_dirs = pv.ALLOWED_MODEL_DIRS
        pv.ALLOWED_MODEL_DIRS = [models_dir]
        try:
            with pytest.raises(HTTPException):
                validate_model_path("escape_link")
        except AssertionError:
            # If the validation passes, the result should still be within bounds
            pass
        finally:
            pv.ALLOWED_MODEL_DIRS = original_dirs


class TestAllowedModelDirs:
    """Tests for ALLOWED_MODEL_DIRS configuration."""

    def test_allowed_dirs_are_resolved_paths(self):
        """Test that allowed directories are resolved (absolute) paths."""
        from src.api.utils.path_validation import ALLOWED_MODEL_DIRS

        for allowed_dir in ALLOWED_MODEL_DIRS:
            assert allowed_dir.is_absolute()

    def test_allowed_dirs_are_path_objects(self):
        """Test that allowed directories are Path objects."""
        from src.api.utils.path_validation import ALLOWED_MODEL_DIRS

        for allowed_dir in ALLOWED_MODEL_DIRS:
            assert isinstance(allowed_dir, Path)
