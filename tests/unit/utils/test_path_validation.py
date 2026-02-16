"""Tests for src.api.utils.path_validation module."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.utils.path_validation import validate_model_path


class TestValidateModelPath:
    """Tests for validate_model_path function."""

    def test_rejects_absolute_posix_path(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            validate_model_path("/etc/passwd")
        # On POSIX: 400 (absolute path rejected), on Windows: 404 (not found)
        assert exc_info.value.status_code in (400, 404)

    def test_rejects_absolute_windows_path(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            validate_model_path("C:\\Windows\\System32")
        assert exc_info.value.status_code == 400

    def test_rejects_parent_traversal(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            validate_model_path("../../../etc/passwd")
        assert exc_info.value.status_code == 400
        assert "parent directory" in exc_info.value.detail.lower()

    def test_rejects_embedded_parent_traversal(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            validate_model_path("models/../../secret.txt")
        assert exc_info.value.status_code == 400

    def test_rejects_invalid_path_type(self) -> None:
        """Path creation from None should raise 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_model_path(None)  # type: ignore[arg-type]
        assert exc_info.value.status_code == 400

    def test_nonexistent_relative_path_raises_404(self) -> None:
        """A valid relative path that doesn't exist should raise 404."""
        with pytest.raises(HTTPException) as exc_info:
            validate_model_path("nonexistent_model.urdf")
        assert exc_info.value.status_code == 404
