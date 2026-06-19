"""Tests for src.api.utils.path_validation module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from src.api.utils.path_validation import (
    resolve_output_path,
    validate_model_path,
)


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

    def test_path_validation_rejects_parent_traversal(self) -> None:
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


class TestResolveOutputPath:
    """Tests for resolve_output_path — write-path containment guard (#7710)."""

    def test_returns_resolved_path_for_nonexistent_target_under_root(
        self, tmp_path: Path
    ) -> None:
        """A not-yet-existing file inside an allowed root is accepted."""
        target = tmp_path / "results" / "out.json"
        result = resolve_output_path(target, [tmp_path])
        assert result == target.resolve()
        # The target itself need not exist.
        assert not result.exists()

    def test_accepts_target_directly_in_allowed_root(self, tmp_path: Path) -> None:
        target = tmp_path / "out.csv"
        assert resolve_output_path(target, [tmp_path]) == target.resolve()

    def test_rejects_path_escaping_all_allowed_roots(self, tmp_path: Path) -> None:
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside" / "evil.json"
        with pytest.raises(HTTPException) as exc_info:
            resolve_output_path(outside, [allowed])
        assert exc_info.value.status_code == 400
        assert "escapes" in exc_info.value.detail.lower()

    def test_rejects_parent_traversal_out_of_root(self, tmp_path: Path) -> None:
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        escaping = allowed / ".." / "secret.json"
        with pytest.raises(HTTPException) as exc_info:
            resolve_output_path(escaping, [allowed])
        assert exc_info.value.status_code == 400

    def test_rejects_when_no_allowed_dirs(self, tmp_path: Path) -> None:
        with pytest.raises(HTTPException) as exc_info:
            resolve_output_path(tmp_path / "out.json", [])
        assert exc_info.value.status_code == 400

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="symlink creation requires privileges on Windows",
    )
    def test_rejects_symlink_traversal_of_existing_parent(self, tmp_path: Path) -> None:
        """A symlinked parent component is rejected as defense-in-depth even
        when the resolved target still lands inside the allowed root (#6926).
        ``allowed/link`` -> ``allowed/real`` keeps containment intact, so only
        the symlink-traversal guard can reject it."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        real = allowed / "real"
        real.mkdir()
        link = allowed / "link"
        link.symlink_to(real, target_is_directory=True)
        target = link / "out.json"
        with pytest.raises(HTTPException) as exc_info:
            resolve_output_path(target, [allowed])
        assert exc_info.value.status_code == 400
        assert "symlink" in exc_info.value.detail.lower()
