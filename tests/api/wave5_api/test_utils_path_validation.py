"""Tests for src/api/utils/path_validation.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from src.api.utils import path_validation as pv

pytestmark = pytest.mark.unit


def test_absolute_posix_path_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        pv.validate_model_path("/etc/passwd")
    assert exc.value.status_code == 400
    assert "absolute" in exc.value.detail.lower()


def test_absolute_windows_path_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        pv.validate_model_path("C:/Windows/system32")
    assert exc.value.status_code == 400


def test_parent_traversal_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        pv.validate_model_path("../../../etc/passwd")
    assert exc.value.status_code == 400
    assert "parent" in exc.value.detail.lower()


def test_parent_traversal_in_parts_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        pv.validate_model_path("foo/../bar")
    assert exc.value.status_code == 400


def test_nonexistent_path_returns_404() -> None:
    with pytest.raises(HTTPException) as exc:
        pv.validate_model_path("definitely_not_a_real_file_xyz.urdf")
    assert exc.value.status_code == 404


def test_resolve_contained_path_outside_raises_404(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    candidate = other / "file.txt"
    candidate.write_text("x")
    with pytest.raises(HTTPException) as exc:
        pv.resolve_contained_path(candidate, [allowed])
    assert exc.value.status_code == 404


def test_resolve_contained_path_inside_returns_resolved(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    f = allowed / "model.urdf"
    f.write_text("<robot/>")
    result = pv.resolve_contained_path(f, [allowed])
    assert result == f.resolve()


def test_resolve_contained_path_not_exists_skips(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    candidate = allowed / "missing.urdf"  # exists() returns False
    with pytest.raises(HTTPException) as exc:
        pv.resolve_contained_path(candidate, [allowed])
    assert exc.value.status_code == 404


def test_validate_model_path_finds_existing_in_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_root = tmp_path / "models"
    fake_root.mkdir()
    target = fake_root / "swing.urdf"
    target.write_text("<robot/>")
    monkeypatch.setattr(pv, "ALLOWED_MODEL_DIRS", [fake_root.resolve()])
    out = pv.validate_model_path("swing.urdf")
    assert "swing.urdf" in out
