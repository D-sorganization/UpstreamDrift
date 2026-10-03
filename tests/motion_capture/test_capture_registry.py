"""Unit tests for the shared capture registry (#11162)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.motion_capture.capture_registry import (
    CaptureDataUnavailable,
    CaptureInfo,
    CaptureIntegrityError,
    UnknownCaptureError,
    capture_info,
    list_captures,
    require_capture,
    resolve_capture,
)

pytestmark = pytest.mark.unit


def test_list_captures() -> None:
    captures = list_captures()
    expected = [
        "capture-A",
        "capture-B",
        "capture-O",
        "club-workbook-main",
        "club-workbook-wiffle",
    ]
    for cid in expected:
        assert cid in captures


def test_capture_info_public() -> None:
    info_a = capture_info("capture-A")
    assert isinstance(info_a, CaptureInfo)
    assert info_a.id == "capture-A"
    assert info_a.where == "public"
    assert info_a.relative_path == "data/C3D_TA_Driver.c3d"
    assert (
        info_a.sha256
        == "cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d"
    )
    assert info_a.bytes == 402432
    assert info_a.sample_rate_hz == 360.0
    assert info_a.frame_count == 654

    info_b = capture_info("capture-B")
    assert info_b.id == "capture-B"
    assert info_b.where == "public"
    assert info_b.relative_path == "data/C3D_TA_Iron.c3d"
    assert (
        info_b.sha256
        == "00a70c1ec0a887c28f9c0397766904eb9e79dee9f6d2bbd0cbb0062445adf561"
    )
    assert info_b.bytes == 404480
    assert info_b.sample_rate_hz == 360.0
    assert info_b.frame_count == 657


def test_capture_info_private() -> None:
    info_o = capture_info("capture-O")
    assert info_o.id == "capture-O"
    assert info_o.where == "private"
    assert info_o.relative_path == "datasets/capture-O/capture-O_driver.c3d"
    assert (
        info_o.sha256
        == "2569659eff1e0b00294fd3c213d332cb609d8e37c31853f4bce9eb860e99b160"
    )
    assert info_o.bytes == 225280
    assert info_o.sample_rate_hz == 240.0
    assert info_o.frame_count == 367

    info_main = capture_info("club-workbook-main")
    assert info_main.id == "club-workbook-main"
    assert info_main.where == "private"
    assert info_main.relative_path == "club_workbooks/Club_Data/e91f760171f5.xlsx"
    assert (
        info_main.sha256
        == "5d9183e1d01ea7c6f9c162375dd6855c076ee26a96b76c90e59e9cf2679dde25"
    )
    assert info_main.bytes is None

    info_wiffle = capture_info("club-workbook-wiffle")
    assert info_wiffle.id == "club-workbook-wiffle"
    assert info_wiffle.where == "private"
    assert (
        info_wiffle.relative_path
        == "club_workbooks/Wiffle_ProV1_club_3D_data/d61999450688.xlsx"
    )
    assert (
        info_wiffle.sha256
        == "88d3eb31541d886031f6c5ad7c82493b0e3ee4e3f73a8652372277cc14d02234"
    )
    assert info_wiffle.bytes == 907474


def test_capture_info_unknown() -> None:
    with pytest.raises(UnknownCaptureError):
        capture_info("unknown-capture-id")


def test_resolve_capture_public() -> None:
    path_a = resolve_capture("capture-A")
    assert path_a.is_file()
    assert path_a.name == "C3D_TA_Driver.c3d"

    path_b = resolve_capture("capture-B")
    assert path_b.is_file()
    assert path_b.name == "C3D_TA_Iron.c3d"


def test_resolve_capture_unknown() -> None:
    with pytest.raises(UnknownCaptureError):
        resolve_capture("non-existent-id")


def test_resolve_capture_private_unavailable_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAPTURE_DATA_DIR", raising=False)
    with pytest.raises(CaptureDataUnavailable):
        resolve_capture("capture-O")

    with pytest.raises(CaptureDataUnavailable):
        resolve_capture("club-workbook-main")


def test_resolve_capture_private_unavailable_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTURE_DATA_DIR", str(tmp_path))
    with pytest.raises(CaptureDataUnavailable):
        resolve_capture("capture-O")


def test_resolve_capture_private_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Set up mock private capture-O file matching expected relative path and hash
    rel_path = Path("datasets/capture-O/capture-O_driver.c3d")
    target = tmp_path / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    # Expected sha256 for capture-O
    info = capture_info("capture-O")
    # Write dummy file with wrong hash first to test integrity error
    target.write_bytes(b"corrupted content")
    with pytest.raises(CaptureIntegrityError):
        resolve_capture("capture-O", data_dir=tmp_path)

    # Now write content matching sha256
    # For testing, we mock hashlib or create a test registry entry,
    # or write matching hash via a custom entry / mock
    with patch(
        "src.motion_capture.capture_registry._hash_file",
        return_value=info.sha256,
    ):
        resolved = resolve_capture("capture-O", data_dir=tmp_path)
        assert resolved == target


def test_require_capture_skips_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAPTURE_DATA_DIR", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_capture("capture-O")


def test_capture_info_video_kind_schema() -> None:
    """Capture registry supports video kind and defaults to c3d for legacy entries."""
    from src.motion_capture.capture_registry import register_capture

    # Legacy dictionary without kind defaults to c3d
    legacy_info = CaptureInfo.from_dict(
        {
            "id": "legacy-item",
            "where": "public",
            "relative_path": "path/test.c3d",
            "sha256": "0" * 64,
            "bytes": 100,
        }
    )
    assert legacy_info.kind == "c3d"

    # Video kind is preserved
    video_info = CaptureInfo(
        id="capture-O-video/cov-01",
        kind="video",
        where="private",
        relative_path="capture-O-video/originals/cov-01.mp4",
        sha256="1" * 64,
        bytes=1000,
        sample_rate_hz=30.0,
        frame_count=300,
    )
    assert video_info.kind == "video"
    roundtrip = CaptureInfo.from_dict(video_info.to_dict())
    assert roundtrip.kind == "video"
    assert roundtrip.id == "capture-O-video/cov-01"


def test_register_capture(tmp_path: Path) -> None:
    """register_capture persists and updates registry manifest correctly."""
    from src.motion_capture.capture_registry import register_capture

    repo_dir = tmp_path / "repo"
    data_dir = repo_dir / "data"
    data_dir.mkdir(parents=True)
    manifest_file = data_dir / "capture_registry.json"
    manifest_file.write_text("{}", encoding="utf-8")

    info = CaptureInfo(
        id="capture-O-video/cov-test",
        kind="video",
        where="private",
        relative_path="capture-O-video/originals/cov-test.mp4",
        sha256="2" * 64,
        bytes=500,
        sample_rate_hz=60.0,
        frame_count=180,
    )
    register_capture(info, repo_root=repo_dir)

    loaded = capture_info("capture-O-video/cov-test", repo_root=repo_dir)
    assert loaded.id == "capture-O-video/cov-test"
    assert loaded.kind == "video"
    assert loaded.sample_rate_hz == 60.0
