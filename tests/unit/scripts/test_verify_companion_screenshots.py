"""Unit tests for screenshot manifest byte and dimension verification (#9191)."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from scripts.verify_companion_screenshots import (
    main,
    verify_screenshot_records,
)

pytestmark = pytest.mark.unit


def _make_png_bytes(width: int = 3, height: int = 2) -> bytes:
    """Generate a valid, minimal PNG image in pure Python via struct/zlib."""
    signature = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk: width (4), height (4), bit depth (1), color type (1 = RGB: 2),
    # compression (1), filter (1), interlace (1)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    # IDAT chunk: filter byte (0 = none) followed by RGB pixels per row
    raw_rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    compressed = zlib.compress(raw_rows)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed))
    idat_chunk = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc

    # IEND chunk
    iend_crc = struct.pack(">I", zlib.crc32(b"IEND"))
    iend_chunk = struct.pack(">I", 0) + b"IEND" + iend_crc

    return signature + ihdr_chunk + idat_chunk + iend_chunk


def _valid_captured_record(
    png_path: Path, repo_root: Path, width: int = 3, height: int = 2
) -> dict[str, Any]:
    if png_path.is_file():
        data = png_path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
    else:
        sha256 = "0" * 64
    try:
        rel_path = png_path.relative_to(repo_root).as_posix()
    except ValueError:
        rel_path = str(png_path)
    return {
        "id": "prog-primary",
        "program_id": "prog",
        "status": "captured",
        "path": rel_path,
        "sha256": sha256,
        "width": width,
        "height": height,
        "viewport": {"width": width, "height": height},
        "theme": "dark",
        "capture_workflow_id": "wf",
        "capture_step_id": "step",
        "alt_text": "A descriptive screenshot alt text",
        "caption": "A screenshot caption",
        "visible_limitations": [],
        "artifact_class": "illustrative",
        "reason": None,
    }


def _valid_pending_record() -> dict[str, Any]:
    return {
        "id": "prog-primary",
        "program_id": "prog",
        "status": "pending",
        "path": None,
        "sha256": None,
        "width": None,
        "height": None,
        "viewport": None,
        "theme": None,
        "capture_workflow_id": None,
        "capture_step_id": None,
        "alt_text": None,
        "caption": None,
        "visible_limitations": [],
        "artifact_class": "illustrative",
        "reason": "No governed capture exists at this commit (#9191).",
    }


def test_correct_captured_record_passes(tmp_path: Path) -> None:
    png_file = tmp_path / "screenshot.png"
    png_file.write_bytes(_make_png_bytes(3, 2))
    payload = {"records": [_valid_captured_record(png_file, tmp_path, 3, 2)]}
    violations = verify_screenshot_records(payload, tmp_path)
    assert violations == []


def test_correct_pending_record_passes(tmp_path: Path) -> None:
    payload = {"records": [_valid_pending_record()]}
    violations = verify_screenshot_records(payload, tmp_path)
    assert violations == []


def test_captured_wrong_hash_fails(tmp_path: Path) -> None:
    png_file = tmp_path / "screenshot.png"
    png_file.write_bytes(_make_png_bytes(3, 2))
    record = _valid_captured_record(png_file, tmp_path, 3, 2)
    record["sha256"] = "0" * 64
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "sha256 mismatch" in violations[0]


def test_captured_wrong_width_fails(tmp_path: Path) -> None:
    png_file = tmp_path / "screenshot.png"
    png_file.write_bytes(_make_png_bytes(3, 2))
    record = _valid_captured_record(png_file, tmp_path, 3, 2)
    record["width"] = 999
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "width mismatch" in violations[0]


def test_captured_wrong_height_fails(tmp_path: Path) -> None:
    png_file = tmp_path / "screenshot.png"
    png_file.write_bytes(_make_png_bytes(3, 2))
    record = _valid_captured_record(png_file, tmp_path, 3, 2)
    record["height"] = 999
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "height mismatch" in violations[0]


def test_captured_missing_file_fails(tmp_path: Path) -> None:
    record = _valid_captured_record(tmp_path / "missing.png", tmp_path, 3, 2)
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "file does not exist" in violations[0]


def test_captured_parent_relative_path_fails(tmp_path: Path) -> None:
    record = _valid_captured_record(tmp_path / "sample.png", tmp_path, 3, 2)
    record["path"] = "../escaped.png"
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "resolves outside repo_root" in violations[0]


def test_captured_absolute_path_fails(tmp_path: Path) -> None:
    record = _valid_captured_record(tmp_path / "sample.png", tmp_path, 3, 2)
    record["path"] = "/etc/screenshot.png"
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "must not be absolute" in violations[0]


def test_captured_non_png_bytes_fails(tmp_path: Path) -> None:
    bad_file = tmp_path / "not_png.png"
    bad_file.write_bytes(b"NOT A REAL PNG FILE")
    record = _valid_captured_record(bad_file, tmp_path, 3, 2)
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "unsupported image format" in violations[0]


def test_captured_empty_alt_text_fails(tmp_path: Path) -> None:
    png_file = tmp_path / "screenshot.png"
    png_file.write_bytes(_make_png_bytes(3, 2))
    record = _valid_captured_record(png_file, tmp_path, 3, 2)
    record["alt_text"] = ""
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "alt_text must be a non-empty string" in violations[0]


def test_pending_with_non_null_sha256_fails(tmp_path: Path) -> None:
    record = _valid_pending_record()
    record["sha256"] = "a" * 64
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "pending screenshot sha256 must be null" in violations[0]


def test_pending_with_empty_reason_fails(tmp_path: Path) -> None:
    record = _valid_pending_record()
    record["reason"] = "   "
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "pending screenshot reason must be a non-empty string" in violations[0]


def test_unknown_status_fails(tmp_path: Path) -> None:
    record = _valid_pending_record()
    record["status"] = "unknown_status"
    violations = verify_screenshot_records({"records": [record]}, tmp_path)
    assert len(violations) == 1
    assert "unknown status 'unknown_status'" in violations[0]


def test_failure_messages_are_distinct(tmp_path: Path) -> None:
    png_file = tmp_path / "valid.png"
    png_file.write_bytes(_make_png_bytes(3, 2))
    bad_file = tmp_path / "bad.png"
    bad_file.write_bytes(b"not png")

    rec_wrong_hash = _valid_captured_record(png_file, tmp_path, 3, 2)
    rec_wrong_hash["sha256"] = "1" * 64

    rec_wrong_width = _valid_captured_record(png_file, tmp_path, 3, 2)
    rec_wrong_width["width"] = 100

    rec_wrong_height = _valid_captured_record(png_file, tmp_path, 3, 2)
    rec_wrong_height["height"] = 100

    rec_missing_file = _valid_captured_record(
        tmp_path / "does_not_exist.png", tmp_path, 3, 2
    )

    rec_escape_path = _valid_captured_record(png_file, tmp_path, 3, 2)
    rec_escape_path["path"] = "../out.png"

    rec_abs_path = _valid_captured_record(png_file, tmp_path, 3, 2)
    rec_abs_path["path"] = "/var/screenshot.png"

    rec_bad_format = _valid_captured_record(bad_file, tmp_path, 3, 2)

    rec_empty_alt = _valid_captured_record(png_file, tmp_path, 3, 2)
    rec_empty_alt["alt_text"] = ""

    rec_pending_sha = _valid_pending_record()
    rec_pending_sha["sha256"] = "e" * 64

    rec_pending_no_reason = _valid_pending_record()
    rec_pending_no_reason["reason"] = ""

    rec_unknown_status = _valid_pending_record()
    rec_unknown_status["status"] = "other"

    cases = [
        rec_wrong_hash,
        rec_wrong_width,
        rec_wrong_height,
        rec_missing_file,
        rec_escape_path,
        rec_abs_path,
        rec_bad_format,
        rec_empty_alt,
        rec_pending_sha,
        rec_pending_no_reason,
        rec_unknown_status,
    ]
    messages = []
    for case in cases:
        v = verify_screenshot_records({"records": [case]}, tmp_path)
        assert len(v) == 1
        messages.append(v[0])

    assert len(set(messages)) == len(messages)


def test_dbc_preconditions_raise_type_error(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="payload must be a Mapping"):
        verify_screenshot_records("not a mapping", tmp_path)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="records must be a list"):
        verify_screenshot_records({"records": "not a list"}, tmp_path)


def test_main_cli_exit_codes_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Exit code 0 on valid payload
    good_manifest = tmp_path / "good.json"
    good_manifest.write_text(
        json.dumps({"records": [_valid_pending_record()]}), encoding="utf-8"
    )
    stderr_buf = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr_buf)
    rc = main(["--manifest", str(good_manifest), "--repo-root", str(tmp_path)])
    assert rc == 0
    assert stderr_buf.getvalue() == ""

    # 2. Exit code 1 on violations (printed one per line to stderr)
    bad_manifest = tmp_path / "bad.json"
    bad_rec = _valid_pending_record()
    bad_rec["sha256"] = "non-null"
    bad_manifest.write_text(json.dumps({"records": [bad_rec]}), encoding="utf-8")
    stderr_buf = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr_buf)
    rc = main(["--manifest", str(bad_manifest), "--repo-root", str(tmp_path)])
    assert rc == 1
    assert "pending screenshot sha256 must be null" in stderr_buf.getvalue()

    # 3. Exit code 2 on unreadable file
    missing_manifest = tmp_path / "nonexistent.json"
    stderr_buf = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr_buf)
    rc = main(["--manifest", str(missing_manifest), "--repo-root", str(tmp_path)])
    assert rc == 2

    # 4. Exit code 2 on invalid JSON syntax
    corrupt_manifest = tmp_path / "corrupt.json"
    corrupt_manifest.write_text("{this is not json", encoding="utf-8")
    stderr_buf = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr_buf)
    rc = main(["--manifest", str(corrupt_manifest), "--repo-root", str(tmp_path)])
    assert rc == 2

    # 5. Exit code 2 on invalid JSON schema / DbC TypeError (e.g. JSON array)
    array_manifest = tmp_path / "array.json"
    array_manifest.write_text("[1, 2, 3]", encoding="utf-8")
    stderr_buf = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr_buf)
    rc = main(["--manifest", str(array_manifest), "--repo-root", str(tmp_path)])
    assert rc == 2
