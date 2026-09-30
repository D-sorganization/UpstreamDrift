"""Governed companion screenshot registry and asset verification tests (#9191)."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS_SCHEMA_PATH = (
    REPO_ROOT / "docs/api/contracts/upstreamdrift-companion-screenshots-v1.schema.json"
)
MANIFEST_SCHEMA_PATH = (
    REPO_ROOT / "docs/api/contracts/upstreamdrift-companion-v1.schema.json"
)
REGISTRY_PATH = REPO_ROOT / "scripts/config/companion_screenshots.v1.json"

pytestmark = pytest.mark.unit


def _make_png_bytes(
    width: int = 4, height: int = 4, color: tuple[int, int, int] = (255, 0, 0)
) -> bytes:
    """Generate valid PNG bytes with exact dimensions."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    raw_row = b"\x00" + bytes(color) * width
    raw_data = raw_row * height
    compressed = zlib.compress(raw_data)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed))
    idat_chunk = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc

    iend_crc = struct.pack(">I", zlib.crc32(b"IEND"))
    iend_chunk = struct.pack(">I", 0) + b"IEND" + iend_crc

    return signature + ihdr_chunk + idat_chunk + iend_chunk


def _screenshots_module():
    from scripts import companion_screenshots

    return companion_screenshots


def test_registry_file_exists_and_is_valid_json() -> None:
    assert REGISTRY_PATH.is_file(), (
        "scripts/config/companion_screenshots.v1.json must exist"
    )
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert data["registry_id"] == "upstreamdrift-companion-screenshots"
    assert data["version"] == "1.0.0"
    assert isinstance(data["records"], list)
    assert len(data["records"]) > 0


def test_parse_registry_produces_valid_payloads() -> None:
    from scripts import companion_catalog

    mod = _screenshots_module()
    cat = companion_catalog.build_catalog(REPO_ROOT, require_clean=False)
    program_ids = {p["id"] for p in cat["programs"]}
    parsed = mod.load_and_parse_registry(
        repo_root=REPO_ROOT,
        source_commit="1" * 40,
        program_ids=program_ids,
        workflow_ids={
            "reference-simulation-plot",
            "companion-export",
            "program-catalog-export",
        },
    )
    assert "records" in parsed
    assert "summary" in parsed
    assert parsed["summary"]["screenshot_records"] == 76
    assert parsed["summary"]["captured_screenshot_records"] == 6
    assert parsed["summary"]["pending_screenshot_records"] == 70


def test_parse_registry_rejects_dangling_program_id() -> None:
    mod = _screenshots_module()
    raw = {
        "registry_id": "upstreamdrift-companion-screenshots",
        "version": "1.0.0",
        "records": [
            {
                "id": "unknown_prog-primary",
                "program_id": "unknown_prog",
                "status": "pending",
                "path": None,
                "sha256": None,
                "width": None,
                "height": None,
                "viewport": None,
                "theme": None,
                "capture_workflow_id": None,
                "capture_step_id": None,
                "capture_environment": None,
                "alt_text": None,
                "caption": None,
                "visible_limitations": [],
                "artifact_class": "illustrative",
                "reason": "Pending capture",
            }
        ],
    }
    with pytest.raises(mod.ScreenshotContractError, match="unknown_prog"):
        mod.parse_registry(
            json.dumps(raw).encode("utf-8"),
            repo_root=REPO_ROOT,
            source_commit="1" * 40,
            program_ids={"pendulum_simulator"},
            workflow_ids={"reference-simulation-plot"},
        )


def test_parse_registry_rejects_dangling_workflow_id() -> None:
    mod = _screenshots_module()
    raw = {
        "registry_id": "upstreamdrift-companion-screenshots",
        "version": "1.0.0",
        "records": [
            {
                "id": "pendulum_simulator-primary",
                "program_id": "pendulum_simulator",
                "status": "captured",
                "path": "docs/screenshots/test.png",
                "sha256": "0" * 64,
                "width": 100,
                "height": 100,
                "viewport": {"width": 100, "height": 100},
                "theme": "dark",
                "capture_workflow_id": "nonexistent-workflow",
                "capture_step_id": "step-1",
                "capture_environment": "test-env",
                "alt_text": "Sample alt text",
                "caption": "Sample caption",
                "visible_limitations": [],
                "artifact_class": "illustrative",
                "reason": None,
            }
        ],
    }
    with pytest.raises(mod.ScreenshotContractError, match="nonexistent-workflow"):
        mod.parse_registry(
            json.dumps(raw).encode("utf-8"),
            repo_root=REPO_ROOT,
            source_commit="1" * 40,
            program_ids={"pendulum_simulator"},
            workflow_ids={"reference-simulation-plot"},
        )


def test_captured_assets_exist_and_match_declared_hashes_and_dimensions() -> None:
    from scripts import companion_catalog

    mod = _screenshots_module()
    cat = companion_catalog.build_catalog(REPO_ROOT, require_clean=False)
    program_ids = {p["id"] for p in cat["programs"]}
    parsed = mod.load_and_parse_registry(
        repo_root=REPO_ROOT,
        source_commit="1" * 40,
        program_ids=program_ids,
        workflow_ids={
            "reference-simulation-plot",
            "companion-export",
            "program-catalog-export",
        },
    )
    captured = [r for r in parsed["records"] if r["status"] == "captured"]
    assert len(captured) == 6, "Must have exactly 6 representative captured screenshots"

    for record in captured:
        asset_path = REPO_ROOT / record["path"]
        assert asset_path.is_file(), f"Asset {record['path']} must exist"
        data = asset_path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        assert actual_hash == record["sha256"], f"SHA256 mismatch for {record['id']}"

        # Check dimensions via PNG header
        assert len(data) >= 24
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        w, h = struct.unpack(">II", data[16:24])
        assert w == record["width"]
        assert h == record["height"]
        assert record["viewport"]["width"] == w or record["viewport"]["width"] >= w
        assert record["viewport"]["height"] == h or record["viewport"]["height"] >= h
        assert record["alt_text"] and len(record["alt_text"].strip()) > 0
        assert record["caption"] and len(record["caption"].strip()) > 0
        assert record["theme"] in ("light", "dark")
        assert record["artifact_class"] in (
            "illustrative",
            "raw_plot",
            "qualification_evidence",
        )
        assert (
            record["capture_environment"]
            and len(record["capture_environment"].strip()) > 0
        )
