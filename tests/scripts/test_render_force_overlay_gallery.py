"""Tests for the force overlay gallery generation script (FTO-30, #11315)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_gallery_smoke_generates_manifest_and_index(tmp_path: Path) -> None:
    """Gallery generation in headless mode creates index.html and manifest.json."""
    from scripts.render_force_overlay_gallery import render_gallery

    manifest_path, index_path = render_gallery(
        out_dir=tmp_path,
        synthetic_only=True,
    )

    assert index_path.is_file()
    assert manifest_path.is_file()

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert "schema_version" in manifest
    assert "generated_at" in manifest
    assert "entries" in manifest
    assert isinstance(manifest["entries"], list)
    assert len(manifest["entries"]) > 0

    # Verify each entry has required DbC fields
    for entry in manifest["entries"]:
        assert "engine" in entry
        assert "status" in entry
        assert entry["status"] in ("rendered", "skipped")
        if entry["status"] == "rendered":
            assert "image_file" in entry
            assert (tmp_path / entry["image_file"]).is_file()
            assert "receipt" in entry
        else:
            assert "skip_reason" in entry

    index_content = index_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in index_content
    assert "Force Overlay Gallery" in index_content
