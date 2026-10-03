"""Smoke and contract tests for scripts/render_force_overlay_gallery.py (FTO-30, #11315)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = [pytest.mark.unit]


def test_render_gallery_synthetic_smoke(tmp_path: Path) -> None:
    """Verify that render_force_overlay_gallery generates index.html and manifest.json in synthetic mode."""
    from scripts.render_force_overlay_gallery import render_gallery

    manifest_path = render_gallery(output_dir=tmp_path, synthetic_only=True)

    assert manifest_path.is_file(), f"Expected manifest file at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "schema_version" in manifest
    assert "generated_at" in manifest
    assert "git_commit" in manifest
    assert "engines" in manifest
    assert isinstance(manifest["engines"], dict)
    assert "media" in manifest
    assert isinstance(manifest["media"], list)
    assert len(manifest["media"]) > 0

    index_html = tmp_path / "index.html"
    assert index_html.is_file(), f"Expected index.html at {index_html}"
    html_text = index_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "Force / Torque Overlay Gallery" in html_text


def test_gallery_cli_invocation(tmp_path: Path) -> None:
    """Verify scripts/render_force_overlay_gallery.py runs cleanly as a CLI entry point."""
    out_dir = tmp_path / "cli_out"
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "render_force_overlay_gallery.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--out",
        str(out_dir),
        "--synthetic-only",
    ]

    result = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Script failed with code {result.returncode}:\nStdout: {result.stdout}\nStderr: {result.stderr}"
    )

    assert (out_dir / "index.html").is_file()
    assert (out_dir / "manifest.json").is_file()
