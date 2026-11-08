"""Tests for SHOW-A10 evidence and summary deliverable (#14735)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.tools.native_viewer_export.key_frames import COLORS, GREEN_TARGET

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

EVIDENCE_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "development"
    / "full_body_models"
    / "evidence"
    / "show_a10"
)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_show_a10_summary_structure() -> None:
    summary_path = EVIDENCE_DIR / "SUMMARY.md"
    assert summary_path.is_file(), f"SUMMARY.md missing at {summary_path}"
    content = summary_path.read_text(encoding="utf-8")

    assert "SHOW-A10" in content
    assert "mujoco_driver" in content
    assert "4bd0ae5099bca6f51c26e7bd0b356d2a6176b0c7" in content
    assert "REJECTED (G3)" in content
    assert "fa7a9b7075" in content
    assert "## Outputs (SHA-256)" in content
    assert "## Inputs (SHA-256)" in content
    assert "## Acceptance Comparison vs 11-04 Baseline (`fa7a9b7075`)" in content
    assert "## Visual Artifacts" in content
    assert "## Reproduction Command" in content


def test_show_a10_output_pngs_valid_and_hashes_match() -> None:
    summary_content = (EVIDENCE_DIR / "SUMMARY.md").read_text(encoding="utf-8")

    expected_files = [
        "down_the_line_address_1280.png",
        "down_the_line_impact_1280.png",
        "down_the_line_top_1280.png",
        "face_on_address_1280.png",
        "face_on_impact_1280.png",
        "face_on_top_1280.png",
        "side_by_side_address_dtl.png",
        "side_by_side_address_face_on.png",
        "side_by_side_top_dtl.png",
        "side_by_side_top_face_on.png",
    ]

    for fname in expected_files:
        png_path = EVIDENCE_DIR / fname
        assert png_path.is_file(), f"Expected output {fname} not found"
        # Verify valid PNG header
        header = png_path.read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n", f"{fname} is not a valid PNG"

        # Verify image opens and has dimensions
        img = Image.open(png_path)
        assert img.width > 0 and img.height > 0

        # Verify hash matches SUMMARY.md
        digest = _sha256(png_path)
        assert f"`{fname}` | `{digest}`" in summary_content, (
            f"Hash for {fname} ({digest}) not in SUMMARY.md"
        )


def test_show_a10_face_on_has_grip_clock_inset() -> None:
    fo_addr = Image.open(EVIDENCE_DIR / "face_on_address_1280.png")
    arr = np.asarray(fo_addr)
    # Clock is at bottom right: center around w - 110, h - 120
    # Check green tick (60, 220, 60)
    has_green = np.any((arr[:, :, 1] > 200) & (arr[:, :, 0] < 100))
    assert has_green, "face_on_address should contain green target tick"

    # Check red lead hand (255, 60, 60)
    has_red = np.any((arr[:, :, 0] > 200) & (arr[:, :, 2] < 100))
    assert has_red, "face_on_address should contain red lead needle"

    # Check blue trail hand (80, 120, 255)
    has_blue = np.any((arr[:, :, 2] > 200) & (arr[:, :, 0] < 120))
    assert has_blue, "face_on_address should contain blue trail needle"


def test_show_a10_side_by_side_composition() -> None:
    sbs = Image.open(EVIDENCE_DIR / "side_by_side_address_face_on.png")
    # Base (720x720) + Candidate (720x720) = 1440x720
    assert sbs.size == (1440, 720)
