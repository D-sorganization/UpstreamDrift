"""Tests for load_source format-hint validation (#6930)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.loader import load_source


def test_load_source_unknown_format_hint_raises(tmp_path: Path) -> None:
    """A non-auto hint that matches no adapter must raise ValueError.

    This stops the silent fallthrough to content auto-detection (#6930).
    """
    src = tmp_path / "capture.dat"
    src.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(ValueError, match="Unknown source format"):
        load_source(src, format_hint="totally_not_a_format")


def test_load_source_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.c3d"
    with pytest.raises(FileNotFoundError):
        load_source(missing, format_hint="c3d")
