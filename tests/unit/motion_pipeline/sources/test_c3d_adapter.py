"""Tests for the C3D adapter (ezc3d optional dependency)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources import c3d_adapter as _mod
from src.shared.python.motion_pipeline.sources.c3d_adapter import C3DAdapter

_HAS_EZC3D = _mod._HAS_EZC3D


@pytest.mark.skipif(not _HAS_EZC3D, reason="ezc3d not installed")
def test_c3d_supports_only_when_ezc3d(tmp_path: Path) -> None:
    p = tmp_path / "x.c3d"
    p.write_bytes(b"\x00")
    # supports() returns True only because ezc3d is installed and the file
    # exists with the right extension. We don't open it here; that is the
    # job of load().
    assert C3DAdapter.supports(p) is True
