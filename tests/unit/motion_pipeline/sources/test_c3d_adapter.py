"""Tests for the C3D adapter (ezc3d optional dependency)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_rust_load_surfaces_events_in_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c3d_file = tmp_path / "events.c3d"
    c3d_file.write_bytes(b"synthetic")

    fake_rust = SimpleNamespace(
        parse_c3d=lambda _path: {
            "labels": ["M01"],
            "positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "n_frames": 1,
            "n_markers": 1,
            "fps": 100.0,
            "units": "m",
            "events": [
                {"label": "FootStrike", "context": "Left", "time_s": 0.5},
                {"label": "ToeOff", "context": "Right", "time_s": 1.25},
            ],
        }
    )

    monkeypatch.setattr(_mod, "_rust_io", fake_rust)
    monkeypatch.setattr(_mod, "_HAS_RUST", True)

    trajectory = C3DAdapter()._load_via_rust(c3d_file, None)

    assert trajectory.metadata["events"] == [
        {"label": "FootStrike", "context": "Left", "time_s": 0.5},
        {"label": "ToeOff", "context": "Right", "time_s": 1.25},
    ]
