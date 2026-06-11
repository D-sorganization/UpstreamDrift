"""Tests for the C3D adapter (ezc3d optional dependency)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_pipeline.sources import c3d_adapter as _mod
from src.shared.python.motion_pipeline.sources.base import AdapterContractError
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
            "analog": {
                "labels": ["Fx", "Fy"],
                "units": ["N", "N"],
                "values": np.array([[10.0, 20.0]], dtype=np.float32),
                "n_frames": 1,
                "samples_per_frame": 1,
                "n_channels": 2,
                "rate": 100.0,
            },
            "force_platforms": [
                {
                    "type": 2,
                    "channels": [1, 2, 3, 4, 5, 6],
                    "corners": [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    "origin": [0.0, 0.0, -0.05],
                }
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
    assert trajectory.metadata["analog"] == {
        "labels": ["Fx", "Fy"],
        "units": ["N", "N"],
        "n_frames": 1,
        "samples_per_frame": 1,
        "n_channels": 2,
        "rate": 100.0,
    }
    assert trajectory.metadata["force_platforms"] == [
        {
            "type": 2,
            "channels": [1, 2, 3, 4, 5, 6],
            "corners": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "origin": [0.0, 0.0, -0.05],
        }
    ]


def test_c3d_ezc3d_load_converts_cm_to_meters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c3d_file = tmp_path / "centimeters.c3d"
    c3d_file.write_bytes(b"synthetic")
    points = np.array(
        [
            [[100.0], [200.0]],
            [[50.0], [75.0]],
            [[25.0], [125.0]],
            [[1.0], [1.0]],
        ]
    )
    fake_c3d = {
        "parameters": {
            "POINT": {
                "LABELS": {"value": ["M01", "M02"]},
                "RATE": {"value": [100.0]},
                "UNITS": {"value": ["cm"]},
            }
        },
        "data": {"points": points},
    }

    monkeypatch.setattr(_mod, "_HAS_RUST", False)
    monkeypatch.setattr(_mod, "_HAS_EZC3D", True)
    monkeypatch.setattr(_mod, "_HAS_C3D_BACKEND", True)
    monkeypatch.setattr(_mod, "_ezc3d", SimpleNamespace(c3d=lambda _path: fake_c3d))

    trajectory = C3DAdapter().load(c3d_file)

    first_marker = trajectory.frames[0].markers["M01"]
    assert first_marker.x == pytest.approx(1.0)
    assert first_marker.y == pytest.approx(0.5)
    assert first_marker.z == pytest.approx(0.25)
    assert trajectory.metadata["units"] == "cm"


def test_c3d_rust_load_rejects_units_that_rust_prescales_unsafely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c3d_file = tmp_path / "centimeters.c3d"
    c3d_file.write_bytes(b"synthetic")
    fake_rust = SimpleNamespace(
        parse_c3d=lambda _path: {
            "labels": ["M01"],
            "positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "n_frames": 1,
            "fps": 100.0,
            "units": "cm",
        }
    )

    monkeypatch.setattr(_mod, "_rust_io", fake_rust)
    monkeypatch.setattr(_mod, "_HAS_RUST", True)

    with pytest.raises(AdapterContractError, match="C3D Rust backend cannot trust"):
        C3DAdapter()._load_via_rust(c3d_file, None)


def test_c3d_ezc3d_metadata_warns_when_defaulting_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    c3d_file = tmp_path / "zero_rate.c3d"
    c3d_file.write_bytes(b"synthetic")
    fake_c3d = {
        "parameters": {
            "POINT": {
                "LABELS": {"value": ["M01"]},
                "RATE": {"value": [0.0]},
                "UNITS": {"value": ["mm"]},
            }
        },
        "data": {"points": np.zeros((4, 1, 1))},
    }

    monkeypatch.setattr(_mod, "_HAS_RUST", False)
    monkeypatch.setattr(_mod, "_HAS_EZC3D", True)
    monkeypatch.setattr(_mod, "_HAS_C3D_BACKEND", True)
    monkeypatch.setattr(_mod, "_ezc3d", SimpleNamespace(c3d=lambda _path: fake_c3d))

    with caplog.at_level("WARNING"):
        md = C3DAdapter().metadata(c3d_file)

    assert md.fps == pytest.approx(30.0)
    assert "defaulting C3D fps to 30.0" in caplog.text
