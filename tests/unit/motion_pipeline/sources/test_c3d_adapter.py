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

pytestmark = pytest.mark.unit


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


def _load_via_ezc3d_scalar_loop_reference(
    points: np.ndarray,
    labels: list[str],
    fps: float,
    scale: float,
) -> list[dict[str, tuple[float, float, float]]]:
    """Reference implementation mirroring the pre-#8933 scalar-index loop.

    Kept local to the test so the parity check is independent of the
    production code path it is verifying against.
    """
    from src.shared.python.motion_pipeline.sources.c3d_adapter import (
        has_nan_coordinate,
    )

    n_frames = int(points.shape[2])
    result: list[dict[str, tuple[float, float, float]]] = []
    for fi in range(n_frames):
        markers: dict[str, tuple[float, float, float]] = {}
        for mi, name in enumerate(labels):
            if mi >= points.shape[1]:
                break
            x = float(points[0, mi, fi]) * scale
            y = float(points[1, mi, fi]) * scale
            z = float(points[2, mi, fi]) * scale
            if has_nan_coordinate(x, y, z):
                continue
            markers[name] = (x, y, z)
        result.append(markers)
    return result


def test_c3d_ezc3d_vectorized_matches_scalar_loop_with_occlusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parity test for #8933 item 1.

    Compares the vectorized ``_load_via_ezc3d`` marker construction against
    a reference scalar-index-loop implementation (the pre-fix behavior),
    including a NaN-occluded marker and a labels array longer than the
    marker axis, on random point-cloud data.
    """
    rng = np.random.default_rng(8933)
    n_markers = 5
    n_frames = 7
    points = rng.normal(size=(4, n_markers, n_frames))
    # Occlude one marker in one frame with NaN, as ezc3d does for gaps.
    points[0, 2, 3] = np.nan
    points[1, 2, 3] = np.nan
    points[2, 2, 3] = np.nan

    labels = [f"M{i:02d}" for i in range(n_markers)]
    # Extra label with no corresponding marker column — exercises the
    # min(len(labels), points.shape[1]) truncation.
    labels_with_extra = [*labels, "EXTRA"]

    fps = 120.0
    scale = 0.001  # e.g. mm -> m

    reference = _load_via_ezc3d_scalar_loop_reference(
        points, labels_with_extra, fps, scale
    )

    c3d_file = tmp_path / "parity.c3d"
    c3d_file.write_bytes(b"synthetic")
    fake_c3d = {
        "parameters": {
            "POINT": {
                "LABELS": {"value": labels_with_extra},
                "RATE": {"value": [fps]},
                "UNITS": {"value": ["mm"]},
            }
        },
        "data": {"points": points},
    }

    monkeypatch.setattr(_mod, "_HAS_RUST", False)
    monkeypatch.setattr(_mod, "_HAS_EZC3D", True)
    monkeypatch.setattr(_mod, "_HAS_C3D_BACKEND", True)
    monkeypatch.setattr(_mod, "_ezc3d", SimpleNamespace(c3d=lambda _path: fake_c3d))

    trajectory = C3DAdapter().load(c3d_file)

    assert len(trajectory.frames) == n_frames
    for fi, frame in enumerate(trajectory.frames):
        expected = reference[fi]
        assert set(frame.markers.keys()) == set(expected.keys())
        for name, (ex, ey, ez) in expected.items():
            marker = frame.markers[name]
            assert marker.x == pytest.approx(ex)
            assert marker.y == pytest.approx(ey)
            assert marker.z == pytest.approx(ez)
            # model_construct-built markers still carry the documented
            # defaults for fields the C3D source doesn't populate.
            assert marker.residual is None
            assert marker.occluded is False

    # The occluded marker (index 2) must be dropped from frame 3, matching
    # the scalar-loop reference's has_nan_coordinate skip.
    assert "M02" not in trajectory.frames[3].markers
