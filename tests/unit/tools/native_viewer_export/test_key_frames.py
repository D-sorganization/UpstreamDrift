"""Tests for the key-frame PNG writer and grip-clock inset helper (SHOW-A10, #14735)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.tools.native_viewer_export.key_frames import (
    compose_side_by_side,
    detect_key_frame_indices,
    draw_grip_clock,
    render_a10_summary,
    write_key_frame_stills,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_draw_grip_clock_renders_dial_and_hands() -> None:
    img = Image.new("RGB", (640, 480), (20, 20, 20))
    rolls = {"lead": 15.0, "trail": -165.0}
    out = draw_grip_clock(img, rolls, cx=500, cy=380, r=60)
    assert out.size == (640, 480)
    arr = np.asarray(out)
    # Check that pixels were modified in the clock region
    clock_region = arr[320:440, 440:560]
    assert np.any(clock_region != [20, 20, 20])
    # Check green target tick (60, 220, 60)
    has_green = np.any((clock_region[:, :, 1] > 200) & (clock_region[:, :, 0] < 100))
    assert has_green, "clock should contain green target tick"
    # Check red lead hand (255, 60, 60)
    has_red = np.any((clock_region[:, :, 0] > 200) & (clock_region[:, :, 2] < 100))
    assert has_red, "clock should contain red lead needle"
    # Check blue trail hand (80, 120, 255)
    has_blue = np.any((clock_region[:, :, 2] > 200) & (clock_region[:, :, 0] < 120))
    assert has_blue, "clock should contain blue trail needle"


def test_detect_key_frame_indices_from_trajectory() -> None:
    # Synthetic swing: 100 frames, t in [0, 1.0]
    n = 100
    times = np.linspace(0.0, 1.0, n)
    # Clubhead x moves back (top at k=60), then swings forward to ball at impact (k=80)
    head = np.zeros((n, 3))
    head[:60, 0] = np.linspace(0.0, -1.0, 60)
    head[60:80, 0] = np.linspace(-1.0, 0.0, 20)
    head[80:, 0] = np.linspace(0.0, 0.8, 20)

    indices = detect_key_frame_indices(
        times, head_positions=head, impact_time_s=times[80]
    )
    assert indices["address"] == 0
    assert indices["top"] == 60
    assert indices["impact"] == 80
    assert 0 == indices["address"] < indices["top"] < indices["impact"] < n


def test_detect_key_frame_indices_fallback_without_head() -> None:
    times = np.linspace(0.0, 1.0, 101)
    indices = detect_key_frame_indices(times, impact_time_s=0.8)
    assert indices["address"] == 0
    assert indices["top"] < indices["impact"]
    assert indices["impact"] == 80


def test_compose_side_by_side() -> None:
    left = Image.new("RGB", (400, 300), (40, 40, 40))
    right = Image.new("RGB", (400, 300), (60, 60, 60))
    composed = compose_side_by_side(
        left, right, label_left="Baseline (11-04)", label_right="A10 Improved"
    )
    assert composed.size == (800, 300)
    arr = np.asarray(composed)
    assert np.any(arr[:, :400] != arr[:, 400:])


def test_write_key_frame_stills_creates_valid_pngs(tmp_path: Path) -> None:
    frames = {
        "address": np.zeros((100, 100, 3), dtype=np.uint8),
        "top": np.full((100, 100, 3), 50, dtype=np.uint8),
        "impact": np.full((100, 100, 3), 100, dtype=np.uint8),
    }
    rolls = {"lead": 15.0, "trail": -165.0}
    paths = write_key_frame_stills(
        frames,
        rolls,
        tmp_path,
        prefix="driver_mujoco_face_on_",
        r=20,
    )
    assert len(paths) == 3
    for phase in ("address", "top", "impact"):
        assert phase in paths
        p = paths[phase]
        assert p.exists()
        header = p.read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n", f"{p} is not a valid PNG"


def test_render_a10_summary() -> None:
    input_hashes = {"bundle": "a" * 64}
    output_hashes = {
        "face_on_address_1280.png": "b" * 64,
        "face_on_top_1280.png": "c" * 64,
    }
    summary = render_a10_summary(
        model_id="mujoco_driver",
        source_commit="deadbeef" * 5,
        receipt_verdict="PASS (SHOW-A10)",
        input_hashes=input_hashes,
        output_hashes=output_hashes,
        lead_roll_deg=15.0,
        trail_roll_deg=-165.0,
    )
    assert "mujoco_driver" in summary
    assert "deadbeef" in summary
    assert "PASS (SHOW-A10)" in summary
    assert "+15.0" in summary
    assert "b" * 64 in summary
