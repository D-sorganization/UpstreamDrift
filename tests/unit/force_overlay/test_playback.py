"""Tests for generic force overlay playback (ADR-0052, #11302, FTO-17)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

pytestmark = [pytest.mark.unit]

from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    ForceTorqueSeries,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle
from src.shared.python.force_overlay.playback import (
    PlaybackOptions,
    PlaybackReceipt,
    SegmentSeries,
    load_segment_series,
    render_force_playback,
    save_segment_series,
)
from src.shared.python.video_timing.frame_schedule import FrameSchedule


def _make_synthetic_playback_data() -> tuple[ForceTorqueSeries, SegmentSeries]:
    """Create a 3-frame series with 1 tension segment, 1 compression segment, and 1 reaction force."""
    times = [0.0, 0.1, 0.2]
    frames = []

    for t in times:
        # Segment 0 is under tension (+500 N), segment 1 is under compression (-500 N)
        axial = AxialLoadFrame(
            time_s=t,
            values_n={"thigh": 500.0, "shank": -500.0},
            source="synthetic",
        )
        wrench = OverlayWrench(
            kind=WrenchKind.JOINT_REACTION,
            label="reaction:knee",
            body="shank",
            point_m=(0.0, 0.0, 0.5),
            force_n=(0.0, 0.0, 100.0),
            torque_nm=None,
            source="synthetic",
        )
        frames.append(
            ForceTorqueFrame(
                time_s=t,
                engine="synthetic",
                wrenches=(wrench,),
                axial_loads=axial,
            )
        )

    series = ForceTorqueSeries(frames=tuple(frames))

    # Segment endpoints: thigh from (0,0,1) to (0,0,0.5); shank from (0,0,0.5) to (0,0,0)
    names = ("thigh", "shank")
    proximal = np.zeros((3, 2, 3), dtype=np.float64)
    distal = np.zeros((3, 2, 3), dtype=np.float64)

    for i in range(3):
        # thigh
        proximal[i, 0] = [0.0, 0.0, 1.0]
        distal[i, 0] = [0.0, 0.0, 0.5]
        # shank
        proximal[i, 1] = [0.0, 0.0, 0.5]
        distal[i, 1] = [0.0, 0.0, 0.0]

    segments = SegmentSeries(names=names, proximal=proximal, distal=distal)
    return series, segments


def test_segment_series_npz_roundtrip(tmp_path: Path) -> None:
    _, segments = _make_synthetic_playback_data()
    npz_path = tmp_path / "segments.npz"
    save_segment_series(segments, npz_path)

    loaded = load_segment_series(npz_path)
    assert loaded.names == segments.names
    np.testing.assert_allclose(loaded.proximal, segments.proximal)
    np.testing.assert_allclose(loaded.distal, segments.distal)


def test_playback_renders_png_frames_and_receipt(tmp_path: Path) -> None:
    series, segments = _make_synthetic_playback_data()
    out_dir = tmp_path / "frames"

    color_scale = ForceColorScale(
        enabled=True,
        tension_limit_n=1000.0,
        compression_limit_n=1000.0,
        tension_color="#0000ff",
        compression_color="#ff0000",
    )

    receipt = render_force_playback(
        series,
        segments,
        style=ForceGlyphStyle(),
        color_scale=color_scale,
        out_path=out_dir,
        fps=10,
        size_px=(400, 300),
    )

    assert isinstance(receipt, PlaybackReceipt)
    assert receipt.frame_count == 3
    assert receipt.encoder == "png"
    assert receipt.size_px == (400, 300)
    assert receipt.fps == 10
    assert receipt.segment_count == 2

    # Check generated PNG files
    png_files = sorted(out_dir.glob("frame_*.png"))
    assert len(png_files) == 3


def test_playback_frame_pixel_colors(tmp_path: Path) -> None:
    series, segments = _make_synthetic_playback_data()
    out_dir = tmp_path / "frames_color"

    color_scale = ForceColorScale(
        enabled=True,
        tension_limit_n=1000.0,
        compression_limit_n=1000.0,
        tension_color="#0000ff",
        compression_color="#ff0000",
    )

    render_force_playback(
        series,
        segments,
        color_scale=color_scale,
        out_path=out_dir,
        fps=10,
        size_px=(600, 600),
    )

    # Load first rendered frame
    img = Image.open(out_dir / "frame_0000.png").convert("RGB")
    arr = np.array(img)

    # Verify tension segment (bluish pixels: B > R + 30 and B > 50)
    blue_mask = (arr[:, :, 2] > arr[:, :, 0] + 30) & (arr[:, :, 2] > 50)
    assert np.any(blue_mask), "Expected bluish tension pixels in rendered frame"

    # Verify compression segment (reddish pixels: R > B + 30 and R > 50)
    red_mask = (arr[:, :, 0] > arr[:, :, 2] + 30) & (arr[:, :, 0] > 50)
    assert np.any(red_mask), "Expected reddish compression pixels in rendered frame"

    # Verify joint reaction arrow (greenish pixels: G > R + 30 and G > B + 30)
    green_mask = (
        (arr[:, :, 1] > arr[:, :, 0] + 20)
        & (arr[:, :, 1] > arr[:, :, 2] + 20)
        & (arr[:, :, 1] > 80)
    )
    assert np.any(green_mask), (
        "Expected greenish joint reaction arrow pixels in rendered frame"
    )


def test_playback_validates_inputs(tmp_path: Path) -> None:
    series, segments = _make_synthetic_playback_data()

    # Mismatched length between series and segments
    short_segments = SegmentSeries(
        names=segments.names,
        proximal=segments.proximal[:1],
        distal=segments.distal[:1],
    )
    with pytest.raises(ValueError, match="mismatch|length"):
        render_force_playback(series, short_segments, out_path=tmp_path / "out", fps=10)

    # Nonpositive fps
    with pytest.raises(ValueError, match="fps must be positive"):
        render_force_playback(series, segments, out_path=tmp_path / "out", fps=0)

    # Nonpositive speed
    with pytest.raises(ValueError, match="speed must be positive"):
        render_force_playback(
            series, segments, out_path=tmp_path / "out2", fps=10, speed=0.0
        )


def test_default_fps_is_sixty_the_unified_default() -> None:
    """GCV-14 (#11720): capture-rig/export/visual_layer all default to 60 fps."""
    assert PlaybackOptions().fps == 60
    assert PlaybackOptions().speed == 1.0


def test_render_force_playback_is_time_based_not_index_based(tmp_path: Path) -> None:
    """One output frame per FrameSchedule sample over series.times_s, not one
    per series sample (GCV-14, #11720)."""
    series, segments = _make_synthetic_playback_data()
    out_dir = tmp_path / "frames_time"

    receipt = render_force_playback(series, segments, out_path=out_dir, fps=60)

    expected = FrameSchedule(np.array(series.times_s), 60, 1.0).n_frames
    assert expected != len(series)  # the behaviour actually changed
    assert receipt.frame_count == expected
    png_files = sorted(out_dir.glob("frame_*.png"))
    assert len(png_files) == expected


def test_render_force_playback_half_speed_roughly_doubles_frames(
    tmp_path: Path,
) -> None:
    series, segments = _make_synthetic_playback_data()
    full = render_force_playback(
        series, segments, out_path=tmp_path / "full", fps=60, speed=1.0
    )
    half = render_force_playback(
        series, segments, out_path=tmp_path / "half", fps=60, speed=0.5
    )
    assert abs(half.frame_count - 2 * full.frame_count) <= 1


def test_render_force_playback_fps_ten_matches_legacy_frame_count(
    tmp_path: Path,
) -> None:
    """At fps=10 the series' own 0.1 s spacing lines up with the schedule
    exactly, so the frame count matches the pre-GCV-14 index-based count."""
    series, segments = _make_synthetic_playback_data()
    receipt = render_force_playback(series, segments, out_path=tmp_path / "ten", fps=10)
    assert receipt.frame_count == len(series) == 3
