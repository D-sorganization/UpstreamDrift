"""Case comparison grid video and phase freeze-frame sheets (#15722, SRP-3).

The owner asked for the ZTCF, ZVCF and total renders laid out in one grid
so players can compare the components, plus freeze frames at key phases for
all three cases, each labelled and coloured by its case.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.force_overlay.palette import hex_to_rgba
from src.shared.python.force_overlay.tip_to_tail import TipToTailStyle
from src.tools.native_viewer_export.frame_times import write_frame_times
from src.tools.swing_report_package.composite import (
    COMPOSITE_SETTINGS_PATH,
    case_banner,
    compose_grid_video,
    load_composite_settings,
    phase_sheet,
)
from src.tools.swing_report_package.phases import KeyPhase

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
CASES = ("ztcf", "zvcf", "total")
W, H = 64, 48


def _rgb(hex_code: str) -> np.ndarray:
    return np.array([round(255 * c) for c in hex_to_rgba(hex_code)[:3]])


def _clips(tmp_path, frames: int = 6, fps: int = 10):
    """One tiny clip per case, each frame a flat grey of its index."""
    imageio = pytest.importorskip("imageio.v2")
    clips = {}
    for case in CASES:
        path = tmp_path / f"{case}.mp4"
        with imageio.get_writer(str(path), fps=fps, macro_block_size=8) as w:
            for k in range(frames):
                w.append_data(np.full((H, W, 3), 20 + 30 * k, np.uint8))
        write_frame_times({case: path}, [0.1 * k for k in range(frames)])
        clips[case] = path
    return clips


def _small():
    settings = load_composite_settings()
    return settings.scaled(tile_width_px=W, cell_width_px=W)


def test_a_banner_wears_its_case_colour_from_the_one_style() -> None:
    settings = _small()
    look = TipToTailStyle()
    for case, field in zip(
        CASES, ("drift_hex", "control_hex", "total_hex"), strict=True
    ):
        img = case_banner(W * 4, case, settings)
        assert img.shape[:2] == (settings.banner_height_px, W * 4)
        corner = img[1, 1].astype(int)
        assert np.abs(corner - _rgb(getattr(look, field))).max() <= 1
    with pytest.raises(ValueError, match="case"):
        case_banner(W, "sideways", settings)


def test_the_grid_video_has_every_case_under_its_banner(tmp_path) -> None:
    imageio = pytest.importorskip("imageio.v2")
    out = tmp_path / "grid.mp4"
    compose_grid_video(_clips(tmp_path), out, _small(), ("Driver, Face-On",))
    with imageio.get_reader(str(out)) as reader:
        frames = [np.asarray(f) for f in reader.iter_data()]
    assert len(frames) == 6
    settings = _small()
    banner = settings.banner_height_px
    # 2x2 layout: three cases and an info tile
    assert frames[0].shape[1] == 2 * W
    assert frames[0].shape[0] == 2 * (banner + H)
    top_left = frames[0][2, 2].astype(int)
    assert np.abs(top_left - _rgb(TipToTailStyle().drift_hex)).max() <= 12


def test_clips_of_different_lengths_are_refused(tmp_path) -> None:
    clips = _clips(tmp_path)
    write_frame_times({"x": clips["zvcf"]}, [0.0, 0.1])
    with pytest.raises(ValueError, match="frame times"):
        compose_grid_video(clips, tmp_path / "g.mp4", _small(), ())


def test_the_phase_sheet_puts_each_case_at_each_phase(tmp_path) -> None:
    pytest.importorskip("imageio.v2")
    from PIL import Image

    phases = (KeyPhase("top", "Top", 2, 0.2), KeyPhase("impact", "Impact", 4, 0.4))
    out = tmp_path / "sheet.png"
    stills = phase_sheet(_clips(tmp_path), phases, out, _small())
    sheet = np.asarray(Image.open(out).convert("RGB"))
    settings = _small()
    assert sheet.shape[1] == settings.row_label_width_px + 3 * W
    assert sheet.shape[0] == settings.banner_height_px + 2 * H
    # the impact row of the ZVCF column shows the impact frame's grey (20 + 30*4)
    y = settings.banner_height_px + H + H // 2
    x = settings.row_label_width_px + W + W // 2
    assert abs(int(sheet[y, x, 0]) - 140) <= 12
    # one still per case per phase, named by both
    assert {p.name for p in stills} == {
        f"{c}_{p.id}.png" for c in CASES for p in phases
    }


def test_settings_are_checked(tmp_path) -> None:
    doc = json.loads(COMPOSITE_SETTINGS_PATH.read_text(encoding="utf-8"))
    doc["case_order"] = ["ztcf", "sideways"]
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="case"):
        load_composite_settings(bad)


def test_a_long_row_label_wraps_inside_its_column() -> None:
    """The first HQ sheet cut "Follow-Through (Shaft Parallel)" off at the
    column edge; labels wrap at spaces so every word is shown."""
    from src.tools.swing_report_package.composite import label_lines, text_width

    s = load_composite_settings()
    text = "Follow-Through (Shaft Parallel)"
    lines = label_lines(text, s.row_label_width_px, s.row_label_px)
    assert len(lines) > 1 and " ".join(lines) == text
    margin = s.row_label_px  # half on each side
    for line in lines:
        assert text_width(line, s.row_label_px) <= s.row_label_width_px - margin
    assert label_lines("Top", s.row_label_width_px, s.row_label_px) == ("Top",)


def test_the_grid_info_tile_reads_the_swing_time_and_current_phase() -> None:
    """The fourth grid tile was nearly empty; it now tells the player where
    in the swing the frame is, then how to read the three cases."""
    from src.tools.swing_report_package.composite import grid_info_lines

    s = load_composite_settings()
    phases = (KeyPhase("address", "Address", 0, 0.0), KeyPhase("top", "Top", 9, 0.9))
    lines = grid_info_lines(0.95, phases, ("Title",), s)
    assert lines[0] == "Title"
    assert "0.950 s" in lines[1] and lines[2].endswith("Top")
    assert lines[3:] == tuple(s.guide)
    assert grid_info_lines(0.5, phases, (), s)[1].endswith("Address")
    assert all("Phase" not in x for x in grid_info_lines(0.5, (), (), s))
    assert s.info_text_px > s.subtitle_px
