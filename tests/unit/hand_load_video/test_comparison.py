"""Side-by-side null-space comparison frames (#15494, #15725).

The compositor aligns panels by time, lays them out as baseline against
least-force and ZTCF | ZVCF | Total, and stamps each frame. These tests use
arrays only: no ffmpeg, no encoded video, and no report-catalog import.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont

from src.hand_load_video import (
    COMPONENTS,
    OBJECTIVE,
    PHASES,
    STRATEGIES,
    UNAVAILABLE,
    GridSpec,
    Provenance,
    VideoWriterUnavailable,
    align,
    compose,
    load_source,
    panel_from_arrays,
    phase_sheets,
    select_phases,
    stack_sheets,
    write_video,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_RED = (180, 0, 0)
_GREEN = (0, 140, 0)
_BLUE = (0, 0, 160)
_CYAN = (0, 140, 140)
_FONT = ImageFont.load_default()


def _solid(rgb: tuple[int, int, int], size: int = 6) -> np.ndarray:
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[:] = rgb
    return frame


def _panel(strategy, component, times, colors, *, badge, units="N"):
    frames = [None if color is None else _solid(color) for color in colors]
    return panel_from_arrays(
        strategy, component, times, frames, badge=badge, units=units
    )


def _inked(
    image: np.ndarray, origin: tuple[int, int], text: str, rgb: tuple[int, int, int]
) -> bool:
    """True when ``text`` is stamped at ``origin`` in exact ``rgb`` pixels."""
    mask = _FONT.getmask(text, mode="1")
    bitmap = np.asarray(Image.frombytes("L", mask.size, bytes(mask)))
    ys, xs = np.nonzero(bitmap)
    ys = ys + origin[1]
    xs = xs + origin[0]
    if (
        len(xs) == 0
        or int(xs.max()) >= image.shape[1]
        or int(ys.max()) >= image.shape[0]
    ):
        return False
    return bool(np.all(image[ys, xs] == rgb))


def test_panels_align_by_time_not_by_frame_index() -> None:
    baseline = _panel(
        "baseline",
        "ztcf",
        (0.0, 0.1, 0.4),
        (_RED, _GREEN, _BLUE),
        badge="allocation-only",
    )
    # Its first stored frame is 0.1 s, so index 0 is not the clock's index 0.
    least = _panel(
        "least-force",
        "ztcf",
        (0.1, 0.4, 0.9),
        (_CYAN, _BLUE, _RED),
        badge="forward-replayed",
    )
    synced = align((baseline, least), (0.0, 0.1, 0.4), tolerance_s=0.04)
    assert [frame.time_s for frame in synced] == [0.0, 0.1, 0.4]
    at_start, at_tenth, at_late = synced
    assert at_start.cell("least-force", "ztcf").missing
    assert at_start.cell("baseline", "ztcf").pixels is not None
    assert np.all(at_tenth.cell("baseline", "ztcf").pixels[0, 0] == _GREEN)
    assert np.all(at_tenth.cell("least-force", "ztcf").pixels[0, 0] == _CYAN)
    for frame in synced:
        for cell in frame.cells:
            if not cell.missing:
                assert cell.time_s is not None
                assert abs(cell.time_s - frame.time_s) <= 0.04
    assert at_late.cell("least-force", "zvcf").missing  # channel was never supplied
    assert at_late.cell("least-force", "zvcf").units == UNAVAILABLE


def test_grid_places_strategies_against_components() -> None:
    spec = GridSpec(cell_w=10, cell_h=8, gutter=3, banner_h=5, footer_h=6, label_w=20)
    assert spec.rows == STRATEGIES
    assert spec.columns == COMPONENTS
    assert STRATEGIES == ("baseline", "least-force")
    assert COMPONENTS == ("ztcf", "zvcf", "total")
    assert spec.width == 20 + 3 * 10 + 4 * 3
    assert spec.height == 2 * 5 + 2 * (3 + 5 + 8) + 3 + 6
    boxes = [spec.cell_box(row, col) for row in range(2) for col in range(3)]
    for index, box in enumerate(boxes):
        for other in boxes[index + 1 :]:
            assert (
                box[0] + box[2] <= other[0]
                or other[0] + other[2] <= box[0]
                or (box[1] + box[3] <= other[1] or other[1] + other[3] <= box[1])
            )
    left, right = spec.cell_box(0, 0), spec.cell_box(0, 1)
    assert right[0] == left[0] + left[2] + spec.gutter
    assert spec.cell_box(1, 2)[1] > spec.cell_box(0, 2)[1]


def test_freeze_sheets_use_the_seven_phases_without_guessing() -> None:
    times = (0.0, 0.5, 1.0, 1.15, 1.25, 1.3, 1.6)
    colors = (_RED, _GREEN, _BLUE, _CYAN, (10, 10, 10), (90, 20, 20), (20, 20, 90))
    panel = _panel("baseline", "total", times, colors, badge="forward-replayed")
    synced = align((panel,), times, tolerance_s=0.0)
    asked = {
        "address": 0.0,
        "shaft_parallel_back": 0.5,
        "top": 1.0,
        "lead_arm_parallel": 1.15,
        "shaft_parallel_down": 1.25,
        "impact": 1.3,
        "follow_through": 1.6,
    }
    picks = select_phases(synced, asked, tolerance_s=0.02)
    assert [pick.name for pick in picks] == list(PHASES)
    assert len(PHASES) == 7
    for pick, stamp, color in zip(picks, times, colors, strict=True):
        assert pick.frame is not None
        assert pick.frame.time_s == stamp
        assert np.all(pick.frame.cell("baseline", "total").pixels[0, 0] == color)
    withheld = select_phases(synced, {"impact": None, "top": 9.0}, tolerance_s=0.02)
    by_name = {pick.name: pick for pick in withheld}
    assert by_name["impact"].frame is None
    assert by_name["top"].frame is None
    assert by_name["address"].frame is None
    near = select_phases(synced, {"shaft_parallel_down": 1.27}, tolerance_s=0.03)
    chosen = {pick.name: pick for pick in near}["shaft_parallel_down"]
    assert chosen.frame is not None and chosen.frame.time_s == 1.25
    with pytest.raises(ValueError, match="unknown swing phase"):
        select_phases(synced, {"not-a-phase": 0.0}, tolerance_s=0.0)


def test_every_frame_carries_the_objective_badge_and_footer() -> None:
    spec = GridSpec()
    provenance = Provenance("fixture-swing", "toy-model", "abc123def")
    panel = _panel(
        "baseline", "ztcf", (0.0,), (_RED,), badge="allocation-only", units="N"
    )
    other = _panel(
        "least-force", "zvcf", (0.0,), (_GREEN,), badge="forward-replayed", units="N.m"
    )
    image, stamp = compose(
        align((panel, other), (0.0,), tolerance_s=0.0)[0], provenance, spec
    )
    assert stamp.objective == OBJECTIVE
    assert "least squared force" in stamp.objective
    assert "opposing-force cancellation" in stamp.objective
    lowered = stamp.objective.lower()
    assert "wasted energy" not in lowered
    assert "biological optimality" not in lowered
    assert stamp.footer == "source=fixture-swing model=toy-model sha=abc123def"
    assert stamp.badges == ("allocation-only", "forward-replayed")
    assert _inked(image, spec.text_origin("objective"), OBJECTIVE, spec.yellow)
    assert _inked(image, spec.text_origin("footer"), stamp.footer, spec.white)
    assert _inked(image, spec.badge_text_origin(0), "allocation-only", spec.black)
    assert _inked(image, spec.badge_text_origin(1), "forward-replayed", spec.black)
    assert tuple(image[spec.header_sample(0)]) == spec.component_rgb["ztcf"]
    assert tuple(image[spec.header_sample(1)]) == spec.component_rgb["zvcf"]
    assert tuple(image[spec.header_sample(2)]) == spec.component_rgb["total"]
    assert tuple(image[spec.badge_sample(0)]) == spec.badge_rgb["allocation-only"]
    assert tuple(image[spec.badge_sample(1)]) == spec.badge_rgb["forward-replayed"]
    sheets = phase_sheets(
        select_phases(
            align((panel,), (0.0,), tolerance_s=0.0), {"address": 0.0}, tolerance_s=0.0
        ),
        provenance,
        spec,
    )
    assert list(sheets) == list(PHASES)
    assert _inked(sheets["address"], (2, 1), "Address", spec.yellow)
    assert _inked(sheets["impact"], (2, 1), "Impact UNAVAILABLE", spec.yellow)
    stacked = stack_sheets([sheets[name] for name in PHASES])
    assert stacked.shape[0] == sum(sheet.shape[0] for sheet in sheets.values()) + 2 * 6


def test_unavailable_pixels_stay_visible_and_zero_force_does_not() -> None:
    spec = GridSpec()
    black = np.zeros((6, 6, 3), dtype=np.uint8)
    present = panel_from_arrays(
        "baseline",
        "total",
        (0.0, 0.2),
        (black, None),
        badge="allocation-only",
        units="N",
    )
    synced = align((present,), (0.0, 0.2), tolerance_s=0.0)
    shown, stamp = compose(synced[0], Provenance("src", "model", "sha"), spec)
    missing, missing_stamp = compose(synced[1], Provenance("src", "model", "sha"), spec)
    y, x = spec.content_sample(0, 2)
    assert tuple(shown[y, x]) == (0, 0, 0)
    assert missing_stamp.missing[2] is True
    assert not stamp.missing[2]
    cell = spec.cell_box(0, 2)
    region = missing[cell[1] : cell[1] + cell[3], cell[0] : cell[0] + cell[2]]
    assert np.any(np.all(region == spec.purple, axis=-1))
    assert _inked(missing, spec.unavailable_origin(0, 2), UNAVAILABLE, spec.yellow)
    assert missing_stamp.units[2] == "N"
    assert synced[0].cell("least-force", "ztcf").units == UNAVAILABLE


def test_local_source_is_the_catalog_seam_and_the_writer_is_optional(
    monkeypatch,
) -> None:
    panel = _panel("baseline", "total", (0.0,), (_RED,), badge="forward-replayed")

    class _Source:
        def sequences(self):
            return (panel,)

    assert load_source(_Source())[0] is panel
    root = Path(__file__).resolve().parents[3] / "src" / "hand_load_video"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "swing_report_package" not in text
        assert "native_viewer_export" not in text

    def _missing():
        raise VideoWriterUnavailable("imageio is not installed")

    monkeypatch.setattr("src.hand_load_video.compare._load_imageio", _missing)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(VideoWriterUnavailable, match="imageio"):
        write_video((frame,), "clip.mp4", fps=10)

    stored: list[np.ndarray] = []

    class _Writer:
        def append_data(self, data: np.ndarray) -> None:
            stored.append(data)

        def close(self) -> None:
            stored.append(np.zeros((1, 1, 3), dtype=np.uint8))

    class _IO:
        def get_writer(self, path, *, fps, codec, format):  # noqa: A002
            assert fps == 10 and codec == "libx264" and format == "FFMPEG"
            assert str(path).endswith("clip.mp4")
            return _Writer()

    monkeypatch.setattr("src.hand_load_video.compare._load_imageio", lambda: _IO())
    write_video((frame,), "clip.mp4", fps=10)
    assert len(stored) == 2
    with pytest.raises(ValueError, match="fps"):
        write_video((frame,), "clip.mp4", fps=0)
