"""Case comparison grid video and phase freeze-frame sheets (#15722, SRP-3).

The ZTCF (drift), ZVCF (control) and total renders of one swing and view are
laid out side by side under banners in their case colours, so a player sees
what each component does at every instant; the freeze-frame sheet puts each
case at each key phase. Layout is ``composite_settings.json``; case colours
come from :class:`TipToTailStyle`, the one source the arrows also use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.shared.python.force_overlay.palette import hex_to_rgba
from src.shared.python.force_overlay.tip_to_tail import TipToTailStyle
from src.tools.native_viewer_export.compositor import _font, draw_label
from src.tools.native_viewer_export.frame_times import frame_at_time, load_frame_times

__all__ = [
    "CASES",
    "COMPOSITE_SETTINGS_PATH",
    "CompositeSettings",
    "case_banner",
    "compose_grid_video",
    "load_composite_settings",
    "phase_sheet",
]

COMPOSITE_SETTINGS_PATH = Path(__file__).with_name("composite_settings.json")
CASES = ("ztcf", "zvcf", "total")
_CASE_FIELD = {"ztcf": "drift_hex", "zvcf": "control_hex", "total": "total_hex"}
_INT_KEYS = (
    "banner_height_px",
    "title_px",
    "subtitle_px",
    "grid_columns",
    "tile_width_px",
    "cell_width_px",
    "row_label_width_px",
    "row_label_px",
    "info_text_px",
    "crf",
)
Image8 = NDArray[np.uint8]


@dataclass(frozen=True)
class CompositeSettings:
    """Layout of the grid video and the phase sheet (see the settings file)."""

    case_order: tuple[str, ...]
    cases: Mapping[str, Mapping[str, str]]
    banner_height_px: int
    title_px: int
    subtitle_px: int
    grid_columns: int
    tile_width_px: int
    cell_width_px: int
    row_label_width_px: int
    row_label_px: int
    info_text_px: int
    guide: tuple[str, ...]
    info_background_hex: str
    crf: int

    def scaled(self, **sizes: int) -> CompositeSettings:
        """These settings with other sizes (e.g. a preview)."""
        return replace(self, **sizes)

    def rgb(self, case: str) -> tuple[int, int, int]:
        """The case colour; ``ValueError`` for an unknown case."""
        if case not in CASES:
            raise ValueError(f"case must be one of {CASES}, got {case!r}")
        hex_code = getattr(TipToTailStyle(), _CASE_FIELD[case])
        r, g, b = (round(255 * c) for c in hex_to_rgba(hex_code)[:3])
        return (r, g, b)


def load_composite_settings(path: Path | None = None) -> CompositeSettings:
    """The composite settings, checked.

    Raises:
        ValueError: an unknown or untitled case, or a size that is not a
            positive integer.
    """
    doc = json.loads((path or COMPOSITE_SETTINGS_PATH).read_text(encoding="utf-8"))
    order = tuple(doc["case_order"])
    for case in order:
        if case not in CASES or not doc["cases"].get(case, {}).get("title"):
            raise ValueError(f"composite case {case!r} is unknown or has no title")
    for key in _INT_KEYS:
        if type(doc.get(key)) is not int or doc[key] <= 0:
            raise ValueError(f"composite {key} must be a positive integer")
    return CompositeSettings(
        order,
        doc["cases"],
        **{k: doc[k] for k in _INT_KEYS},
        guide=tuple(doc.get("guide", ())),
        info_background_hex=doc["info_background_hex"],
    )


def _text_rgb(rgb: Sequence[int]) -> tuple[int, int, int]:
    """Black on a light background, white on a dark one."""
    luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return (0, 0, 0) if luma > 150 else (255, 255, 255)


def case_banner(width: int, case: str, settings: CompositeSettings) -> Image8:
    """A banner ``width`` wide in the case colour with its title and subtitle."""
    rgb = settings.rgb(case)
    img = np.empty((settings.banner_height_px, width, 3), np.uint8)
    img[:] = rgb
    text = settings.cases[case]
    ink = _text_rgb(rgb)
    pad = max(2, settings.banner_height_px // 12)
    img = draw_label(
        img, text["title"], (2 * pad, pad), size=settings.title_px, fill=ink
    )
    sub_y = pad + int(settings.title_px * 1.2)
    return draw_label(
        img,
        text.get("subtitle", ""),
        (2 * pad, sub_y),
        size=settings.subtitle_px,
        fill=ink,
    )


def _resized(frame: Image8, width: int) -> Image8:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    if frame.shape[:2] == (height, width):
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.LANCZOS))


def _info_tile(
    shape: tuple[int, int], lines: Sequence[str], s: CompositeSettings
) -> Image8:
    tile = np.empty((*shape, 3), np.uint8)
    tile[:] = [round(255 * c) for c in hex_to_rgba(s.info_background_hex)[:3]]
    size, x = s.info_text_px, s.banner_height_px // 3
    y = s.banner_height_px // 2
    for line in lines:
        for part in label_lines(line, shape[1] - x, size) if line else ("",):
            tile = draw_label(tile, part, (x, y), size=size)
            y += int(size * 1.3)
        y += size // 2
    return tile


def grid_info_lines(
    time_s: float,
    phases: Sequence[Any],
    base: Sequence[str],
    s: CompositeSettings,
) -> tuple[str, ...]:
    """Info tile text at swing time ``time_s``: ``base``, the swing time and
    the latest phase reached (when phases are known), then the guide."""
    reached = [p for p in phases if p.time_s <= time_s + 1e-9]
    now: tuple[str, ...] = ()
    if phases:
        label = reached[-1].label if reached else phases[0].label
        now = (f"Swing time {time_s:.3f} s", f"Phase: {label}")
    return (*base, *now, *s.guide)


def _grid(
    frames: Mapping[str, Image8], lines: Sequence[str], s: CompositeSettings
) -> Image8:
    tiles = [
        np.vstack(
            [case_banner(s.tile_width_px, c, s), _resized(frames[c], s.tile_width_px)]
        )
        for c in s.case_order
    ]
    while len(tiles) % s.grid_columns:
        tiles.append(_info_tile(tiles[0].shape[:2], lines, s))
    rows = [
        np.hstack(tiles[i : i + s.grid_columns])
        for i in range(0, len(tiles), s.grid_columns)
    ]
    return np.vstack(rows)


def _checked_clips(clips: Mapping[str, Path], s: CompositeSettings) -> list:
    missing = [c for c in s.case_order if c not in clips]
    if missing:
        raise ValueError(f"composite needs a clip per case; missing {missing}")
    times = [load_frame_times(clips[c]) for c in s.case_order]
    if any(t != times[0] for t in times):
        raise ValueError("the case clips have different frame times")
    return times


def compose_grid_video(
    clips: Mapping[str, Path],
    out: Path,
    settings: CompositeSettings | None = None,
    info_lines: Sequence[str] = (),
    writer_factory: Callable[[Path, float], Any] | None = None,
    phases: Sequence[Any] = (),
) -> Path:
    """Write the case grid video of ``clips`` (one per case) to ``out``.

    With ``phases`` the info tile shows each frame's swing time and phase.

    Raises:
        ValueError: a case clip is missing or the clips' frame times differ.
    """
    import imageio.v2 as imageio

    s = settings or load_composite_settings()
    times = _checked_clips(clips, s)[0]
    readers = [imageio.get_reader(str(clips[c])) for c in s.case_order]
    fps = float(readers[0].get_meta_data()["fps"])
    if writer_factory is None:
        from src.tools.native_viewer_export.core import imageio_writer

        def writer_factory(path: Path, rate: float) -> Any:
            return imageio_writer(path, rate, crf=s.crf)

    writer = writer_factory(out, fps)
    try:
        for t, *frames in zip(times, *readers, strict=True):
            lines = grid_info_lines(t, phases, info_lines, s)
            writer.append_data(
                _grid(dict(zip(s.case_order, frames, strict=True)), lines, s)
            )
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    return out


def text_width(text: str, size: int) -> float:
    """Drawn width (px) of ``text`` at font ``size``."""
    return float(_font(size).getlength(text))


def label_lines(text: str, width_px: int, size: int) -> tuple[str, ...]:
    """``text`` wrapped at spaces to fit ``width_px`` less a ``size`` margin.

    A single word wider than the column stays on its own line.
    """
    room = width_px - size
    lines: list[str] = []
    for word in text.split():
        joined = f"{lines[-1]} {word}" if lines else word
        if lines and text_width(joined, size) <= room:
            lines[-1] = joined
        else:
            lines.append(word)
    return tuple(lines)


def _row_label(height: int, text: str, s: CompositeSettings) -> Image8:
    tile = _info_tile((height, s.row_label_width_px), (), s)
    lines = label_lines(text, s.row_label_width_px, s.row_label_px)
    step = round(1.25 * s.row_label_px)
    top = height // 2 - (len(lines) * step) // 2
    for i, line in enumerate(lines):
        tile = draw_label(
            tile, line, (s.row_label_px // 2, top + i * step), size=s.row_label_px
        )
    return tile


def phase_sheet(
    clips: Mapping[str, Path],
    phases: Sequence[Any],
    out: Path,
    settings: CompositeSettings | None = None,
) -> list[Path]:
    """Write the freeze-frame sheet (phases by cases) to ``out``.

    Each case's still at each phase is also saved beside ``out`` as
    ``<case>_<phase id>.png``; the paths are returned.

    Raises:
        ValueError: a case clip is missing, the clips' frame times differ, or
            a phase lies outside the clips.
    """
    import imageio.v2 as imageio

    s = settings or load_composite_settings()
    times = _checked_clips(clips, s)[0]
    stills: list[Path] = []
    rows = []
    for phase in phases:
        k = frame_at_time(times, phase.time_s)
        cells = []
        for case in s.case_order:
            with imageio.get_reader(str(clips[case])) as reader:
                frame = np.asarray(reader.get_data(k))
            path = out.with_name(f"{case}_{phase.id}.png")
            Image.fromarray(frame).save(path)
            stills.append(path)
            cells.append(_resized(frame, s.cell_width_px))
        label = _row_label(cells[0].shape[0], phase.label, s)
        rows.append(np.hstack([label, *cells]))
    corner = _info_tile((s.banner_height_px, s.row_label_width_px), (), s)
    header = np.hstack(
        [corner, *(case_banner(s.cell_width_px, c, s) for c in s.case_order)]
    )
    Image.fromarray(np.vstack([header, *rows])).save(out)
    return stills
