"""Synchronised inset plot strips for the owner showcase clips (GCV-19, #11725).

Two strips, both built from data the exporter already computes (no new physics):

* ``grf``: vertical ground-reaction force of the left foot, right foot and their
  net in body weights (BW), plus the net centre of pressure (CoP) world X and Y
  in metres, from the ``contact:*`` wrenches of the overlay feed;
* ``grip``: net grip force magnitude (N) and couple magnitude (N m) from the
  feed's grip analyses.

The curves are taken once per swing from the SOURCE samples (the full time
series, not the resampled clip frames) and the strip is rendered once with
matplotlib Agg per frame size. Per displayed frame only a moving time cursor is
drawn at the frame's swing time, so every speed clip and the impact window share
one strip. Unavailable data are NaN (a gap in the line), never zero; when no
curve has any data the builders return a reason string instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tools.native_viewer_export.compositor import Image8, _check_image

MAX_SAMPLES = 600  # source samples used for the curves (stride-thinned)
STRIP_HEIGHT_FRAC = 0.20  # keeps the strip below the impact card at 640x360
STRIP_MARGIN_FRAC = 0.012
HUD_CLEARANCE_PX = 52  # keep the HUD text lines (bottom edge) uncovered
MIN_STRIP_PX = (160, 60)
_DPI = 100
_BACKGROUND = (0.04, 0.04, 0.06, 0.66)
_CURSOR = (255, 235, 90)
_BLUE, _ORANGE, _YELLOW, _GREEN, _PINK = (
    "#4aa3ff",
    "#ff9f43",
    "#ffeb5a",
    "#6fe08a",
    "#ff7fb0",
)
_GRF_LABELS = {
    "left": "contact:grf_left",
    "right": "contact:grf_right",
    "net": "contact:grf_net",
}
_COP_LABEL = "contact:free_moment_net"  # emitted at the net CoP (when it exists)
_BW_UNKNOWN = "body weight is unknown (glyph style has no body-weight scale)"


@dataclass(frozen=True)
class Curve:
    """One named line; NaN entries are gaps (unavailable, never zero)."""

    label: str
    values: NDArray[np.float64]
    color: str


@dataclass(frozen=True)
class Panel:
    """One axes of the strip: curves sharing a y axis."""

    ylabel: str
    curves: tuple[Curve, ...]


@dataclass(frozen=True)
class StripSeries:
    """The time series of one strip. ``times_s`` is the swing time of each sample."""

    kind: str
    times_s: NDArray[np.float64]
    panels: tuple[Panel, ...]
    impact_time_s: float | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("times_s must be 1-D with at least two samples")
        if not np.isfinite(times).all() or (np.diff(times) <= 0.0).any():
            raise ValueError("times_s must be finite and strictly increasing")
        if not self.panels:
            raise ValueError("a strip needs at least one panel")
        for panel in self.panels:
            if not panel.curves:
                raise ValueError("every panel needs at least one curve")
            for curve in panel.curves:
                if curve.values.shape != times.shape:
                    raise ValueError(f"curve {curve.label!r} length differs from times")
        if self.impact_time_s is not None and not math.isfinite(self.impact_time_s):
            raise ValueError("impact_time_s must be finite")

    def record(self) -> dict[str, Any]:
        """JSON-friendly availability record for the manifest."""
        return {
            "kind": self.kind,
            "t_start_s": float(self.times_s[0]),
            "t_end_s": float(self.times_s[-1]),
            "samples": int(self.times_s.size),
            "panels": [
                {
                    "ylabel": p.ylabel,
                    "curves": {
                        c.label: int(np.isfinite(c.values).sum()) for c in p.curves
                    },
                }
                for p in self.panels
            ],
        }


@dataclass(frozen=True)
class PanelMap:
    """Pixel mapping of one panel inside the rendered strip (strip coordinates)."""

    t0: float
    t1: float
    x0: int
    x1: int
    y0: int  # top row
    y1: int  # bottom row (exclusive)

    def x_at(self, t_s: float) -> int:
        """Column of time ``t_s`` (clamped to the panel; monotone in ``t_s``)."""
        frac = (t_s - self.t0) / (self.t1 - self.t0)
        frac = min(1.0, max(0.0, frac))
        return min(self.x1 - 1, self.x0 + int(round(frac * (self.x1 - self.x0 - 1))))


@dataclass(frozen=True)
class RenderedStrip:
    """A strip rendered once: straight-alpha RGBA pixels plus the time mapping."""

    rgba: NDArray[np.uint8]
    maps: tuple[PanelMap, ...]

    @property
    def width(self) -> int:
        return int(self.rgba.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgba.shape[0])

    def cursor_xs(self, t_s: float) -> tuple[int, ...]:
        """Cursor column of every panel at swing time ``t_s``."""
        return tuple(m.x_at(t_s) for m in self.maps)


# --- series extraction ---------------------------------------------------------------


def _source_indices(n_source: int) -> NDArray[np.int64]:
    count = min(n_source, MAX_SAMPLES)
    return np.unique(np.linspace(0, n_source - 1, count).round().astype(np.int64))


def _check_source(times: Sequence[float] | NDArray[np.float64], n_source: int) -> Any:
    if not isinstance(n_source, int) or n_source < 2:
        raise ValueError(f"n_source must be an integer >= 2, got {n_source!r}")
    arr = np.asarray(times, dtype=float)
    if arr.shape != (n_source,):
        raise ValueError(f"times must have shape ({n_source},), got {arr.shape}")
    return arr


def _wrench_map(frame: Any) -> dict[str, Any]:
    return {w.label: w for w in getattr(frame, "wrenches", ())}


def grf_series(
    feed: Any,
    n_source: int,
    times: Sequence[float] | NDArray[np.float64],
    impact_time_s: float | None = None,
) -> StripSeries | str:
    """GRF (BW) and net CoP series from the overlay feed, or a reason string.

    ``feed.frame_at(i)`` is read for a stride-thinned subset of the ``n_source``
    source samples (``times`` are their swing times). A foot without a loaded
    contact, and a CoP below the shared force threshold, are NaN gaps.
    """
    times_arr = _check_source(times, n_source)
    if feed is None:
        return "GRF strip unavailable: overlays are disabled"
    style = getattr(feed, "style", None)
    weight_n = getattr(style, "reference_force_n", None)
    if getattr(style, "scale_mode", None) != "body_weight" or not weight_n:
        return _BW_UNKNOWN
    idx = _source_indices(n_source)
    fz = {k: np.full(idx.size, np.nan) for k in _GRF_LABELS}
    cop = np.full((idx.size, 2), np.nan)
    for row, i in enumerate(idx):
        wrenches = _wrench_map(feed.frame_at(int(i)))
        for key, label in _GRF_LABELS.items():
            w = wrenches.get(label)
            if w is not None and w.force_n is not None:
                fz[key][row] = float(w.force_n[2]) / float(weight_n)
        mark = wrenches.get(_COP_LABEL)
        if mark is not None:
            cop[row] = (float(mark.point_m[0]), float(mark.point_m[1]))
    if not np.isfinite(fz["net"]).any():
        return "GRF strip unavailable: no sample has a loaded ground contact"
    t = times_arr[idx]
    panels = [
        Panel(
            "Vertical GRF (BW)",
            (
                Curve("left", fz["left"], _BLUE),
                Curve("right", fz["right"], _ORANGE),
                Curve("net", fz["net"], _YELLOW),
            ),
        )
    ]
    if np.isfinite(cop).any():
        panels.append(
            Panel(
                "Net CoP (m)",
                (Curve("X", cop[:, 0], _GREEN), Curve("Y", cop[:, 1], _PINK)),
            )
        )
    return StripSeries("grf", t, tuple(panels), impact_time_s)


def _norm_or_nan(vec: Any) -> float:
    return float("nan") if vec is None else float(np.linalg.norm(np.asarray(vec)))


def grip_series(
    feed: Any,
    n_source: int,
    times: Sequence[float] | NDArray[np.float64],
    impact_time_s: float | None = None,
) -> StripSeries | str:
    """Net grip force |F| (N) and couple |M| (N m) series, or a reason string."""
    times_arr = _check_source(times, n_source)
    analyses_of = getattr(feed, "grip_analyses", None)
    if analyses_of is None:
        return "grip strip unavailable: the grip overlay is not enabled"
    idx = _source_indices(n_source)
    analyses = analyses_of([int(i) for i in idx])
    if len(analyses) != idx.size:
        raise ValueError("grip_analyses returned a different number of samples")
    force = np.array([_norm_or_nan(a.net_force_n) for a in analyses])
    couple = np.array([_norm_or_nan(a.couple_at_midpoint_nm) for a in analyses])
    if not np.isfinite(force).any() and not np.isfinite(couple).any():
        why = next((a.unavailable_reason for a in analyses if a.unavailable_reason), "")
        return f"grip strip unavailable: {why or 'no grip wrench at any sample'}"
    panels = []
    if np.isfinite(force).any():
        panels.append(Panel("Net grip force (N)", (Curve("|F|", force, _YELLOW),)))
    if np.isfinite(couple).any():
        panels.append(Panel("Grip couple (N m)", (Curve("|M|", couple, _PINK),)))
    return StripSeries("grip", times_arr[idx], tuple(panels), impact_time_s)


# --- rendering -----------------------------------------------------------------------


def _check_size(width_px: int, height_px: int) -> None:
    for name, value, floor in (
        ("width_px", width_px, MIN_STRIP_PX[0]),
        ("height_px", height_px, MIN_STRIP_PX[1]),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < floor:
            raise ValueError(f"{name} must be an integer >= {floor}, got {value!r}")


def _style_axes(ax: Any, panel: Panel, font_pt: float) -> None:
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.5))
    ax.tick_params(colors="white", labelsize=font_pt, length=2, pad=1)
    ax.set_ylabel(panel.ylabel, color="white", fontsize=font_pt, labelpad=2)
    ax.grid(True, color=(1.0, 1.0, 1.0, 0.15), linewidth=0.5)
    ax.set_xlabel("swing time (s)", color="white", fontsize=font_pt, labelpad=1)
    legend = ax.legend(
        loc="upper left",
        fontsize=font_pt,
        ncol=len(panel.curves),
        frameon=False,
        borderpad=0.2,
        handlelength=1.2,
    )
    for text in legend.get_texts():
        text.set_color("white")


def render_strip(series: StripSeries, width_px: int, height_px: int) -> RenderedStrip:
    """Render ``series`` once to a ``(height_px, width_px, 4)`` straight-alpha image.

    Panels sit side by side on a dark translucent background with a legend,
    unit-bearing y labels, a shared swing-time axis and a dashed vertical line
    at the impact time. Raises ``ValueError`` for a bad series or size.
    """
    if not isinstance(series, StripSeries):
        raise TypeError("series must be a StripSeries")
    _check_size(width_px, height_px)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    font_pt = max(6.0, height_px * 0.075) * 72.0 / _DPI
    fig = Figure(
        figsize=((width_px + 0.5) / _DPI, (height_px + 0.5) / _DPI),
        dpi=_DPI,
        facecolor=_BACKGROUND,
    )
    canvas = FigureCanvasAgg(fig)
    n = len(series.panels)
    axes = fig.subplots(1, n, squeeze=False)[0]
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.27, top=0.95, wspace=0.28)
    t0, t1 = float(series.times_s[0]), float(series.times_s[-1])
    for ax, panel in zip(axes, series.panels, strict=True):
        for curve in panel.curves:
            ax.plot(
                series.times_s, curve.values, color=curve.color, lw=1.4,
                label=curve.label,
            )  # fmt: skip
        ax.set_xlim(t0, t1)
        impact = series.impact_time_s
        if impact is not None and t0 <= impact <= t1:
            ax.axvline(impact, color="white", ls="--", lw=1.0, alpha=0.9)
        _style_axes(ax, panel, font_pt)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    if rgba.shape[:2] != (height_px, width_px):
        raise RuntimeError(
            f"strip rendered as {rgba.shape[:2]}, expected {(height_px, width_px)}"
        )
    maps = []
    for ax in axes:
        box = ax.get_window_extent()
        maps.append(
            PanelMap(
                t0,
                t1,
                int(round(box.x0)),
                int(round(box.x1)),
                int(round(height_px - box.y1)),
                int(round(height_px - box.y0)),
            )
        )
    return RenderedStrip(rgba, tuple(maps))


def strip_bbox(
    frame_shape: tuple[int, ...], strip: RenderedStrip
) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` the strip covers in a frame of ``frame_shape``."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if strip.width > width or strip.height > height:
        raise ValueError("strip is larger than the frame")
    x0 = (width - strip.width) // 2
    y1 = max(strip.height, height - HUD_CLEARANCE_PX)
    return x0, y1 - strip.height, x0 + strip.width, y1


def strip_size_for(frame_shape: tuple[int, ...]) -> tuple[int, int]:
    """Strip ``(width, height)`` in pixels for a frame of ``frame_shape``."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    margin = max(4, int(width * STRIP_MARGIN_FRAC))
    return (
        max(MIN_STRIP_PX[0], width - 2 * margin),
        max(MIN_STRIP_PX[1], int(height * STRIP_HEIGHT_FRAC)),
    )


def draw_strip(frame: Image8, strip: RenderedStrip, t_s: float) -> Image8:
    """Return a copy of ``frame`` with the strip and a cursor at ``t_s`` drawn in.

    Only pixels inside :func:`strip_bbox` change: the strip is alpha-blended and
    a 2 px cursor is drawn in every panel at its column for ``t_s``.
    Raises ``ValueError`` for a non-finite ``t_s`` or a bad frame.
    """
    _check_image(frame)
    if not isinstance(strip, RenderedStrip):
        raise TypeError("strip must be a RenderedStrip")
    if not math.isfinite(t_s):
        raise ValueError("t_s must be finite")
    x0, y0, x1, y1 = strip_bbox(frame.shape, strip)
    out = frame.copy()
    alpha = strip.rgba[..., 3:4].astype(np.float32) / 255.0
    base = out[y0:y1, x0:x1].astype(np.float32)
    blended = base * (1.0 - alpha) + strip.rgba[..., :3].astype(np.float32) * alpha
    region = np.rint(blended).astype(np.uint8)
    for m, cx in zip(strip.maps, strip.cursor_xs(t_s), strict=True):
        region[m.y0 : m.y1, max(m.x0, cx - 1) : min(m.x1, cx + 1)] = _CURSOR
    out[y0:y1, x0:x1] = region
    return out


class StripRenderer:
    """Draws one :class:`StripSeries` on frames, rendering once per frame size."""

    def __init__(self, series: StripSeries) -> None:
        if not isinstance(series, StripSeries):
            raise TypeError("series must be a StripSeries")
        self.series = series
        self._rendered: dict[tuple[int, int], RenderedStrip] = {}

    def draw(self, frame: Image8, t_s: float) -> Image8:
        _check_image(frame)
        key = (int(frame.shape[0]), int(frame.shape[1]))
        if key not in self._rendered:
            width, height = strip_size_for(frame.shape)
            self._rendered[key] = render_strip(self.series, width, height)
        return draw_strip(frame, self._rendered[key], t_s)
