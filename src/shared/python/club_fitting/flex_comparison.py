"""Flex comparison sheet: every flex's flight overlaid and a table (#15150 C5).

From a sweep result (``flex_sweep.json``): the ball flight of each flex on one
side view (downrange against height) and one plan view (downrange against
side), one colour per flex, the shaft centrelines at impact in the bend plane
(lead and droop against the distance along the shaft, true aspect with the
deflection exaggerated by a setting), and a table of head speed, ball speed,
carry, side and face per flex. Matplotlib Agg, no Qt; styling from
``src/config/flex_comparison.json``. Same hand motion for every flex (the
golfer does not adapt). The sweep CLI writes it with ``--comparison``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.club_fitting.grip_motion import _positive
from src.shared.python.club_fitting.shaft_report import (
    _UNSAFE,
    Metric,
    SweepResult,
    load_settings_json,
    markdown_table,
    metric_rows,
    metric_text,
)

SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "flex_comparison.json"


@dataclass(frozen=True)
class ComparisonSettings:
    """Validated styling and layout of the flex comparison sheet."""

    sheet_stem: str
    page_size_in: tuple[float, float]
    dpi: int
    font_size: float
    title_font_size: float
    title: str
    grid_alpha: float
    height_ratios: tuple[float, float, float, float]
    line_width: float
    flex_colors: tuple[str, ...]
    distance_scale: float
    downrange_label: str
    height_label: str
    side_label: str
    side_view_title: str
    plan_view_title: str
    no_flight_text: str
    shape_exaggeration: float
    shape_length_scale: float
    shape_along_label: str
    shape_lead_title: str
    shape_droop_title: str
    shape_deflection_label: str
    shape_exaggeration_format: str
    no_shape_text: str
    missing_text: str
    metrics: tuple[Metric, ...]

    def __post_init__(self) -> None:
        for name in ("dpi", "font_size", "title_font_size", "line_width",
                     "distance_scale", "grid_alpha", "shape_exaggeration",
                     "shape_length_scale"):  # fmt: skip
            _positive(name, getattr(self, name))
        for name, size in (("page_size_in", 2), ("height_ratios", 4)):
            if len(getattr(self, name)) != size:
                raise ValueError(f"{name} must hold {size} numbers")
            for value in getattr(self, name):
                _positive(name, value)
        if not self.sheet_stem or _UNSAFE.search(self.sheet_stem):
            raise ValueError("sheet_stem must be a non-empty file-safe name")
        if not self.flex_colors:
            raise ValueError("flex_colors must list at least one colour")
        if not self.metrics:
            raise ValueError("metrics must list at least one entry")

    def color(self, index: int) -> str:
        """Colour of the ``index``-th flex (the list repeats past its end)."""
        return self.flex_colors[index % len(self.flex_colors)]


def load_comparison_settings(path: Path | None = None) -> ComparisonSettings:
    """Settings from ``path`` (default ``src/config/flex_comparison.json``).

    Raises:
        ValueError: a missing or unknown key, or a value out of range.
    """
    keys = ("page_size_in", "height_ratios", "flex_colors")
    return load_settings_json(path or SETTINGS_PATH, ComparisonSettings, keys)


def _plot_flights(side: Any, plan: Any, result: SweepResult, s: ComparisonSettings):
    """Overlay each flex's flight on both views; a note where none exists."""
    for i, row in enumerate(result.rows):
        flight = result.flights.get(row["flex"])
        if flight is None:
            continue
        x = flight.downrange_m * s.distance_scale
        style = {"color": s.color(i), "lw": s.line_width, "label": row["flex"]}
        side.plot(x, flight.height_m * s.distance_scale, **style)
        plan.plot(x, flight.side_m * s.distance_scale, **style)
    views = ((side, s.side_view_title, s.height_label),
             (plan, s.plan_view_title, s.side_label))  # fmt: skip
    for ax, title, ylabel in views:
        ax.set(title=title, xlabel=s.downrange_label, ylabel=ylabel)
        ax.grid(True, alpha=s.grid_alpha)
        ax.tick_params(labelsize=s.font_size)
        if ax.get_lines():
            ax.legend(loc="best", fontsize=s.font_size)
        else:
            ax.text(0.5, 0.5, s.no_flight_text, ha="center", va="center")


def _plot_shapes(lead: Any, droop: Any, result: SweepResult, s: ComparisonSettings):
    """Each flex's centreline at impact; deflection times the exaggeration.

    Both axes share one length unit at equal aspect, and the deflection tick
    labels are divided back by the exaggeration so they read true values.
    """
    from matplotlib.ticker import FuncFormatter

    k = s.shape_exaggeration
    for i, row in enumerate(result.rows):
        shape = result.shapes.get(row["flex"])
        if shape is None:
            continue
        x = shape.along_m * s.shape_length_scale
        style = {"color": s.color(i), "lw": s.line_width, "label": row["flex"]}
        lead.plot(x, shape.lead_m * s.shape_length_scale * k, **style)
        droop.plot(x, shape.droop_m * s.shape_length_scale * k, **style)
    for ax, title in ((lead, s.shape_lead_title), (droop, s.shape_droop_title)):
        ax.set(title=title, xlabel=s.shape_along_label,
               ylabel=s.shape_deflection_label)  # fmt: skip
        ax.grid(True, alpha=s.grid_alpha)
        ax.tick_params(labelsize=s.font_size)
        if not ax.get_lines():
            ax.text(0.5, 0.5, s.no_shape_text, ha="center", va="center")
            continue
        ax.set_aspect("equal", adjustable="datalim")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / k:g}"))
        ax.legend(loc="best", fontsize=s.font_size)
        if k != 1.0:
            ax.text(0.01, 0.02, s.shape_exaggeration_format.format(k),
                    transform=ax.transAxes, fontsize=s.font_size)  # fmt: skip


def _table_cells(records: list[dict], s: ComparisonSettings) -> list[list[str]]:
    return [
        [str(r["flex"]), *(metric_text(r[m.key], m, s.missing_text) for m in s.metrics)]
        for r in records
    ]


def render_comparison(result: SweepResult, settings: ComparisonSettings):
    """Agg figure: side and plan views of every flight, the shaft shapes at
    impact (lead and droop side by side) and the table."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    s = settings
    fig = Figure(figsize=s.page_size_in, dpi=s.dpi, layout="constrained")
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(4, 2, height_ratios=s.height_ratios)
    side, plan = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :])
    lead, droop = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])
    bottom = fig.add_subplot(grid[3, :])
    fig.suptitle(s.title, fontsize=s.title_font_size)
    _plot_flights(side, plan, result, s)
    _plot_shapes(lead, droop, result, s)
    bottom.axis("off")
    records = metric_rows(result, s.metrics)
    table = bottom.table(
        cellText=_table_cells(records, s),
        colLabels=["Flex", *(m.label for m in s.metrics)],
        loc="center",
    )
    table.set_fontsize(s.font_size)
    for i in range(len(records)):
        table[i + 1, 0].set_text_props(color=s.color(i))
    return fig


def write_comparison(
    result: SweepResult, out_dir: Path, settings: ComparisonSettings | None = None
) -> list[Path]:
    """Write ``<stem>.png`` (the sheet) and ``<stem>.md`` (its table)."""
    s = settings or load_comparison_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    png, md = out_dir / f"{s.sheet_stem}.png", out_dir / f"{s.sheet_stem}.md"
    render_comparison(result, s).savefig(png)
    records = metric_rows(result, s.metrics)
    md.write_text(markdown_table(records, s.metrics, s.missing_text), encoding="utf-8")
    return [png, md]
