"""Shaft report: one page per flex and a table across flexes (#15150 C12).

From a sweep result (``flex_sweep.json``), per flex: a page with lead and droop
of the tip against time up to impact over the impact and flight values; plus
``<stem>.json``/``<stem>.md`` tables. Matplotlib Agg, no Qt; styling from
``src/config/shaft_report.json``. Same hand motion for every flex (the golfer
does not adapt). The sweep CLI writes it with ``--report``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
import logging
from pathlib import Path
import re
from typing import Any

from src.shared.python.club_fitting.bend_trace import BendTrace
from src.shared.python.club_fitting.flight_trace import FlightTrace
from src.shared.python.club_fitting.grip_motion import _positive
from src.shared.python.club_fitting.shaft_shape import ShaftShape

logger = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "shaft_report.json"
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass(frozen=True)
class Metric:
    """One table entry: a sweep-row key, its label and a format string."""

    key: str
    label: str
    format: str


@dataclass(frozen=True)
class ReportSettings:
    """Validated styling and layout of the shaft report."""

    page_stem: str
    page_size_in: tuple[float, float]
    dpi: int
    font_size: float
    title_font_size: float
    grid_alpha: float
    height_ratios: tuple[float, float]
    lead_color: str
    droop_color: str
    impact_color: str
    line_width: float
    time_scale: float
    time_label: str
    bend_label: str
    missing_text: str
    metrics: tuple[Metric, ...]

    def __post_init__(self) -> None:
        for name in ("dpi", "font_size", "title_font_size", "line_width",
                     "time_scale", "grid_alpha"):  # fmt: skip
            _positive(name, getattr(self, name))
        for name in ("page_size_in", "height_ratios"):
            if len(getattr(self, name)) != 2:
                raise ValueError(f"{name} must hold two numbers")
            for value in getattr(self, name):
                _positive(name, value)
        if not self.page_stem or _UNSAFE.search(self.page_stem):
            raise ValueError("page_stem must be a non-empty file-safe name")
        if not self.metrics:
            raise ValueError("metrics must list at least one entry")


def load_settings_json(path: Path, cls: type, tuple_keys: tuple[str, ...]) -> Any:
    """``cls`` built from the JSON at ``path``: ``_``-keys dropped, ``metrics``
    as :class:`Metric` entries and ``tuple_keys`` as tuples.

    Raises:
        ValueError: a missing or unknown key, or a value ``cls`` rejects.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    schema = {f.name for f in fields(cls)}
    if set(data) != schema:
        diff = sorted(set(data) ^ schema)
        raise ValueError(f"{path.name} keys differ from the schema: {diff}")
    data["metrics"] = tuple(Metric(**m) for m in data["metrics"])
    data |= {k: tuple(data[k]) for k in tuple_keys}
    return cls(**data)


def load_report_settings(path: Path | None = None) -> ReportSettings:
    """Settings from ``path`` (default ``src/config/shaft_report.json``)."""
    keys = ("page_size_in", "height_ratios")
    return load_settings_json(path or SETTINGS_PATH, ReportSettings, keys)


@dataclass(frozen=True)
class SweepResult:
    """Sweep rows (flat records, one per flex), their bend traces, ball
    flights and impact shaft shapes, where the sweep has them."""

    rows: tuple[dict[str, Any], ...]
    traces: dict[str, BendTrace]
    flights: dict[str, FlightTrace] = field(default_factory=dict)
    shapes: dict[str, ShaftShape] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("the sweep result has no rows")
        if any("flex" not in r for r in self.rows):
            raise ValueError("every sweep row needs a 'flex'")


def load_sweep_result(path: Path) -> SweepResult:
    """Read ``flex_sweep.json``; ``traces``, ``flights`` and ``shapes`` (absent
    from sweeps written before #15150 C5b) are optional."""
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = {k: BendTrace.from_dict(v) for k, v in data.get("traces", {}).items()}
    flights = {k: FlightTrace.from_dict(v) for k, v in data.get("flights", {}).items()}
    shapes = {k: ShaftShape.from_dict(v) for k, v in data.get("shapes", {}).items()}
    return SweepResult(tuple(data.get("rows", ())), traces, flights, shapes)


def summary_rows(result: SweepResult, settings: ReportSettings) -> list[dict]:
    """One record per flex: ``flex`` and every settings metric (``None`` if
    the sweep does not provide it)."""
    return metric_rows(result, settings.metrics)


def metric_rows(result: SweepResult, metrics: tuple[Metric, ...]) -> list[dict]:
    """:func:`summary_rows` for any metric list."""
    out = []
    for row in result.rows:
        trace = result.traces.get(row["flex"])
        lead, droop = (None, None) if trace is None else trace.at_impact()
        known = {**row, "lead_at_impact_mm": lead, "droop_at_impact_mm": droop}
        out.append({"flex": row["flex"], **{m.key: known.get(m.key) for m in metrics}})
    return out


def metric_text(value: Any, metric: Metric, missing: str) -> str:
    """``value`` in the metric's format, ``missing`` for ``None``."""
    return missing if value is None else metric.format.format(value)


def _plot_trace(ax: Any, trace: BendTrace | None, settings: ReportSettings) -> None:
    s = settings
    if trace is None:
        ax.text(0.5, 0.5, "No bend trace in this sweep", ha="center", va="center")
        return
    t = trace.time_s * s.time_scale
    ax.plot(t, trace.lead_mm, color=s.lead_color, lw=s.line_width, label="Lead")
    ax.plot(t, trace.droop_mm, color=s.droop_color, lw=s.line_width, label="Droop")
    ax.axvline(0.0, color=s.impact_color, ls="--", lw=s.line_width, label="Impact")
    ax.set(xlabel=s.time_label, ylabel=s.bend_label)
    ax.grid(True, alpha=s.grid_alpha)
    ax.legend(loc="best")


def render_page(record: dict, trace: BendTrace | None, settings: ReportSettings):
    """Agg figure for one flex: bend against time over a table of values."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    s = settings
    fig = Figure(figsize=s.page_size_in, dpi=s.dpi)
    FigureCanvasAgg(fig)
    top, bottom = fig.subplots(2, 1, gridspec_kw={"height_ratios": s.height_ratios})
    fig.suptitle(f"{record['flex']} Shaft", fontsize=s.title_font_size)
    _plot_trace(top, trace, s)
    bottom.axis("off")
    cells = [[m.label, metric_text(record.get(m.key), m, s.missing_text)]
             for m in s.metrics]  # fmt: skip
    table = bottom.table(cellText=cells, colLabels=["Quantity", "Value"], loc="center")
    table.set_fontsize(s.font_size)
    for ax in (top, bottom):
        ax.tick_params(labelsize=s.font_size)
    return fig


def markdown_table(
    records: list[dict], metrics: tuple[Metric, ...], missing: str
) -> str:
    """Markdown table: a flex column then one column per metric."""
    head = ["Flex", *(m.label for m in metrics)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in records:
        cells = [metric_text(r[m.key], m, missing) for m in metrics]
        lines.append("| " + " | ".join([str(r["flex"]), *cells]) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    result: SweepResult, out_dir: Path, settings: ReportSettings | None = None
) -> list[Path]:
    """Write one PNG page per flex, then ``<stem>.json`` and ``<stem>.md``."""
    s = settings or load_report_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = summary_rows(result, s)
    paths = []
    for record in records:
        fig = render_page(record, result.traces.get(record["flex"]), s)
        path = out_dir / f"{s.page_stem}_{_UNSAFE.sub('_', str(record['flex']))}.png"
        fig.savefig(path)
        paths.append(path)
    table = {"columns": ["flex", *(m.key for m in s.metrics)], "rows": records}
    json_path, md_path = out_dir / f"{s.page_stem}.json", out_dir / f"{s.page_stem}.md"
    json_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    md_path.write_text(
        markdown_table(records, s.metrics, s.missing_text), encoding="utf-8"
    )
    return [*paths, json_path, md_path]
