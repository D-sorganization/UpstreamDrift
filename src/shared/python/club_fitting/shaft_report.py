"""Shaft report: one page per flex and a table across flexes (#15150 C12).

From a sweep result (``flex_sweep.json``), per flex: a page with lead and droop
of the tip against time up to impact over the impact and flight values; plus
``<stem>.json``/``<stem>.md`` tables. Matplotlib Agg, no Qt; styling from
``src/config/shaft_report.json``. Same hand motion for every flex (the golfer
does not adapt). The sweep CLI writes it with ``--report``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import logging
from pathlib import Path
import re
from typing import Any

from src.shared.python.club_fitting.bend_trace import BendTrace
from src.shared.python.club_fitting.grip_motion import _positive

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


_SETTING_KEYS = {f.name for f in fields(ReportSettings)}


def load_report_settings(path: Path | None = None) -> ReportSettings:
    """Settings from ``path`` (default ``src/config/shaft_report.json``).

    Raises:
        ValueError: a missing or unknown key, or a value out of range.
    """
    data = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    if set(data) != _SETTING_KEYS:
        diff = sorted(set(data) ^ _SETTING_KEYS)
        raise ValueError(f"shaft_report settings keys differ from the schema: {diff}")
    data["metrics"] = tuple(Metric(**m) for m in data["metrics"])
    data["page_size_in"] = tuple(data["page_size_in"])
    data["height_ratios"] = tuple(data["height_ratios"])
    return ReportSettings(**data)


@dataclass(frozen=True)
class SweepResult:
    """Sweep rows (flat records, one per flex) and their bend traces, if any."""

    rows: tuple[dict[str, Any], ...]
    traces: dict[str, BendTrace]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("the sweep result has no rows")
        if any("flex" not in r for r in self.rows):
            raise ValueError("every sweep row needs a 'flex'")


def load_sweep_result(path: Path) -> SweepResult:
    """Read ``flex_sweep.json``; ``traces`` is optional (older sweeps)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = {k: BendTrace.from_dict(v) for k, v in data.get("traces", {}).items()}
    return SweepResult(tuple(data.get("rows", ())), traces)


def summary_rows(result: SweepResult, settings: ReportSettings) -> list[dict]:
    """One record per flex: ``flex`` and every settings metric (``None`` if
    the sweep does not provide it)."""
    out = []
    for row in result.rows:
        trace = result.traces.get(row["flex"])
        lead, droop = (None, None) if trace is None else trace.at_impact()
        known = {**row, "lead_at_impact_mm": lead, "droop_at_impact_mm": droop}
        out.append(
            {"flex": row["flex"], **{m.key: known.get(m.key) for m in settings.metrics}}
        )
    return out


def _text(value: Any, metric: Metric, settings: ReportSettings) -> str:
    return settings.missing_text if value is None else metric.format.format(value)


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
    cells = [[m.label, _text(record.get(m.key), m, s)] for m in s.metrics]
    table = bottom.table(cellText=cells, colLabels=["Quantity", "Value"], loc="center")
    table.set_fontsize(s.font_size)
    for ax in (top, bottom):
        ax.tick_params(labelsize=s.font_size)
    return fig


def _markdown(records: list[dict], settings: ReportSettings) -> str:
    head = ["Flex", *(m.label for m in settings.metrics)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in records:
        cells = [_text(r[m.key], m, settings) for m in settings.metrics]
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
    md_path.write_text(_markdown(records, s), encoding="utf-8")
    return [*paths, json_path, md_path]
