"""Title Case markdown summary of a swing report package (#12327).

Style follows ``swing_comparison.report.comparison_to_markdown``. Advertised
quantities show sample counts and a value range per series; withheld
quantities are listed with their reason only, never with any number from an
engine.
"""

from __future__ import annotations

import logging

from .contract import Quantity, Series, SwingReport
from .policy import STATUS_WITHHELD

logger = logging.getLogger(__name__)


def _title(text: str) -> str:
    return text.replace("_", " ").title()


def _range(series: Series) -> str:
    finite = [v for v in series.values if v is not None]
    if not finite:
        return "N/A"
    return f"{min(finite):.3f} to {max(finite):.3f}"


def _series_row(series: Series) -> str:
    missing = sum(v is None for v in series.values)
    engine = series.engine + (f" ({series.channel})" if series.channel else "")
    return (
        f"| {engine} | {series.units} | {series.frame} | {len(series.values)} "
        f"| {missing} | {_range(series)} |"
    )


def _advertised_block(quantity: Quantity) -> list[str]:
    unc = quantity.uncertainty
    lines = [f"### {_title(quantity.quantity_id)}", ""]
    if unc is not None:
        lines += [
            f"Uncertainty: {_title(unc.kind)}, "
            f"{_title(unc.budget_source)} ({unc.units}).",
            "",
        ]
    lines += [
        "| Engine | Units | Frame | Samples | Missing | Range |",
        "| :--- | :--- | :--- | ---: | ---: | :--- |",
        *(_series_row(s) for s in quantity.series),
        "",
    ]
    return lines


def _provenance_lines(report: SwingReport) -> list[str]:
    prov = report.provenance
    return [
        "## Provenance",
        "",
        f"- Report ID: {report.report_id}",
        f"- Scope: {report.scope}",
        f"- Generated: {report.generated_utc}",
        f"- Software: {prov.software_name} {prov.software_version}",
        f"- Capture: {prov.capture.capture_id} ({prov.capture.visibility})",
        "",
        "## Runs",
        "",
        "| Run ID | Engine | Version | Run Class |",
        "| :--- | :--- | :--- | :--- |",
        *(
            f"| {r.run_id} | {r.engine} | {r.engine_version} | {r.run_class} |"
            for r in report.runs
        ),
        "",
    ]


def _plot_lines(report: SwingReport) -> list[str]:
    lines = ["## Plots", ""]
    for plot in report.plots:
        if plot.render == "values":
            lines.append(f"- {plot.title}: plots/{plot.plot_id}.png")
        else:
            lines.append(f"- {plot.title}: unavailable ({plot.unavailable_reason})")
    return [*lines, ""]


def render_summary(report: SwingReport) -> str:
    """Render ``report`` as Title Case GitHub-flavored markdown."""
    advertised = [q for q in report.quantities if q.status != STATUS_WITHHELD]
    withheld = [q for q in report.quantities if q.status == STATUS_WITHHELD]
    lines = ["# Swing Report", "", *_provenance_lines(report)]
    lines += ["## Advertised Quantities", ""]
    if not advertised:
        lines += ["None: no quantity has comparable data from two engines.", ""]
    for quantity in advertised:
        lines += _advertised_block(quantity)
    lines += ["## Withheld Quantities", ""]
    lines += [
        f"- {_title(q.quantity_id)}: {q.reason or 'not validated'}" for q in withheld
    ]
    lines.append("")
    lines += _plot_lines(report)
    notes = [s.notes for s in report.sections if s.notes]
    if notes:
        lines += ["## Notes", "", *(f"- {n}" for n in notes), ""]
    return "\n".join(lines)
