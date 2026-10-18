"""View model of the Variability & Consistency tab (NSX-100, #12729).

Pure data, no PyQt, matplotlib or numpy.  Everything shown is read from the
``sensitivity`` panels of an :class:`NSXPlotModel` (plot model 2.0.0, NSX-99):
decision labels, ``UNVALIDATED`` badges and hypothesis chips are the payload's
own text (via :func:`~.nsx_view.evidence_text`), a withheld item keeps its
reason and gets no number, and an unavailable panel says ``UNAVAILABLE:`` with
the engine's reason.  Nothing is computed, gated or defaulted here.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.loop_nullspace.plot_model import DecisionView, NSXPlotModel
from src.shared.python.loop_nullspace.plot_model_sensitivity import (
    DispersionEntry,
    KernelHeatmap,
    SensitivityPanels,
    StrategyParetoPanel,
    VarianceBudgetPanel,
)
from src.tools.grip_wrench_plots.nsx_view import evidence_text

__all__ = [
    "NOT_REQUESTED",
    "SECTION_NAMES",
    "PlotView",
    "SectionView",
    "VariabilityView",
    "variability_view",
]

SECTION_NAMES = (
    "kernel_heatmap",
    "variance_budget",
    "strategy_pareto",
    "face_dispersion",
)
TITLES = {
    "kernel_heatmap": "Kernel Heatmap",
    "variance_budget": "Variance Budget",
    "strategy_pareto": "Strategy Pareto",
    "face_dispersion": "Face Dispersion",
}
NOT_REQUESTED = "Variability analysis was not requested for this run."
UNAVAILABLE = "UNAVAILABLE: "
SHOWN = "shown"
WITHHELD = "withheld"
NO_NUMBER = "unavailable"


@dataclass(frozen=True)
class PlotView:
    """What to draw: ``kind`` is heatmap, bars, scatter or interval."""

    kind: str
    title: str
    xlabel: str
    ylabel: str
    labels: tuple[str, ...] = ()
    row_labels: tuple[str, ...] = ()
    x: tuple[float | None, ...] = ()
    y: tuple[float | None, ...] = ()
    y2: tuple[float | None, ...] = ()
    grid: tuple[tuple[float | None, ...], ...] = ()


@dataclass(frozen=True)
class SectionView:
    """One panel: evidence badge, status, table, notes and (if shown) a plot."""

    name: str
    title: str
    badge: str
    status: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    notes: tuple[str, ...]
    plot: PlotView | None


@dataclass(frozen=True)
class VariabilityView:
    """Header lines, the four sections and the non-claims."""

    status: str
    header: tuple[str, ...]
    sections: tuple[SectionView, ...]
    non_claims: tuple[str, ...]

    def section(self, name: str) -> SectionView:
        for sec in self.sections:
            if sec.name == name:
                return sec
        raise ValueError(f"unknown section {name!r}; expected one of {SECTION_NAMES}")

    def text(self) -> str:
        """All text of the view, one line per cell group (for tests and copy)."""
        lines = [self.status, *self.header, *self.non_claims]
        for sec in self.sections:
            lines += [sec.title, sec.badge, sec.status, *sec.columns, *sec.notes]
            lines += [cell for row in sec.rows for cell in row]
        return "\n".join(lines)


def fmt(value: float | None) -> str:
    """A number for display; no number is the word, never a zero."""
    return NO_NUMBER if value is None else f"{value:.5g}"


def _badge(decisions: list[DecisionView | None]) -> str:
    texts = dict.fromkeys(evidence_text(d) for d in decisions)
    return "\n".join(texts) if texts else "no quantity"


def _status(reason: str) -> str:
    return f"{WITHHELD}: {reason}"


def _kernel(k: KernelHeatmap) -> SectionView:
    cols = ("Time (s)", *k.input_labels)
    badge = _badge([k.decision])
    title = f"{k.outcome} ({k.units}) per input"
    if not k.shown:
        return SectionView(
            "kernel_heatmap",
            TITLES["kernel_heatmap"],
            badge,
            _status(k.reason),
            cols,
            (),
            (),
            None,
        )
    rows = []
    for t, lo, hi in zip(k.step_times_s, k.lower, k.upper, strict=True):
        cells = [
            f"{fmt(a)}..{fmt(b)}" if k.is_band else fmt(a)
            for a, b in zip(lo, hi, strict=True)
        ]
        rows.append((fmt(t), *cells))
    notes = (
        "family band (lower..upper)" if k.is_band else "allocation-invariant kernel",
    )
    plot = PlotView(
        "heatmap",
        title,
        "input",
        "time (s)",
        k.input_labels,
        tuple(fmt(t) for t in k.step_times_s),
        grid=k.lower,
    )
    return SectionView(
        "kernel_heatmap",
        TITLES["kernel_heatmap"],
        badge,
        SHOWN,
        cols,
        tuple(rows),
        notes,
        plot,
    )


def _budget(b: VarianceBudgetPanel) -> SectionView:
    cols = ("Input Class", "Variance", "Share")
    badge = _badge([b.decision])
    if not b.shown:
        return SectionView(
            "variance_budget",
            TITLES["variance_budget"],
            badge,
            _status(b.reason),
            cols,
            (),
            (),
            None,
        )
    rows = tuple((p.input_class, fmt(p.variance), fmt(p.share)) for p in b.parts)
    plot = PlotView("bars", f"{b.outcome} variance ({b.units})", "input class", b.units,
                    tuple(p.input_class for p in b.parts), y=tuple(p.variance for p in b.parts))  # fmt: skip
    return SectionView(
        "variance_budget", TITLES["variance_budget"], badge, SHOWN, cols, rows, (), plot
    )


PARETO_COLUMNS = (
    "Arm", "State", "Rank", "Variance Ratio", "Clubhead Speed (m/s)", "Effort", "Reason", "Evidence",
)  # fmt: skip


def _pareto(p: StrategyParetoPanel) -> SectionView:
    if p.unavailable_reason:
        return SectionView(
            "strategy_pareto",
            TITLES["strategy_pareto"],
            _badge([]),
            UNAVAILABLE + p.unavailable_reason,
            PARETO_COLUMNS,
            (),
            (),
            None,
        )
    rows = []
    for pt in p.points:
        effort = f"{fmt(pt.effort)} ({pt.effort_basis})" if pt.shown else NO_NUMBER
        rows.append((
            pt.label,
            SHOWN if pt.shown else WITHHELD,
            "" if pt.rank is None else str(pt.rank),
            fmt(pt.variance_ratio),
            fmt(pt.chs_mps),
            effort,
            pt.reason,
            evidence_text(pt.decision),
        ))  # fmt: skip
    for sk in p.skipped:
        rows.append(
            (sk.label, sk.status, "", NO_NUMBER, NO_NUMBER, NO_NUMBER, sk.reason, "")
        )
    notes = [f"ranking basis: {p.ranking_basis}", f"mode: {p.mode}"]
    notes += [
        f"front {f.name}: {', '.join(f.labels) or 'none'} (withheld: {f.withheld_count})"
        for f in p.fronts
    ]
    if p.winner:
        notes.append(f"winner: {p.winner}")
    notes += list(p.assumptions)
    if p.non_claim:
        notes.append(p.non_claim)
    shown = [pt for pt in p.points if pt.shown]
    plot = PlotView("scatter", "Variance ratio vs clubhead speed", "clubhead speed (m/s)", "variance ratio",
                    tuple(pt.label for pt in shown), x=tuple(pt.chs_mps for pt in shown), y=tuple(pt.variance_ratio for pt in shown)) if shown else None  # fmt: skip
    badge = _badge([pt.decision for pt in p.points])
    return SectionView(
        "strategy_pareto",
        TITLES["strategy_pareto"],
        badge,
        "",
        PARETO_COLUMNS,
        tuple(rows),
        tuple(notes),
        plot,
    )


DISPERSION_COLUMNS = (
    "Source",
    "State",
    "Lower",
    "Upper",
    "Units",
    "Reason",
    "Evidence",
)


def _entry_row(e: DispersionEntry) -> tuple[str, ...]:
    return (
        e.label,
        SHOWN if e.shown else WITHHELD,
        fmt(e.lower),
        fmt(e.upper),
        e.units,
        e.reason,
        evidence_text(e.decision),
    )


def _dispersion(panel: SensitivityPanels) -> SectionView:
    d = panel.face_dispersion
    name, title = "face_dispersion", TITLES["face_dispersion"]
    if d.unavailable_reason:
        return SectionView(
            name,
            title,
            _badge([]),
            UNAVAILABLE + d.unavailable_reason,
            DISPERSION_COLUMNS,
            (),
            (),
            None,
        )
    entries = [e for e in (d.linear, *d.arms) if e is not None]
    rows = [_entry_row(e) for e in entries]
    notes = []
    decisions: list[DecisionView | None] = [e.decision for e in entries]
    for mc in d.mc:
        decisions.append(mc.decision)
        state = SHOWN if mc.shown else WITHHELD
        rows.append(
            (
                f"mc:{mc.outcome}",
                state,
                fmt(mc.mc_sd),
                fmt(mc.predicted_sd),
                d.units,
                mc.reason,
                evidence_text(mc.decision),
            )
        )
        notes.append(
            f"Monte Carlo {mc.outcome} ({mc.status}): predicted {fmt(mc.predicted_sd)}, Monte Carlo {fmt(mc.mc_sd)}, gap {fmt(mc.gap)}, failure rate {fmt(mc.failure_rate)}"
        )
    shown = [e for e in entries if e.shown]
    plot = PlotView("interval", f"{d.outcome} standard deviation ({d.units})", "source", d.units,
                    tuple(e.label for e in shown), y=tuple(e.lower for e in shown), y2=tuple(e.upper for e in shown)) if shown else None  # fmt: skip
    return SectionView(
        name,
        title,
        _badge(decisions),
        "",
        DISPERSION_COLUMNS,
        tuple(rows),
        tuple(notes),
        plot,
    )


def _header(s: SensitivityPanels) -> tuple[str, ...]:
    if s.status != "complete":
        return (UNAVAILABLE + str(s.reason), f"stage status: {s.status}")
    mech = ", ".join(s.declared_mechanisms) or "none"
    out = [
        f"Primary outcome: {s.primary_outcome}",
        f"Outcomes: {', '.join(s.outcomes)}",
        f"Declared mechanisms: {mech}",
        f"Noise hypotheses: {', '.join(s.noise_hypotheses) or 'none'}",
        f"Linear validity: {s.linear_validity}",
        f"Cache identity: {s.cache_identity}",
    ]
    if s.skipped_priority:
        out.append(
            f"Outcomes skipped from the priority: {', '.join(s.skipped_priority)}"
        )
    return tuple(out)


def _unavailable_sections(s: SensitivityPanels) -> tuple[SectionView, ...]:
    reason = UNAVAILABLE + str(s.reason)
    return tuple(
        SectionView(n, TITLES[n], _badge([]), reason, (), (), (), None)
        for n in SECTION_NAMES
    )


def variability_view(model: NSXPlotModel) -> VariabilityView:
    """The view of ``model``'s sensitivity panels.

    A model with no ``sensitivity`` (a 1.0.0 model, or a run that did not ask)
    reads ``NOT_REQUESTED``.  A stage that did not run reads ``UNAVAILABLE:`` plus
    its reason in the status, the header and every section.

    Raises:
        TypeError: ``model`` is not an :class:`NSXPlotModel`.
    """
    if not isinstance(model, NSXPlotModel):
        raise TypeError("model must be an NSXPlotModel")
    s = model.sensitivity
    if s is None:
        return VariabilityView(NOT_REQUESTED, (), (), ())
    if s.status != "complete":
        status = UNAVAILABLE + str(s.reason)
        return VariabilityView(
            status, _header(s), _unavailable_sections(s), s.non_claims
        )
    sections = (
        _kernel(s.kernel_heatmap),
        _budget(s.variance_budget),
        _pareto(s.strategy_pareto),
        _dispersion(s),
    )
    return VariabilityView("complete", _header(s), sections, s.non_claims)
