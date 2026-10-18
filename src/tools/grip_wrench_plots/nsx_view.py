"""View model of the NSX tabs: what each panel shows, from the plot model only.

NSX-43 (#12575).  Pure data, no PyQt, matplotlib or numpy: the Qt tabs draw
exactly the :class:`PanelView` this module derives from an
:class:`~src.shared.python.loop_nullspace.plot_model.NSXPlotModel`.  Nothing is
computed or gated here -- every shown / withheld / band / reason state is the
model's, i.e. the ``display_decision`` outcome recorded in it.  A value the
model does not mark shown is never given a number, and an unknown state is
withheld.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from src.shared.python.loop_nullspace.plot_model import (
    Band,
    DecisionView,
    FrameValue,
    NSXPlotModel,
    Overlay,
    ValueSeries,
    WithheldItem,
)

__all__ = [
    "PANEL_TABS",
    "CurveView",
    "LineView",
    "PanelView",
    "RowView",
    "SegmentView",
    "evidence_text",
    "panel_view",
    "panel_views",
]

PANEL_TABS = ("invariants", "envelopes", "hand_split", "diagnostics")
TAB_TITLES = {
    "invariants": "Invariants",
    "envelopes": "Envelopes",
    "hand_split": "Hand Split",
    "diagnostics": "Diagnostics",
}
NO_DECISION = "no display decision"
SHOWN = "shown"
PARTIAL = "partial"
WITHHELD = "withheld"


@dataclass(frozen=True)
class SegmentView:
    """A band segment: frames, display class and the rows binding it."""

    first_frame_index: int
    last_frame_index: int
    display_class: str
    sign_class: str | None
    binding_hypotheses: tuple[str, ...]


@dataclass(frozen=True)
class LineView:
    """One drawn line; ``None`` entries are frames with no number."""

    label: str
    values: tuple[float | None, ...]


@dataclass(frozen=True)
class CurveView:
    """One axes: lines, or a band (``lower``/``upper``) with its overlay."""

    name: str
    units: str
    frames: tuple[int, ...]
    lines: tuple[LineView, ...]
    lower: tuple[float | None, ...] | None
    upper: tuple[float | None, ...] | None
    segments: tuple[SegmentView, ...]
    overlay: LineView | None
    infeasible_frames: tuple[int, ...]


@dataclass(frozen=True)
class RowView:
    """One listed quantity with its evidence and why it is not drawn."""

    name: str
    units: str
    state: str
    evidence: str
    reason: str
    detail: str


@dataclass(frozen=True)
class PanelView:
    """Everything one tab shows."""

    panel: str
    title: str
    badge: str
    status: str
    rows: tuple[RowView, ...]
    curves: tuple[CurveView, ...]
    notes: tuple[str, ...]


def evidence_text(decision: DecisionView | None) -> str:
    """The evidence badge line of one decision (``display_decision`` outcome).

    ``action: label | badge, badge | hypotheses: ... | conditioning: ...``; a
    missing decision reads ``no display decision``.
    """
    if decision is None:
        return NO_DECISION
    head = f"{decision.action}: {decision.label}" if decision.label else decision.action
    parts = [head]
    if decision.badges:
        parts.append(", ".join(decision.badges))
    if decision.hypothesis_chips:
        parts.append("hypotheses: " + ", ".join(decision.hypothesis_chips))
    if decision.conditioning_chips:
        parts.append("conditioning: " + ", ".join(decision.conditioning_chips))
    if decision.preset:
        parts.append(f"preset: {decision.preset}")
    return " | ".join(parts)


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _badge(decisions: Sequence[DecisionView | None]) -> str:
    texts = _unique([evidence_text(d) for d in decisions])
    return "\n".join(texts) if texts else "no quantity"


def _withheld_row(item: WithheldItem) -> RowView:
    return RowView(item.quantity_id, "", WITHHELD, item.action, item.reason, "")


def _band_segments(band: Band) -> tuple[SegmentView, ...]:
    return tuple(
        SegmentView(
            s.first_frame_index,
            s.last_frame_index,
            s.display_class.value,
            s.sign_class,
            s.binding_hypotheses,
        )
        for s in band.segments
    )


def _band_curve(band: Band, overlay: Overlay | None) -> CurveView:
    return CurveView(
        band.quantity_id,
        band.units,
        band.frame_indices,
        (),
        band.lower,
        band.upper,
        _band_segments(band),
        None if overlay is None else LineView("representative", overlay.values),
        band.infeasible_frame_indices,
    )


def _band_row(band: Band, overlay: Overlay | None) -> RowView:
    evidence = evidence_text(band.decision)
    detail = ", ".join(sorted({h for s in band.segments for h in s.binding_hypotheses}))
    if overlay is not None:
        evidence += " || " + evidence_text(overlay.decision)
    return RowView(
        band.quantity_id,
        band.units,
        SHOWN,
        evidence,
        band.decision.reason,
        f"binding: {detail}" if detail else "",
    )


def _overlay_map(model: NSXPlotModel) -> dict[str, Overlay]:
    return {o.band_id: o for o in model.allocation.overlays}


def _allocation_notes(model: NSXPlotModel) -> tuple[str, ...]:
    alloc = model.allocation
    notes: list[str] = []
    if alloc.unavailable_reason:
        notes.append(f"allocation unavailable: {alloc.unavailable_reason}")
    elif alloc.status is not None:
        notes.append(f"allocation: {alloc.status}")
    if alloc.non_claim:
        notes.append(alloc.non_claim)
    return tuple(notes)


def _band_panel(
    panel: str,
    bands: Sequence[Band],
    withheld: Sequence[WithheldItem],
    unavailable: str | None,
    overlays: dict[str, Overlay],
    notes: tuple[str, ...],
) -> PanelView:
    rows = tuple(_band_row(b, overlays.get(b.quantity_id)) for b in bands) + tuple(
        _withheld_row(w) for w in withheld
    )
    curves = tuple(_band_curve(b, overlays.get(b.quantity_id)) for b in bands)
    status = f"unavailable: {unavailable}" if unavailable else ""
    return PanelView(
        panel,
        TAB_TITLES[panel],
        _badge([b.decision for b in bands]),
        status,
        rows,
        curves,
        notes,
    )


def _envelopes_view(model: NSXPlotModel) -> PanelView:
    env = model.envelopes
    notes = _allocation_notes(model) + tuple(
        f"allocation withheld {w.quantity_id}: {w.reason}"
        for w in model.allocation.withheld
    )
    return _band_panel(
        "envelopes",
        env.bands,
        env.withheld,
        env.unavailable_reason,
        _overlay_map(model),
        notes,
    )


def _hand_split_view(model: NSXPlotModel) -> PanelView:
    split = model.hand_split
    return _band_panel(
        "hand_split",
        split.bands,
        split.withheld,
        split.unavailable_reason,
        _overlay_map(model),
        (f"frame: {split.frame}",),
    )


def _frame_row(name: str, frames: Sequence[FrameValue]) -> RowView:
    shown = [f for f in frames if f.shown]
    state = WITHHELD if not shown else SHOWN if len(shown) == len(frames) else PARTIAL
    evidence = " / ".join(_unique([evidence_text(f.decision) for f in frames]))
    reason = " / ".join(_unique([f.reason for f in frames if f.reason]))
    return RowView(name, "", state, evidence, reason, f"{len(shown)}/{len(frames)}")


def _invariants_view(model: NSXPlotModel) -> PanelView:
    inv = model.invariants
    rows: list[RowView] = []
    curves: list[CurveView] = []
    decisions: list[DecisionView | None] = []
    for series in inv.series:
        frames = series.frames
        decisions.extend(f.decision for f in frames)
        row = _frame_row(series.name, frames)
        rows.append(replace(row, units=series.units))
        width = max((len(f.values) for f in frames), default=0)
        lines = tuple(
            LineView(
                series.name if width == 1 else f"{series.name}[{k}]",
                tuple(
                    f.values[k] if f.shown and k < len(f.values) else None
                    for f in frames
                ),
            )
            for k in range(width)
        )
        if lines:
            indices = tuple(f.frame_index for f in frames)
            curves.append(
                CurveView(
                    series.name, series.units, indices, lines, None, None, (), None, ()
                )
            )
    status = f"unavailable: {inv.unavailable_reason}" if inv.unavailable_reason else ""
    return PanelView(
        "invariants",
        TAB_TITLES["invariants"],
        _badge(decisions),
        status,
        tuple(rows),
        tuple(curves),
        (),
    )


def _residual_row(series: ValueSeries) -> RowView:
    return RowView(
        f"residual {series.name}",
        series.units,
        SHOWN if series.shown else WITHHELD,
        evidence_text(series.decision),
        series.reason,
        "",
    )


def _residual_curve(series: ValueSeries, frames: tuple[int, ...]) -> CurveView:
    return CurveView(
        f"residual {series.name}",
        series.units,
        frames,
        (LineView(series.name, series.values),),
        None,
        None,
        (),
        None,
        (),
    )


def _diagnostic_rows(model: NSXPlotModel) -> tuple[RowView, ...]:
    diag = model.diagnostics
    rows: list[RowView] = []
    adm = diag.admission
    if adm is not None:
        reasons = "; ".join(adm.reasons)
        detail = f"derivatives: {adm.derivative_method}, noise floor {adm.noise_floor}"
        rows.append(RowView("admission", "", adm.verdict, "", reasons, detail))
    for f in diag.frames:
        detail = (
            f"rank A {f.rank_A}, nullity {f.nullity}, rank J_c {f.rank_J_c}, "
            f"condition {f.condition_number}"
        )
        rows.append(RowView(f"frame {f.index}", "", SHOWN, "", "", detail))
    kkt = diag.kkt
    if kkt is not None:
        detail = f"primal {kkt.primal}, dual {kkt.dual}, complementarity {kkt.complementarity}"
        rows.append(RowView("allocation KKT", "", SHOWN, "", "", detail))
    return tuple(rows)


def _diagnostics_view(model: NSXPlotModel) -> PanelView:
    diag = model.diagnostics
    frames = tuple(f.index for f in diag.frames)
    rows = _diagnostic_rows(model) + tuple(_residual_row(r) for r in diag.residuals)
    curves = [
        _residual_curve(r, model.frame_indices) for r in diag.residuals if r.shown
    ]
    condition = tuple(f.condition_number for f in diag.frames)
    if frames and any(c is not None for c in condition):
        line = LineView("condition number", condition)
        curves.insert(
            0,
            CurveView(
                "condition number", "1", frames, (line,), None, None, (), None, ()
            ),
        )
    return PanelView(
        "diagnostics",
        TAB_TITLES["diagnostics"],
        _badge([r.decision for r in diag.residuals]),
        "",
        rows,
        tuple(curves),
        (),
    )


_BUILDERS = {
    "invariants": _invariants_view,
    "envelopes": _envelopes_view,
    "hand_split": _hand_split_view,
    "diagnostics": _diagnostics_view,
}


def panel_view(model: NSXPlotModel, panel: str) -> PanelView:
    """The view of one tab.

    Raises:
        TypeError: ``model`` is not an :class:`NSXPlotModel`.
        ValueError: ``panel`` is not one of :data:`PANEL_TABS`.
    """
    if not isinstance(model, NSXPlotModel):
        raise TypeError("model must be an NSXPlotModel")
    if panel not in _BUILDERS:
        raise ValueError(f"unknown NSX tab {panel!r}; expected one of {PANEL_TABS}")
    return _BUILDERS[panel](model)


def panel_views(model: NSXPlotModel) -> dict[str, PanelView]:
    """All tab views of ``model``, keyed by panel name in tab order."""
    return {panel: panel_view(model, panel) for panel in PANEL_TABS}
