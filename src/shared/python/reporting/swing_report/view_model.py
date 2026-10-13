"""Qt-free view-model of a validated swing report (#12329, epic #11419).

The report dialog consumes only this model, never raw engine output. A row
exposes labels and text, never data: a withheld quantity therefore cannot leak
a number through the UI (fail-closed policy of #10382, invariants I5/I8).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
import logging
from pathlib import Path

from .contract import PlotDescriptor, Quantity, SwingReport
from .exporter import export_package
from .policy import STATUS_WITHHELD
from .selection import PlotSelection, select_plots

logger = logging.getLogger(__name__)

MODE_FULL = "full"
MODE_CUSTOM = "custom"
NOT_VALIDATED_LABEL = "Not Validated"
ADVERTISED_LABEL = "Advertised With Uncertainty"


@dataclass(frozen=True)
class PlotRow:
    """One plot as the dialog shows it (text only, no numeric data)."""

    plot_id: str
    title: str
    kind: str
    enabled: bool
    status_label: str
    detail: str
    unavailable_reason: str | None = None


def _uncertainty_text(quantity: Quantity | None) -> str:
    unc = None if quantity is None else quantity.uncertainty
    if unc is None:
        return ""
    return f"Uncertainty: {unc.kind} ({unc.budget_source}), units {unc.units}"


def _row(plot: PlotDescriptor, quantity: Quantity | None) -> PlotRow:
    withheld = plot.render == "unavailable" or (
        quantity is not None and quantity.status == STATUS_WITHHELD
    )
    if withheld:
        reason = plot.unavailable_reason or "Not validated"
        return PlotRow(
            plot.plot_id,
            plot.title,
            plot.kind,
            False,
            NOT_VALIDATED_LABEL,
            reason,
            reason,
        )
    return PlotRow(
        plot.plot_id,
        plot.title,
        plot.kind,
        plot.is_selectable,
        ADVERTISED_LABEL,
        _uncertainty_text(quantity),
    )


class ReportViewModel:
    """Selection state over a validated ``SwingReport``.

    Invariants: ``selected_ids`` only ever holds enabled plot ids; in Full mode
    it is every enabled plot; a withheld plot can never be selected.
    """

    def __init__(self, report: SwingReport) -> None:
        by_id = {q.quantity_id: q for q in report.quantities}
        self._report = report
        self._rows = tuple(_row(p, by_id.get(p.quantity_id)) for p in report.plots)
        self._enabled = tuple(r.plot_id for r in self._rows if r.enabled)
        self._mode = MODE_FULL
        self._custom: frozenset[str] = frozenset(
            p.plot_id
            for p in report.plots
            if p.plot_id in self._enabled and p.default_selected
        )

    @property
    def rows(self) -> tuple[PlotRow, ...]:
        return self._rows

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def selected_ids(self) -> tuple[str, ...]:
        """Selected enabled plot ids in report order."""
        chosen = set(self._enabled) if self._mode == MODE_FULL else self._custom
        return tuple(pid for pid in self._enabled if pid in chosen)

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_FULL, MODE_CUSTOM):
            raise ValueError(f"unknown mode: {mode!r}")
        self._mode = mode

    def set_custom_selection(self, plot_ids: Iterable[str]) -> None:
        """Replace the Custom selection; withheld/unknown ids are rejected."""
        ids = frozenset(plot_ids)
        bad = sorted(ids - set(self._enabled))
        if bad:
            raise ValueError(f"plots cannot be selected: {bad}")
        self._custom = ids

    def can_export(self) -> bool:
        return bool(self.selected_ids)

    def select(self) -> PlotSelection:
        """Resolve the selection through ``select_plots``."""
        if not self.can_export():
            raise ValueError("no plots selected")
        ids: Collection[str] | None = (
            None if self._mode == MODE_FULL else self.selected_ids
        )
        return select_plots(self._report, ids)

    def export(self, out_dir: Path | str) -> dict[str, str]:
        """Export the selection as a package; return ``path -> sha256``."""
        return export_package(self.select().report, out_dir)
