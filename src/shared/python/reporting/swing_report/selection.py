"""Full/Custom plot selection over a validated swing report (#12326).

``select_plots`` is the single entry point for the exporter (S4) and the
dialog (S5): they pass a ``SwingReport`` and plot ids and get back a validated
report whose ``plots``/``sections`` are the resolved selection, so neither
needs to reach into the registry or the quantities.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
import logging

from .contract import PlotDescriptor, Section, SwingReport
from .plots import PLOT_REGISTRY, PlotSpec, resolve_descriptor
from .validator import validate_swing_report

logger = logging.getLogger(__name__)


class UnknownPlotError(ValueError):
    """Raised when a requested plot id is not in the registry."""


def _specs(
    registry: Mapping[str, PlotSpec], plot_ids: Collection[str] | None
) -> list[PlotSpec]:
    """Registry-ordered specs for ``plot_ids`` (all when ``None``)."""
    if plot_ids is None:
        return list(registry.values())
    if not plot_ids:
        raise ValueError("custom selection needs at least one plot id")
    unknown = sorted(set(plot_ids) - set(registry))
    if unknown:
        raise UnknownPlotError(f"unknown plot ids: {unknown}")
    wanted = set(plot_ids)
    return [spec for pid, spec in registry.items() if pid in wanted]


def resolve_plots(
    report: SwingReport,
    plot_ids: Collection[str] | None = None,
    *,
    registry: Mapping[str, PlotSpec] = PLOT_REGISTRY,
) -> tuple[PlotDescriptor, ...]:
    """Resolve registry plots (or ``plot_ids``) against ``report``.

    Order is registry order, independent of the request order; duplicates in
    ``plot_ids`` collapse. Unavailable plots are kept, with their reason.

    Raises:
        UnknownPlotError: a requested id is not in ``registry``.
        ValueError: ``plot_ids`` is given but empty.
    """
    return tuple(resolve_descriptor(s, report) for s in _specs(registry, plot_ids))


@dataclass(frozen=True)
class PlotSelection:
    """Result of ``select_plots``.

    ``report`` is the validated report holding the selection. ``unresolved``
    maps plot ids to the reason they cannot be listed in it: the contract
    requires every plot's quantity to exist in the report (I7), so a plot
    whose quantity the builder did not emit (e.g. single-engine, no spread)
    is reported here instead of being silently dropped.
    """

    report: SwingReport
    unresolved: Mapping[str, str]


def _note(unresolved: Mapping[str, str]) -> str | None:
    if not unresolved:
        return None
    return "; ".join(f"{pid}: {why}" for pid, why in unresolved.items())


def _sections(
    plots: tuple[PlotDescriptor, ...], scope: str, notes: str | None
) -> tuple[Section, ...]:
    if scope == "custom":
        return (
            Section("selected", "Selected", tuple(p.plot_id for p in plots), notes),
        )
    shown = tuple(p.plot_id for p in plots if p.render == "values")
    hidden = tuple(p.plot_id for p in plots if p.render == "unavailable")
    return (
        Section("overview", "Overview", shown, notes),
        Section("not_validated", "Not Validated", hidden),
    )


def select_plots(
    report: SwingReport,
    plot_ids: Collection[str] | None = None,
    *,
    registry: Mapping[str, PlotSpec] = PLOT_REGISTRY,
) -> PlotSelection:
    """Return ``report`` restricted to a Full (``None``) or Custom selection.

    Postconditions: ``result.report`` passes ``validate_swing_report`` and has
    ``scope`` ``full`` or ``custom``; a withheld plot is never dropped from a
    custom selection, it is listed ``unavailable`` with its reason; a plot
    that cannot be represented at all is in ``result.unresolved`` (and noted
    in its section); every requested plot is in exactly one of the two;
    ``report`` is not mutated.

    Raises:
        UnknownPlotError: a requested id is not in ``registry``.
        ValueError: ``plot_ids`` is given but empty.
        SwingReportContractError: the result is invalid (a bug).
    """
    scope = "full" if plot_ids is None else "custom"
    resolved = resolve_plots(report, plot_ids, registry=registry)
    present = {q.quantity_id for q in report.quantities}
    plots = tuple(d for d in resolved if d.quantity_id in present)
    unresolved = {
        d.plot_id: d.unavailable_reason or "unavailable"
        for d in resolved
        if d.quantity_id not in present
    }
    sections = _sections(plots, scope, _note(unresolved))
    selected = replace(report, scope=scope, plots=plots, sections=sections)
    return PlotSelection(validate_swing_report(selected.to_dict()), unresolved)
