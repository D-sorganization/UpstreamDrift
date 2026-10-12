"""Swing report builder over existing engine results (#12325, epic #11419).

``build_swing_report`` assembles a contract document from ``EngineResult``
inputs under the fail-closed policy of #10382:

* only ``ADVERTISED`` quantities carry values, and only when at least two
  engines supply comparable series (uncertainty is the provisional
  cross-engine spread, so a single engine is not advertised);
* every ``WITHHELD`` quantity is emitted as a value-free stub regardless of
  what the engines produced;
* missing samples stay ``None`` (never zero) and an engine failure degrades to
  a recorded note, never an exception.

The result is always passed through ``validate_swing_report`` (postcondition).
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import logging
import math

from src.shared.python.data_io.provenance import ProvenanceInfo
from src.shared.python.motion_matching.provenance import engine_package_version

from .adapters_base import EngineResult, QuantityData, finite_or_none
from .contract import (
    Capture,
    EngineRun,
    PlotDescriptor,
    Quantity,
    ReportProvenance,
    Section,
    Series,
    SwingReport,
    Uncertainty,
)
from .policy import (
    ADVERTISED,
    NOT_VALIDATED_LABEL,
    STATUS_ADVERTISED,
    STATUS_WITHHELD,
    UNCERTAINTY_BUDGET_SOURCE,
    UNCERTAINTY_KIND,
    WITHHELD,
)
from .validator import load_schema, validate_swing_report

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"
SCOPES = ("full", "custom")
EVIDENCE_REF = "DV-10382"
UNKNOWN_HASH = "0" * 64  # sentinel for a run whose model hash is unavailable
MIN_ENGINES_FOR_SPREAD = 2
_PLOT_KIND = {"kinematic_sequence": "sequence_timeline"}
# Import name differs from distribution name for some engines.
_DISTRIBUTIONS = {"pinocchio": ("pin",), "drake": ("drake",)}


def _schema_enum(name: str) -> frozenset[str]:
    return frozenset(load_schema()["$defs"][name]["enum"])


def _check_inputs(
    results: Sequence[EngineResult], capture: Capture, scope: str
) -> None:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    if not results:
        raise ValueError("results must contain at least one engine result")
    run_ids = [r.run_id for r in results]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"run_id values must be unique, got {run_ids}")
    if capture.visibility == "private" and capture.capture_sha256 is not None:
        raise ValueError("private captures must not carry capture_sha256")


def _run_for(result: EngineResult) -> EngineRun:
    version = result.engine_version or engine_package_version(
        None, *_DISTRIBUTIONS.get(result.engine, (result.engine,))
    )
    return EngineRun(
        run_id=result.run_id,
        engine=result.engine,
        engine_version=version,
        model_file_hash=result.model_file_hash or UNKNOWN_HASH,
        run_class=result.run_class,
        force_provenance=result.force_provenance,
    )


def _data_problem(data: QuantityData) -> str | None:
    """Why ``data`` cannot enter a report, or ``None`` if it is usable."""
    if data.units not in _schema_enum("Units"):
        return f"unsupported units {data.units!r}"
    if data.frame not in _schema_enum("Frame"):
        return f"unsupported frame {data.frame!r}"
    if data.channel in WITHHELD:
        return f"channel {data.channel!r} is withheld"
    if not data.time_s or len(data.time_s) != len(data.values):
        return "time_s and values must be non-empty and equal length"
    times = data.time_s
    if not all(math.isfinite(t) for t in times):
        return "time_s must be finite"
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        return "time_s must be strictly increasing"
    return None


def _comparable(
    candidates: list[tuple[EngineResult, QuantityData]], notes: list[str]
) -> list[tuple[EngineResult, QuantityData]]:
    """Keep candidates sharing the first usable candidate's grid, units, frame."""
    usable = []
    for result, data in candidates:
        problem = _data_problem(data)
        if problem:
            notes.append(f"{result.engine}: excluded ({problem})")
        else:
            usable.append((result, data))
    if not usable:
        return []
    ref = usable[0][1]
    kept = []
    for result, data in usable:
        key = (data.units, data.frame, data.time_s)
        if key == (ref.units, ref.frame, ref.time_s):
            kept.append((result, data))
        else:
            notes.append(f"{result.engine}: excluded (units/frame/time grid differ)")
    return kept


def _spread(
    columns: list[tuple[float | None, ...]],
) -> tuple[list[list[float | None]], list[float], list[float]]:
    """Per-sample cross-engine envelope (min, max), in the series units.

    Samples without two finite engines have their values nulled in every
    series (a value is never advertised without a spread); their bounds are a
    placeholder 0.0 that no consumer can read as a value.
    """
    n = len(columns[0])
    values: list[list[float | None]] = [list(c) for c in columns]
    lower, upper = [0.0] * n, [0.0] * n
    for i in range(n):
        finite = [c[i] for c in columns if c[i] is not None]
        if len(finite) < MIN_ENGINES_FOR_SPREAD:
            for col in values:
                col[i] = None
            continue
        lower[i], upper[i] = min(finite), max(finite)
    return values, lower, upper


def _advertised(
    qid: str, healthy: list[EngineResult], notes: list[str]
) -> Quantity | None:
    candidates = [(r, r.quantities[qid]) for r in healthy if qid in r.quantities]
    if len(candidates) < MIN_ENGINES_FOR_SPREAD:
        reason = "single engine: no spread" if candidates else "no engine data"
        notes.append(f"{qid}: not advertised ({reason})")
        return None
    local: list[str] = []
    kept = _comparable(candidates, local)
    notes.extend(f"{qid}: {n}" for n in local)
    if len(kept) < MIN_ENGINES_FOR_SPREAD:
        notes.append(f"{qid}: not advertised (fewer than two comparable engines)")
        return None
    columns = [tuple(finite_or_none(v) for v in d.values) for _, d in kept]
    values, lower, upper = _spread(columns)
    ref = kept[0][1]
    series = tuple(
        Series(
            engine=r.engine,
            run_id=r.run_id,
            units=d.units,
            frame=d.frame,
            time_s=tuple(float(t) for t in d.time_s),
            values=tuple(col),
            channel=d.channel,
        )
        for (r, d), col in zip(kept, values, strict=True)
    )
    return Quantity(
        quantity_id=qid,
        status=STATUS_ADVERTISED,
        series=series,
        uncertainty=Uncertainty(
            kind=UNCERTAINTY_KIND,
            budget_source=UNCERTAINTY_BUDGET_SOURCE,
            units=ref.units,
            lower=tuple(lower),
            upper=tuple(upper),
        ),
    )


def _withheld(qid: str, healthy: list[EngineResult]) -> Quantity:
    dropped = any(qid in r.quantities for r in healthy)
    reason = "not validated (#10382)"
    if dropped:
        reason += "; engine output dropped"
    return Quantity(
        quantity_id=qid,
        status=STATUS_WITHHELD,
        label=NOT_VALIDATED_LABEL,
        reason=reason,
        evidence_ref=EVIDENCE_REF,
    )


def _title(qid: str) -> str:
    return qid.replace("_", " ").title()


def _plot_for(quantity: Quantity, run_engines: tuple[str, ...]) -> PlotDescriptor:
    qid = quantity.quantity_id
    kind = _PLOT_KIND.get(qid, "time_series")
    if quantity.status == STATUS_WITHHELD:
        return PlotDescriptor(
            plot_id=f"{qid}_plot",
            title=_title(qid),
            quantity_id=qid,
            kind=kind,
            engines=run_engines,
            render="unavailable",
            unavailable_reason=quantity.reason or "not validated",
            selectable=True,
            default_selected=False,
        )
    first = quantity.series[0]
    return PlotDescriptor(
        plot_id=f"{qid}_plot",
        title=_title(qid),
        quantity_id=qid,
        kind=kind,
        engines=tuple(s.engine for s in quantity.series),
        render="values",
        units=first.units,
        frame=first.frame,
        uncertainty_fields=("lower", "upper"),
        selectable=True,
        default_selected=True,
    )


def _sections(plots: list[PlotDescriptor], notes: list[str]) -> tuple[Section, ...]:
    shown = tuple(p.plot_id for p in plots if p.render == "values")
    hidden = tuple(p.plot_id for p in plots if p.render == "unavailable")
    return (
        Section(
            section_id="overview",
            title="Overview",
            plot_ids=shown,
            notes="; ".join(notes) if notes else None,
        ),
        Section(section_id="not_validated", title="Not Validated", plot_ids=hidden),
    )


def _report_id(info: ProvenanceInfo, capture: Capture, runs: list[EngineRun]) -> str:
    seed = "|".join([info.timestamp_utc, capture.capture_id, *(r.run_id for r in runs)])
    return f"swing-report-{hashlib.sha256(seed.encode()).hexdigest()[:16]}"


def build_swing_report(
    results: Sequence[EngineResult],
    capture: Capture,
    scope: str,
    *,
    provenance: ProvenanceInfo | None = None,
) -> SwingReport:
    """Build a validated ``SwingReport`` from per-engine results.

    Preconditions: ``scope`` is ``full`` or ``custom``; ``results`` is
    non-empty with unique ``run_id`` values; a private ``capture`` carries no
    ``capture_sha256``. ``provenance`` defaults to ``ProvenanceInfo.capture()``.

    Postconditions: the report passes ``validate_swing_report``; no withheld
    quantity holds any value; no advertised quantity exists without
    cross-engine spread; no missing sample is zero-filled.

    Raises:
        ValueError: on a violated precondition.
        SwingReportContractError: if the assembled document is invalid (a bug).
    """
    _check_inputs(results, capture, scope)
    info = provenance if provenance is not None else ProvenanceInfo.capture()
    runs = [_run_for(r) for r in results]
    healthy = [r for r in results if r.healthy]
    notes = [f"{r.engine}: unavailable ({r.error})" for r in results if not r.healthy]

    quantities: list[Quantity] = []
    for qid in sorted(ADVERTISED):
        advertised = _advertised(qid, healthy, notes)
        if advertised is not None:
            quantities.append(advertised)
    quantities.extend(_withheld(qid, healthy) for qid in sorted(WITHHELD))

    run_engines = tuple(dict.fromkeys(r.engine for r in runs))
    plots = [_plot_for(q, run_engines) for q in quantities]
    report = SwingReport(
        schema_version=SCHEMA_VERSION,
        report_id=_report_id(info, capture, runs),
        generated_utc=info.timestamp_utc,
        scope=scope,
        provenance=ReportProvenance(
            timestamp_utc=info.timestamp_utc,
            software_name=info.software_name,
            software_version=info.software_version,
            git_commit_sha=info.git_commit_sha,
            git_is_dirty=info.git_is_dirty,
            capture=capture,
        ),
        runs=tuple(runs),
        quantities=tuple(quantities),
        plots=tuple(plots),
        sections=_sections(plots, notes),
    )
    return validate_swing_report(report.to_dict())
