"""Cross-engine comparison service for backend-agnostic simulation traces.

The service runs the same input through selected simulation backends and
returns a structured, provenance-stamped report over the shared ``Trace``
schema. It deliberately stays above concrete adapter implementations: engines
only need the ``SimulationBackend`` Protocol, with ZTCF/ZVCF panels enabled
when a backend also satisfies ``DynamicsProvider``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import check_finite, require

from .protocol import DynamicsProvider, SimState, SimulationBackend, Trace
from .ztcf_zvcf import evaluate_ztcf_along_trajectory, zvcf_acceleration

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "ComparisonInput",
    "ComparisonPanel",
    "ComparisonReport",
    "ComparisonThresholds",
    "DivergenceAnnotation",
    "DivergenceRegistry",
    "DivergenceRegistryEntry",
    "EngineRun",
    "SeriesSummary",
    "compare",
    "compare_runs",
    "compare_traces",
    "render_markdown_report",
    "write_report",
]

_DEFAULT_DOCS = "docs/simulation_backends/cross_engine_comparison.md"


@dataclass(frozen=True)
class ComparisonInput:
    """Input shared by every selected backend.

    Args:
        horizon: Number of integration steps to run.
        dt: Integration step size in seconds.
        controls: Optional control history passed to every backend.
        initial_state: Optional state used to reset every backend before rollout.
    """

    horizon: int
    dt: float
    controls: np.ndarray | None = None
    initial_state: SimState | None = None

    def __post_init__(self) -> None:
        """Validate public constructor preconditions."""
        require(self.horizon > 0, "horizon must be positive", value=self.horizon)
        require(
            self.dt > 0.0 and np.isfinite(self.dt),
            "dt must be a positive finite float",
            value=self.dt,
        )
        if self.controls is not None:
            controls = np.asarray(self.controls, dtype=float)
            require(
                controls.ndim == 2,
                "controls must be a 2-D array shaped (horizon, nu)",
                value=controls.shape,
            )
            require(
                controls.shape[0] == self.horizon,
                "controls row count must equal horizon",
                value=(controls.shape[0], self.horizon),
            )
            require(
                check_finite(controls),
                "controls must contain only finite values",
                value=controls,
            )
            object.__setattr__(self, "controls", controls)


@dataclass(frozen=True)
class ComparisonThresholds:
    """Per-panel divergence thresholds used for annotations."""

    kinematics: float = 1e-6
    kinetics: float = 1e-6
    ztcf: float = 1e-6
    zvcf: float = 1e-6
    wrench: float = 1e-6

    def for_panel(self, panel_name: str) -> float:
        """Return the configured tolerance for ``panel_name``."""
        mapping = {
            "kinematics": self.kinematics,
            "kinetics": self.kinetics,
            "ztcf": self.ztcf,
            "zvcf": self.zvcf,
            "wrench": self.wrench,
        }
        return mapping[panel_name]


@dataclass(frozen=True)
class DivergenceRegistryEntry:
    """Registry metadata linked from each divergence annotation."""

    key: str
    title: str
    description: str
    docs_anchor: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation."""
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "docs_anchor": self.docs_anchor,
        }


@dataclass(frozen=True)
class DivergenceRegistry:
    """Small engine-agnostic registry for known comparison divergence types."""

    entries: Mapping[str, DivergenceRegistryEntry]

    @classmethod
    def default(cls) -> DivergenceRegistry:
        """Return the canonical CC-27 divergence registry."""
        entries = {
            "kinematics.q": DivergenceRegistryEntry(
                key="kinematics.q",
                title="Generalized position drift",
                description="Position trajectories disagree after alignment.",
                docs_anchor=f"{_DEFAULT_DOCS}#kinematics",
            ),
            "kinematics.v": DivergenceRegistryEntry(
                key="kinematics.v",
                title="Generalized velocity drift",
                description="Velocity trajectories disagree after alignment.",
                docs_anchor=f"{_DEFAULT_DOCS}#kinematics",
            ),
            "kinetics.control": DivergenceRegistryEntry(
                key="kinetics.control",
                title="Applied generalized force drift",
                description="Applied controls or recorded torques disagree.",
                docs_anchor=f"{_DEFAULT_DOCS}#kinetics",
            ),
            "ztcf.acceleration": DivergenceRegistryEntry(
                key="ztcf.acceleration",
                title="Zero-torque counterfactual drift",
                description="Instantaneous ZTCF accelerations disagree.",
                docs_anchor=f"{_DEFAULT_DOCS}#counterfactuals",
            ),
            "zvcf.acceleration": DivergenceRegistryEntry(
                key="zvcf.acceleration",
                title="Zero-velocity counterfactual drift",
                description="Instantaneous ZVCF accelerations disagree.",
                docs_anchor=f"{_DEFAULT_DOCS}#counterfactuals",
            ),
            "wrench.contact": DivergenceRegistryEntry(
                key="wrench.contact",
                title="Contact wrench drift",
                description="Six-axis contact wrench histories disagree.",
                docs_anchor=f"{_DEFAULT_DOCS}#wrench",
            ),
        }
        return cls(entries=entries)

    def resolve(self, panel_name: str, metric_name: str) -> DivergenceRegistryEntry:
        """Return registry metadata for a panel metric."""
        key = f"{panel_name}.{metric_name}"
        if key in self.entries:
            return self.entries[key]
        return DivergenceRegistryEntry(
            key=key,
            title=f"{panel_name} {metric_name}",
            description="Unregistered comparison divergence.",
            docs_anchor=_DEFAULT_DOCS,
        )


@dataclass(frozen=True)
class SeriesSummary:
    """Compact side-by-side summary for one numeric time series."""

    samples: int
    columns: int
    minimum: float
    maximum: float
    mean_abs: float
    peak_abs: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-compatible representation."""
        return {
            "samples": self.samples,
            "columns": self.columns,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean_abs": self.mean_abs,
            "peak_abs": self.peak_abs,
        }


@dataclass(frozen=True)
class DivergenceAnnotation:
    """A thresholded comparison between one engine and the baseline."""

    panel: str
    metric: str
    engine: str
    baseline: str
    max_abs_delta: float
    rms_delta: float
    tolerance: float
    registry: DivergenceRegistryEntry
    severity: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "panel": self.panel,
            "metric": self.metric,
            "engine": self.engine,
            "baseline": self.baseline,
            "max_abs_delta": self.max_abs_delta,
            "rms_delta": self.rms_delta,
            "tolerance": self.tolerance,
            "registry": self.registry.to_dict(),
            "severity": self.severity,
        }


@dataclass(frozen=True)
class EngineRun:
    """A named trace and optional engine object used for derived panels."""

    name: str
    trace: Trace
    engine: object | None = None

    def __post_init__(self) -> None:
        """Validate public constructor preconditions."""
        require(bool(self.name.strip()), "engine run name must be non-empty")

    def provenance(self) -> dict[str, object]:
        """Return deterministic panel provenance for this engine run."""
        provenance: dict[str, object] = {
            "engine": self.trace.backend,
            "label": self.name,
            "schema_version": self.trace.schema_version,
            "dt": float(self.trace.dt),
            "num_steps": self.trace.num_steps,
        }
        flat = {
            key.removeprefix("provenance_"): value
            for key, value in self.trace.meta.items()
            if key.startswith("provenance_")
        }
        if flat:
            provenance["stamp"] = dict(sorted(flat.items()))
        elif self.trace.meta:
            provenance["meta"] = dict(sorted(self.trace.meta.items()))
        return provenance


@dataclass(frozen=True)
class ComparisonPanel:
    """One report panel with per-engine summaries and annotations."""

    name: str
    title: str
    units: str
    metrics: Mapping[str, Mapping[str, SeriesSummary]]
    provenance_by_engine: Mapping[str, Mapping[str, object]]
    annotations: tuple[DivergenceAnnotation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "title": self.title,
            "units": self.units,
            "metrics": {
                metric: {
                    engine: summary.to_dict()
                    for engine, summary in engine_summaries.items()
                }
                for metric, engine_summaries in self.metrics.items()
            },
            "provenance_by_engine": {
                engine: dict(provenance)
                for engine, provenance in self.provenance_by_engine.items()
            },
            "annotations": [annotation.to_dict() for annotation in self.annotations],
        }


@dataclass(frozen=True)
class ComparisonReport:
    """Structured side-by-side cross-engine comparison report."""

    baseline: str
    engines: tuple[str, ...]
    panels: tuple[ComparisonPanel, ...]

    @property
    def divergences(self) -> tuple[DivergenceAnnotation, ...]:
        """Flatten all panel annotations into report order."""
        return tuple(
            annotation
            for panel in self.panels
            for annotation in panel.annotations
            if annotation.severity != "within_tolerance"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "baseline": self.baseline,
            "engines": list(self.engines),
            "panels": [panel.to_dict() for panel in self.panels],
            "divergences": [item.to_dict() for item in self.divergences],
        }


def compare(
    engines: Sequence[SimulationBackend],
    input_data: ComparisonInput,
    *,
    labels: Sequence[str] | None = None,
    registry: DivergenceRegistry | None = None,
    thresholds: ComparisonThresholds | None = None,
) -> ComparisonReport:
    """Run identical input across engines and return a comparison report.

    Args:
        engines: Selected backends. At least two are required.
        input_data: Shared rollout input.
        labels: Optional user-facing labels matching ``engines``.
        registry: Optional divergence registry override.
        thresholds: Optional per-panel divergence thresholds.

    Returns:
        A structured report with side-by-side panels and provenance.
    """
    require(len(engines) >= 2, "at least two engines are required")
    names = _resolve_labels(engines, labels)
    runs: list[EngineRun] = []
    for name, engine in zip(names, engines, strict=True):
        if input_data.initial_state is not None:
            engine.reset(input_data.initial_state.copy())
        else:
            engine.reset(None)
        trace = engine.rollout(input_data.controls, input_data.horizon, input_data.dt)
        runs.append(EngineRun(name=name, trace=trace, engine=engine))
    return compare_runs(runs, registry=registry, thresholds=thresholds)


def compare_traces(
    traces: Mapping[str, Trace],
    *,
    registry: DivergenceRegistry | None = None,
    thresholds: ComparisonThresholds | None = None,
) -> ComparisonReport:
    """Compare precomputed traces when rerunning engines is not needed."""
    require(len(traces) >= 2, "at least two traces are required")
    runs = tuple(EngineRun(name=name, trace=trace) for name, trace in traces.items())
    return compare_runs(runs, registry=registry, thresholds=thresholds)


def compare_runs(
    runs: Sequence[EngineRun],
    *,
    registry: DivergenceRegistry | None = None,
    thresholds: ComparisonThresholds | None = None,
) -> ComparisonReport:
    """Build report panels from already collected engine runs."""
    require(len(runs) >= 2, "at least two engine runs are required")
    reg = registry or DivergenceRegistry.default()
    limits = thresholds or ComparisonThresholds()
    baseline = runs[0].name
    panels = (
        _build_panel(
            "kinematics",
            "Kinematics",
            "rad, rad/s",
            {"q": _trace_q, "v": _trace_v},
            runs,
            reg,
            limits,
        ),
        _build_panel(
            "kinetics",
            "Kinetics",
            "N*m",
            {"control": _trace_kinetics},
            runs,
            reg,
            limits,
        ),
        _build_panel(
            "ztcf",
            "ZTCF",
            "rad/s^2",
            {"acceleration": _ztcf_series},
            runs,
            reg,
            limits,
        ),
        _build_panel(
            "zvcf",
            "ZVCF",
            "rad/s^2",
            {"acceleration": _zvcf_series},
            runs,
            reg,
            limits,
        ),
        _build_panel(
            "wrench",
            "Wrench",
            "N, N*m",
            {"contact": _trace_wrench},
            runs,
            reg,
            limits,
        ),
    )
    return ComparisonReport(
        baseline=baseline,
        engines=tuple(run.name for run in runs),
        panels=panels,
    )


def render_markdown_report(report: ComparisonReport) -> str:
    """Render ``report`` as a deterministic Markdown document."""
    lines = [
        "# Cross-Engine Comparison Report",
        "",
        f"Baseline: `{report.baseline}`",
        f"Engines: {', '.join(f'`{engine}`' for engine in report.engines)}",
        "",
    ]
    for panel in report.panels:
        lines.extend([f"## {panel.title}", "", f"Units: `{panel.units}`", ""])
        for metric, summaries in panel.metrics.items():
            lines.extend(
                [
                    f"### {metric}",
                    "",
                    "| Engine | Samples | Columns | Mean abs | Peak abs |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for engine, summary in summaries.items():
                lines.append(
                    "| "
                    f"{engine} | {summary.samples} | {summary.columns} | "
                    f"{summary.mean_abs:.6g} | {summary.peak_abs:.6g} |"
                )
            lines.append("")
        lines.extend(["Provenance:", ""])
        for engine, provenance in panel.provenance_by_engine.items():
            payload = json.dumps(provenance, sort_keys=True, default=str)
            lines.append(f"- `{engine}`: `{payload}`")
        lines.append("")
        if panel.annotations:
            lines.extend(
                [
                    "Annotations:",
                    "",
                    "| Engine | Metric | Severity | Max abs delta | RMS delta | Registry |",
                    "|---|---|---|---:|---:|---|",
                ]
            )
            for item in panel.annotations:
                lines.append(
                    "| "
                    f"{item.engine} | {item.metric} | {item.severity} | "
                    f"{item.max_abs_delta:.6g} | {item.rms_delta:.6g} | "
                    f"[{item.registry.key}]({item.registry.docs_anchor}) |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(
    report: ComparisonReport,
    path: str | Path,
    *,
    format: str = "markdown",
) -> None:
    """Write ``report`` to ``path`` as Markdown or JSON."""
    if format == "markdown":
        content = render_markdown_report(report)
    elif format == "json":
        content = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    else:
        raise ValueError(f"unsupported report format: {format!r}")
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _resolve_labels(
    engines: Sequence[SimulationBackend], labels: Sequence[str] | None
) -> tuple[str, ...]:
    if labels is not None:
        require(
            len(labels) == len(engines),
            "labels must match engine count",
            value=(len(labels), len(engines)),
        )
        raw_names = [str(label) for label in labels]
    else:
        raw_names = [_backend_name(engine) for engine in engines]
    counts: dict[str, int] = {}
    resolved = []
    for raw in raw_names:
        require(bool(raw.strip()), "engine label must be non-empty")
        count = counts.get(raw, 0) + 1
        counts[raw] = count
        resolved.append(raw if count == 1 else f"{raw}#{count}")
    return tuple(resolved)


def _backend_name(engine: SimulationBackend) -> str:
    caps = engine.capabilities
    return str(caps.name)


def _build_panel(
    name: str,
    title: str,
    units: str,
    metric_getters: Mapping[str, object],
    runs: Sequence[EngineRun],
    registry: DivergenceRegistry,
    thresholds: ComparisonThresholds,
) -> ComparisonPanel:
    metrics: dict[str, dict[str, SeriesSummary]] = {}
    series_by_metric: dict[str, dict[str, np.ndarray]] = {}
    for metric, getter in metric_getters.items():
        typed_getter = getter  # narrow for readability below
        summaries: dict[str, SeriesSummary] = {}
        series_by_engine: dict[str, np.ndarray] = {}
        for run in runs:
            data = typed_getter(run)  # type: ignore[operator]
            if data is None:
                continue
            arr = _as_matrix(metric, data)
            summaries[run.name] = _summarize(arr)
            series_by_engine[run.name] = arr
        if summaries:
            metrics[metric] = summaries
            series_by_metric[metric] = series_by_engine

    annotations: list[DivergenceAnnotation] = []
    tolerance = thresholds.for_panel(name)
    for metric, series_by_engine in series_by_metric.items():
        annotations.extend(
            _annotate_metric(name, metric, runs, series_by_engine, tolerance, registry)
        )
    provenance = {run.name: run.provenance() for run in runs}
    return ComparisonPanel(
        name=name,
        title=title,
        units=units,
        metrics=metrics,
        provenance_by_engine=provenance,
        annotations=tuple(annotations),
    )


def _annotate_metric(
    panel_name: str,
    metric_name: str,
    runs: Sequence[EngineRun],
    series_by_engine: Mapping[str, np.ndarray],
    tolerance: float,
    registry: DivergenceRegistry,
) -> list[DivergenceAnnotation]:
    baseline_name = runs[0].name
    baseline = series_by_engine.get(baseline_name)
    if baseline is None:
        return []
    out: list[DivergenceAnnotation] = []
    for run in runs[1:]:
        candidate = series_by_engine.get(run.name)
        if candidate is None:
            continue
        max_abs, rms = _delta_stats(baseline, candidate)
        severity = _severity(max_abs, tolerance)
        out.append(
            DivergenceAnnotation(
                panel=panel_name,
                metric=metric_name,
                engine=run.name,
                baseline=baseline_name,
                max_abs_delta=max_abs,
                rms_delta=rms,
                tolerance=tolerance,
                registry=registry.resolve(panel_name, metric_name),
                severity=severity,
            )
        )
    return out


def _severity(max_abs_delta: float, tolerance: float) -> str:
    if max_abs_delta <= tolerance:
        return "within_tolerance"
    if max_abs_delta <= tolerance * 10.0:
        return "minor"
    return "major"


def _delta_stats(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    rows = min(left.shape[0], right.shape[0])
    cols = min(left.shape[1], right.shape[1])
    require(rows > 0 and cols > 0, "series must share at least one aligned value")
    delta = left[:rows, :cols] - right[:rows, :cols]
    max_abs = float(np.max(np.abs(delta)))
    rms = float(np.sqrt(np.vdot(delta, delta) / delta.size))
    return max_abs, rms


def _as_matrix(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    require(arr.ndim == 2, f"{name} must be one- or two-dimensional", value=arr.shape)
    require(arr.shape[0] > 0, f"{name} must have samples", value=arr.shape)
    require(check_finite(arr), f"{name} must contain only finite values", value=arr)
    return arr


def _summarize(value: np.ndarray) -> SeriesSummary:
    arr = _as_matrix("summary", value)
    abs_arr = np.abs(arr)
    return SeriesSummary(
        samples=int(arr.shape[0]),
        columns=int(arr.shape[1]),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        mean_abs=float(np.mean(abs_arr)),
        peak_abs=float(np.max(abs_arr)),
    )


def _trace_q(run: EngineRun) -> np.ndarray:
    return run.trace.q


def _trace_v(run: EngineRun) -> np.ndarray:
    return run.trace.v


def _trace_kinetics(run: EngineRun) -> np.ndarray | None:
    if run.trace.torques is not None:
        return run.trace.torques
    return run.trace.u


def _trace_wrench(run: EngineRun) -> np.ndarray | None:
    return run.trace.wrench


def _ztcf_series(run: EngineRun) -> np.ndarray | None:
    if not isinstance(run.engine, DynamicsProvider):
        return None
    return evaluate_ztcf_along_trajectory(run.engine, run.trace.q, run.trace.v)


def _zvcf_series(run: EngineRun) -> np.ndarray | None:
    if not isinstance(run.engine, DynamicsProvider):
        return None
    tau = run.trace.torques
    if tau is None:
        tau = run.trace.u
    if tau is None:
        tau = np.zeros_like(run.trace.q)
    tau_arr = _as_matrix("tau", tau)
    if tau_arr.shape[1] != run.trace.q.shape[1]:
        return None
    rows = min(run.trace.q.shape[0], tau_arr.shape[0])
    out = np.empty((rows, run.trace.q.shape[1]), dtype=float)
    for idx in range(rows):
        out[idx] = zvcf_acceleration(run.engine, run.trace.q[idx], tau_arr[idx])
    return out
