"""Frozen data model for the swing report contract v1 (#12324).

Instances are only produced by ``validate_swing_report`` (which enforces the
schema and invariants); ``to_dict`` reproduces the validated source document.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _opt(**fields: Any) -> dict[str, Any]:
    """Drop ``None`` entries so optional keys stay absent on export."""
    return {k: v for k, v in fields.items() if v is not None}


@dataclass(frozen=True)
class Capture:
    capture_id: str
    visibility: str
    capture_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _opt(
            capture_id=self.capture_id,
            visibility=self.visibility,
            capture_sha256=self.capture_sha256,
        )


@dataclass(frozen=True)
class ReportProvenance:
    timestamp_utc: str
    software_name: str
    software_version: str
    git_commit_sha: str | None
    capture: Capture
    git_is_dirty: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "software_name": self.software_name,
            "software_version": self.software_version,
            "git_commit_sha": self.git_commit_sha,
            **_opt(git_is_dirty=self.git_is_dirty),
            "capture": self.capture.to_dict(),
        }


@dataclass(frozen=True)
class EngineRun:
    run_id: str
    engine: str
    engine_version: str
    model_file_hash: str
    run_class: str
    force_provenance: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _opt(
            run_id=self.run_id,
            engine=self.engine,
            engine_version=self.engine_version,
            model_file_hash=self.model_file_hash,
            run_class=self.run_class,
            force_provenance=self.force_provenance,
        )


@dataclass(frozen=True)
class Series:
    engine: str
    run_id: str
    units: str
    frame: str
    time_s: tuple[float, ...]
    values: tuple[float | None, ...]
    channel: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "run_id": self.run_id,
            "units": self.units,
            "frame": self.frame,
            **_opt(channel=self.channel),
            "time_s": list(self.time_s),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class Uncertainty:
    kind: str
    budget_source: str
    units: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "budget_source": self.budget_source,
            "units": self.units,
            "lower": list(self.lower),
            "upper": list(self.upper),
        }


@dataclass(frozen=True)
class Quantity:
    """A report quantity. Withheld quantities hold no series or uncertainty."""

    quantity_id: str
    status: str
    series: tuple[Series, ...] = ()
    uncertainty: Uncertainty | None = None
    label: str | None = None
    reason: str | None = None
    evidence_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = _opt(
            quantity_id=self.quantity_id,
            status=self.status,
            label=self.label,
            reason=self.reason,
            evidence_ref=self.evidence_ref,
        )
        if self.series:
            out["series"] = [s.to_dict() for s in self.series]
        if self.uncertainty is not None:
            out["uncertainty"] = self.uncertainty.to_dict()
        return out


@dataclass(frozen=True)
class PlotDescriptor:
    plot_id: str
    title: str
    quantity_id: str
    kind: str
    engines: tuple[str, ...]
    render: str
    units: str | None = None
    frame: str | None = None
    uncertainty_fields: tuple[str, ...] | None = None
    selectable: bool | None = None
    default_selected: bool | None = None
    unavailable_reason: str | None = None

    @property
    def is_selectable(self) -> bool:
        """Schema default for ``selectable`` is true."""
        return True if self.selectable is None else self.selectable

    def to_dict(self) -> dict[str, Any]:
        fields = (
            self.uncertainty_fields
            if self.uncertainty_fields is None
            else list(self.uncertainty_fields)
        )
        return _opt(
            plot_id=self.plot_id,
            title=self.title,
            quantity_id=self.quantity_id,
            kind=self.kind,
            units=self.units,
            frame=self.frame,
            engines=list(self.engines),
            uncertainty_fields=fields,
            selectable=self.selectable,
            default_selected=self.default_selected,
            render=self.render,
            unavailable_reason=self.unavailable_reason,
        )


@dataclass(frozen=True)
class Section:
    section_id: str
    title: str
    plot_ids: tuple[str, ...]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _opt(
            section_id=self.section_id,
            title=self.title,
            plot_ids=list(self.plot_ids),
            notes=self.notes,
        )


@dataclass(frozen=True)
class SwingReport:
    schema_version: str
    report_id: str
    generated_utc: str
    scope: str
    provenance: ReportProvenance
    runs: tuple[EngineRun, ...]
    quantities: tuple[Quantity, ...]
    plots: tuple[PlotDescriptor, ...]
    sections: tuple[Section, ...]
    extensions: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_utc": self.generated_utc,
            "scope": self.scope,
            "provenance": self.provenance.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
            "quantities": [q.to_dict() for q in self.quantities],
            "plots": [p.to_dict() for p in self.plots],
            "sections": [s.to_dict() for s in self.sections],
        }
        if self.extensions is not None:
            out["extensions"] = dict(self.extensions)
        return out


def _series(d: Mapping[str, Any]) -> Series:
    return Series(
        engine=d["engine"],
        run_id=d["run_id"],
        units=d["units"],
        frame=d["frame"],
        time_s=tuple(d["time_s"]),
        values=tuple(d["values"]),
        channel=d.get("channel"),
    )


def _uncertainty(d: Mapping[str, Any] | None) -> Uncertainty | None:
    if d is None:
        return None
    return Uncertainty(
        kind=d["kind"],
        budget_source=d["budget_source"],
        units=d["units"],
        lower=tuple(d["lower"]),
        upper=tuple(d["upper"]),
    )


def _quantity(d: Mapping[str, Any]) -> Quantity:
    return Quantity(
        quantity_id=d["quantity_id"],
        status=d["status"],
        series=tuple(_series(s) for s in d.get("series", ())),
        uncertainty=_uncertainty(d.get("uncertainty")),
        label=d.get("label"),
        reason=d.get("reason"),
        evidence_ref=d.get("evidence_ref"),
    )


def _plot(d: Mapping[str, Any]) -> PlotDescriptor:
    fields = d.get("uncertainty_fields")
    return PlotDescriptor(
        plot_id=d["plot_id"],
        title=d["title"],
        quantity_id=d["quantity_id"],
        kind=d["kind"],
        engines=tuple(d["engines"]),
        render=d["render"],
        units=d.get("units"),
        frame=d.get("frame"),
        uncertainty_fields=None if fields is None else tuple(fields),
        selectable=d.get("selectable"),
        default_selected=d.get("default_selected"),
        unavailable_reason=d.get("unavailable_reason"),
    )


def report_from_dict(d: Mapping[str, Any]) -> SwingReport:
    """Build a ``SwingReport`` from a document that already passed validation."""
    prov = d["provenance"]
    cap = prov["capture"]
    return SwingReport(
        schema_version=d["schema_version"],
        report_id=d["report_id"],
        generated_utc=d["generated_utc"],
        scope=d["scope"],
        provenance=ReportProvenance(
            timestamp_utc=prov["timestamp_utc"],
            software_name=prov["software_name"],
            software_version=prov["software_version"],
            git_commit_sha=prov["git_commit_sha"],
            git_is_dirty=prov.get("git_is_dirty"),
            capture=Capture(
                capture_id=cap["capture_id"],
                visibility=cap["visibility"],
                capture_sha256=cap.get("capture_sha256"),
            ),
        ),
        runs=tuple(EngineRun(**r) for r in d["runs"]),
        quantities=tuple(_quantity(q) for q in d["quantities"]),
        plots=tuple(_plot(p) for p in d["plots"]),
        sections=tuple(
            Section(
                section_id=s["section_id"],
                title=s["title"],
                plot_ids=tuple(s["plot_ids"]),
                notes=s.get("notes"),
            )
            for s in d["sections"]
        ),
        extensions=d.get("extensions"),
    )
