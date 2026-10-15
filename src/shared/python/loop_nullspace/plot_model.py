"""Framework-free NSX plot model for PyQt6 and React parity (NSX-42, #12574).

Constructs a serializable 5-panel plot model (:class:`NSXPlotModel`) from an
NSX analysis report and output-policy display decisions:

1. ``invariants``: bare-club net wrench, summed loop power, outside-loop
   torques, tangent load.
2. ``envelopes``: band-first series with per-segment ``display_class`` in
   ``{"DETERMINED_SIGN", "INDETERMINATE", "UNBOUNDED"}`` and
   ``binding_hypotheses`` extracted from active constraint duals.
3. ``hand_split``: 6 grip-frame split coordinates ``s1 .. s6``.
4. ``allocation_representative``: overlaid on the envelope band, never alone
   (Design-by-Contract raises ``ValueError`` if a representative is requested
   without its corresponding envelope band).
5. ``diagnostics``: rank, condition number, admission residuals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any, Literal

from src.shared.python.loop_nullspace.output_policy_integration import (
    DisplayDecision,
    display_decision,
)

__all__ = [
    "EnvelopeBandSeries",
    "EnvelopeSegment",
    "NSXPlotModel",
    "PlotLineSeries",
    "PlotPanelModel",
    "SegmentDisplayClass",
    "build_plot_model",
]

SegmentDisplayClass = Literal["DETERMINED_SIGN", "INDETERMINATE", "UNBOUNDED"]

_HAND_SPLIT_IDS: frozenset[str] = frozenset(
    {
        "s1_axial_internal_force_n",
        "s2_transverse_v_force_shear_couple_n",
        "s3_transverse_w_force_shear_couple_n",
        "s4_axial_twist_diff_moment_nm",
        "s5_bending_v_diff_moment_nm",
        "s6_bending_w_diff_moment_nm",
    }
)
_DUAL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class EnvelopeSegment:
    """Contiguous frame range sharing a sign class and binding hypotheses."""

    start_index: int
    end_index: int
    display_class: SegmentDisplayClass
    binding_hypotheses: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize segment to JSON-compatible dict."""
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "display_class": self.display_class,
            "binding_hypotheses": list(self.binding_hypotheses),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvelopeSegment:
        """Rebuild an :class:`EnvelopeSegment` from a mapping."""
        raw_cls = str(data["display_class"])
        if raw_cls not in {"DETERMINED_SIGN", "INDETERMINATE", "UNBOUNDED"}:
            raise ValueError(f"Invalid segment display_class: {raw_cls!r}")
        disp_cls: SegmentDisplayClass = raw_cls  # type: ignore[assignment]
        return cls(
            start_index=int(data["start_index"]),
            end_index=int(data["end_index"]),
            display_class=disp_cls,
            binding_hypotheses=tuple(
                str(h) for h in data.get("binding_hypotheses", ())
            ),
        )


@dataclass(frozen=True)
class PlotLineSeries:
    """Single line or overlaid representative series with provenance badge."""

    quantity_id: str
    label: str
    units: str
    values: tuple[float | None, ...]
    badge_text: str
    render_mode: str
    envelope_ref_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize line series to JSON-compatible dict."""
        return {
            "quantity_id": self.quantity_id,
            "label": self.label,
            "units": self.units,
            "values": list(self.values),
            "badge_text": self.badge_text,
            "render_mode": self.render_mode,
            "envelope_ref_id": self.envelope_ref_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlotLineSeries:
        """Rebuild a :class:`PlotLineSeries` from a mapping."""
        ref = data.get("envelope_ref_id")
        return cls(
            quantity_id=str(data["quantity_id"]),
            label=str(data["label"]),
            units=str(data["units"]),
            values=tuple(
                None if v is None else float(v) for v in data.get("values", ())
            ),
            badge_text=str(data.get("badge_text", "")),
            render_mode=str(data.get("render_mode", "")),
            envelope_ref_id=str(ref) if ref is not None else None,
        )


@dataclass(frozen=True)
class EnvelopeBandSeries:
    """Band-first feasible envelope series with optional overlaid representative."""

    quantity_id: str
    label: str
    units: str
    lower: tuple[float | None, ...]
    upper: tuple[float | None, ...]
    chebyshev_center: tuple[float | None, ...]
    chebyshev_radius: tuple[float | None, ...]
    segments: tuple[EnvelopeSegment, ...]
    badge_text: str
    render_mode: str
    representative: tuple[float | None, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize envelope band series to JSON-compatible dict."""
        return {
            "quantity_id": self.quantity_id,
            "label": self.label,
            "units": self.units,
            "lower": list(self.lower),
            "upper": list(self.upper),
            "chebyshev_center": list(self.chebyshev_center),
            "chebyshev_radius": list(self.chebyshev_radius),
            "segments": [s.to_dict() for s in self.segments],
            "badge_text": self.badge_text,
            "render_mode": self.render_mode,
            "representative": (
                list(self.representative) if self.representative is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvelopeBandSeries:
        """Rebuild an :class:`EnvelopeBandSeries` from a mapping."""
        rep = data.get("representative")
        return cls(
            quantity_id=str(data["quantity_id"]),
            label=str(data["label"]),
            units=str(data["units"]),
            lower=tuple(None if v is None else float(v) for v in data["lower"]),
            upper=tuple(None if v is None else float(v) for v in data["upper"]),
            chebyshev_center=tuple(
                None if v is None else float(v) for v in data["chebyshev_center"]
            ),
            chebyshev_radius=tuple(
                None if v is None else float(v) for v in data["chebyshev_radius"]
            ),
            segments=tuple(
                EnvelopeSegment.from_dict(s) for s in data.get("segments", ())
            ),
            badge_text=str(data.get("badge_text", "")),
            render_mode=str(data.get("render_mode", "")),
            representative=(
                tuple(None if v is None else float(v) for v in rep)
                if rep is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PlotPanelModel:
    """One panel of the framework-free NSX plot model."""

    panel_id: str
    title: str
    line_series: tuple[PlotLineSeries, ...] = ()
    envelope_series: tuple[EnvelopeBandSeries, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize panel to JSON-compatible dict."""
        return {
            "panel_id": self.panel_id,
            "title": self.title,
            "line_series": [s.to_dict() for s in self.line_series],
            "envelope_series": [e.to_dict() for e in self.envelope_series],
            "metrics": json.loads(json.dumps(dict(self.metrics))),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlotPanelModel:
        """Rebuild a :class:`PlotPanelModel` from a mapping."""
        return cls(
            panel_id=str(data["panel_id"]),
            title=str(data["title"]),
            line_series=tuple(
                PlotLineSeries.from_dict(s) for s in data.get("line_series", ())
            ),
            envelope_series=tuple(
                EnvelopeBandSeries.from_dict(e) for e in data.get("envelope_series", ())
            ),
            metrics=dict(data.get("metrics", {})),
        )


@dataclass(frozen=True)
class NSXPlotModel:
    """Framework-free plot model shared identically by PyQt6 and React."""

    schema_version: str
    report_id: str
    preset_name: str
    time_s: tuple[float, ...]
    panels: Mapping[str, PlotPanelModel]

    def to_dict(self) -> dict[str, Any]:
        """Serialize plot model to JSON-compatible dict."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "preset_name": self.preset_name,
            "time_s": list(self.time_s),
            "panels": {k: v.to_dict() for k, v in self.panels.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NSXPlotModel:
        """Rebuild an :class:`NSXPlotModel` from a mapping."""
        if "panels" not in data or "time_s" not in data:
            raise ValueError("plot model dictionary requires 'panels' and 'time_s'")
        panels_raw = data["panels"]
        return cls(
            schema_version=str(data.get("schema_version", "1.0.0")),
            report_id=str(data["report_id"]),
            preset_name=str(data["preset_name"]),
            time_s=tuple(float(t) for t in data["time_s"]),
            panels={str(k): PlotPanelModel.from_dict(v) for k, v in panels_raw.items()},
        )


def _map_sign_to_display_class(
    sign_label: str, lower_val: float | None, upper_val: float | None
) -> SegmentDisplayClass:
    if sign_label == "UNBOUNDED" or lower_val is None or upper_val is None:
        return "UNBOUNDED"
    if sign_label == "INDETERMINATE" or (lower_val < 0.0 < upper_val):
        return "INDETERMINATE"
    return "DETERMINED_SIGN"


def _binding_from_duals(dual_map: Mapping[str, Any]) -> tuple[str, ...]:
    active = [
        str(hid) for hid, val in dual_map.items() if abs(float(val)) > _DUAL_TOLERANCE
    ]
    return tuple(sorted(active))


def _build_segments(
    lower: Sequence[float | None],
    upper: Sequence[float | None],
    sign_classes: Sequence[str],
    duals: Sequence[Mapping[str, Any]],
) -> tuple[EnvelopeSegment, ...]:
    n = len(lower)
    if n == 0:
        return ()
    segments: list[EnvelopeSegment] = []
    cur_cls = _map_sign_to_display_class(sign_classes[0], lower[0], upper[0])
    cur_hyps = _binding_from_duals(duals[0])
    start = 0
    for idx in range(1, n):
        cls_i = _map_sign_to_display_class(sign_classes[idx], lower[idx], upper[idx])
        hyps_i = _binding_from_duals(duals[idx])
        if cls_i != cur_cls or hyps_i != cur_hyps:
            segments.append(
                EnvelopeSegment(
                    start_index=start,
                    end_index=idx - 1,
                    display_class=cur_cls,
                    binding_hypotheses=cur_hyps,
                )
            )
            start = idx
            cur_cls = cls_i
            cur_hyps = hyps_i
    segments.append(
        EnvelopeSegment(
            start_index=start,
            end_index=n - 1,
            display_class=cur_cls,
            binding_hypotheses=cur_hyps,
        )
    )
    return tuple(segments)


def _resolve_decision(
    qid: str,
    record: Mapping[str, Any],
    decisions: Mapping[str, DisplayDecision] | None,
    preset_name: str,
) -> DisplayDecision:
    if decisions is not None and qid in decisions:
        return decisions[qid]
    return display_decision(record, "research_view", preset_name=preset_name)


def _envelope_band_from_record(
    qid: str,
    env: Mapping[str, Any],
    decision: DisplayDecision,
    rep_values: Sequence[float | None] | None,
) -> EnvelopeBandSeries:
    lower = tuple(None if v is None else float(v) for v in env["lower"])
    upper = tuple(None if v is None else float(v) for v in env["upper"])
    center = tuple(None if v is None else float(v) for v in env["chebyshev_center"])
    radius = tuple(None if v is None else float(v) for v in env["chebyshev_radius"])
    segments = _build_segments(
        lower, upper, env["sign_class"], env["active_constraint_duals"]
    )
    rep_tuple = (
        tuple(None if v is None else float(v) for v in rep_values)
        if rep_values is not None
        else None
    )
    return EnvelopeBandSeries(
        quantity_id=qid,
        label=str(env["label"]),
        units=str(env["units"]),
        lower=lower,
        upper=upper,
        chebyshev_center=center,
        chebyshev_radius=radius,
        segments=segments,
        badge_text=decision.badge_text,
        render_mode=decision.render_mode,
        representative=rep_tuple,
    )


def _build_invariant_lines(
    invariants_map: Mapping[str, Any],
    display_decisions: Mapping[str, DisplayDecision] | None,
    preset_name: str,
) -> tuple[PlotLineSeries, ...]:
    lines: list[PlotLineSeries] = []
    for qid, rec in invariants_map.items():
        dec = _resolve_decision(qid, rec, display_decisions, preset_name)
        if dec.allowed:
            lines.append(
                PlotLineSeries(
                    quantity_id=qid,
                    label=str(rec["label"]),
                    units=str(rec["units"]),
                    values=tuple(
                        None if v is None else float(v) for v in rec["values"]
                    ),
                    badge_text=dec.badge_text,
                    render_mode=dec.render_mode,
                )
            )
    return tuple(lines)


def _partition_envelope_and_allocation_series(
    envelopes_map: Mapping[str, Any],
    reps_map: Mapping[str, Any],
    display_decisions: Mapping[str, DisplayDecision] | None,
    preset_name: str,
) -> tuple[
    tuple[EnvelopeBandSeries, ...],
    tuple[EnvelopeBandSeries, ...],
    tuple[EnvelopeBandSeries, ...],
    tuple[PlotLineSeries, ...],
]:
    torque_bands: list[EnvelopeBandSeries] = []
    split_bands: list[EnvelopeBandSeries] = []
    alloc_bands: list[EnvelopeBandSeries] = []
    alloc_lines: list[PlotLineSeries] = []
    for qid, env in envelopes_map.items():
        dec = _resolve_decision(qid, env, display_decisions, preset_name)
        if not dec.allowed:
            continue
        rep_rec = reps_map.get(qid)
        rep_vals = rep_rec["values"] if isinstance(rep_rec, Mapping) else None
        band = _envelope_band_from_record(qid, env, dec, rep_vals)
        if qid in _HAND_SPLIT_IDS:
            split_bands.append(band)
        else:
            torque_bands.append(band)
        if rep_rec is not None and rep_vals is not None:
            alloc_bands.append(band)
            alloc_lines.append(
                PlotLineSeries(
                    quantity_id=qid,
                    label=str(rep_rec["label"]),
                    units=str(rep_rec["units"]),
                    values=tuple(None if v is None else float(v) for v in rep_vals),
                    badge_text=dec.badge_text,
                    render_mode="BAND_WITH_REPRESENTATIVE",
                    envelope_ref_id=qid,
                )
            )
    return (
        tuple(torque_bands),
        tuple(split_bands),
        tuple(alloc_bands),
        tuple(alloc_lines),
    )


def build_plot_model(
    report: Mapping[str, Any] | Any,
    display_decisions: Mapping[str, DisplayDecision] | None = None,
    *,
    preset_name: str = "capacity_min_effort",
) -> NSXPlotModel:
    """Build the 5-panel framework-free NSX plot model (NSX-42, #12574).

    Raises:
        ValueError: If ``report`` is missing required sections or if an
            allocation representative is present without its corresponding
            envelope band (DbC band-first underlay invariant).
    """
    rep_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    invariants_map = rep_dict.get("invariants", {})
    envelopes_map = rep_dict.get("envelopes", {})
    allocation_sec = rep_dict.get("allocation")
    reps_map: Mapping[str, Any] = (
        allocation_sec.get("representatives", {})
        if isinstance(allocation_sec, Mapping)
        else {}
    )
    for qid in reps_map:
        if qid not in envelopes_map:
            raise ValueError(
                f"Allocation representative {qid!r} cannot be plotted without its "
                "corresponding envelope band"
            )

    inv_lines = _build_invariant_lines(invariants_map, display_decisions, preset_name)
    torque_bands, split_bands, alloc_bands, alloc_lines = (
        _partition_envelope_and_allocation_series(
            envelopes_map, reps_map, display_decisions, preset_name
        )
    )
    diag = dict(rep_dict.get("diagnostics", {}))
    adm = dict(rep_dict.get("admission", {}))
    panels = {
        "invariants": PlotPanelModel(
            panel_id="invariants",
            title="Allocation-Invariant Quantities",
            line_series=inv_lines,
        ),
        "envelopes": PlotPanelModel(
            panel_id="envelopes",
            title="Feasible Loop-Torque Envelopes",
            envelope_series=torque_bands,
        ),
        "hand_split": PlotPanelModel(
            panel_id="hand_split",
            title="Grip-Frame Hand-Split Coordinates (s1..s6)",
            envelope_series=split_bands,
        ),
        "allocation_representative": PlotPanelModel(
            panel_id="allocation_representative",
            title="Prior-Conditioned Representatives Overlaid on Envelopes",
            line_series=alloc_lines,
            envelope_series=alloc_bands,
        ),
        "diagnostics": PlotPanelModel(
            panel_id="diagnostics",
            title="Closure Rank, Conditioning, and Admission Residuals",
            metrics={
                "rank_jc": list(diag.get("rank_jc", [])),
                "condition_number_jc": list(diag.get("condition_number_jc", [])),
                "equality_nullity": list(diag.get("equality_nullity", [])),
                "chebyshev_radius": list(diag.get("chebyshev_radius", [])),
                "phi_scaled": list(adm.get("phi_scaled", [])),
                "phidot_scaled": list(adm.get("phidot_scaled", [])),
                "phiddot_scaled": list(adm.get("phiddot_scaled", [])),
            },
        ),
    }
    return NSXPlotModel(
        schema_version="1.0.0",
        report_id=str(rep_dict["report_id"]),
        preset_name=preset_name,
        time_s=tuple(float(t) for t in rep_dict["time_s"]),
        panels=panels,
    )
