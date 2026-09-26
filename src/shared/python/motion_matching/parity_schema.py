"""Unified Cross-Engine Parity Report schema and data models (MS-70, #10350).

Defines:
1. ComparisonClass: Three explicit comparison categories (same-model numerical parity,
   native-model observable agreement, experimental accuracy).
2. PointwiseDifference: Pointwise trajectory differences and gate evaluation.
3. EngineParityRow: Per-engine status, metrics, work, wall-clock, and assumptions.
4. UnifiedParityReport: Complete serializable cross-engine comparison document with
   Markdown generation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from typing import Any

from src.shared.python.contracts import postcondition, precondition

PARITY_REPORT_SCHEMA_VERSION = "matched-swing-parity-report-v1"


class ComparisonClass(str, Enum):
    """Explicit categories for cross-engine evaluation."""

    SAME_MODEL_NUMERICAL_PARITY = "same_model_numerical_parity"
    NATIVE_MODEL_OBSERVABLE_AGREEMENT = "native_model_observable_agreement"
    EXPERIMENTAL_ACCURACY = "experimental_accuracy"


@dataclass(frozen=True)
class PointwiseDifference:
    """Pointwise discrepancy across time steps for a specific physical metric."""

    metric_name: str
    max_abs_diff: float
    rms_diff: float
    mean_diff: float
    unit: str = "m"
    pass_gate: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_abs_diff) or self.max_abs_diff < 0.0:
            raise ValueError(
                f"max_abs_diff must be finite non-negative: {self.max_abs_diff}"
            )
        if not math.isfinite(self.rms_diff) or self.rms_diff < 0.0:
            raise ValueError(f"rms_diff must be finite non-negative: {self.rms_diff}")
        if not math.isfinite(self.mean_diff) or self.mean_diff < 0.0:
            raise ValueError(f"mean_diff must be finite non-negative: {self.mean_diff}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "max_abs_diff": self.max_abs_diff,
            "rms_diff": self.rms_diff,
            "mean_diff": self.mean_diff,
            "unit": self.unit,
            "pass_gate": self.pass_gate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PointwiseDifference:
        return cls(
            metric_name=str(data["metric_name"]),
            max_abs_diff=float(data["max_abs_diff"]),
            rms_diff=float(data["rms_diff"]),
            mean_diff=float(data["mean_diff"]),
            unit=str(data.get("unit", "m")),
            pass_gate=bool(data.get("pass_gate", True)),
        )


@dataclass(frozen=True)
class EngineParityRow:
    """Per-engine parity assessment row."""

    engine: str
    status: str  # "qualified" | "unverified" | "unavailable" | "rejected"
    comparison_class: ComparisonClass = ComparisonClass.SAME_MODEL_NUMERICAL_PARITY
    model_name: str = ""
    model_sha256: str = ""
    assumptions: dict[str, Any] = field(default_factory=dict)
    shared_metrics: dict[str, float] | None = None
    pointwise_differences: dict[str, PointwiseDifference] = field(default_factory=dict)
    total_work_J: float | None = None
    wall_clock_s: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.comparison_class, str):
            object.__setattr__(
                self, "comparison_class", ComparisonClass(self.comparison_class)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "status": self.status,
            "comparison_class": self.comparison_class.value,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "assumptions": self.assumptions,
            "shared_metrics": self.shared_metrics,
            "pointwise_differences": {
                k: v.to_dict() for k, v in self.pointwise_differences.items()
            },
            "total_work_J": self.total_work_J,
            "wall_clock_s": self.wall_clock_s,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EngineParityRow:
        pt_diffs = {
            k: PointwiseDifference.from_dict(v)
            for k, v in data.get("pointwise_differences", {}).items()
        }
        total_work = data.get("total_work_J")
        wall_clock = data.get("wall_clock_s")
        return cls(
            engine=str(data["engine"]),
            status=str(data["status"]),
            comparison_class=ComparisonClass(
                data.get("comparison_class", "same_model_numerical_parity")
            ),
            model_name=str(data.get("model_name", "")),
            model_sha256=str(data.get("model_sha256", "")),
            assumptions=dict(data.get("assumptions", {})),
            shared_metrics=(
                dict(data["shared_metrics"])
                if data.get("shared_metrics") is not None
                else None
            ),
            pointwise_differences=pt_diffs,
            total_work_J=float(total_work) if total_work is not None else None,
            wall_clock_s=float(wall_clock) if wall_clock is not None else None,
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class UnifiedParityReport:
    """Unified cross-engine evaluation report covering all comparison classes."""

    schema_version: str = PARITY_REPORT_SCHEMA_VERSION
    candidate_id: str = "unknown"
    candidate_sha256: str = ""
    reference_engine: str = "pinocchio"
    reference_model_sha256: str = ""
    created_at: str = ""
    is_parity_accepted: bool = False
    status: str = "PARTIAL"  # "PASSED" | "REJECTED" | "PARTIAL"
    engine_rows: dict[str, EngineParityRow] = field(default_factory=dict)
    pairwise_comparisons: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "reference_engine": self.reference_engine,
            "reference_model_sha256": self.reference_model_sha256,
            "created_at": self.created_at,
            "is_parity_accepted": self.is_parity_accepted,
            "status": self.status,
            "engine_rows": {k: v.to_dict() for k, v in self.engine_rows.items()},
            "pairwise_comparisons": self.pairwise_comparisons,
        }
        if self.note:
            d["note"] = self.note
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UnifiedParityReport:
        rows = {
            k: EngineParityRow.from_dict(v)
            for k, v in data.get("engine_rows", {}).items()
        }
        is_accepted = bool(
            data.get("is_parity_accepted", data.get("is_physically_accepted", False))
        )
        return cls(
            schema_version=str(
                data.get("schema_version", PARITY_REPORT_SCHEMA_VERSION)
            ),
            candidate_id=str(data.get("candidate_id", "unknown")),
            candidate_sha256=str(data.get("candidate_sha256", "")),
            reference_engine=str(data.get("reference_engine", "pinocchio")),
            reference_model_sha256=str(data.get("reference_model_sha256", "")),
            created_at=str(data.get("created_at", "")),
            is_parity_accepted=is_accepted,
            status=str(data.get("status", "PARTIAL")),
            engine_rows=rows,
            pairwise_comparisons=dict(data.get("pairwise_comparisons", {})),
            note=str(data.get("note", "")),
        )

    def render_markdown(self) -> str:
        """Render a clean GitHub Flavored Markdown report."""
        lines: list[str] = [
            "# Unified Cross-Engine Parity Report",
            "",
            f"- **Candidate:** `{self.candidate_id}`",
            f"- **Candidate SHA:** `{self.candidate_sha256}`",
            f"- **Reference Engine:** `{self.reference_engine}`",
            f"- **Generated At:** `{self.created_at}`",
            f"- **Overall Verdict:** `{self.status}`",
            "",
            "## 1. Same-Model Numerical Parity",
            "",
            "Direct pointwise trajectory and torque comparisons against the reference model.",
            "Predeclared targets: **<= 1.0 mm** marker RMS, **<= 2.0 %** torque error (with 1.0 N·m floor).",
            "",
            "| Engine | Status | Marker RMS Diff (mm) | Max Marker Diff (mm) | Torque Diff (%) | Total Work (J) | Wall Clock (s) | Reason |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        same_rows = [
            r
            for r in self.engine_rows.values()
            if r.comparison_class == ComparisonClass.SAME_MODEL_NUMERICAL_PARITY
        ]
        if not same_rows:
            lines.append("| _none_ | - | - | - | - | - | - | - |")
        else:
            for r in same_rows:
                m_diff = r.pointwise_differences.get("marker_diff_m")
                t_diff = r.pointwise_differences.get("torque_diff_pct")
                m_rms = f"{m_diff.rms_diff * 1000.0:.2f}" if m_diff else "-"
                m_max = f"{m_diff.max_abs_diff * 1000.0:.2f}" if m_diff else "-"
                t_val = f"{t_diff.mean_diff:.1f} %" if t_diff else "-"
                work = f"{r.total_work_J:.1f}" if r.total_work_J is not None else "-"
                wall = f"{r.wall_clock_s:.3f}" if r.wall_clock_s is not None else "-"
                reason = r.reason or "qualified"
                lines.append(
                    f"| **{r.engine}** | `{r.status}` | {m_rms} | {m_max} | {t_val} | {work} | {wall} | {reason} |"
                )

        lines.extend(
            [
                "",
                "## 2. Native-Model Observable Agreement",
                "",
                "Evaluates observable kinematic paths (markers, yaw, ground forces) across distinct coordinate architectures.",
                "Non-identical coordinate models do not inherit identical torque requirements.",
                "",
                "| Engine | Status | Whole Marker RMSE (mm) | Pelvis Yaw RMSE (deg) | Model Name | Reason / Notes |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )

        native_rows = [
            r
            for r in self.engine_rows.values()
            if r.comparison_class == ComparisonClass.NATIVE_MODEL_OBSERVABLE_AGREEMENT
        ]
        if not native_rows:
            lines.append("| _none_ | - | - | - | - | - |")
        else:
            for r in native_rows:
                metrics = r.shared_metrics or {}
                w_rmse = (
                    f"{metrics['whole_marker_rmse_m'] * 1000.0:.2f}"
                    if "whole_marker_rmse_m" in metrics
                    else "-"
                )
                yaw_deg = (
                    f"{math.degrees(metrics['pelvis_yaw_rmse_rad']):.2f}°"
                    if "pelvis_yaw_rmse_rad" in metrics
                    else "-"
                )
                model = r.model_name or "native"
                reason = r.reason or "qualified"
                lines.append(
                    f"| **{r.engine}** | `{r.status}` | {w_rmse} | {yaw_deg} | `{model}` | {reason} |"
                )

        lines.extend(
            [
                "",
                "## 3. Assumptions and Model Invariants",
                "",
            ]
        )
        for r in self.engine_rows.values():
            if r.assumptions:
                lines.append(f"### `{r.engine}` Assumptions")
                for k, v in sorted(r.assumptions.items()):
                    lines.append(f"- **{k}**: `{v}`")
                lines.append("")

        return "\n".join(lines) + "\n"
