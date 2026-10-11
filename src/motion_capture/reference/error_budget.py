"""Error budget receipt, guidance derivation, and public summary generation (COV-11, #11279).

Condenses markerless, Necromatcher and engine-fitting comparison results against
capture-O marker ground truth into a versioned, machine-readable error budget.
Derives guidance for downstream consumers (Tiger #11226, Hogan #11229, Necromatcher #11232/#11235)
using frozen rules, and generates a privacy-preserving public neutral summary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.shared.python.core.contracts import require

ERROR_BUDGET_SCHEMA_VERSION = "error-budget/1.0.0"

_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_FRAME_INDEX_PATTERN = re.compile(r"cov-\d+.*frame|frame[ _-]?\d+", re.IGNORECASE)
_ABSOLUTE_PATH_PATTERN = re.compile(
    r"[A-Za-z]:[\\/]|/(?:home|Users|var|tmp|AppData)/", re.IGNORECASE
)
_PRIVATE_FILE_PATTERN = re.compile(
    r"\bVID_\d+\b|\.mp4\b|\.c3d\b|\.mov\b|originals[\\/]", re.IGNORECASE
)


class ErrorBudgetCell(BaseModel):
    """A single error budget cell representing observed accuracy for a configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source: str = Field(min_length=1)
    joint_or_landmark: str = Field(min_length=1)
    phase_bin: str = Field(default="all")
    view: str = Field(default="dtl")
    grade: str = Field(default="A")
    comparison_level: str = Field(min_length=2)
    metric: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    n_swings: int
    n_frames: int
    p50: float
    p95: float
    worst: float
    camera_uncertainty_spread: float | None = None
    resolvable: bool = True
    source_receipt_hashes: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def _validate_cell_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            n_swings = data.get("n_swings")
            require(
                n_swings is not None and n_swings > 0,
                f"n_swings must be positive, got {n_swings}",
            )
            n_frames = data.get("n_frames")
            require(
                n_frames is not None and n_frames > 0,
                f"n_frames must be positive, got {n_frames}",
            )
        return data

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        require(self.n_swings > 0, f"n_swings must be positive, got {self.n_swings}")
        require(self.n_frames > 0, f"n_frames must be positive, got {self.n_frames}")
        require(
            math.isfinite(self.p50)
            and math.isfinite(self.p95)
            and math.isfinite(self.worst),
            "Percentiles and worst must be finite numbers",
        )
        if self.camera_uncertainty_spread is not None:
            require(
                math.isfinite(self.camera_uncertainty_spread)
                and self.camera_uncertainty_spread >= 0.0,
                "camera_uncertainty_spread must be finite and non-negative",
            )
        for h in self.source_receipt_hashes:
            require(
                bool(_SHA256_HEX_PATTERN.fullmatch(h)),
                f"Receipt hash {h!r} must be a 64-char SHA-256 digest",
            )


class NotMeasuredRecord(BaseModel):
    """Explicit record of an unmeasured or omitted joint, metric, or configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source: str = Field(min_length=1)
    joint_or_landmark: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ErrorBudget(BaseModel):
    """Versioned machine-readable error budget condensing observed comparison metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: str = ERROR_BUDGET_SCHEMA_VERSION
    cells: tuple[ErrorBudgetCell, ...]
    not_measured: tuple[NotMeasuredRecord, ...] = Field(default_factory=tuple)
    created_utc: str
    receipt_digest: str

    def to_json(self) -> str:
        """Serialize budget to deterministic, formatted JSON."""
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)


class GuidanceRule(BaseModel):
    """Frozen rule defining thresholds for classifying quantity reliability."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    quantity_name: str = Field(min_length=1)
    target_metric: str = Field(min_length=1)
    max_trustworthy_p95: float = Field(gt=0.0)
    max_indicative_p95: float = Field(gt=0.0)
    unit: str = Field(min_length=1)
    requires_level: str = Field(default="L3")
    is_per_frame: bool = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        require(
            self.max_trustworthy_p95 <= self.max_indicative_p95,
            f"max_trustworthy_p95 ({self.max_trustworthy_p95}) must be <= max_indicative_p95 ({self.max_indicative_p95})",
        )


class GuidanceItem(BaseModel):
    """Derived guidance classification for a specific biomechanical quantity."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    quantity_name: str
    classification: str
    threshold_x: float
    observed_p95: float | None
    unit: str
    comparison_level: str
    source: str
    rationale: str


class GuidanceReport(BaseModel):
    """Collection of derived guidance items for downstream consumers."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: str = ERROR_BUDGET_SCHEMA_VERSION
    guidance_rules_version: str = ERROR_BUDGET_SCHEMA_VERSION
    items: tuple[GuidanceItem, ...]
    created_utc: str


_DEFAULT_GUIDANCE_RULES: dict[str, GuidanceRule] = {
    "hand_path_depth": GuidanceRule(
        quantity_name="hand_path_depth",
        target_metric="depth_error",
        max_trustworthy_p95=40.0,
        max_indicative_p95=80.0,
        unit="mm",
        requires_level="L3",
        is_per_frame=True,
    ),
    "pelvis_rotation_top": GuidanceRule(
        quantity_name="pelvis_rotation_top",
        target_metric="angle_error",
        max_trustworthy_p95=5.0,
        max_indicative_p95=10.0,
        unit="deg",
        requires_level="L3",
        is_per_frame=False,
    ),
    "x_factor": GuidanceRule(
        quantity_name="x_factor",
        target_metric="angle_error",
        max_trustworthy_p95=6.0,
        max_indicative_p95=12.0,
        unit="deg",
        requires_level="L3",
        is_per_frame=False,
    ),
    "kinematic_sequence_order": GuidanceRule(
        quantity_name="kinematic_sequence_order",
        target_metric="timing_error",
        max_trustworthy_p95=0.05,
        max_indicative_p95=0.10,
        unit="phase_pct",
        requires_level="L3",
        is_per_frame=False,
    ),
}


def get_default_guidance_rules() -> dict[str, GuidanceRule]:
    """Return a copy of the authoritative frozen guidance rules."""
    return dict(_DEFAULT_GUIDANCE_RULES)


def _canonical_cell_key(c: ErrorBudgetCell) -> tuple[str, ...]:
    return (
        c.source,
        c.joint_or_landmark,
        c.phase_bin,
        c.view,
        c.grade,
        c.comparison_level,
        c.metric,
    )


def _extract_summary_record(
    s: Mapping[str, Any] | BaseModel,
) -> tuple[ErrorBudgetCell | None, NotMeasuredRecord | None]:
    """Extract an ErrorBudgetCell or NotMeasuredRecord from an input summary."""
    data: dict[str, Any] = s.model_dump() if isinstance(s, BaseModel) else dict(s)
    source = str(data.get("backend") or data.get("source") or "unknown")
    joint = str(data.get("joint_or_landmark") or data.get("joint_name") or "unknown")
    metric = str(data.get("metric") or "residual")
    n_swings = int(data.get("n_swings", 0))
    n_frames = int(data.get("n_frames", 0))

    if n_swings <= 0 or n_frames <= 0:
        reason = str(
            data.get("reason")
            or f"Omitted: 0 valid swings/frames measured for {joint} ({metric})"
        )
        return None, NotMeasuredRecord(
            source=source,
            joint_or_landmark=joint,
            metric=metric,
            reason=reason,
        )

    receipt_hash = data.get("receipt_hash")
    hashes = data.get("source_receipt_hashes")
    if hashes:
        receipt_hashes = tuple(sorted(str(h) for h in hashes))
    elif receipt_hash:
        receipt_hashes = (str(receipt_hash),)
    else:
        receipt_hashes = ()

    cell = ErrorBudgetCell(
        source=source,
        joint_or_landmark=joint,
        phase_bin=str(data.get("phase_bin", "all")),
        view=str(data.get("view", "dtl")),
        grade=str(data.get("grade", "A")),
        comparison_level=str(data.get("level") or data.get("comparison_level", "L2")),
        metric=metric,
        unit=str(data.get("unit", "px")),
        n_swings=n_swings,
        n_frames=n_frames,
        p50=float(data["p50"]),
        p95=float(data["p95"]),
        worst=float(data["worst"]),
        camera_uncertainty_spread=(
            float(data["camera_uncertainty_spread"])
            if data.get("camera_uncertainty_spread") is not None
            else None
        ),
        resolvable=bool(data.get("resolvable", True)),
        source_receipt_hashes=receipt_hashes,
    )
    return cell, None


def build_error_budget(
    summaries: Sequence[Mapping[str, Any] | BaseModel],
    *,
    created_utc: str | None = None,
) -> ErrorBudget:
    """Build a deterministic, validated error budget from comparison summaries."""
    cells: list[ErrorBudgetCell] = []
    not_measured: list[NotMeasuredRecord] = []
    timestamps: list[str] = []

    for s in summaries:
        data: dict[str, Any] = s.model_dump() if isinstance(s, BaseModel) else dict(s)
        ts = data.get("created_utc") or data.get("timestamp")
        if ts and isinstance(ts, str):
            timestamps.append(ts)
        cell, unmeasured = _extract_summary_record(s)
        if cell is not None:
            cells.append(cell)
        if unmeasured is not None:
            not_measured.append(unmeasured)

    sorted_cells = tuple(sorted(cells, key=_canonical_cell_key))
    sorted_unmeasured = tuple(
        sorted(
            not_measured,
            key=lambda u: (u.source, u.joint_or_landmark, u.metric, u.reason),
        )
    )

    if created_utc is None:
        created_utc = max(timestamps) if timestamps else "2026-10-04T00:00:00Z"

    raw_payload = json.dumps(
        {
            "schema_version": ERROR_BUDGET_SCHEMA_VERSION,
            "cells": [c.model_dump(mode="json") for c in sorted_cells],
            "not_measured": [u.model_dump(mode="json") for u in sorted_unmeasured],
        },
        sort_keys=True,
    ).encode("utf-8")
    receipt_digest = hashlib.sha256(raw_payload).hexdigest()

    return ErrorBudget(
        schema_version=ERROR_BUDGET_SCHEMA_VERSION,
        cells=sorted_cells,
        not_measured=sorted_unmeasured,
        created_utc=created_utc,
        receipt_digest=receipt_digest,
    )


def _classify_cell_guidance(
    cell: ErrorBudgetCell,
    rule: GuidanceRule,
) -> GuidanceItem:
    """Classify a single budget cell against a frozen guidance rule."""
    if rule.is_per_frame and cell.comparison_level == "L1":
        raise ValueError(
            f"L1 cell cannot be labelled trustworthy for per-frame quantity '{rule.quantity_name}'"
        )

    thresh_x = rule.max_trustworthy_p95
    thresh_ind = rule.max_indicative_p95
    obs_p95 = cell.p95

    if not cell.resolvable:
        classification = "not recoverable from single view"
        rationale = (
            "Camera uncertainty spread exceeds nominal difference; not resolvable"
        )
    elif obs_p95 < thresh_x:
        classification = f"trustworthy at p95 < {thresh_x}"
        rationale = f"Observed p95 ({obs_p95:.1f} {rule.unit}) meets target bound (< {thresh_x} {rule.unit})"
    elif obs_p95 < thresh_ind:
        classification = "indicative"
        rationale = f"Observed p95 ({obs_p95:.1f} {rule.unit}) meets indicative bound (< {thresh_ind} {rule.unit})"
    else:
        classification = "not recoverable from single view"
        rationale = f"Observed p95 ({obs_p95:.1f} {rule.unit}) exceeds bounds (>= {thresh_ind} {rule.unit})"

    return GuidanceItem(
        quantity_name=rule.quantity_name,
        classification=classification,
        threshold_x=thresh_x,
        observed_p95=obs_p95,
        unit=rule.unit,
        comparison_level=cell.comparison_level,
        source=cell.source,
        rationale=rationale,
    )


def derive_guidance(
    budget: ErrorBudget,
    *,
    rules: Mapping[str, GuidanceRule] | None = None,
    schema_version: str = ERROR_BUDGET_SCHEMA_VERSION,
    created_utc: str | None = None,
) -> GuidanceReport:
    """Derive guidance items from error budget cells enforcing frozen rule invariants."""
    effective_rules = dict(rules if rules is not None else _DEFAULT_GUIDANCE_RULES)

    if rules is not None and rules != _DEFAULT_GUIDANCE_RULES:
        require(
            schema_version != ERROR_BUDGET_SCHEMA_VERSION,
            f"Guidance rules are frozen in schema version {ERROR_BUDGET_SCHEMA_VERSION}. "
            "Modifying thresholds requires bumping schema version.",
        )

    items: list[GuidanceItem] = []
    for cell in budget.cells:
        name = cell.joint_or_landmark
        rule = effective_rules.get(name)
        if rule is None:
            rule = GuidanceRule(
                quantity_name=name,
                target_metric=cell.metric,
                max_trustworthy_p95=30.0 if cell.unit == "mm" else 0.02,
                max_indicative_p95=60.0 if cell.unit == "mm" else 0.04,
                unit=cell.unit,
                requires_level="L2",
                is_per_frame=True,
            )
        items.append(_classify_cell_guidance(cell, rule))

    eff_created = created_utc if created_utc is not None else budget.created_utc
    return GuidanceReport(
        schema_version=schema_version,
        guidance_rules_version=schema_version,
        items=tuple(items),
        created_utc=eff_created,
    )


def _assert_privacy_clean(text: str) -> None:
    """Assert generated text contains no private paths, frame indices, or files."""
    require(
        not _FRAME_INDEX_PATTERN.search(text),
        "Privacy invariant violation: summary contains frame index leaks",
    )
    require(
        not _ABSOLUTE_PATH_PATTERN.search(text),
        "Privacy invariant violation: summary contains absolute filesystem paths",
    )
    require(
        not _PRIVATE_FILE_PATTERN.search(text),
        "Privacy invariant violation: summary contains private store file names",
    )


def generate_public_summary(
    budget: ErrorBudget,
    guidance: GuidanceReport,
    *,
    owner_approved: bool = True,
) -> str:
    """Generate a neutral, privacy-preserving public Markdown summary with owner approval."""
    require(owner_approved, "Public summary cannot be generated without owner approval")

    lines: list[str] = [
        "# Capture-O Single-View Reconstruction Error Budget and Guidance Summary",
        "",
        "Parent epic: #11268 (Capture-O Video Companion).",
        "Subject: `subject-O` (single subject, marker-anchored capture `capture-O`).",
        f"Schema Version: `{budget.schema_version}`",
        f"Receipt Digest: `{budget.receipt_digest}`",
        "",
        "## 1. Executive Summary & Guidance Classification",
        "",
        "Single-view historical video reconstruction is evaluated against owner marker ground truth.",
        "Downstream historical-player programs (#11226, #11229, #11235) cite these empirical bounds.",
        "",
        "| Quantity | Level | Source | Observed p95 | Target Bound | Classification | Rationale |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for item in guidance.items:
        obs_str = (
            f"{item.observed_p95:.2f} {item.unit}"
            if item.observed_p95 is not None
            else "N/A"
        )
        lines.append(
            f"| `{item.quantity_name}` | {item.comparison_level} | `{item.source}` | "
            f"{obs_str} | < {item.threshold_x} {item.unit} | **{item.classification}** | {item.rationale} |"
        )

    lines.extend(
        [
            "",
            "## 2. Aggregate Error Budget Table",
            "",
            "| Source | Quantity | Phase | View | Grade | Level | Metric | Unit | n Swings | p50 | p95 | Worst | Resolvable |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for cell in budget.cells:
        lines.append(
            f"| `{cell.source}` | `{cell.joint_or_landmark}` | {cell.phase_bin} | {cell.view} | {cell.grade} | "
            f"{cell.comparison_level} | {cell.metric} | {cell.unit} | {cell.n_swings} | "
            f"{cell.p50:.2f} | {cell.p95:.2f} | {cell.worst:.2f} | {cell.resolvable} |"
        )

    if budget.not_measured:
        lines.extend(
            [
                "",
                "## 3. Omitted & Unmeasured Channels",
                "",
                "| Source | Quantity | Metric | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for u in budget.not_measured:
            lines.append(
                f"| `{u.source}` | `{u.joint_or_landmark}` | {u.metric} | {u.reason} |"
            )

    lines.extend(
        [
            "",
            "## 4. Governed Limitations",
            "",
            "- **Single Subject:** Measured exclusively on `subject-O`. No general population claim is made.",
            "- **Single Camera View:** Monocular depth ambiguity remains unobservable without multi-view or physical constraints.",
            "- **Unpaired Swings:** Swings without confident pairing are held at Level L0/L1 and excluded from per-frame L2/L3 metrics.",
            "- **Clock Qualification:** Streams with unknown frame clocks report phase-normalized time and omit velocity metrics.",
            "- **Landmark Conventions:** Marker set uses surface skin markers; visual detectors estimate joint centres.",
            "- **Marker Occlusion:** Known occlusions during high-acceleration swing phases are explicitly noted.",
            "",
        ]
    )

    summary_text = "\n".join(lines)
    _assert_privacy_clean(summary_text)
    return summary_text
