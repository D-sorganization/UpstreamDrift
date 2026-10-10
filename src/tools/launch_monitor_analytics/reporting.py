"""Pure report/export builders shared by the desktop GUI and the web API.

Extracted from ``src/tools/launch_monitor_analytics/gui.py``'s Reports tab
(``_refresh_report``, ``export_data``, ``export_manifest``) so the PyQt
desktop app and the FastAPI routes build byte-identical report text, CSV
exports, and reproducibility manifests from one implementation
(issue #11987 slice 7a). This module has no Qt dependency; pandas is
imported at module scope because, unlike an API route handler, nothing here
sits on the API boot path (see
``src/api/routes/launch_monitor_analytics_reports.py`` for the deferred
import at that boundary).
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from src.tools.launch_monitor_model import numeric_metric_columns

SCIENTIFIC_BOUNDARY_TEXT = (
    "Scientific Boundary: Correlation and predictive fit do not establish "
    "causality. Derived metrics and unmatched monitor comparisons require "
    "special care."
)
"""The desktop Reports tab's ``scientific_boundary`` label text.

Single source of truth for both the PyQt label
(``src/tools/launch_monitor_analytics/gui.py``) and the reproducibility
manifest the web API builds, so the two never drift apart.
"""


def build_project_report(
    *,
    project_name: str,
    session_count: int,
    frame: pd.DataFrame,
    import_warning_count: int,
    treatment_action_count: int,
) -> str:
    """Build the Reports tab's plain-text project report.

    Mirrors ``_refresh_report`` in ``src/tools/launch_monitor_analytics/gui.py``
    exactly. The desktop calls this with ``len(self.project.sessions)``,
    ``self.analysis_frame``, the combined import-warning count across every
    imported session, and ``len(self.project.audit_log)``. The web API has no
    import step, so it passes a session count derived from the inline
    records and an ``import_warning_count`` of 0.

    Preconditions: ``project_name`` is non-empty once stripped of
    whitespace; ``session_count``, ``import_warning_count``, and
    ``treatment_action_count`` are non-negative.

    Postcondition: the returned text is byte-identical to what
    ``_refresh_report`` would set on ``report_text`` for the same inputs.
    """
    if not project_name.strip():
        raise ValueError("project_name must not be empty")
    for name, value in (
        ("session_count", session_count),
        ("import_warning_count", import_warning_count),
        ("treatment_action_count", treatment_action_count),
    ):
        if value < 0:
            raise ValueError(f"{name} must not be negative, got {value}")

    source_fields = len(
        [column for column in frame.columns if str(column).startswith("source::")]
    )
    metrics = numeric_metric_columns(frame)
    return (
        "Launch Monitor Analytics Project\n"
        "================================\n"
        f"Project: {project_name}\n"
        f"Sessions: {session_count}\n"
        f"Shots: {len(frame)}\n"
        f"Canonical Numeric Metrics: {len(metrics)}\n"
        f"Retained Source Fields: {source_fields}\n"
        f"Import Warnings: {import_warning_count}\n\n"
        f"Recorded Treatment Actions: {treatment_action_count}\n\n"
        "Scientific Interpretation & Traceability\n"
        "-----------------------------------------\n"
        "Relationships describe association, not causation. Identity-derived "
        "metrics are marked by the metric registry. Matched shots are required "
        "for monitor bias and agreement claims; unmatched comparisons remain "
        "descriptive. Original source columns and per-file SHA-256 provenance "
        "are retained in the project.\n\n"
        "Methodology & Formula Traceability:\n"
        "- Longitudinal trends: Theil-Sen robust linear regression with Mann-Kendall test\n"
        "- Dispersion: 95% bivariate normal confidence ellipse (Hotelling T^2)\n"
        "- Multicollinearity: Variance Inflation Factor (VIF = 1 / (1 - R_i^2))\n"
        "- Strokes Gained: SG = verified E(start state) - 1 - verified E(finish state) "
        "(Broadie 2011/2014, DOI: 10.1287/inte.1110.0594)\n"
    )


def format_canonical_csv_export(
    frame: pd.DataFrame, *, export_id: str, exported_at: str
) -> str:
    """Build the CSV text ``export_data``'s CSV branch writes to disk.

    Mirrors ``export_data`` in ``src/tools/launch_monitor_analytics/gui.py``:
    a leading ``# export_id=... exported_at=...`` comment row, followed by
    ``frame.to_csv(index=False)`` verbatim. ``frame.to_csv()`` builds the
    same character sequence whether it is handed a file handle or returns a
    string, so this concatenation reproduces the desktop's on-disk bytes.

    Preconditions: ``export_id`` and ``exported_at`` are non-empty.
    """
    if not export_id:
        raise ValueError("export_id must not be empty")
    if not exported_at:
        raise ValueError("exported_at must not be empty")
    header = f"# export_id={export_id} exported_at={exported_at}\n"
    return header + frame.to_csv(index=False)


def build_data_export_record(
    *, export_id: str, exported_at: str, data_file: str, data_bytes: bytes
) -> dict[str, str]:
    """Build the ``_last_data_export`` record ``export_data`` stamps on export.

    Preconditions: ``export_id``, ``exported_at``, and ``data_file`` are
    non-empty; ``data_bytes`` is non-empty.

    Postcondition: ``data_sha256`` is the SHA-256 hex digest of
    ``data_bytes`` — the exported artifact's own bytes, mirroring
    ``export_data``'s
    ``hashlib.sha256(destination.read_bytes()).hexdigest()``.
    """
    if not export_id:
        raise ValueError("export_id must not be empty")
    if not exported_at:
        raise ValueError("exported_at must not be empty")
    if not data_file:
        raise ValueError("data_file must not be empty")
    if not data_bytes:
        raise ValueError("data_bytes must not be empty")
    return {
        "export_id": export_id,
        "exported_at": exported_at,
        "data_file": data_file,
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
    }


def build_reproducibility_manifest(
    *,
    project_name: str,
    sessions: list[dict[str, Any]],
    treatment_audit_log: list[dict[str, Any]],
    frame: pd.DataFrame,
    scientific_boundary: str,
    data_export: dict[str, str] | None,
) -> dict[str, Any]:
    """Build the reproducibility-manifest payload ``export_manifest`` writes.

    Mirrors ``export_manifest`` in ``src/tools/launch_monitor_analytics/gui.py``
    key-for-key and in order: ``project``, ``sessions``,
    ``treatment_audit_log``, ``analysis_rows``, ``canonical_metrics``,
    ``scientific_boundary``, ``data_export``.

    Preconditions: ``project_name`` and ``scientific_boundary`` are
    non-empty once stripped of whitespace.

    Postcondition: ``data_export`` passes through unchanged (``None`` when
    no export has run yet).
    """
    if not project_name.strip():
        raise ValueError("project_name must not be empty")
    if not scientific_boundary.strip():
        raise ValueError("scientific_boundary must not be empty")
    return {
        "project": project_name,
        "sessions": sessions,
        "treatment_audit_log": treatment_audit_log,
        "analysis_rows": len(frame),
        "canonical_metrics": numeric_metric_columns(frame),
        "scientific_boundary": scientific_boundary,
        "data_export": data_export,
    }
