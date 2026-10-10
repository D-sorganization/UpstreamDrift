"""Web Reports tab routes for Launch Monitor Analytics (issue #11987 slice 7a).

Sibling to ``src/api/routes/launch_monitor_analytics.py`` (same router prefix
and tag): that module is close to the file-size budget, so the Reports tab's
two endpoints live here instead. They call the same pure builders the PyQt
desktop Reports tab calls (``src/tools/launch_monitor_analytics/reporting.py``,
extracted from ``gui.py``'s ``_refresh_report``/``export_data``/
``export_manifest``), so the web and desktop paths share one implementation.

The web app has no import step — there is no ``ImportManifest`` per session —
so ``session_count`` is derived from distinct ``session_id`` values in the
inline records (0 when the column is absent) and ``import_warning_count`` is
always 0. The reproducibility manifest's ``sessions`` is always ``[]`` for the
same reason. Parquet export stays desktop-only; the web export is CSV only.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.api.middleware.error_handler import handle_api_errors

router = APIRouter(
    prefix="/tools/launch-monitor-analytics",
    tags=["launch-monitor-analytics"],
)


class ReportPayloadV2(BaseModel):
    """Bounded inline records for the web Reports tab.

    Mirrors the inputs ``_refresh_report``/``export_data``/
    ``export_manifest`` read from ``src/tools/launch_monitor_analytics/gui.py``,
    minus the desktop's imported-session state the web app does not have.
    ``project_name`` defaults to the desktop's ``clear_project`` default.
    """

    records: list[dict[str, Any]] = Field(min_length=1, max_length=20_000)
    project_name: str = Field("Untitled Launch Monitor Study", min_length=1)
    treatment_audit_log: list[dict[str, Any]] = Field(default_factory=list)


def _session_count_from_frame(frame: Any) -> int:
    """Count distinct non-null ``session_id`` values in ``frame``.

    Returns 0 when the column is absent. The web app has no import step, so
    there is no ``ImportedSession`` list to count directly; this mirrors the
    desktop's combined analysis frame, where every shot already carries its
    session's id.
    """
    if "session_id" not in frame.columns:
        return 0
    return int(frame["session_id"].dropna().nunique())


@router.post("/v2/report")
@handle_api_errors
async def build_report_v2(payload: ReportPayloadV2) -> dict[str, object]:
    """Build the Reports tab's report text for inline web records.

    Calls :func:`build_project_report`, the same builder the desktop Reports
    tab's ``_refresh_report`` calls, so the API and PyQt paths share one
    report-text contract. ``session_count`` comes from distinct ``session_id``
    values in ``records`` (0 when absent); ``import_warning_count`` is always
    0 because the web app has no import step to generate warnings.
    """

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    from src.tools.launch_monitor_analytics.reporting import build_project_report
    from src.tools.launch_monitor_model import numeric_metric_columns

    frame = pd.DataFrame.from_records(payload.records)
    session_count = _session_count_from_frame(frame)
    treatment_action_count = len(payload.treatment_audit_log)
    report_text = build_project_report(
        project_name=payload.project_name,
        session_count=session_count,
        frame=frame,
        import_warning_count=0,
        treatment_action_count=treatment_action_count,
    )
    return {
        "report_text": report_text,
        "project_name": payload.project_name,
        "session_count": session_count,
        "shot_count": len(frame),
        "canonical_metrics": numeric_metric_columns(frame),
        "treatment_action_count": treatment_action_count,
    }


@router.post("/v2/export")
@handle_api_errors
async def export_report_v2(payload: ReportPayloadV2) -> dict[str, object]:
    """Export the canonical CSV and reproducibility manifest for web records.

    Mirrors the desktop Reports tab's ``export_data`` (CSV branch only —
    Parquet stays desktop-only, since it needs a filesystem destination) and
    ``export_manifest``: a fresh ``export_id``/``exported_at`` pair is
    generated, the CSV text is built with :func:`format_canonical_csv_export`,
    the pair is correlated into ``data_export`` via
    :func:`build_data_export_record`, and the manifest is built with
    :func:`build_reproducibility_manifest`. ``sessions`` is always ``[]``: the
    web app has no imported-session manifests to serialize.
    """

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    from src.api.routes.launch_monitor_analytics import _audit_log_to_json_safe
    from src.tools.launch_monitor_analytics.reporting import (
        SCIENTIFIC_BOUNDARY_TEXT,
        build_data_export_record,
        build_reproducibility_manifest,
        format_canonical_csv_export,
    )

    frame = pd.DataFrame.from_records(payload.records)
    export_id = uuid.uuid4().hex
    exported_at = datetime.now(UTC).isoformat()
    data_file = "launch_monitor_data.csv"
    csv_text = format_canonical_csv_export(
        frame, export_id=export_id, exported_at=exported_at
    )
    data_export = build_data_export_record(
        export_id=export_id,
        exported_at=exported_at,
        data_file=data_file,
        data_bytes=csv_text.encode("utf-8"),
    )
    manifest = build_reproducibility_manifest(
        project_name=payload.project_name,
        sessions=[],
        treatment_audit_log=_audit_log_to_json_safe(tuple(payload.treatment_audit_log)),
        frame=frame,
        scientific_boundary=SCIENTIFIC_BOUNDARY_TEXT,
        data_export=data_export,
    )
    return {"csv": csv_text, "data_export": data_export, "manifest": manifest}
