"""Swing report web routes (issue #12330): React parity with #12329.

The PyQt ``SwingReportDialog`` is canonical. These routes expose the same
``ReportViewModel`` to the web dialog so both surfaces list identical rows and
share one selection and export implementation (DRY):

* ``POST /swing-report/view`` validates a ``swing-report-v1`` document and
  returns the text-only rows plus the default Custom selection.
* ``POST /swing-report/export`` resolves a Full or Custom selection through
  ``select_plots`` and returns the S4 package (``export_package``) as a zip.

The swing report package needs ``jsonschema`` and ``matplotlib``; it is
imported inside the handlers so route discovery never depends on them, and a
runtime without them answers 503.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
import re
import tempfile
from types import ModuleType
from typing import Any
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from src.api.models.swing_report import (
    SwingReportExportRequest,
    SwingReportPlotRow,
    SwingReportViewRequest,
    SwingReportViewResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]")


def package_filename(report_id: str) -> str:
    """Download name for a report package; only ``[A-Za-z0-9_.-]`` survive."""
    return f"swing-report-{_UNSAFE_FILENAME.sub('_', report_id)}.zip"


def _swing_report_modules() -> tuple[ModuleType, ModuleType]:
    """Return the ``(validator, view_model)`` modules, or raise 503."""
    try:
        from src.shared.python.reporting.swing_report import validator, view_model
    except ImportError as exc:
        logger.warning("Swing report runtime unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Swing report support is unavailable in this runtime: {exc}",
        ) from exc
    return validator, view_model


def _load_model(doc: dict[str, Any]) -> Any:
    """Validate ``doc`` and wrap it in a ``ReportViewModel`` (422 if invalid)."""
    validator, view_model = _swing_report_modules()
    try:
        report = validator.validate_swing_report(doc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return view_model.ReportViewModel(report)


@router.post("/swing-report/view", response_model=SwingReportViewResponse)
def view_swing_report(request: SwingReportViewRequest) -> SwingReportViewResponse:
    """Rows of a validated report, exactly as the PyQt dialog lists them.

    Postconditions: ``rows`` equal the Python ``ReportViewModel`` rows (text
    only); ``default_custom_ids`` holds only enabled plot ids.
    """
    model = _load_model(request.report)
    rows = [
        SwingReportPlotRow.model_validate(r, from_attributes=True) for r in model.rows
    ]
    model.set_mode("custom")
    return SwingReportViewResponse(
        report_id=request.report["report_id"],
        rows=rows,
        default_custom_ids=list(model.selected_ids),
    )


def _zip_directory(root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(root).as_posix())
    return buffer.getvalue()


@router.post("/swing-report/export")
def export_swing_report(request: SwingReportExportRequest) -> Response:
    """Export the Full or Custom selection as a zipped S4 package.

    Preconditions: the report is valid and every Custom id is an enabled
    (non-withheld) plot, else 422.
    Postcondition: the zip unpacks to a package ``load_package`` accepts.
    """
    model = _load_model(request.report)
    try:
        if request.mode == "custom":
            model.set_custom_selection(request.plot_ids)
        model.set_mode(request.mode)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "package"
            model.export(out)
            payload = _zip_directory(out)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    name = package_filename(request.report["report_id"])
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
