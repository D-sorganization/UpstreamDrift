"""Web Sessions tab routes for Launch Monitor Analytics (issue #11987).

Sibling to ``src/api/routes/launch_monitor_analytics_reports.py`` (same router
prefix and tag): the two endpoints here call the same
``import_session``/``ImportMappingDialog`` review logic the PyQt desktop
Sessions tab calls (``src/tools/launch_monitor_analytics/gui.py``'s
``_on_import_files``/``import_file``, and
``src/tools/launch_monitor_analytics/widgets.py``'s ``ImportMappingDialog``,
now backed by the Qt-free ``src/tools/launch_monitor_analytics/import_review.py``),
so the web and desktop import paths share one implementation.

Nothing is persisted server side: an upload is written to a
``tempfile.TemporaryDirectory`` as ``upload<suffix>`` and the directory is
removed before the response is built. The returned manifest's
``source_path`` is replaced by the uploaded filename so no server-side temp
path ever leaks into a response. Private-corpus loading
(``load_private_corpus_sessions``, which needs ``LAUNCH_MONITOR_DATA_ROOT``)
stays desktop-only, and removing an imported session is client-side state on
the web — neither has a web route.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from typing import Any

import anyio
import numpy as np  # numpy itself is not pandas and is not deferred (issue #8943)
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from src.api.middleware.error_handler import handle_api_errors
from src.api.middleware.upload_limits import write_upload_file_to_path

router = APIRouter(
    prefix="/tools/launch-monitor-analytics",
    tags=["launch-monitor-analytics"],
)

_SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}
)
_MAX_IMPORT_ROWS = 20_000


class ImportMappingPayload(BaseModel):
    """One reviewed column-mapping override, matching ``ColumnMapping``."""

    source_column: str
    target_column: str
    source_unit: str | None = None
    multiplier: float = 1.0
    measurement_status: str = "reported"


class ImportOptionsPayload(BaseModel):
    """Reviewed import configuration for ``POST /v2/import``.

    Mirrors ``ImportMappingDialog.import_options()``'s inputs minus the
    desktop's per-header widget state: a header with no entry in
    ``mappings`` falls back to the profile's automatic mapping, exactly like
    leaving that header's dialog combo at its auto-detected value.
    """

    profile_id: str | None = None
    mappings: list[ImportMappingPayload] = Field(default_factory=list)
    session_name: str | None = None
    player: str | None = None
    monitor_model: str | None = None
    software_version: str | None = None
    tags: list[str] = Field(default_factory=list)


def _validate_upload_suffix(filename: str | None) -> str:
    """Return the lower-cased file suffix, or raise 415 for an unsupported type."""
    if not filename:
        raise HTTPException(status_code=415, detail="A filename is required")
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix}'. Supported: "
                + ", ".join(sorted(_SUPPORTED_SUFFIXES))
            ),
        )
    return suffix


def _sanitize_temp_path(text: str, tmp_path: Path, filename: str) -> str:
    """Replace every occurrence of the temp upload path with ``filename``.

    ``import_session`` embeds the resolved absolute source path in some of
    its ``ValueError`` messages. Nothing server-side should leak into a web
    response (see module docstring).
    """
    sanitized = text.replace(str(tmp_path), filename)
    return sanitized.replace(str(tmp_path.resolve()), filename)


def _json_safe_shot_records(frame: Any) -> list[dict[str, object]]:
    """Serialize imported shots to JSON-safe row dicts.

    Reuses :func:`src.api.routes.launch_monitor_analytics._frame_to_records`
    for NaN/infinite float and timestamp handling, then converts any
    remaining numpy scalar (e.g. the raw ``source::<column>`` columns, which
    keep their source dtype) to its native Python type.
    """
    from src.api.routes.launch_monitor_analytics import _frame_to_records

    records = _frame_to_records(frame)
    for entry in records:
        for key, value in entry.items():
            if isinstance(value, np.generic):
                entry[key] = value.item()
    return records


def _manifest_payload(manifest: Any, filename: str) -> dict[str, object]:
    """Serialize an ``ImportManifest`` with ``source_path`` set to ``filename``.

    ``ImportManifest.source_path`` is otherwise the resolved server-side temp
    path; nothing server-side should leak into a web response (see module
    docstring).
    """
    payload = dataclasses.asdict(manifest)
    payload["source_path"] = filename
    return payload


async def _write_and_read_headers(
    file: UploadFile, tmp_path: Path, filename: str
) -> list[str]:
    """Write the upload to ``tmp_path`` and return its column headers.

    Raises:
        HTTPException: 422 when the file's headers cannot be read; the
            message names ``filename``, never the server temp path.
    """
    from src.tools.launch_monitor_analytics.import_review import read_headers

    await write_upload_file_to_path(file, tmp_path)
    try:
        return read_headers(tmp_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=_sanitize_temp_path(str(exc), tmp_path, filename),
        ) from exc


@router.post("/v2/import/preview")
@handle_api_errors
async def preview_import_v2(file: UploadFile = File(...)) -> dict[str, object]:
    """Read headers and detect a vendor profile for an uploaded export.

    Mirrors the construction half of the desktop Sessions tab's
    ``ImportMappingDialog``: reads the headers, runs ``detect_profile``, and
    computes every profile's automatic mapping so a client can offer the
    same profile switcher without a second round trip. Nothing is written to
    durable storage; the upload is parsed from a
    ``tempfile.TemporaryDirectory`` that is removed before the response is
    built.
    """
    from src.tools.launch_monitor_analytics.import_review import (
        MEASUREMENT_STATUSES,
        auto_mappings,
        mapping_targets,
    )
    from src.tools.launch_monitor_model import PROFILES, detect_profile

    suffix = _validate_upload_suffix(file.filename)
    filename = file.filename or f"upload{suffix}"
    with tempfile.TemporaryDirectory(prefix="lm_import_preview_") as tmp_dir:
        tmp_path = Path(tmp_dir) / f"upload{suffix}"
        headers = await _write_and_read_headers(file, tmp_path, filename)
        detection = detect_profile(headers)
        auto_mappings_by_profile = {
            profile_id: auto_mappings(profile_id, headers) for profile_id in PROFILES
        }

    return {
        "filename": filename,
        "headers": headers,
        "detected_profile_id": detection.profile_id,
        "detection_confidence": detection.confidence,
        "profiles": [
            {"profile_id": profile_id, "vendor": profile.vendor}
            for profile_id, profile in PROFILES.items()
        ],
        "auto_mappings": auto_mappings_by_profile,
        "targets": mapping_targets(),
        "measurement_statuses": list(MEASUREMENT_STATUSES),
    }


@router.post("/v2/import")
@handle_api_errors
async def import_session_v2(
    file: UploadFile = File(...),
    options: str = Form("{}"),
) -> dict[str, object]:
    """Import an uploaded launch-monitor export with ``import_session``.

    Calls the same ``import_session`` the desktop Sessions tab's
    ``import_file`` calls, so the web and desktop import paths share one
    implementation. See the module docstring for what is, and is not,
    persisted server side.
    """
    from src.tools.launch_monitor_analytics.import_review import (
        build_import_options,
    )
    from src.tools.launch_monitor_model import (
        ImportedSession,
        detect_profile,
        import_session,
    )

    suffix = _validate_upload_suffix(file.filename)
    filename = file.filename or f"upload{suffix}"
    try:
        request = ImportOptionsPayload.model_validate_json(options)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid options: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="lm_import_") as tmp_dir:
        tmp_path = Path(tmp_dir) / f"upload{suffix}"
        headers = await _write_and_read_headers(file, tmp_path, filename)

        profile_id = request.profile_id or detect_profile(headers).profile_id
        rows = [
            (
                mapping.source_column,
                mapping.target_column,
                mapping.source_unit or "",
                mapping.multiplier,
                mapping.measurement_status,
            )
            for mapping in request.mappings
        ]
        try:
            base_options = build_import_options(
                profile_id=profile_id,
                rows=rows,
                session_name=request.session_name or "",
                default_session_name=Path(filename).stem,
                player=request.player or "",
                monitor_model=request.monitor_model or "",
                software_version=request.software_version or "",
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        import_options = dataclasses.replace(base_options, tags=tuple(request.tags))

        def _run_import() -> ImportedSession:
            return import_session(tmp_path, import_options)

        try:
            imported: object = await anyio.to_thread.run_sync(_run_import)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=_sanitize_temp_path(str(exc), tmp_path, filename),
            ) from exc
        if not isinstance(imported, ImportedSession):
            raise TypeError("import_session returned an invalid result")

        row_count = len(imported.shots)
        if row_count > _MAX_IMPORT_ROWS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Import produced {row_count} rows, exceeding the "
                    f"{_MAX_IMPORT_ROWS} row limit"
                ),
            )
        columns = [str(column) for column in imported.shots.columns]
        records = _json_safe_shot_records(imported.shots)
        manifest = _manifest_payload(imported.manifest, filename)

    return {
        "session_id": imported.session_id,
        "name": imported.name,
        "manifest": manifest,
        "columns": columns,
        "records": records,
        "row_count": row_count,
    }
