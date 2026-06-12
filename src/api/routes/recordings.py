"""Recording persistence and export routes (issue #7451).

Export/recording parity with the desktop app: the PyQt6 dashboard records
sessions via ``GenericPhysicsRecorder`` and exports HDF5/CSV/MAT/C3D/JSON
through ``src.shared.python.data_io.export``. These routes expose the same
recorder data and the *same* export serializers over HTTP so the web client
gets byte-identical artifacts.

Endpoints:
    - ``POST /recordings`` — finalize the active session recorder to disk.
    - ``GET /recordings`` — list persisted recordings with metadata.
    - ``GET /recordings/{id}`` — single recording metadata.
    - ``DELETE /recordings/{id}`` — remove a recording.
    - ``GET /recordings/{id}/export?format=...`` — stream an export artifact.
    - ``GET /export/formats`` — honest enumeration of available formats.

All dependencies are injected via FastAPI's Depends() mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from src.api.services.recording_service import (
    InvalidRecordingIdError,
    RecordingNotFoundError,
    RecordingStore,
    exportable_formats,
)

from ..dependencies import get_simulation_service

if TYPE_CHECKING:
    from ..services.simulation_service import SimulationService

router = APIRouter(tags=["export"])

_MEDIA_TYPES: dict[str, str] = {
    "json": "application/json",
    "csv": "text/csv",
    "mat": "application/octet-stream",
    "hdf5": "application/x-hdf5",
    "c3d": "application/octet-stream",
}


def get_recording_store(request: Request) -> RecordingStore:
    """Retrieve (or lazily create) the RecordingStore from app state."""
    state = request.app.state
    store = getattr(state, "recording_store", None)
    if store is None:
        store = RecordingStore()
        state.recording_store = store
    return store


def _http_404(recording_id: str) -> HTTPException:
    return HTTPException(
        status_code=404, detail=f"Recording '{recording_id}' not found"
    )


def _http_invalid_id(exc: InvalidRecordingIdError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@router.get("/export/formats")
async def list_export_formats() -> dict[str, Any]:
    """Enumerate export formats with honest availability flags.

    Derived from the same registry the desktop Export tab uses
    (``get_available_export_formats``), whose availability flags probe
    importability of the optional dependencies (scipy, h5py, ezc3d).
    """
    return {"formats": exportable_formats()}


@router.post("/recordings", status_code=201)
async def create_recording(
    simulation_service: SimulationService = Depends(get_simulation_service),
    store: RecordingStore = Depends(get_recording_store),
) -> dict[str, Any]:
    """Finalize and persist the active session recorder to disk."""
    session = simulation_service.get_session_recording()
    if session is None:
        raise HTTPException(
            status_code=409,
            detail="No recorded simulation session available to persist. "
            "Run a simulation first.",
        )
    recorder, context = session
    data_dict = recorder.get_data_dict()
    recording_id = await anyio.to_thread.run_sync(store.persist, data_dict, context)
    return store.get_metadata(recording_id)


@router.get("/recordings")
async def list_recordings(
    store: RecordingStore = Depends(get_recording_store),
) -> dict[str, Any]:
    """List persisted recordings (id + metadata), newest first."""
    return {"recordings": store.list_recordings()}


@router.get("/recordings/{recording_id}")
async def get_recording(
    recording_id: str,
    store: RecordingStore = Depends(get_recording_store),
) -> dict[str, Any]:
    """Return metadata for a single recording."""
    try:
        return store.get_metadata(recording_id)
    except InvalidRecordingIdError as exc:
        raise _http_invalid_id(exc) from exc
    except RecordingNotFoundError as exc:
        raise _http_404(recording_id) from exc


@router.delete("/recordings/{recording_id}")
async def delete_recording(
    recording_id: str,
    store: RecordingStore = Depends(get_recording_store),
) -> dict[str, Any]:
    """Delete a persisted recording and its export artifacts."""
    try:
        store.delete(recording_id)
    except InvalidRecordingIdError as exc:
        raise _http_invalid_id(exc) from exc
    except RecordingNotFoundError as exc:
        raise _http_404(recording_id) from exc
    return {"deleted": recording_id}


@router.get("/recordings/{recording_id}/export")
async def export_recording(
    recording_id: str,
    format: str = "json",
    store: RecordingStore = Depends(get_recording_store),
) -> FileResponse:
    """Export a recording in the requested format and stream the file.

    Calls the recorder's own export serializers
    (``src.shared.python.data_io.export``) — no parallel serialization code.
    """
    try:
        path = await anyio.to_thread.run_sync(store.export, recording_id, format)
    except InvalidRecordingIdError as exc:
        raise _http_invalid_id(exc) from exc
    except RecordingNotFoundError as exc:
        raise _http_404(recording_id) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path,
        media_type=_MEDIA_TYPES.get(format, "application/octet-stream"),
        filename=f"{recording_id}{path.suffix}",
    )
