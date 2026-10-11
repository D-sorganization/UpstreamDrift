"""OpenCap session inspection and import routes (#11409).

Exposes REST endpoints to inspect an OpenCap session directory, list trials,
and load a trial for downstream OpenSim musculoskeletal analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.shared.python.motion_pipeline.sources.opencap_session import (
    OpenCapSessionMetadata,
    inspect_opencap_session,
    load_opencap_session,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/opencap", tags=["opencap"])


class InspectSessionRequest(BaseModel):
    """Request model for inspecting an OpenCap session."""

    session_dir: str = Field(..., description="Path to the OpenCap session directory")


class SubjectResponse(BaseModel):
    """Subject anthropometry and model choice."""

    mass_kg: float | None = None
    height_m: float | None = None
    sex: str | None = None
    opensim_model: str | None = None
    subject_id: str | None = None


class InspectSessionResponse(BaseModel):
    """Response model for session inspection."""

    session_dir: str
    trials: list[str]
    subject: SubjectResponse
    model_file: str | None = None
    kinematics_trials: list[str]
    notes: list[str]


class ImportTrialRequest(BaseModel):
    """Request model for importing a single trial from an OpenCap session."""

    session_dir: str = Field(..., description="Path to the OpenCap session directory")
    trial: str | None = Field(
        None, description="Trial name to load (defaults to first motion trial)"
    )


class ImportTrialResponse(BaseModel):
    """Response model for trial import."""

    session_dir: str
    trial: str
    trials: list[str]
    subject: SubjectResponse
    model_file: str | None = None
    has_kinematics: bool
    kinematics_columns: list[str]
    notes: list[str]


def _to_subject_response(subject: Any) -> SubjectResponse:
    return SubjectResponse(
        mass_kg=subject.mass_kg,
        height_m=subject.height_m,
        sex=subject.sex,
        opensim_model=subject.opensim_model,
        subject_id=subject.subject_id,
    )


def _validate_session_dir(session_dir: str) -> Path:
    path = Path(session_dir)
    if not path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"OpenCap session directory not found: {session_dir}",
        )
    return path


@router.post("/inspect", response_model=InspectSessionResponse)
async def inspect_session(request: InspectSessionRequest) -> InspectSessionResponse:
    """Inspect an OpenCap session directory without loading large observation arrays."""
    path = _validate_session_dir(request.session_dir)

    try:
        meta: OpenCapSessionMetadata = inspect_opencap_session(path)
        return InspectSessionResponse(
            session_dir=str(meta.session_dir),
            trials=meta.trials,
            subject=_to_subject_response(meta.subject),
            model_file=str(meta.model_file) if meta.model_file else None,
            kinematics_trials=list(meta.kinematics_trials),
            notes=list(meta.notes),
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "Failed to inspect OpenCap session %s: %s", request.session_dir, exc
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.post("/import", response_model=ImportTrialResponse)
async def import_trial(request: ImportTrialRequest) -> ImportTrialResponse:
    """Load a specific trial from an OpenCap session directory."""
    path = _validate_session_dir(request.session_dir)

    try:
        session = load_opencap_session(path, trial=request.trial)
        kinematics_columns: list[str] = []
        if session.kinematics is not None:
            skeleton = session.kinematics.skeleton
            kinematics_columns = list(skeleton.joints)

        return ImportTrialResponse(
            session_dir=str(path),
            trial=session.trial,
            trials=session.trials,
            subject=_to_subject_response(session.subject),
            model_file=str(session.model_file) if session.model_file else None,
            has_kinematics=session.kinematics is not None,
            kinematics_columns=kinematics_columns,
            notes=list(session.notes),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.warning(
            "Failed to import OpenCap trial from %s: %s", request.session_dir, exc
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
