"""Cross-engine robustness comparison API routes (issue #7455).

Provides ``POST /analysis/cross-engine`` as an async task endpoint — the
perturbation study can take several seconds per engine, so results are
delivered via the standard task-manager pattern (start → poll status → get).

Design by Contract
------------------
- ``POST /analysis/cross-engine`` requires at least one valid engine name.
- ``GET  /analysis/cross-engine/status/{task_id}`` requires a non-empty task ID.
- Results are JSON-serialisable; no numpy arrays survive the boundary.

DRY
---
All computation delegates to
``src.shared.python.analysis.cross_engine.run_cross_engine_study``.  No
physics logic lives here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from src.shared.python.analysis.cross_engine import ENGINE_NAMES
from src.shared.python.logging_pkg.logging_config import get_logger

from ..utils.datetime_compat import UTC
from ..dependencies import get_task_manager

logger = get_logger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CrossEnginePerturbationConfig(BaseModel):
    """Perturbation study configuration.

    All fields match ``CrossEngineSimConfig`` from the service layer.
    """

    t_end: float = Field(
        default=1.0, gt=0.0, description="Simulation horizon (seconds)"
    )
    dt: float = Field(
        default=0.01, gt=0.0, description="Integration timestep (seconds)"
    )
    noise_amplitude: float = Field(
        default=0.05, ge=0.0, description="Perturbation amplitude"
    )
    n_trials: int = Field(
        default=10, ge=1, le=200, description="Number of perturbation trials"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")

    @model_validator(mode="after")
    def validate_dt_lt_t_end(self) -> CrossEnginePerturbationConfig:
        if self.dt >= self.t_end:
            raise ValueError(
                f"dt ({self.dt}) must be strictly less than t_end ({self.t_end})"
            )
        return self


class CrossEngineStudyRequest(BaseModel):
    """Request body for POST /analysis/cross-engine."""

    engines: list[str] = Field(
        default_factory=lambda: ["pendulum_stub"],
        min_length=1,
        description="Engine names to compare; each must be a recognised engine.",
    )
    config: CrossEnginePerturbationConfig = CrossEnginePerturbationConfig()
    allow_stub_substitution: bool = Field(
        default=True,
        description=(
            "When false, the study fails instead of silently running a "
            "2-DOF stub for a requested engine whose real backend is "
            "unavailable (#8817). Either way the result declares each "
            "engine's backend ('real' or 'stub_2dof') and lists "
            "'stubbed_engines'."
        ),
    )

    @field_validator("engines")
    @classmethod
    def engines_must_be_known(cls, names: list[str]) -> list[str]:
        """Pre: every engine name must be in ENGINE_NAMES."""
        unknown = [n for n in names if n not in ENGINE_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown engine name(s) {unknown}; supported: {sorted(ENGINE_NAMES)}"
            )
        return names


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_study_background(
    task_id: str,
    request: CrossEngineStudyRequest,
    task_manager: Any,
) -> None:
    """Execute the cross-engine study and persist results in the task manager."""
    task_manager.update_progress(task_id, 0)
    try:
        from src.shared.python.analysis.cross_engine import run_cross_engine_study
        from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
            CrossEngineSimConfig,
        )

        cfg = request.config
        sim_config = CrossEngineSimConfig(
            t_end=cfg.t_end,
            dt=cfg.dt,
            noise_amplitude=cfg.noise_amplitude,
            n_trials=cfg.n_trials,
            seed=cfg.seed,
        )
        result = run_cross_engine_study(
            request.engines,
            sim_config,
            allow_stub_substitution=request.allow_stub_substitution,
        )
        task_manager.mark_completed(task_id, result)
        logger.info("Cross-engine study %s completed", task_id)
    except (ValueError, RuntimeError, ImportError) as exc:
        logger.exception("Cross-engine study %s failed", task_id)
        task_manager.mark_failed(task_id, str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/analysis/cross-engine")
async def start_cross_engine_study(
    payload: CrossEngineStudyRequest,
    background_tasks: BackgroundTasks,
    task_manager: Any = Depends(get_task_manager),
) -> dict[str, str]:
    """Start an async cross-engine perturbation study.

    Launches the compute in a background task and returns a task ID
    immediately.  Poll ``GET /analysis/cross-engine/status/{task_id}``
    for results.

    Args:
        payload: Engines to compare and perturbation config.
        background_tasks: FastAPI background task runner.
        task_manager: Injected task manager for lifecycle tracking.

    Returns:
        ``{"task_id": str, "status": "started"}``

    Raises:
        HTTPException 400: invalid engine names (caught by Pydantic validator).
    """
    task_id = str(uuid.uuid4())
    task_manager.set(
        task_id,
        {
            "status": "started",
            "created_at": datetime.now(UTC).isoformat(),
            "engines": payload.engines,
        },
    )
    background_tasks.add_task(
        _run_study_background,
        task_id,
        payload,
        task_manager,
    )
    logger.info(
        "Cross-engine study started task=%s engines=%s", task_id, payload.engines
    )
    return {"task_id": task_id, "status": "started"}


@router.get("/analysis/cross-engine/status/{task_id}")
async def get_cross_engine_status(
    task_id: str,
    task_manager: Any = Depends(get_task_manager),
) -> dict[str, Any]:
    """Poll the status of a cross-engine study.

    Args:
        task_id: The task identifier returned by POST /analysis/cross-engine.
        task_manager: Injected task manager.

    Returns:
        Current task status dict (may include ``result`` when completed).

    Raises:
        HTTPException 400: task_id is blank.
        HTTPException 404: task not found.
    """
    if not task_id or not task_id.strip():
        raise HTTPException(status_code=400, detail="task_id must be non-empty")
    if not task_manager.exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    task_data = task_manager.get(task_id)
    return dict(task_data) if task_data else {}
