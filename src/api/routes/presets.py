"""Parameter preset routes.

Provides endpoints for saving and loading simulation parameter presets.
Presets are stored as JSON files in ~/.upstream_modeling_suite/presets/.

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..dependencies import get_logger

router = APIRouter()

# Parameter bounds from physics_parameters.py / ParameterPanel
PARAM_BOUNDS: dict[str, dict[str, float]] = {
    "duration": {"min": 0.1, "max": 60.0},
    "timestep": {"min": 0.001, "max": 0.01},
}

_SAFE_NAME_RE = re.compile(r"^[\w\- ]{1,64}$")


def _presets_dir() -> Path:
    """Return the presets storage directory, creating it if needed.

    Returns:
        Path to the presets directory.
    """
    directory = Path.home() / ".upstream_modeling_suite" / "presets"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_preset_path(name: str) -> Path | None:
    """Resolve a preset file path, rejecting unsafe names.

    Args:
        name: Preset name to resolve.

    Returns:
        Safe path, or None if the name is unsafe.
    """
    if not name or not _SAFE_NAME_RE.match(name):
        return None
    slug = name.strip().replace(" ", "_")
    directory = _presets_dir()
    candidate = (directory / slug).with_suffix(".json")
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate


def _validate_param_bounds(params: dict[str, Any]) -> list[str]:
    """Check parameter values against known min/max bounds.

    Args:
        params: Dictionary of parameter values to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []
    for key, bounds in PARAM_BOUNDS.items():
        if key not in params:
            continue
        value = params[key]
        if not isinstance(value, int | float):
            errors.append(f"{key} must be numeric")
            continue
        if value < bounds["min"]:
            errors.append(f"{key} must be >= {bounds['min']} (got {value})")
        if value > bounds["max"]:
            errors.append(f"{key} must be <= {bounds['max']} (got {value})")
    return errors


class PresetSaveRequest(BaseModel):
    """Request body for saving a parameter preset.

    Preconditions:
        - name must be 1-64 characters, alphanumeric/hyphen/space/underscore
        - params must not be empty
    """

    name: str = Field(..., description="Preset name", min_length=1, max_length=64)
    params: dict[str, Any] = Field(..., description="Simulation parameters to save")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Precondition: name must use safe characters."""
        stripped = v.strip()
        if not _SAFE_NAME_RE.match(stripped):
            raise ValueError(
                "Preset name must be 1-64 characters "
                "(alphanumeric, hyphens, underscores, spaces)"
            )
        return stripped

    @field_validator("params")
    @classmethod
    def validate_params_not_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Precondition: params dict must not be empty."""
        if not v:
            raise ValueError("params must not be empty")
        return v


class PresetEntry(BaseModel):
    """A single preset entry returned in list responses."""

    name: str = Field(..., description="Preset name")
    params: dict[str, Any] = Field(..., description="Saved parameters")


class PresetsListResponse(BaseModel):
    """Response model for listing all presets."""

    presets: list[PresetEntry] = Field(
        default_factory=list, description="All saved presets"
    )


@router.get("/presets", response_model=PresetsListResponse)
async def list_presets(
    logger: Any = Depends(get_logger),
) -> PresetsListResponse:
    """List all saved parameter presets.

    Returns:
        List of preset entries with name and parameters.
    """
    directory = _presets_dir()
    presets: list[PresetEntry] = []

    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data.get("name", path.stem.replace("_", " "))
            params = data.get("params", {})
            presets.append(PresetEntry(name=name, params=params))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            if logger:
                logger.warning("Skipping malformed preset %s: %s", path.name, exc)

    return PresetsListResponse(presets=presets)


@router.post("/presets", status_code=201)
async def save_preset(
    request: PresetSaveRequest,
    logger: Any = Depends(get_logger),
) -> dict[str, str]:
    """Save a parameter preset.

    Args:
        request: Preset name and parameter values.
        logger: Injected logger.

    Returns:
        Confirmation with preset name.

    Raises:
        HTTPException 400: If parameters fail bounds validation.
        HTTPException 422: If the name contains unsafe characters.
    """
    validation_errors = _validate_param_bounds(request.params)
    if validation_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Parameter validation failed",
                "errors": validation_errors,
            },
        )

    preset_path = _safe_preset_path(request.name)
    if preset_path is None:
        raise HTTPException(
            status_code=422,
            detail="Preset name contains unsafe characters",
        )

    payload = {"name": request.name, "params": request.params}
    try:
        preset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        if logger:
            logger.error("Failed to write preset %s: %s", request.name, exc)
        raise HTTPException(status_code=500, detail="Failed to save preset") from exc

    if logger:
        logger.info("Saved preset: %s", request.name)
    return {"status": "saved", "name": request.name}
