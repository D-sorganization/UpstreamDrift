"""Swing Objective Lab API routes (issue #9128).

Provides REST endpoints for evaluating competing downswing mechanisms under a
shared effort budget and returning versioned comparison matrices (schema 1.0.0).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.middleware.error_handler import handle_api_errors
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/tools/swing-objectives", tags=["swing-objectives"])

COMPARISON_SCHEMA_VERSION = "1.0.0"


def _get_swing_objectives_module() -> Any:
    """Dynamically resolve double_pendulum_golf.swing_objectives."""
    repo_root = Path(__file__).resolve().parents[3]
    tools_paths = [
        (repo_root.parent / "Tools" / "src" / "pendulum_simulator" / "src").resolve(),
        (
            repo_root / "vendor" / "ud-tools" / "src" / "pendulum_simulator" / "src"
        ).resolve(),
    ]
    custom_root = os.environ.get("TOOLS_REPO_ROOT")
    if custom_root:
        tools_paths.insert(
            0, (Path(custom_root) / "src" / "pendulum_simulator" / "src").resolve()
        )

    for path in tools_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        import double_pendulum_golf

        for path in tools_paths:
            pkg_path = path / "double_pendulum_golf"
            if (
                pkg_path.exists()
                and hasattr(double_pendulum_golf, "__path__")
                and str(pkg_path) not in double_pendulum_golf.__path__
            ):
                double_pendulum_golf.__path__.insert(0, str(pkg_path))

        import double_pendulum_golf.swing_objectives as so

        return so
    except Exception as exc:
        logger.error(f"Failed to import double_pendulum_golf.swing_objectives: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"Swing objective optimization engine unavailable: {exc}",
        ) from exc


class GolferPresetInfo(BaseModel):
    """Golfer preset description and initial physical parameters."""

    name: str = Field(..., description="Human-readable preset name")
    arm_mass_kg: float = Field(5.0, gt=0.0, description="Lumped arm mass in kg")
    shaft_mass_kg: float = Field(..., gt=0.0, description="Equivalent shaft mass in kg")
    clubhead_mass_kg: float = Field(
        ..., gt=0.0, description="Equivalent head mass in kg"
    )
    arm_length_m: float = Field(
        0.65, gt=0.0, description="Arm length hub-to-hands in m"
    )
    club_length_m: float = Field(
        1.10, gt=0.0, description="Club length hands-to-head in m"
    )
    top_arm_angle_rad: float = Field(
        2.618, description="Arm angle at top of backswing in rad"
    )
    top_wrist_cock_rad: float = Field(
        1.745, description="Wrist cock at top of backswing in rad"
    )
    duration_s: float = Field(
        0.28, gt=0.0, description="Default downswing duration in s"
    )
    hub_torque_nm: float = Field(
        250.0, gt=0.0, description="Default peak hub torque in N*m"
    )
    wrist_torque_nm: float = Field(
        20.0, gt=0.0, description="Default peak wrist torque in N*m"
    )
    node_count: int = Field(
        21, ge=5, le=101, description="Default collocation node count"
    )


class PresetListResponse(BaseModel):
    """Available golfer presets."""

    presets: list[GolferPresetInfo]


class SwingObjectiveCompareRequest(BaseModel):
    """Request to optimize downswings across competing objectives."""

    preset_name: str | None = Field(None, description="Optional preset key or name")
    arm_mass_kg: float = Field(5.0, gt=0.0, description="Arm mass in kg")
    shaft_mass_kg: float | None = Field(
        None, gt=0.0, description="Equivalent shaft mass in kg"
    )
    clubhead_mass_kg: float | None = Field(
        None, gt=0.0, description="Equivalent head mass in kg"
    )
    arm_length_m: float = Field(0.65, gt=0.0, description="Arm length in m")
    club_length_m: float = Field(1.10, gt=0.0, description="Club length in m")
    top_arm_angle_rad: float = Field(2.618, description="Top arm angle in rad")
    top_wrist_cock_rad: float = Field(1.745, description="Top wrist cock angle in rad")
    duration_s: float = Field(
        0.28, gt=0.05, le=1.5, description="Downswing duration in s"
    )
    hub_torque_nm: float = Field(
        250.0, gt=1.0, le=2000.0, description="Max hub torque in N*m"
    )
    wrist_torque_nm: float = Field(
        20.0, gt=0.1, le=500.0, description="Max wrist torque in N*m"
    )
    node_count: int = Field(21, ge=5, le=101, description="Collocation node count")
    objective_keys: list[str] | None = Field(
        None,
        description="Optional subset of objective keys to compare (minimum 2)",
    )


class SwingComparisonResponse(BaseModel):
    """Wire payload matching comparison schema 1.0.0."""

    schema_version: str = COMPARISON_SCHEMA_VERSION
    objective_keys: list[str]
    units: dict[str, str]
    raw_values: dict[str, dict[str, float]]
    matrix: list[list[float]]
    torque_saturation: dict[str, list[float]]
    swing_distance: list[list[float]]
    is_degenerate: bool
    diagnostics: dict[str, dict[str, Any]]


@router.get("/presets", response_model=PresetListResponse)
@handle_api_errors
async def list_presets() -> PresetListResponse:
    """Return available golfer presets for downswing comparison."""
    so = _get_swing_objectives_module()
    default_preset = getattr(so, "DEFAULT_PRESET", None)
    if default_preset is None:
        raise HTTPException(status_code=500, detail="Default golfer preset not defined")

    presets = [
        GolferPresetInfo(
            name=default_preset.name,
            arm_mass_kg=float(default_preset.arm_mass_kg),
            shaft_mass_kg=float(default_preset.shaft_mass_kg),
            clubhead_mass_kg=float(default_preset.clubhead_mass_kg),
            arm_length_m=float(default_preset.arm_length_m),
            club_length_m=float(default_preset.club_length_m),
            top_arm_angle_rad=float(default_preset.top_arm_angle_rad),
            top_wrist_cock_rad=float(default_preset.top_wrist_cock_rad),
            duration_s=float(default_preset.duration_s),
            hub_torque_nm=float(default_preset.hub_torque_nm),
            wrist_torque_nm=float(default_preset.wrist_torque_nm),
            node_count=int(default_preset.node_count),
        )
    ]
    return PresetListResponse(presets=presets)


@router.post("/compare", response_model=SwingComparisonResponse)
@handle_api_errors
async def compare_swing_objectives(
    request: SwingObjectiveCompareRequest,
) -> SwingComparisonResponse:
    """Run direct collocation downswing comparison across multiple objectives."""
    so = _get_swing_objectives_module()

    default_preset = so.DEFAULT_PRESET
    shaft_m = (
        request.shaft_mass_kg
        if request.shaft_mass_kg is not None
        else float(default_preset.shaft_mass_kg)
    )
    clubhead_m = (
        request.clubhead_mass_kg
        if request.clubhead_mass_kg is not None
        else float(default_preset.clubhead_mass_kg)
    )

    preset = so.GolferPreset(
        name=request.preset_name or "Configured Golfer",
        arm_mass_kg=request.arm_mass_kg,
        shaft_mass_kg=shaft_m,
        clubhead_mass_kg=clubhead_m,
        arm_length_m=request.arm_length_m,
        club_length_m=request.club_length_m,
        top_arm_angle_rad=request.top_arm_angle_rad,
        top_wrist_cock_rad=request.top_wrist_cock_rad,
        duration_s=request.duration_s,
        hub_torque_nm=request.hub_torque_nm,
        wrist_torque_nm=request.wrist_torque_nm,
        node_count=request.node_count,
    )

    budget = so.SwingBudget(
        duration_s=request.duration_s,
        hub_torque_nm=request.hub_torque_nm,
        wrist_torque_nm=request.wrist_torque_nm,
        node_count=request.node_count,
    )

    try:
        config = so.build_config(budget=budget, preset=preset)
        keys = request.objective_keys or None
        comp = so.compare_objectives(config, objective_keys=keys)
        payload = so.comparison_to_payload(comp)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Swing objective comparison failed")
        raise HTTPException(
            status_code=500, detail=f"Comparison optimization failed: {exc}"
        ) from exc

    return SwingComparisonResponse(**payload)
