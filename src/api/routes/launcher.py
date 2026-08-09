"""Launcher manifest API routes.

Serves the shared launcher manifest to the Tauri/React frontend,
enabling both launchers to derive their tile lists from a single source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.config.launcher_manifest_loader import ASSETS_DIR, LauncherManifest
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

from ..models.responses import LauncherManifestResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/launcher", tags=["launcher"])


# Cache the manifest in memory (singleton holder avoids 'global')
_launcher_state: dict[str, LauncherManifest | None] = {"manifest": None}


def _get_manifest() -> LauncherManifest:
    """Get or load the launcher manifest (singleton).

    Returns:
        The loaded LauncherManifest

    Raises:
        HTTPException: If manifest cannot be loaded
    """
    if _launcher_state["manifest"] is None:
        try:
            _launcher_state["manifest"] = LauncherManifest.load()
        except (FileNotFoundError, ValueError) as e:
            logger.exception("Failed to load launcher manifest")
            raise HTTPException(
                status_code=500,
                detail=f"Launcher manifest error: {e}",
            ) from e
    # Guaranteed non-None after the check above
    manifest = _launcher_state["manifest"]
    assert manifest is not None  # for mypy
    return manifest


@router.get(
    "/manifest",
    response_model=LauncherManifestResponse,
    response_model_exclude_none=True,
)
async def get_manifest() -> dict[str, Any]:
    """Get the complete launcher manifest.

    The response is validated against ``LauncherManifestResponse`` so the
    TypeScript contract generated from the OpenAPI schema covers this
    payload (issue #7447). ``response_model_exclude_none`` preserves the
    historical ``to_dict`` behavior of omitting unset optional keys.

    Returns:
        Full manifest with all tiles, ordered by display order.
    """
    manifest = _get_manifest()
    return manifest.to_dict()


@router.get("/tiles")
async def get_tiles() -> list[dict[str, Any]]:
    """Get all launcher tiles in display order.

    Returns:
        List of tile dictionaries.
    """
    manifest = _get_manifest()
    return [t.to_dict() for t in manifest.tiles]


@router.get("/tiles/{tile_id}")
@precondition(
    lambda tile_id: tile_id is not None and len(tile_id.strip()) > 0,
    "Tile ID must be a non-empty string",
)
async def get_tile(tile_id: str) -> dict[str, Any]:
    """Get a specific tile by ID.

    Args:
        tile_id: The tile identifier.

    Returns:
        Tile dictionary.

    Raises:
        HTTPException: If tile not found.
    """
    manifest = _get_manifest()
    tile = manifest.get_tile(tile_id)
    if tile is None:
        raise HTTPException(status_code=404, detail=f"Tile not found: {tile_id}")
    return tile.to_dict()


@router.get("/engines")
async def get_engines() -> list[dict[str, Any]]:
    """Get only physics engine tiles.

    Returns:
        List of physics engine tile dictionaries.
    """
    manifest = _get_manifest()
    return [t.to_dict() for t in manifest.physics_engines]


@router.get("/tools")
async def get_tools() -> list[dict[str, Any]]:
    """Get only tool/utility tiles.

    Returns:
        List of tool tile dictionaries.
    """
    manifest = _get_manifest()
    return [t.to_dict() for t in manifest.tools]


@router.get("/logos/validate")
async def validate_logos() -> dict[str, Any]:
    """Validate that all tile logos exist on disk.

    Returns:
        Validation report with missing and present logo lists.
    """
    manifest = _get_manifest()
    missing = manifest.validate_logos()
    total = len(manifest.tiles)
    present = total - len(missing)

    return {
        "total": total,
        "present": present,
        "missing_count": len(missing),
        "missing_tiles": missing,
        "all_valid": len(missing) == 0,
    }


@router.get("/logos/{filename:path}")
@precondition(
    lambda filename: filename is not None and len(filename.strip()) > 0,
    "Logo filename must be a non-empty string",
)
async def get_logo(filename: str) -> FileResponse:
    """Serve a tile logo file.

    Args:
        filename: Logo filename (e.g., 'mujoco_humanoid.svg' or relative logo path).

    Returns:
        The logo file as an image response.

    Raises:
        HTTPException: If logo not found or invalid filename.
    """
    # DBC Precondition: prevent path traversal
    if ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    logo_path = ASSETS_DIR / filename
    if not logo_path.exists():
        manifest = _get_manifest()
        for tile in manifest.tiles:
            if (
                tile.logo == filename
                or Path(tile.logo).name == filename
                or tile.logo_path.name == filename
            ) and tile.logo_exists:
                logo_path = tile.logo_path
                break

    if not logo_path.exists():
        raise HTTPException(status_code=404, detail=f"Logo not found: {filename}")

    # Determine media type
    suffix = logo_path.suffix.lower()
    media_types = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(logo_path),
        media_type=media_type,
        filename=logo_path.name,
    )


# --- Engine Capabilities ---

# Registry of known engine capability profiles.
# Engines register their capabilities here so the API can serve them.
_capabilities_state: dict[str, dict[str, dict[str, str]] | None] = {"cache": None}


def _capability_profile(engine_name: str, **levels: Any) -> Any:
    from src.engines.common.capabilities import CapabilityLevel, EngineCapabilities

    defaults: dict[str, Any] = {
        "mass_matrix": CapabilityLevel.FULL,
        "jacobian": CapabilityLevel.FULL,
        "contact_forces": CapabilityLevel.FULL,
        "inverse_dynamics": CapabilityLevel.FULL,
        "drift_acceleration": CapabilityLevel.FULL,
        "parameter_gradients": CapabilityLevel.PARTIAL,
        "state_control_gradients": CapabilityLevel.PARTIAL,
        "forward_sim": CapabilityLevel.FULL,
        "contact_step": CapabilityLevel.PARTIAL,
        "trajectory_opt": CapabilityLevel.PARTIAL,
        "video_export": CapabilityLevel.PARTIAL,
        "dataset_export": CapabilityLevel.FULL,
        "force_visualization": CapabilityLevel.PARTIAL,
        "model_positioning": CapabilityLevel.FULL,
        "measurements": CapabilityLevel.FULL,
    }
    defaults.update(levels)
    return EngineCapabilities(engine_name=engine_name, **defaults)


def _primary_engine_profiles() -> dict[str, Any]:
    from src.engines.common.capabilities import CapabilityLevel

    F = CapabilityLevel.FULL
    P = CapabilityLevel.PARTIAL

    return {
        "mujoco": _capability_profile(
            "MuJoCo",
            contact_step=F,
            video_export=F,
            force_visualization=F,
        ),
        "drake": _capability_profile(
            "Drake",
            contact_forces=P,
            state_control_gradients=F,
            contact_step=F,
            trajectory_opt=F,
        ),
        "pinocchio": _capability_profile("Pinocchio", contact_step=P),
        "opensim": _capability_profile(
            "OpenSim",
            contact_forces=P,
            model_positioning=P,
        ),
    }


def _specialized_engine_profiles() -> dict[str, Any]:
    from src.engines.common.capabilities import CapabilityLevel

    F = CapabilityLevel.FULL
    P = CapabilityLevel.PARTIAL
    N = CapabilityLevel.NONE

    return {
        "myosuite": _capability_profile(
            "MyoSuite",
            contact_forces=P,
            parameter_gradients=N,
            state_control_gradients=N,
            contact_step=F,
            trajectory_opt=N,
            model_positioning=P,
        ),
        "jaxsim": _capability_profile(
            "JaxSim",
            contact_forces=P,
            drift_acceleration=P,
            parameter_gradients=F,
            state_control_gradients=F,
            video_export=N,
            dataset_export=P,
            force_visualization=N,
            model_positioning=P,
            measurements=N,
        ),
        "pendulum": _capability_profile(
            "Pendulum",
            contact_forces=N,
            parameter_gradients=F,
            state_control_gradients=F,
            contact_step=N,
            trajectory_opt=F,
            force_visualization=F,
        ),
        "putting_green": _capability_profile(
            "Putting Green",
            contact_forces=P,
            parameter_gradients=N,
            state_control_gradients=N,
            trajectory_opt=N,
        ),
    }


def _build_engine_profiles() -> dict[str, Any]:
    return {
        **_primary_engine_profiles(),
        **_specialized_engine_profiles(),
    }


def _get_engine_capabilities() -> dict[str, dict[str, str]]:
    """Get capability profiles for all known engines.

    Returns:
        Dictionary mapping engine_id to capability dict.
    """
    if _capabilities_state["cache"] is not None:
        return _capabilities_state["cache"]

    profiles = _build_engine_profiles()
    _capabilities_state["cache"] = {k: v.to_dict() for k, v in profiles.items()}

    if not (_capabilities_state["cache"] is not None):  # Ensure not None for mypy
        raise ValueError("DbC Blocked: Precondition failed.")
    return _capabilities_state["cache"]


@router.get("/engines/capabilities")
async def get_all_engine_capabilities() -> dict[str, dict[str, str]]:
    """Get capability profiles for all known engines.

    Returns:
        Dictionary mapping engine_id to capability profile.
    """
    return _get_engine_capabilities()


@router.get("/engines/{engine_id}/capabilities")
@precondition(
    lambda engine_id: engine_id is not None and len(engine_id.strip()) > 0,
    "Engine ID must be a non-empty string",
)
async def get_engine_capabilities(engine_id: str) -> dict[str, str]:
    """Get capability profile for a specific engine.

    Args:
        engine_id: Engine identifier (e.g., 'mujoco', 'drake')

    Returns:
        Capability profile dictionary

    Raises:
        HTTPException: If engine not found
    """
    caps = _get_engine_capabilities()
    if engine_id not in caps:
        raise HTTPException(
            status_code=404,
            detail=f"Engine not found: {engine_id}. Available: {list(caps.keys())}",
        )
    return caps[engine_id]
