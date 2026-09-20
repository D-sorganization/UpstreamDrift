"""Launcher manifest API routes.

Serves the shared launcher manifest to the Tauri/React frontend,
enabling both launchers to derive their tile lists from a single source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, cast

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.api.launcher_manifest_cache import (
    get_cached_manifest,
    get_cached_manifest_async,
)
from src.config.capability_state import resolve_canonical_display_name
from src.config.launcher_manifest_loader import ASSETS_DIR, LauncherManifest
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

from ..models.responses import LauncherManifestResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/launcher", tags=["launcher"])


def _get_manifest() -> LauncherManifest:
    """Get the launcher manifest via the shared process-level cache.

    The cache (``src.api.launcher_manifest_cache``) is invalidated when the
    manifest or model registry files change on disk (issue #8937).

    Returns:
        The loaded LauncherManifest

    Raises:
        HTTPException: If manifest cannot be loaded
    """
    try:
        return get_cached_manifest()
    except (FileNotFoundError, ValueError) as e:
        logger.exception("Failed to load launcher manifest")
        raise HTTPException(
            status_code=500,
            detail=f"Launcher manifest error: {e}",
        ) from e


async def _get_manifest_async() -> LauncherManifest:
    """Async variant of :func:`_get_manifest` that never blocks the loop."""
    try:
        return await get_cached_manifest_async()
    except (FileNotFoundError, ValueError) as e:
        logger.exception("Failed to load launcher manifest")
        raise HTTPException(
            status_code=500,
            detail=f"Launcher manifest error: {e}",
        ) from e


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
    manifest = await _get_manifest_async()
    return manifest.to_dict()


@router.get("/tiles")
async def get_tiles() -> list[dict[str, Any]]:
    """Get all visible launcher tiles in display order.

    Hidden tiles (legacy aliases) are excluded, matching the manifest
    invariant documented on ``LauncherManifest.to_dict`` (issue #8863).

    Returns:
        List of tile dictionaries.
    """
    manifest = await _get_manifest_async()
    return [t.to_dict() for t in manifest.visible_tiles]


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
    manifest = await _get_manifest_async()
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
    manifest = await _get_manifest_async()
    return [t.to_dict() for t in manifest.physics_engines]


@router.get("/tools")
async def get_tools() -> list[dict[str, Any]]:
    """Get only tool/utility tiles.

    Returns:
        List of tool tile dictionaries.
    """
    manifest = await _get_manifest_async()
    return [t.to_dict() for t in manifest.tools]


@router.get("/logos/validate")
async def validate_logos() -> dict[str, Any]:
    """Validate that all tile logos exist on disk.

    Returns:
        Validation report with missing and present logo lists.
    """
    manifest = await _get_manifest_async()
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
        manifest = await _get_manifest_async()
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

# Authoritative canonical engine display names
_CANONICAL_ENGINE_NAMES: Final[dict[str, str]] = {
    "mujoco": "MuJoCo",
    "drake": "Drake",
    "pinocchio": "Pinocchio",
    "opensim": "OpenSim",
    "myosuite": "MyoSuite",
    "myosim": "MyoSuite",
    "simscape": "Simscape",
    "jaxsim": "JaxSim",
    "putting_green": "Putting Green",
    "double_pendulum": "Double Pendulum",
    "pendulum": "Pendulum",
    "matlab": "MATLAB",
}

_capabilities_state: dict[str, dict[str, dict[str, str]] | None] = {"cache": None}


def _load_engine_capability_matrix() -> dict[str, Any]:
    """Load engine capability matrix from disk."""
    from src.config.launcher_manifest_loader import CONFIG_DIR

    matrix_file = CONFIG_DIR / "engine_capability_matrix.json"
    if matrix_file.is_file():
        try:
            return cast(
                dict[str, Any], json.loads(matrix_file.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load engine capability matrix: %s", exc)
    return {}


def _get_engine_capabilities() -> dict[str, dict[str, str]]:
    """Get capability profiles for all known engines dynamically from matrix.

    Returns:
        Dictionary mapping engine_id to capability dict.
    """
    if _capabilities_state["cache"] is not None:
        return _capabilities_state["cache"]

    matrix = _load_engine_capability_matrix()
    profiles = matrix.get("profiles", {})
    result: dict[str, dict[str, str]] = {}

    for engine_id, prof in profiles.items():
        caps = dict(prof.get("capabilities", {}))
        display_name = _CANONICAL_ENGINE_NAMES.get(
            engine_id, resolve_canonical_display_name(engine_id)
        )
        caps["engine_name"] = display_name
        caps.setdefault("spatial_jacobian_order", "translation_rotation")
        result[engine_id] = caps

    # Support legacy alias 'pendulum' -> 'double_pendulum' if present
    if "double_pendulum" in result and "pendulum" not in result:
        pendulum_caps = dict(result["double_pendulum"])
        pendulum_caps["engine_name"] = "Pendulum"
        result["pendulum"] = pendulum_caps

    _capabilities_state["cache"] = result
    return result


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
