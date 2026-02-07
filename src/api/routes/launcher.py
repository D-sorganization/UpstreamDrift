"""Launcher manifest API routes.

Serves the shared launcher manifest to the Tauri/React frontend,
enabling both launchers to derive their tile lists from a single source.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.config.launcher_manifest_loader import ASSETS_DIR, LauncherManifest
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/launcher", tags=["launcher"])

# Cache the manifest in memory (loaded once at startup)
_manifest: LauncherManifest | None = None


def _get_manifest() -> LauncherManifest:
    """Get or load the launcher manifest (singleton).

    Returns:
        The loaded LauncherManifest

    Raises:
        HTTPException: If manifest cannot be loaded
    """
    global _manifest
    if _manifest is None:
        try:
            _manifest = LauncherManifest.load()
        except (FileNotFoundError, ValueError) as e:
            logger.error("Failed to load launcher manifest: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Launcher manifest error: {e}",
            ) from e
    return _manifest


@router.get("/manifest")
async def get_manifest() -> dict:  # type: ignore[type-arg]
    """Get the complete launcher manifest.

    Returns:
        Full manifest with all tiles, ordered by display order.
    """
    manifest = _get_manifest()
    return manifest.to_dict()


@router.get("/tiles")
async def get_tiles() -> list[dict]:  # type: ignore[type-arg]
    """Get all launcher tiles in display order.

    Returns:
        List of tile dictionaries.
    """
    manifest = _get_manifest()
    return [t.to_dict() for t in manifest.tiles]


@router.get("/tiles/{tile_id}")
async def get_tile(tile_id: str) -> dict:  # type: ignore[type-arg]
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
async def get_engines() -> list[dict]:  # type: ignore[type-arg]
    """Get only physics engine tiles.

    Returns:
        List of physics engine tile dictionaries.
    """
    manifest = _get_manifest()
    return [t.to_dict() for t in manifest.physics_engines]


@router.get("/tools")
async def get_tools() -> list[dict]:  # type: ignore[type-arg]
    """Get only tool/utility tiles.

    Returns:
        List of tool tile dictionaries.
    """
    manifest = _get_manifest()
    return [t.to_dict() for t in manifest.tools]


@router.get("/logos/validate")
async def validate_logos() -> dict:  # type: ignore[type-arg]
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


@router.get("/logos/{filename}")
async def get_logo(filename: str) -> FileResponse:
    """Serve a tile logo file.

    Args:
        filename: Logo filename (e.g., 'mujoco_humanoid.svg').

    Returns:
        The logo file as an image response.

    Raises:
        HTTPException: If logo not found or invalid filename.
    """
    # DBC Precondition: prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    logo_path = ASSETS_DIR / filename
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
        filename=filename,
    )
