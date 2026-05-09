"""Launcher Manifest Loader — Single source of truth for launcher tiles.

This module loads the shared launcher manifest (launcher_manifest.json) and
provides typed access for both PyQt and API consumers. The Tauri/React
frontend can also read this manifest via the API endpoint.

Design by Contract:
    Preconditions:
        - Manifest file must exist at the expected path
        - Manifest must be valid JSON conforming to the schema
    Postconditions:
        - All returned tiles have valid, non-empty id, name, and category
        - Tile order is deterministic (sorted by 'order' field)
    Invariants:
        - Manifest is immutable after loading (frozen dataclass)
        - Logo file references are relative to ASSETS_DIR
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.launchers.launcher_provider_compatibility import is_engine_runtime_available
from src.shared.python.config.model_pack_manifest import LauncherPresentationMetadata
from src.shared.python.config.model_registry import ModelConfig, ModelRegistry
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Paths
CONFIG_DIR = Path(__file__).parent
MANIFEST_PATH = CONFIG_DIR / "launcher_manifest.json"
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets" / "logos"
REGISTRY_PATH = CONFIG_DIR / "models.yaml"
_DEFAULT_PROVIDER_LOGO = "golf_logo.svg"
_ENGINE_LOGOS = {
    "drake": "drake.svg",
    "mujoco": "mujoco_humanoid.svg",
    "myosuite": "myosim.svg",
    "opensim": "opensim.svg",
    "pinocchio": "pinocchio.svg",
    "putting_green": "putting_green.svg",
}


def _has_provider_metadata(model: ModelConfig) -> bool:
    """Return True when a registry entry comes from provider-aware metadata."""
    if model.provider not in (None, "", "local"):
        return True
    return bool(model.source_root)


def _legacy_launcher_metadata(model: ModelConfig) -> LauncherPresentationMetadata:
    """Provide a migration bridge for models without explicit launcher metadata."""
    if model.engine_type:
        category = "physics_engine"
        logo = _ENGINE_LOGOS.get(model.engine_type, _DEFAULT_PROVIDER_LOGO)
        status = "provider_ready"
    else:
        category = "external"
        logo = _DEFAULT_PROVIDER_LOGO
        status = "external"
    return LauncherPresentationMetadata(
        category=category,
        logo=logo,
        status=status,
    )


def _build_provider_tile(model: ModelConfig) -> LauncherTile:
    """Adapt a provider-backed model registry entry into a launcher tile."""
    metadata = model.launcher or _legacy_launcher_metadata(model)
    status = metadata.status
    source_root = (
        Path(model.source_root) if isinstance(model.source_root, str) else None
    )
    if source_root is not None and not source_root.exists():
        status = "provider_unavailable"
    elif not is_engine_runtime_available(model.engine_type):
        status = "runtime_unavailable"

    return LauncherTile(
        id=model.id,
        name=model.name,
        description=model.description,
        category=metadata.category,
        type=model.type,
        path=model.path,
        logo=metadata.logo,
        status=status,
        capabilities=model.capabilities,
        order=model.order,
        engine_type=model.engine_type,
        provider=model.provider,
        source_root=model.source_root,
        web_route=metadata.web_route,
        hidden=model.hidden,
    )


@dataclass(frozen=True)
class LauncherTile:
    """A single launcher tile definition.

    Attributes:
        id: Unique identifier for the tile
        name: Display name shown in both launchers
        description: Brief description shown under the tile
        category: One of: physics_engine, tool, external
        type: Engine/handler type for launch dispatch
        path: Relative path to the script/entry point
        logo: Logo filename (relative to assets dir)
        status: Status chip text (gui_ready, engine_ready, utility, etc.)
        capabilities: List of capability tags for filtering/display
        order: Display order (1 = first)
        engine_type: Optional engine type identifier for physics engines
        web_route: Optional URL path for tiles that open web tools
    """

    id: str
    name: str
    description: str
    category: str
    type: str
    path: str
    logo: str
    status: str
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    order: int = 99
    engine_type: str | None = None
    provider: str | None = None
    source_root: str | None = None
    web_route: str | None = None
    hidden: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LauncherTile:
        """Create a LauncherTile from a manifest dict entry.

        Args:
            data: Dictionary with tile properties from the manifest

        Returns:
            LauncherTile instance

        Raises:
            ValueError: If required fields are missing
        """
        required = {"id", "name", "description", "category", "type", "path", "logo"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Manifest entry missing required fields: {missing}")

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=data["category"],
            type=data["type"],
            path=data["path"],
            logo=data["logo"],
            status=data.get("status", "unknown"),
            capabilities=tuple(data.get("capabilities", [])),
            tags=tuple(data.get("tags", [])),
            order=data.get("order", 99),
            engine_type=data.get("engine_type"),
            web_route=data.get("web_route"),
            hidden=bool(data.get("hidden", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "type": self.type,
            "path": self.path,
            "logo": self.logo,
            "status": self.status,
            "capabilities": list(self.capabilities),
            "order": self.order,
        }
        if self.engine_type:
            result["engine_type"] = self.engine_type
        if self.web_route:
            result["web_route"] = self.web_route
        if self.tags:
            result["tags"] = list(self.tags)
        if self.hidden:
            result["hidden"] = True
        return result

    @property
    def logo_path(self) -> Path:
        """Absolute path to the logo file."""
        return ASSETS_DIR / self.logo

    @property
    def logo_exists(self) -> bool:
        """Check if the logo file exists on disk."""
        return self.logo_path.exists()

    @property
    def is_physics_engine(self) -> bool:
        """Check if this tile represents a physics engine."""
        return self.category == "physics_engine"

    @property
    def is_tool(self) -> bool:
        """Check if this tile represents a tool/utility."""
        return self.category == "tool"


@dataclass
class LauncherManifest:
    """The complete launcher manifest.

    Invariant: tiles are always sorted by order.
    """

    version: str
    tiles: tuple[LauncherTile, ...]
    description: str = ""

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        include_provider_tiles: bool = True,
        registry_path: Path | None = None,
    ) -> LauncherManifest:
        """Load the launcher manifest from disk.

        Args:
            path: Optional override path. Defaults to MANIFEST_PATH.
            include_provider_tiles: Whether to augment the base manifest with
                provider-backed tiles from the shared model registry.
            registry_path: Optional override for the shared model registry path.

        Returns:
            Loaded LauncherManifest

        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ValueError: If manifest format is invalid
        """
        manifest_path = path or MANIFEST_PATH

        # DBC Precondition
        if not manifest_path.exists():
            raise FileNotFoundError(f"Launcher manifest not found: {manifest_path}")

        logger.info("Loading launcher manifest from %s", manifest_path)

        with open(manifest_path, encoding="utf-8") as f:
            raw = json.load(f)

        if "tiles" not in raw:
            raise ValueError("Manifest missing 'tiles' array")

        tiles_raw = raw["tiles"]
        if not isinstance(tiles_raw, list):
            raise ValueError("Manifest 'tiles' must be a list")

        tiles = [LauncherTile.from_dict(t) for t in tiles_raw]
        if include_provider_tiles:
            tiles.extend(
                cls._load_provider_tiles(
                    registry_path=registry_path or REGISTRY_PATH,
                    existing_ids={tile.id for tile in tiles},
                )
            )

        sorted_tiles: tuple[LauncherTile, ...] = tuple(
            sorted(tiles, key=lambda t: (t.order, t.id))
        )

        manifest = cls(
            version=raw.get("version", "0.0.0"),
            tiles=sorted_tiles,
            description=raw.get("description", ""),
        )

        # DBC Postcondition: verify all tiles have unique IDs
        ids = [t.id for t in sorted_tiles]
        duplicates = [tid for tid in ids if ids.count(tid) > 1]
        if duplicates:
            raise ValueError(f"Duplicate tile IDs in manifest: {set(duplicates)}")

        logger.info(
            "Loaded %d tiles (v%s): %s",
            len(sorted_tiles),
            manifest.version,
            ", ".join(t.id for t in tiles),
        )

        return manifest

    @staticmethod
    def _load_provider_tiles(
        *,
        registry_path: Path,
        existing_ids: set[str],
    ) -> list[LauncherTile]:
        """Load dynamic provider-backed tiles from the shared model registry."""
        if not registry_path.exists():
            return []

        registry = ModelRegistry(config_path=registry_path)
        provider_tiles: list[LauncherTile] = []

        for model in registry.get_all_models():
            if model.id in existing_ids or not _has_provider_metadata(model):
                continue
            provider_tiles.append(_build_provider_tile(model))

        if provider_tiles:
            logger.info(
                "Augmented launcher manifest with %d provider-backed tiles",
                len(provider_tiles),
            )

        return provider_tiles

    def get_tile(self, tile_id: str) -> LauncherTile | None:
        """Get a tile by its ID.

        Args:
            tile_id: The tile identifier

        Returns:
            LauncherTile if found, None otherwise
        """
        if not (tile_id is not None):
            raise ValueError("tile_id must be provided")
        for tile in self.tiles:
            if tile.id == tile_id:
                return tile
        return None

    def get_tiles_by_category(
        self, category: str, *, include_hidden: bool = False
    ) -> list[LauncherTile]:
        """Get all tiles in a category.

        Args:
            category: Category to filter by (physics_engine, tool, external)
            include_hidden: When False (default), tiles flagged ``hidden`` are
                excluded so legacy aliases do not appear as duplicate launcher
                cards.

        Returns:
            List of matching tiles, ordered by their order field
        """
        return [
            t
            for t in self.tiles
            if t.category == category and (include_hidden or not t.hidden)
        ]

    @property
    def visible_tiles(self) -> list[LauncherTile]:
        """Tiles excluding entries flagged ``hidden`` (legacy aliases)."""
        return [t for t in self.tiles if not t.hidden]

    @property
    def physics_engines(self) -> list[LauncherTile]:
        """Get all physics engine tiles (excluding hidden aliases)."""
        return self.get_tiles_by_category("physics_engine")

    @property
    def tools(self) -> list[LauncherTile]:
        """Get all tool tiles (excluding hidden aliases)."""
        return self.get_tiles_by_category("tool")

    @property
    def tile_ids(self) -> list[str]:
        """Get ordered list of all tile IDs."""
        return [t.id for t in self.tiles]

    @property
    def ordered_ids(self) -> list[str]:
        """Get tile IDs in display order (alias for tile_ids)."""
        return self.tile_ids

    def to_dict(self, *, include_hidden: bool = False) -> dict[str, Any]:
        """Serialize manifest for API responses.

        Args:
            include_hidden: When False (default), tiles flagged ``hidden`` are
                excluded so legacy aliases do not appear as duplicate launcher
                cards. The web dashboard renders tiles by category without
                filtering ``hidden``, so the API must exclude them.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        tiles = self.tiles if include_hidden else self.visible_tiles
        return {
            "version": self.version,
            "description": self.description,
            "tiles": [t.to_dict() for t in tiles],
        }

    def validate_logos(self) -> list[str]:
        """Check which tiles have missing logo files.

        Returns:
            List of tile IDs with missing logos
        """
        missing: list[str] = []
        for tile in self.tiles:
            if not tile.logo_exists:
                logger.warning("Missing logo for tile '%s': %s", tile.id, tile.logo)
                missing.append(tile.id)
        return missing
