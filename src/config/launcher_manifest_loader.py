"""Launcher Manifest Loader — typed access to the single tile registry.

``LauncherManifest.load()`` builds the web/API tile surface directly from
``src/config/models.yaml`` (native ``models`` + ``web_catalog``; see
:mod:`src.config.tile_registry`, issue #9412). ``launcher_manifest.json`` is a
generated projection of that registry kept for external consumers and is not
read at runtime; passing an explicit ``path`` still loads a JSON overlay
document (used by tests and external model packs).

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.launchers.launcher_provider_compatibility import is_engine_runtime_available
from src.shared.python.config.model_pack_manifest import LauncherPresentationMetadata
from src.shared.python.config.model_registry import ModelConfig, ModelRegistry
from src.shared.python.config.tile_target_resolution import module_string_to_relpath
from src.shared.python.config.tools_vendor_authority import (
    inspect_tools_vendor_authority,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Paths
CONFIG_DIR = Path(__file__).parent
MANIFEST_PATH = CONFIG_DIR / "launcher_manifest.json"
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets" / "logos"
# The PyQt desktop launcher keeps its tile artwork under src/launchers/;
# models.yaml logo values like "assets/foo.png" are relative to that dir.
PYQT_ASSETS_ROOT = Path(__file__).parent.parent / "launchers"
REGISTRY_PATH = CONFIG_DIR / "models.yaml"
REPO_ROOT = CONFIG_DIR.parents[1]
_DEFAULT_PROVIDER_LOGO = "golf_logo.svg"
# The desktop launcher keeps legacy PNG artwork under src/launchers/assets;
# the web catalog serves SVG-only logos from assets/logos. Registry-surfaced
# tiles translate their desktop artwork to the SVG equivalent for the web.
_WEB_LOGO_BY_DESKTOP_PNG = {
    "golf_logo.png": "golf_logo.svg",
    "mujoco.png": "mujoco_humanoid.svg",
    "drake.png": "drake.svg",
    "pinocchio.png": "pinocchio.svg",
    "opensim.png": "opensim.svg",
    "myosim.png": "myosim.svg",
    "putting_green_modern.png": "putting_green.svg",
    "c3d_viewer_modern.png": "c3d_icon.svg",
    "data_explorer_modern.png": "data_explorer.svg",
    "video_analyzer_modern.png": "video_analyzer.svg",
    "matlab_logo.png": "matlab_logo.svg",
    "project_map.png": "project_map.svg",
    "urdf_icon.png": "urdf_icon.svg",
    "bunkershot_icon.png": "bunkershot3d.svg",
    "training_controller.png": "project_map.svg",
    "openpose.png": "video_analyzer.svg",
    "mediapipe.png": "video_analyzer.svg",
}


def _web_logo(logo: str) -> str:
    """Return an SVG logo for the web catalog, translating desktop PNGs."""
    if logo.endswith(".svg"):
        return logo
    return _WEB_LOGO_BY_DESKTOP_PNG.get(Path(logo).name, _DEFAULT_PROVIDER_LOGO)


_ENGINE_LOGOS = {
    "drake": "drake.svg",
    "mujoco": "mujoco_humanoid.svg",
    "myosuite": "myosim.svg",
    "opensim": "opensim.svg",
    "pinocchio": "pinocchio.svg",
    "putting_green": "putting_green.svg",
}
LAUNCHER_CATEGORY_LABELS: dict[str, str] = {
    "physics_engine": "Physics Engines",
    "biomechanics": "Biomechanics",
    "simulation": "Simulation",
    "motion_matching": "Motion Matching",
    "motion_capture": "Motion Capture",
    "analysis": "Analysis",
    "documentation": "Documentation",
    "external": "External Providers",
    "developer_tools": "Developer Tools",
    "tool": "Tools",
}
LAUNCHER_CATEGORIES = frozenset(LAUNCHER_CATEGORY_LABELS)
WEB_LAUNCH_MODES = frozenset({"route", "native-window", "unavailable"})
TOOL_LIKE_CATEGORIES = frozenset(
    {
        "tool",
        "biomechanics",
        "simulation",
        "motion_matching",
        "motion_capture",
        "analysis",
        "documentation",
        "developer_tools",
    }
)


def _has_provider_metadata(model: ModelConfig) -> bool:
    """Return True when a registry entry comes from provider-aware metadata."""
    if model.provider not in (None, "", "local"):
        return True
    return bool(model.source_root)


def _normalize_launch_path(path: str) -> str:
    """Normalize a manifest ``path`` field to a repo-relative file path.

    Legacy manifests stored dotted Python module strings (e.g.
    ``src.tools.simulation_backends_launcher.__main__``) in the filesystem
    ``path`` field, which launch handlers treat as a file and fail to find
    (issue #8860). Dotted module strings are resolved to their ``.py`` file;
    real file paths pass through unchanged.
    """
    as_module = module_string_to_relpath(path)
    return as_module if as_module is not None else path


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


def _provider_status(
    model: Any,
    status: str,
    repo_root: Path,
    *,
    check_runtime: bool = True,
) -> tuple[str, str | None]:
    """Return an availability-aware (status, detail) without resolved paths.

    Postcondition:
        When the status is ``provider_unavailable`` because the pinned Tools
        vendor authority failed, the detail names the concrete reason (e.g.
        ``"unavailable: Tools pin stale (expected X, found Y)"``) so the
        degradation is explicit to the user, never silent (issue #8852).
    """
    if model.provider == "tools":
        authority = inspect_tools_vendor_authority(repo_root)
        if not authority.available:
            detail = f"unavailable: {authority.reason or 'Tools authority failed'}"
            logger.warning(
                "Launcher tile '%s' degraded to provider_unavailable: %s",
                model.id,
                detail,
            )
            return "provider_unavailable", detail
    elif isinstance(model.source_root, str) and not Path(model.source_root).exists():
        return "provider_unavailable", None
    if check_runtime and not is_engine_runtime_available(model.engine_type):
        return "runtime_unavailable", None
    return status, None


def _build_provider_tile(
    model: ModelConfig, *, repo_root: Path = REPO_ROOT
) -> LauncherTile:
    """Adapt a provider-backed model registry entry into a launcher tile."""
    metadata = model.launcher or _legacy_launcher_metadata(model)
    status, status_detail = _provider_status(model, metadata.status, repo_root)

    return LauncherTile(
        id=model.id,
        name=model.name,
        description=model.description,
        category=metadata.category,
        type=model.type,
        path=model.path,
        logo=_web_logo(metadata.logo),
        status=status,
        status_detail=status_detail,
        capabilities=model.capabilities,
        order=model.order,
        engine_type=model.engine_type,
        provider=model.provider,
        source_root=None if model.provider == "tools" else model.source_root,
        web_route=metadata.web_route,
        web=WebLaunchContract.derive(
            web_route=metadata.web_route,
            path=model.path,
        ),
        default_launch=metadata.default_launch,
        hidden=model.hidden,
        hidden_reason=model.hidden_reason,
        hidden_owner=model.hidden_owner,
    )


def _with_native_pyqt6_semantics(
    tile: LauncherTile, model: ModelConfig | None, *, repo_root: Path = REPO_ROOT
) -> LauncherTile:
    """Derive shared tile semantics from the primary PyQt6 registry entry.

    ``models.yaml`` owns the category, launch target, engine identity, and
    status used by the native launcher.  The shared manifest keeps web-only
    navigation such as ``web_route`` and capability tags used by the web
    catalog; neither substitutes for the native launch contract.
    """
    if model is None or model.launcher is None:
        return tile

    status, status_detail = _provider_status(
        model, model.launcher.status, repo_root, check_runtime=False
    )
    return replace(
        tile,
        category=model.launcher.category,
        status=status,
        status_detail=status_detail,
        type=model.type,
        path=model.path,
        engine_type=model.engine_type,
    )


def _tile_from_record(
    record: Any, model: ModelConfig | None, *, repo_root: Path = REPO_ROOT
) -> LauncherTile:
    """Adapt a ``TileRecord`` from the single registry into a launcher tile.

    Native entries (present in ``ModelRegistry``) keep the exact PyQt6
    semantics via :func:`_with_native_pyqt6_semantics`; web-catalog entries
    only get provider availability applied (no engine runtime probe).
    """
    tile = LauncherTile.from_dict(record.to_manifest_dict())
    if model is not None and model.launcher is not None:
        return _with_native_pyqt6_semantics(tile, model, repo_root=repo_root)
    status, status_detail = _provider_status(
        record, record.status, repo_root, check_runtime=False
    )
    return replace(tile, status=status, status_detail=status_detail)


@dataclass(frozen=True)
class WebLaunchContract:
    """How a tile is reachable from the web (browser/Tauri) app.

    Modes (issue #7461):
        route: The tile opens as an in-app React route. ``route`` is required
            and must start with "/".
        native-window: Launching spawns a native (Qt) window on the machine
            running the API server. Only meaningful when that machine is the
            user's machine (Tauri mode or localhost API).
        unavailable: The tile has no web affordance. ``reason`` is required so
            the dashboard can render an honest badge instead of a dead button.
    """

    mode: str
    route: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the contract invariants (DbC)."""
        if self.mode not in WEB_LAUNCH_MODES:
            raise ValueError(
                f"web.mode must be one of {sorted(WEB_LAUNCH_MODES)}, got {self.mode!r}"
            )
        if self.mode == "route":
            if not isinstance(self.route, str) or not self.route.startswith("/"):
                raise ValueError(
                    "web.route is required for mode 'route' and must start "
                    f"with '/', got {self.route!r}"
                )
        elif self.route is not None:
            raise ValueError(
                f"web.route is only valid for mode 'route', not {self.mode!r}"
            )
        if self.mode == "unavailable" and (
            not isinstance(self.reason, str) or not self.reason.strip()
        ):
            raise ValueError("web.reason is required for mode 'unavailable'")

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, tile_id: str = "?"
    ) -> WebLaunchContract:
        """Parse and validate a ``web`` manifest entry.

        Args:
            data: The ``web`` mapping from the manifest.
            tile_id: Tile ID for error messages.

        Returns:
            Validated WebLaunchContract.

        Raises:
            ValueError: If the contract is malformed.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Tile '{tile_id}': 'web' must be a mapping")
        try:
            return cls(
                mode=data.get("mode", ""),
                route=data.get("route"),
                reason=data.get("reason"),
            )
        except ValueError as exc:
            raise ValueError(f"Tile '{tile_id}': {exc}") from exc

    @classmethod
    def derive(cls, *, web_route: str | None, path: str | None) -> WebLaunchContract:
        """Derive an honest default contract for tiles without a declaration.

        Used for dynamically generated tiles (e.g. provider-backed registry
        entries). Manifest tiles must declare ``web`` explicitly — enforced by
        tests/config/launcher_manifest/test_parity.py.
        """
        if isinstance(web_route, str) and web_route.startswith("/"):
            return cls(mode="route", route=web_route)
        if path:
            return cls(mode="native-window")
        return cls(
            mode="unavailable",
            reason="No web route or native entry point declared",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses."""
        result: dict[str, Any] = {"mode": self.mode}
        if self.route is not None:
            result["route"] = self.route
        if self.reason is not None:
            result["reason"] = self.reason
        return result


@dataclass(frozen=True)
class LauncherTile:
    """A single launcher tile definition.

    Attributes:
        id: Unique identifier for the tile
        name: Display name shown in both launchers
        description: Brief description shown under the tile
        category: One of the canonical launcher categories in
            LAUNCHER_CATEGORIES.
        type: Engine/handler type for launch dispatch
        path: Relative path to the script/entry point
        logo: Logo filename (relative to assets dir)
        status: Status chip text (gui_ready, engine_ready, utility, etc.)
        capabilities: List of capability tags for filtering/display
        order: Display order (1 = first)
        engine_type: Optional engine type identifier for physics engines
        web_route: Optional URL path for tiles that open web tools (legacy;
            superseded by ``web`` for reachability decisions)
        web: Web launch contract declaring how (or whether) the tile is
            reachable from the browser/Tauri app (issue #7461)
    """

    id: str
    name: str
    description: str
    category: str
    type: str
    path: str
    logo: str
    status: str
    status_detail: str | None = None
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    order: int = 99
    engine_type: str | None = None
    provider: str | None = None
    source_root: str | None = None
    working_dir: str | None = None
    python_paths: tuple[str, ...] = ()
    web_route: str | None = None
    web: WebLaunchContract | None = None
    default_launch: str = "tab"
    shell_surfaces: tuple[str, ...] = ()
    hidden: bool = False
    hidden_reason: str | None = None
    hidden_owner: str | None = None

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
        hidden = bool(data.get("hidden", False))
        hidden_reason = data.get("hidden_reason")
        hidden_owner = data.get("hidden_owner")
        if hidden:
            if not isinstance(hidden_reason, str) or not hidden_reason.strip():
                raise ValueError(
                    f"Hidden launcher tile '{data.get('id')}' must define hidden_reason"
                )
            if not isinstance(hidden_owner, str) or not hidden_owner.strip():
                raise ValueError(
                    f"Hidden launcher tile '{data.get('id')}' must define hidden_owner"
                )

        web_raw = data.get("web")
        if web_raw is not None:
            web = WebLaunchContract.from_dict(web_raw, tile_id=str(data.get("id")))
        else:
            web = WebLaunchContract.derive(
                web_route=data.get("web_route"),
                path=data.get("path"),
            )

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=data["category"],
            type=data["type"],
            path=_normalize_launch_path(data["path"]),
            logo=data["logo"],
            status=data.get("status", "unknown"),
            status_detail=data.get("status_detail"),
            capabilities=tuple(data.get("capabilities", [])),
            tags=tuple(data.get("tags", [])),
            order=data.get("order", 99),
            engine_type=data.get("engine_type"),
            provider=data.get("provider"),
            source_root=data.get("source_root"),
            working_dir=data.get("working_dir"),
            python_paths=tuple(data.get("python_paths", [])),
            web_route=data.get("web_route"),
            web=web,
            default_launch=data.get("default_launch", "tab"),
            shell_surfaces=tuple(data.get("shell_surfaces", [])),
            hidden=hidden,
            hidden_reason=(
                hidden_reason.strip() if isinstance(hidden_reason, str) else None
            ),
            hidden_owner=(
                hidden_owner.strip() if isinstance(hidden_owner, str) else None
            ),
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
        if self.status_detail:
            result["status_detail"] = self.status_detail
        if self.engine_type:
            result["engine_type"] = self.engine_type
        if self.provider:
            result["provider"] = self.provider
        if self.source_root:
            result["source_root"] = self.source_root
        if self.working_dir:
            result["working_dir"] = self.working_dir
        if self.python_paths:
            result["python_paths"] = list(self.python_paths)
        if self.web_route:
            result["web_route"] = self.web_route
        if self.web is not None:
            result["web"] = self.web.to_dict()
        if self.default_launch:
            result["default_launch"] = self.default_launch
        if self.shell_surfaces:
            result["shell_surfaces"] = list(self.shell_surfaces)
        if self.tags:
            result["tags"] = list(self.tags)
        if self.hidden:
            result["hidden"] = True
            result["hidden_reason"] = self.hidden_reason
            result["hidden_owner"] = self.hidden_owner
        return result

    @property
    def logo_path(self) -> Path:
        """Absolute path to the logo file."""
        direct = ASSETS_DIR / self.logo
        if direct.exists():
            return direct
        basename_direct = ASSETS_DIR / Path(self.logo).name
        if basename_direct.exists():
            return basename_direct
        # models.yaml tiles declare logos relative to the PyQt launcher
        # assets dir (src/launchers/assets/...); resolve those too so
        # registry-surfaced tiles keep their desktop artwork on the web.
        pyqt_direct = PYQT_ASSETS_ROOT / self.logo
        if pyqt_direct.exists():
            return pyqt_direct
        if self.source_root:
            sr_path = (REPO_ROOT / self.source_root).resolve()
            candidates = [
                sr_path / self.logo,
                sr_path / "assets" / Path(self.logo).name,
                (REPO_ROOT.parent / Path(self.source_root).name / self.logo).resolve(),
                (
                    REPO_ROOT.parent
                    / Path(self.source_root).name
                    / "assets"
                    / Path(self.logo).name
                ).resolve(),
            ]
            for c in candidates:
                if c.exists():
                    return c
        for sibling_name in ("Tools", "Movement-Optimizer", "vendor/ud-tools"):
            sibling_base = (
                (REPO_ROOT / sibling_name).resolve()
                if "vendor" in sibling_name
                else (REPO_ROOT.parent / sibling_name).resolve()
            )
            if sibling_base.exists():
                c1 = sibling_base / self.logo
                if c1.exists():
                    return c1
                c2 = sibling_base / "assets" / Path(self.logo).name
                if c2.exists():
                    return c2
        return direct

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
        return self.category in TOOL_LIKE_CATEGORIES


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
        if path is None:
            # Default: build straight from the single registry (issue
            # #9412). launcher_manifest.json is a generated projection of
            # models.yaml and is never read at runtime.
            return cls._load_from_registry(
                registry_path=registry_path or REGISTRY_PATH,
                include_provider_tiles=include_provider_tiles,
            )

        # Explicit overlay file: legacy JSON + registry augmentation, kept
        # for callers/tests that supply their own manifest documents.
        manifest_path = path

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
        manifest_repo_root = manifest_path.parents[2]
        registry = ModelRegistry(config_path=registry_path or REGISTRY_PATH)
        native_models = {model.id: model for model in registry.get_all_models()}
        tiles = [
            _with_native_pyqt6_semantics(
                tile, native_models.get(tile.id), repo_root=manifest_repo_root
            )
            for tile in tiles
        ]
        if include_provider_tiles:
            tiles.extend(
                cls._load_provider_tiles(
                    registry=registry,
                    existing_ids={tile.id for tile in tiles},
                    repo_root=manifest_repo_root,
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

    @classmethod
    def _load_from_registry(
        cls, *, registry_path: Path, include_provider_tiles: bool
    ) -> LauncherManifest:
        """Build the manifest from ``models.yaml`` (native + web catalog)."""
        from src.config.tile_registry import (
            MANIFEST_DESCRIPTION,
            MANIFEST_VERSION,
            load_tile_registry,
        )

        registry_path = Path(registry_path)
        tile_registry = load_tile_registry(registry_path)
        repo_root = registry_path.resolve().parents[2]
        model_registry = ModelRegistry(config_path=registry_path)
        native_models = {model.id: model for model in model_registry.get_all_models()}
        tiles = [
            _tile_from_record(record, native_models.get(record.id), repo_root=repo_root)
            for record in tile_registry.tiles
        ]
        if include_provider_tiles:
            # Sibling-repo / env-discovered provider models (hybrid discovery)
            # still surface as tiles, exactly as before.
            tiles.extend(
                cls._load_provider_tiles(
                    registry=model_registry,
                    existing_ids={tile.id for tile in tiles},
                    repo_root=repo_root,
                )
            )
        sorted_tiles: tuple[LauncherTile, ...] = tuple(
            sorted(tiles, key=lambda t: (t.order, t.id))
        )
        ids = [t.id for t in sorted_tiles]
        duplicates = {tid for tid in ids if ids.count(tid) > 1}
        if duplicates:
            raise ValueError(f"Duplicate tile IDs in registry: {duplicates}")
        logger.info(
            "Loaded %d tiles from the tile registry %s",
            len(sorted_tiles),
            registry_path,
        )
        return cls(
            version=MANIFEST_VERSION,
            tiles=sorted_tiles,
            description=MANIFEST_DESCRIPTION,
        )

    @staticmethod
    def _load_provider_tiles(
        *,
        existing_ids: set[str],
        registry: ModelRegistry | None = None,
        registry_path: Path | None = None,
        repo_root: Path = REPO_ROOT,
    ) -> list[LauncherTile]:
        """Load dynamic provider-backed tiles from the shared model registry."""
        if registry is None:
            resolved_registry_path = registry_path or REGISTRY_PATH
            if not resolved_registry_path.exists():
                return []
            registry = ModelRegistry(config_path=resolved_registry_path)

        provider_tiles: list[LauncherTile] = []

        # Every registry model surfaces as a tile, not just provider-backed
        # ones: gating on provider metadata structurally excluded repo-local
        # tools (sidekick, pose_subscriber_demo, ...) from the web catalog
        # while the desktop launcher showed them (issue #8853).
        for model in registry.get_all_models():
            if model.id in existing_ids:
                continue
            provider_tiles.append(_build_provider_tile(model, repo_root=repo_root))

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
            category: Category to filter by. Must be a canonical launcher
                category from LAUNCHER_CATEGORIES.
            include_hidden: When False (default), tiles flagged ``hidden`` are
                excluded so legacy aliases do not appear as duplicate launcher
                cards.

        Returns:
            List of matching tiles, ordered by their order field
        """
        if category not in LAUNCHER_CATEGORIES:
            raise ValueError(f"Unknown launcher category: {category}")
        return [
            t
            for t in self.tiles
            if t.category == category and (include_hidden or not t.hidden)
        ]

    @property
    def categories(self) -> dict[str, list[LauncherTile]]:
        """Visible tiles grouped by canonical launcher category."""
        return {
            category: self.get_tiles_by_category(category)
            for category in LAUNCHER_CATEGORY_LABELS
        }

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
        """Get all non-engine utility tiles (excluding hidden aliases)."""
        return [t for t in self.visible_tiles if t.category in TOOL_LIKE_CATEGORIES]

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
            "category_labels": dict(LAUNCHER_CATEGORY_LABELS),
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
