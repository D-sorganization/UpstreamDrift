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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.config.capability_state import (
    _DEFAULT_PROVIDER_LOGO,
    _ENGINE_LOGOS,
    _WEB_LOGO_BY_DESKTOP_PNG,
    _web_logo,
    KNOWN_PHYSICS_ENGINES,
    VALID_MATURITY_LEVELS,
    CapabilityAvailability,
    CapabilityQualification,
    SurfaceAvailability,
    adapt_engine_matrix_qualification,
    resolve_canonical_display_name,
)
from src.config.engine_probe_cache import is_cached_engine_runtime_available
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
_READY_STATUSES = frozenset({"ready", "engine_ready", "release_ready", "gui_ready"})
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

WEB_CATALOG_ONLY_TILES: dict[str, str] = {
    "chat_assistant": "web chat page (/chat); desktop equivalent is the sidekick dock",
    "dataset_generator": "web page (/tools/dataset); native path is the MATLAB chooser",
    "character_builder": "web page (/tools/character-builder); native side is a CLI",
    "analysis_tools_api": "web page (/tools/analysis) over REST endpoints",
    "motion_pipeline": "REST pipeline service; no desktop tile",
    "perturbation_analysis": "API-backed catalog entry; no launchable surface",
    "force_overlays": "API-backed catalog entry; no launchable surface",
    "realtime_ws": "WebSocket endpoint catalog entry; no launchable surface",
    "aip": "API-backed catalog entry; no launchable surface",
    "actuator_controls": "API-backed catalog entry; no launchable surface",
    "unreal_integration": "integration library catalog entry; no launchable surface",
    "robotics_module": "python module catalog entry; no launchable surface",
    "tools_calculator_hub": "alias surface over the Tools data processor",
    "pid_generator": "Tools-ported CLI (generate-pid); no GUI yet",
}


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


_ORIGINAL_IS_ENGINE_RUNTIME_AVAILABLE = is_engine_runtime_available


def _check_engine_runtime(engine_type: str | None) -> bool:
    """Check runtime availability using cache unless monkeypatched in tests."""
    if is_engine_runtime_available is not _ORIGINAL_IS_ENGINE_RUNTIME_AVAILABLE:
        return is_engine_runtime_available(engine_type)
    return is_cached_engine_runtime_available(engine_type)


def _provider_status(
    model: ModelConfig,
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
    if check_runtime and not _check_engine_runtime(model.engine_type):
        return "runtime_unavailable", None
    if model.engine_type and model.engine_type in KNOWN_PHYSICS_ENGINES:
        qual = adapt_engine_matrix_qualification(
            engine_name=model.engine_type,
            is_engine=True,
        )
        if not qual.is_qualified and status in _READY_STATUSES:
            return "experimental", None
    return status, None


@dataclass(frozen=True)
class ProviderContext:
    """Provider metadata for surface availability derivation."""

    provider: str | None = None
    source_root: str | None = None
    engine_type: str | None = None


def _derive_desktop_availability(
    tile_id: str,
    path: str | None,
    status: str,
    status_detail: str | None = None,
    provider_ctx: ProviderContext | None = None,
    *,
    repo_root: Path = REPO_ROOT,
) -> SurfaceAvailability:
    """Derive desktop surface availability."""
    provider = provider_ctx.provider if provider_ctx else None
    source_root = provider_ctx.source_root if provider_ctx else None
    engine_type = provider_ctx.engine_type if provider_ctx else None

    if provider == "tools":
        auth = inspect_tools_vendor_authority(repo_root)
        if not auth.available:
            return SurfaceAvailability(
                available=False,
                reason=f"Tools vendor authority unavailable: {auth.reason or 'Pin stale'}",
                remediation="Run 'python -m scripts.sync_vendor_tools' to synchronize the vendor submodule",
            )
        return SurfaceAvailability(available=True)
    if status == "provider_unavailable":
        return SurfaceAvailability(
            available=False,
            reason=status_detail or f"Provider source root missing: {source_root}",
            remediation="Clone provider repository or configure valid source_root in models.yaml",
        )
    if status == "runtime_unavailable":
        return SurfaceAvailability(
            available=False,
            reason=f"Runtime unavailable: engine runtime '{engine_type}' is not installed in Python environment",
            remediation=f"Install runtime dependencies for '{engine_type}' in the active Python environment",
        )
    if tile_id in WEB_CATALOG_ONLY_TILES:
        return SurfaceAvailability(
            available=False,
            reason=f"Tile '{tile_id}' is a web-only affordance with no desktop window",
            remediation="Access this capability via the web catalog or dashboard",
        )
    if not path or not path.strip() or path.startswith("virtual/"):
        return SurfaceAvailability(
            available=False,
            reason=f"Capability '{tile_id}' has no native desktop entry point",
            remediation="Configure a valid script path or launch via web",
        )
    return SurfaceAvailability(available=True)


def _derive_web_availability(
    tile_id: str,
    web: WebLaunchContract | None,
    path: str | None,
) -> SurfaceAvailability:
    """Derive web surface availability."""
    if web is None:
        return SurfaceAvailability(
            available=False,
            reason=f"No web launch contract configured for tile '{tile_id}'",
            remediation="Launch from PyQt desktop launcher or use command-line interface",
        )
    if web.mode == "route":
        return SurfaceAvailability(available=True)
    if web.mode == "native-window":
        return SurfaceAvailability(
            available=False,
            reason="Native-window launch target requires desktop launcher environment; unavailable directly in browser",
            remediation=(
                f"Launch from PyQt desktop launcher or run: python {path}"
                if path
                else "Launch from desktop launcher"
            ),
        )
    reason_text = web.reason or f"No web affordance declared for tile '{tile_id}'"
    remediation_text = (
        "Use desktop Qt launcher or track release milestone in issue tracker"
        if "planned" in reason_text.lower()
        else "Launch from PyQt desktop launcher or use command-line interface"
    )
    return SurfaceAvailability(
        available=False,
        reason=reason_text,
        remediation=remediation_text,
    )


def _derive_cli_availability(
    tile_id: str,
    path: str | None,
) -> SurfaceAvailability:
    """Derive CLI surface availability."""
    if path and path.endswith(".py"):
        return SurfaceAvailability(available=True)
    return SurfaceAvailability(
        available=False,
        reason=f"Capability '{tile_id}' has no standalone CLI entry point",
        remediation="Launch via desktop launcher or web interface",
    )


def _build_default_availability(
    tile_id: str,
    path: str | None,
    web: WebLaunchContract | None,
    status: str,
    status_detail: str | None = None,
    provider_ctx: ProviderContext | None = None,
    *,
    repo_root: Path = REPO_ROOT,
) -> CapabilityAvailability:
    """Derive orthogonal per-surface availability for desktop, web, api, and cli."""
    desktop = _derive_desktop_availability(
        tile_id=tile_id,
        path=path,
        status=status,
        status_detail=status_detail,
        provider_ctx=provider_ctx,
        repo_root=repo_root,
    )
    web_avail = _derive_web_availability(tile_id, web, path)
    api_avail = SurfaceAvailability(available=True)
    cli_avail = _derive_cli_availability(tile_id, path)

    return CapabilityAvailability(
        surfaces={
            "desktop": desktop,
            "web": web_avail,
            "api": api_avail,
            "cli": cli_avail,
        }
    )


def _qualify_engine_status(
    engine_name: str | None,
    is_engine: bool,
    status: str,
) -> tuple[CapabilityQualification, str]:
    """Adapt engine matrix qualification and demote unverified status to experimental."""
    qualification = adapt_engine_matrix_qualification(
        engine_name=engine_name if is_engine else None,
        is_engine=is_engine,
    )
    if (
        is_engine
        and status not in ("runtime_unavailable", "provider_unavailable")
        and not qualification.is_qualified
        and status in _READY_STATUSES
    ):
        status = "experimental"
    return qualification, status


def _build_provider_tile(
    model: ModelConfig, *, repo_root: Path = REPO_ROOT
) -> LauncherTile:
    """Adapt a provider-backed model registry entry into a launcher tile."""
    metadata = model.launcher or _legacy_launcher_metadata(model)
    status, status_detail = _provider_status(model, metadata.status, repo_root)

    web_route = metadata.web_route
    if (
        model.id in ("movement_optimizer", "tools_movement_optimizer")
        or web_route == "/tools/movement-optimizer"
    ):
        web_route = None

    is_engine = bool(
        (model.engine_type and model.engine_type in KNOWN_PHYSICS_ENGINES)
        or metadata.category == "physics_engine"
    )
    qualification, status = _qualify_engine_status(model.engine_type, is_engine, status)

    web = WebLaunchContract.derive(
        web_route=web_route,
        path=model.path,
    )
    availability = _build_default_availability(
        tile_id=model.id,
        path=model.path,
        web=web,
        status=status,
        status_detail=status_detail,
        provider_ctx=ProviderContext(
            provider=model.provider,
            source_root=model.source_root,
            engine_type=model.engine_type,
        ),
        repo_root=repo_root,
    )
    maturity = (
        "experimental"
        if (is_engine and not qualification.is_qualified)
        or status in ("experimental", "unadvertised_experimental")
        else "stable"
    )
    canonical_name = resolve_canonical_display_name(model.id, model.name)

    return LauncherTile(
        id=model.id,
        name=canonical_name,
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
        web_route=web_route,
        web=web,
        default_launch=metadata.default_launch,
        hidden=model.hidden,
        hidden_reason=model.hidden_reason,
        hidden_owner=model.hidden_owner,
        maturity=maturity,
        availability=availability,
        qualification=qualification,
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
    is_engine = bool(
        (model.engine_type and model.engine_type in KNOWN_PHYSICS_ENGINES)
        or model.launcher.category == "physics_engine"
    )
    qualification, status = _qualify_engine_status(model.engine_type, is_engine, status)

    maturity = getattr(model.launcher, "maturity", None)
    if not maturity or maturity not in VALID_MATURITY_LEVELS:
        if status in ("experimental", "unadvertised_experimental") or (
            is_engine and not qualification.is_qualified
        ):
            maturity = "experimental"
        else:
            maturity = "stable"

    availability = _build_default_availability(
        tile_id=tile.id,
        path=model.path,
        web=tile.web,
        status=status,
        status_detail=status_detail,
        provider_ctx=ProviderContext(
            provider=model.provider,
            source_root=model.source_root,
            engine_type=model.engine_type,
        ),
        repo_root=repo_root,
    )
    canonical_name = resolve_canonical_display_name(tile.id, model.name or tile.name)

    return replace(
        tile,
        name=canonical_name,
        category=model.launcher.category,
        status=status,
        status_detail=status_detail,
        type=model.type,
        path=model.path,
        engine_type=model.engine_type,
        maturity=maturity,
        availability=availability,
        qualification=qualification,
    )


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


def _validate_hidden_tile(
    tile_id: Any,
    hidden: bool,
    hidden_reason: Any,
    hidden_owner: Any,
) -> tuple[str | None, str | None]:
    """Validate and normalize hidden tile attributes."""
    if not hidden:
        return (
            hidden_reason.strip() if isinstance(hidden_reason, str) else None,
            hidden_owner.strip() if isinstance(hidden_owner, str) else None,
        )
    if not isinstance(hidden_reason, str) or not hidden_reason.strip():
        raise ValueError(f"Hidden launcher tile '{tile_id}' must define hidden_reason")
    if not isinstance(hidden_owner, str) or not hidden_owner.strip():
        raise ValueError(f"Hidden launcher tile '{tile_id}' must define hidden_owner")
    return hidden_reason.strip(), hidden_owner.strip()


def _parse_tile_state(
    data: dict[str, Any],
    web: WebLaunchContract,
) -> tuple[str, str, CapabilityAvailability, CapabilityQualification]:
    """Parse maturity, availability, qualification, and status for LauncherTile."""
    avail_raw = data.get("availability")
    qual_raw = data.get("qualification")
    availability = (
        CapabilityAvailability.from_dict(avail_raw)
        if isinstance(avail_raw, dict)
        else None
    )
    qualification = (
        CapabilityQualification.from_dict(qual_raw)
        if isinstance(qual_raw, dict)
        else None
    )

    is_engine = bool(
        data.get("engine_type")
        or data.get("category") == "physics_engine"
        or data.get("type")
        in (
            "mujoco",
            "drake",
            "pinocchio",
            "opensim",
            "myosim",
            "custom_humanoid",
        )
    )
    if qualification is None:
        qualification = adapt_engine_matrix_qualification(
            engine_name=data.get("engine_type"),
            is_engine=is_engine,
        )

    status_val = data.get("status", "unknown")
    if (
        is_engine
        and not qualification.is_qualified
        and status_val in ("ready", "engine_ready", "release_ready")
    ):
        status_val = "experimental"

    maturity_raw = data.get("maturity")
    if maturity_raw in VALID_MATURITY_LEVELS:
        maturity = maturity_raw
    elif status_val in ("experimental", "unadvertised_experimental"):
        maturity = "experimental"
    elif status_val == "prototype":
        maturity = "prototype"
    elif status_val == "deprecated":
        maturity = "deprecated"
    elif is_engine and not qualification.is_qualified:
        maturity = "experimental"
    else:
        maturity = "stable"

    if availability is None:
        availability = _build_default_availability(
            tile_id=str(data["id"]),
            path=data.get("path"),
            web=web,
            status=status_val,
            status_detail=data.get("status_detail"),
            provider_ctx=ProviderContext(
                provider=data.get("provider"),
                source_root=data.get("source_root"),
                engine_type=data.get("engine_type"),
            ),
        )

    return status_val, maturity, availability, qualification


@dataclass(frozen=True)
class LauncherTile:
    """A tile or model that can be launched from the GUI launcher."""

    id: str
    name: str
    description: str
    category: str
    type: str
    path: str
    logo: str
    status: str = "experimental"
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

    maturity: str = "stable"
    availability: CapabilityAvailability | None = None
    qualification: CapabilityQualification | None = None

    def __post_init__(self) -> None:
        """Ensure availability and qualification contracts are always initialized."""
        if self.availability is None:
            avail = _build_default_availability(
                tile_id=self.id,
                path=self.path,
                web=self.web,
                status=self.status,
                status_detail=self.status_detail,
                provider_ctx=ProviderContext(
                    provider=self.provider,
                    source_root=self.source_root,
                    engine_type=self.engine_type,
                ),
            )
            object.__setattr__(self, "availability", avail)

        if self.qualification is None:
            is_eng = bool(
                self.engine_type
                or self.category == "physics_engine"
                or (self.type in KNOWN_PHYSICS_ENGINES and self.category != "tool")
            )
            qual = adapt_engine_matrix_qualification(
                engine_name=self.engine_type,
                is_engine=is_eng,
            )
            object.__setattr__(self, "qualification", qual)

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
        hidden_reason, hidden_owner = _validate_hidden_tile(
            data.get("id"), hidden, data.get("hidden_reason"), data.get("hidden_owner")
        )

        web_raw = data.get("web")
        if web_raw is not None:
            web = WebLaunchContract.from_dict(web_raw, tile_id=str(data.get("id")))
        else:
            web = WebLaunchContract.derive(
                web_route=data.get("web_route"),
                path=data.get("path"),
            )

        status_val, maturity, availability, qualification = _parse_tile_state(data, web)
        canonical_name = resolve_canonical_display_name(data["id"], data["name"])

        return cls(
            id=data["id"],
            name=canonical_name,
            description=data["description"],
            category=data["category"],
            type=data["type"],
            path=_normalize_launch_path(data["path"]),
            logo=data["logo"],
            status=status_val,
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
            hidden_reason=hidden_reason,
            hidden_owner=hidden_owner,
            maturity=maturity,
            availability=availability,
            qualification=qualification,
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
        result["maturity"] = self.maturity
        if self.availability is not None:
            result["availability"] = self.availability.to_dict()
        else:
            result["availability"] = _build_default_availability(
                tile_id=self.id,
                path=self.path,
                web=self.web,
                status=self.status,
                status_detail=self.status_detail,
                provider_ctx=ProviderContext(
                    provider=self.provider,
                    source_root=self.source_root,
                    engine_type=self.engine_type,
                ),
            ).to_dict()
        if self.qualification is not None:
            result["qualification"] = self.qualification.to_dict()
        else:
            is_eng = bool(
                self.engine_type
                or self.category == "physics_engine"
                or (self.type in KNOWN_PHYSICS_ENGINES and self.category != "tool")
            )
            result["qualification"] = adapt_engine_matrix_qualification(
                engine_name=self.engine_type,
                is_engine=is_eng,
            ).to_dict()
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
