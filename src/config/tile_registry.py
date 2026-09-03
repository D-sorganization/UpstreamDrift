"""Single-source tile registry (issue #9412).

``src/config/models.yaml`` is the ONE registry both launchers read:

* ``models:`` — native, launchable entries. The PyQt6 launcher reads them
  through :class:`~src.shared.python.config.model_registry.ModelRegistry`
  (Tools-owned, unchanged); this module reads the same entries directly.
* ``web_catalog:`` — entries that exist only on the web/API surface. Every
  one carries ``surfaces: [web]`` and a ``surface_reason``. ``ModelRegistry``
  ignores this list, so the PyQt launcher never sees them.
* ``excluded_packages:`` — ``src/tools`` packages that deliberately have no
  launcher tile (issue #8863), each with a reason.

Every tile carries the readiness dimensions the registry previously lacked:

* ``maturity``: ``ready | beta | experimental | hidden``;
* ``surfaces``: subset of ``{pyqt, web}`` (``surface_reason`` required when
  a tile is missing from either surface);
* ``help``: repo-relative help page or ``null``;
* ``feature_id``: the ``feature_parity.json`` feature the tile belongs to;
* ``web``: the honest web launch contract (``route | native-window |
  unavailable``).

``src/config/launcher_manifest.json`` and the ``tiles`` arrays of
``src/config/feature_parity.json`` are *generated* projections of this file
(``python -m scripts.registry.generate_registry_artifacts``); nothing at
runtime reads the JSON manifest by default.

This module deliberately imports only the standard library and PyYAML so the
generator can run in the dependency-light ``repo-structure-gates`` CI job.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).resolve().parent / "models.yaml"
REPO_ROOT = REGISTRY_PATH.parents[2]

MATURITY_LEVELS: frozenset[str] = frozenset({"ready", "beta", "experimental", "hidden"})
SURFACES: tuple[str, ...] = ("pyqt", "web")
WEB_MODES: frozenset[str] = frozenset({"route", "native-window", "unavailable"})
# Pathless model types whose launch target is a virtual namespace; keep in
# sync with ``_METADATA_ONLY_TYPES`` in
# src/shared/python/config/model_registry.py (Tools-owned, not importable
# from the dependency-light generator).
METADATA_ONLY_TYPES: frozenset[str] = frozenset(
    {"biomech_exercise", "physics_informed"}
)
MANIFEST_VERSION = "1.0.0"
MANIFEST_DESCRIPTION = (
    "GENERATED from src/config/models.yaml by "
    "scripts/registry/generate_registry_artifacts.py - do not edit by hand. "
    "Web projection of the single tile registry (issue #9412)."
)

# The desktop launcher keeps legacy PNG artwork under src/launchers/assets;
# the web catalog serves SVG-only logos from assets/logos. Registry tiles
# translate their desktop artwork to the SVG equivalent for the web unless
# they declare an explicit ``web_logo``.
_WEB_LOGO_BY_DESKTOP_PNG: dict[str, str] = {
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
_DEFAULT_WEB_LOGO = "golf_logo.svg"


class TileRegistryError(ValueError):
    """Raised when models.yaml violates the single-registry contract."""


def web_logo_for(logo: str) -> str:
    """Return the SVG logo the web catalog serves for a desktop logo value."""
    if logo.endswith(".svg"):
        return logo
    return _WEB_LOGO_BY_DESKTOP_PNG.get(Path(logo).name, _DEFAULT_WEB_LOGO)


@dataclass(frozen=True)
class TileRecord:
    """One tile as declared in models.yaml (native or web-catalog)."""

    id: str
    name: str
    description: str
    type: str
    path: str
    category: str
    logo: str
    status: str
    order: int
    maturity: str
    surfaces: tuple[str, ...]
    web: dict[str, str]
    source: str  # "models" | "web_catalog"
    help: str | None = None
    feature_id: str | None = None
    surface_reason: str | None = None
    maturity_reason: str | None = None
    web_route: str | None = None
    web_logo: str | None = None
    default_launch: str = "tab"
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    shell_surfaces: tuple[str, ...] = ()
    engine_type: str | None = None
    provider: str | None = None
    source_root: str | None = None
    working_dir: str | None = None
    python_paths: tuple[str, ...] = ()
    embed_adapter: str | None = None
    hidden: bool = False
    hidden_reason: str | None = None
    hidden_owner: str | None = None

    @property
    def visible(self) -> bool:
        return not self.hidden and self.maturity != "hidden"

    def on_surface(self, surface: str) -> bool:
        """True when the tile is declared for ``surface`` ("pyqt" or "web")."""
        return surface in self.surfaces

    @property
    def effective_web_logo(self) -> str:
        return self.web_logo or web_logo_for(self.logo)

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize to the launcher_manifest.json tile shape."""
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "type": self.type,
            "path": self.path,
        }
        if self.engine_type:
            data["engine_type"] = self.engine_type
        if self.provider:
            data["provider"] = self.provider
        # Tools-provided tiles resolve through the pinned vendor gitlink, not
        # a sibling checkout, so the manifest never advertises a source root.
        if self.source_root and self.provider != "tools":
            data["source_root"] = self.source_root
        if self.working_dir:
            data["working_dir"] = self.working_dir
        if self.python_paths:
            data["python_paths"] = list(self.python_paths)
        data["logo"] = self.effective_web_logo
        data["status"] = self.status
        data["maturity"] = self.maturity
        if self.maturity_reason:
            data["maturity_reason"] = self.maturity_reason
        data["surfaces"] = list(self.surfaces)
        if self.surface_reason:
            data["surface_reason"] = self.surface_reason
        data["help"] = self.help
        data["feature_id"] = self.feature_id
        data["capabilities"] = list(self.capabilities)
        if self.tags:
            data["tags"] = list(self.tags)
        if self.shell_surfaces:
            data["shell_surfaces"] = list(self.shell_surfaces)
        data["order"] = self.order
        data["default_launch"] = self.default_launch
        if self.web_route:
            data["web_route"] = self.web_route
        data["web"] = dict(self.web)
        if self.hidden:
            data["hidden"] = True
            data["hidden_reason"] = self.hidden_reason
            data["hidden_owner"] = self.hidden_owner
        return data


@dataclass(frozen=True)
class ExcludedPackage:
    package: str
    reason: str
    issue: str | None = None


@dataclass(frozen=True)
class TileRegistry:
    """The parsed single registry."""

    tiles: tuple[TileRecord, ...]
    excluded_packages: tuple[ExcludedPackage, ...] = ()
    path: Path = REGISTRY_PATH
    _by_id: dict[str, TileRecord] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_id", {tile.id: tile for tile in self.tiles})

    def get(self, tile_id: str) -> TileRecord | None:
        return self._by_id.get(tile_id)

    @property
    def ids(self) -> list[str]:
        return [tile.id for tile in self.tiles]

    def native_tiles(self) -> list[TileRecord]:
        return [tile for tile in self.tiles if tile.source == "models"]

    def web_catalog_tiles(self) -> list[TileRecord]:
        return [tile for tile in self.tiles if tile.source == "web_catalog"]

    def surface_ids(self, surface: str, *, include_hidden: bool = False) -> set[str]:
        """Tile ids declared for ``surface`` (visible only unless asked)."""
        if surface not in SURFACES:
            raise ValueError(f"unknown surface {surface!r}; expected one of {SURFACES}")
        return {
            tile.id
            for tile in self.tiles
            if tile.on_surface(surface) and (include_hidden or tile.visible)
        }

    def maturity_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(sorted(MATURITY_LEVELS), 0)
        for tile in self.tiles:
            counts[tile.maturity] += 1
        return counts

    def web_manifest_dict(self) -> dict[str, Any]:
        """The launcher_manifest.json document (web projection, order-sorted)."""
        ordered = sorted(self.tiles, key=lambda t: (t.order, t.id))
        return {
            "version": MANIFEST_VERSION,
            "description": MANIFEST_DESCRIPTION,
            "tiles": [tile.to_manifest_dict() for tile in ordered],
        }

    def feature_bindings(self) -> dict[str, list[str]]:
        """feature_id -> sorted tile ids (the feature_parity.json ``tiles``)."""
        bindings: dict[str, list[str]] = {}
        for tile in self.tiles:
            if tile.feature_id:
                bindings.setdefault(tile.feature_id, []).append(tile.id)
        return {key: sorted(value) for key, value in bindings.items()}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _string(
    entry: Mapping[str, Any], key: str, tile_id: str, *, required: bool
) -> str | None:
    value = entry.get(key)
    if value is None:
        if required:
            raise TileRegistryError(f"tile {tile_id!r}: missing required field {key!r}")
        return None
    if not isinstance(value, str):
        raise TileRegistryError(f"tile {tile_id!r}: field {key!r} must be a string")
    stripped = value.strip()
    if required and not stripped and key != "path":
        raise TileRegistryError(f"tile {tile_id!r}: field {key!r} must be non-empty")
    return stripped


def _string_tuple(entry: Mapping[str, Any], key: str, tile_id: str) -> tuple[str, ...]:
    value = entry.get(key) or ()
    if not isinstance(value, (list, tuple)):
        raise TileRegistryError(f"tile {tile_id!r}: field {key!r} must be a list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TileRegistryError(
                f"tile {tile_id!r}: {key!r} entries must be non-empty strings"
            )
        out.append(item.strip())
    return tuple(out)


def _parse_web(
    entry: Mapping[str, Any], tile_id: str, *, path: str, web_route: str | None
) -> dict[str, str]:
    raw = entry.get("web")
    if raw is None:
        # Derivation mirrors WebLaunchContract.derive so undeclared tiles keep
        # the historical behaviour; the generator always writes it explicitly.
        if web_route:
            return {"mode": "route", "route": web_route}
        if path:
            return {"mode": "native-window"}
        return {"mode": "unavailable", "reason": "No launch path and no web page"}
    if not isinstance(raw, Mapping):
        raise TileRegistryError(f"tile {tile_id!r}: 'web' must be a mapping")
    mode = raw.get("mode")
    if mode not in WEB_MODES:
        raise TileRegistryError(
            f"tile {tile_id!r}: web.mode must be one of {sorted(WEB_MODES)}"
        )
    web: dict[str, str] = {"mode": str(mode)}
    route = raw.get("route")
    reason = raw.get("reason")
    if mode == "route":
        if not isinstance(route, str) or not route.startswith("/"):
            raise TileRegistryError(f"tile {tile_id!r}: web.route must start with '/'")
        web["route"] = route
    elif route is not None:
        raise TileRegistryError(
            f"tile {tile_id!r}: web.route is only valid for mode 'route'"
        )
    if mode == "unavailable":
        if not isinstance(reason, str) or not reason.strip():
            raise TileRegistryError(
                f"tile {tile_id!r}: web mode 'unavailable' requires a reason"
            )
        web["reason"] = reason.strip()
    elif reason is not None:
        raise TileRegistryError(
            f"tile {tile_id!r}: web.reason is only valid for mode 'unavailable'"
        )
    return web


def _parse_launcher(entry: Mapping[str, Any], tile_id: str) -> Mapping[str, Any]:
    launcher = entry.get("launcher")
    if not isinstance(launcher, Mapping):
        raise TileRegistryError(f"tile {tile_id!r}: missing 'launcher' block")
    for key in ("category", "logo", "status"):
        if not isinstance(launcher.get(key), str) or not str(launcher[key]).strip():
            raise TileRegistryError(
                f"tile {tile_id!r}: launcher.{key} must be a non-empty string"
            )
    return launcher


def _parse_surfaces(
    entry: Mapping[str, Any], tile_id: str, *, source: str
) -> tuple[tuple[str, ...], str | None]:
    surfaces = _string_tuple(entry, "surfaces", tile_id)
    if not surfaces:
        raise TileRegistryError(
            f"tile {tile_id!r}: 'surfaces' must list at least one of {SURFACES}"
        )
    unknown = [s for s in surfaces if s not in SURFACES]
    if unknown:
        raise TileRegistryError(
            f"tile {tile_id!r}: unknown surfaces {unknown}; expected {SURFACES}"
        )
    surface_reason = _string(entry, "surface_reason", tile_id, required=False)
    if set(surfaces) != set(SURFACES) and not surface_reason:
        raise TileRegistryError(
            f"tile {tile_id!r}: surfaces {list(surfaces)} omit a surface; "
            "'surface_reason' is required"
        )
    if source == "web_catalog" and "pyqt" in surfaces:
        raise TileRegistryError(
            f"tile {tile_id!r}: web_catalog entries cannot declare the pyqt surface"
        )
    return surfaces, surface_reason


def _parse_hidden(
    entry: Mapping[str, Any], tile_id: str, maturity: str
) -> tuple[bool, str | None, str | None]:
    hidden = bool(entry.get("hidden", False))
    if hidden != (maturity == "hidden"):
        raise TileRegistryError(
            f"tile {tile_id!r}: 'hidden: true' and 'maturity: hidden' must agree "
            f"(hidden={hidden}, maturity={maturity!r})"
        )
    return (
        hidden,
        _string(entry, "hidden_reason", tile_id, required=hidden),
        _string(entry, "hidden_owner", tile_id, required=hidden),
    )


def _parse_launch_target(
    entry: Mapping[str, Any],
    tile_id: str,
    *,
    source: str,
    launcher: Mapping[str, Any],
) -> tuple[str, str | None, dict[str, str]]:
    path = _string(entry, "path", tile_id, required=False) or ""
    entry_type = _string(entry, "type", tile_id, required=True) or ""
    if not path and entry_type in METADATA_ONLY_TYPES:
        # Mirrors ModelRegistry._normalize_legacy_model_entry: pathless
        # exercise presets / PINN modes get a virtual namespace dispatched by
        # dedicated launcher handlers.
        path = f"virtual/{entry_type}/{tile_id}"
    if source == "models" and not path:
        raise TileRegistryError(f"tile {tile_id!r}: native tiles need a non-empty path")
    launcher_route = launcher.get("web_route")
    web_route = _string(entry, "web_route", tile_id, required=False) or (
        launcher_route.strip() if isinstance(launcher_route, str) else None
    )
    web = _parse_web(entry, tile_id, path=path, web_route=web_route)
    if web["mode"] == "route" and not web_route:
        web_route = web["route"]
    return path, web_route, web


def _parse_tile(entry: Mapping[str, Any], *, source: str) -> TileRecord:
    if not isinstance(entry, Mapping):
        raise TileRegistryError(
            f"{source} entries must be mappings, got {type(entry).__name__}"
        )
    tile_id = entry.get("id")
    if not isinstance(tile_id, str) or not tile_id.strip():
        raise TileRegistryError(f"{source} entry without a non-empty 'id': {entry!r}")
    tile_id = tile_id.strip()
    launcher = _parse_launcher(entry, tile_id)

    maturity = entry.get("maturity")
    if maturity not in MATURITY_LEVELS:
        raise TileRegistryError(
            f"tile {tile_id!r}: maturity must be one of {sorted(MATURITY_LEVELS)}, "
            f"got {maturity!r}"
        )
    surfaces, surface_reason = _parse_surfaces(entry, tile_id, source=source)
    hidden, hidden_reason, hidden_owner = _parse_hidden(entry, tile_id, str(maturity))

    order = entry.get("order")
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
        raise TileRegistryError(
            f"tile {tile_id!r}: 'order' must be a non-negative integer"
        )
    if "feature_id" not in entry or "help" not in entry:
        raise TileRegistryError(
            f"tile {tile_id!r}: 'feature_id' and 'help' must be declared (null allowed)"
        )
    path, web_route, web = _parse_launch_target(
        entry, tile_id, source=source, launcher=launcher
    )

    return TileRecord(
        id=tile_id,
        name=_string(entry, "name", tile_id, required=True) or "",
        description=_string(entry, "description", tile_id, required=True) or "",
        type=_string(entry, "type", tile_id, required=True) or "",
        path=path,
        category=str(launcher["category"]).strip(),
        logo=str(launcher["logo"]).strip(),
        status=str(launcher["status"]).strip(),
        order=order,
        maturity=str(maturity),
        surfaces=surfaces,
        web=web,
        source=source,
        help=_string(entry, "help", tile_id, required=False),
        feature_id=_string(entry, "feature_id", tile_id, required=False),
        surface_reason=surface_reason,
        maturity_reason=_string(entry, "maturity_reason", tile_id, required=False),
        web_route=web_route,
        web_logo=_string(entry, "web_logo", tile_id, required=False),
        default_launch=str(launcher.get("default_launch", "tab")),
        capabilities=_string_tuple(entry, "capabilities", tile_id),
        tags=_string_tuple(entry, "tags", tile_id),
        shell_surfaces=_string_tuple(entry, "shell_surfaces", tile_id),
        engine_type=_string(entry, "engine_type", tile_id, required=False),
        provider=_string(entry, "provider", tile_id, required=False),
        source_root=_string(entry, "source_root", tile_id, required=False),
        working_dir=_string(entry, "working_dir", tile_id, required=False),
        python_paths=_string_tuple(entry, "python_paths", tile_id),
        embed_adapter=_string(entry, "embed_adapter", tile_id, required=False),
        hidden=hidden,
        hidden_reason=hidden_reason,
        hidden_owner=hidden_owner,
    )


def _parse_exclusions(raw: Any) -> tuple[ExcludedPackage, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TileRegistryError("'excluded_packages' must be a list")
    out: list[ExcludedPackage] = []
    for entry in raw:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("package"), str):
            raise TileRegistryError(
                f"excluded_packages entry needs a 'package': {entry!r}"
            )
        reason = entry.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            raise TileRegistryError(
                f"excluded package {entry['package']!r} needs a substantive reason (>= 20 chars)"
            )
        issue = entry.get("issue")
        out.append(
            ExcludedPackage(
                package=entry["package"].strip(),
                reason=" ".join(reason.split()),
                issue=str(issue).strip() if issue is not None else None,
            )
        )
    return tuple(out)


def parse_tile_registry(
    data: Mapping[str, Any], *, path: Path = REGISTRY_PATH
) -> TileRegistry:
    """Build a validated :class:`TileRegistry` from parsed YAML."""
    if not isinstance(data, Mapping) or "models" not in data:
        raise TileRegistryError(f"{path}: root must be a mapping with a 'models' list")
    models_raw = data.get("models") or []
    web_raw = data.get("web_catalog") or []
    if not isinstance(models_raw, list) or not isinstance(web_raw, list):
        raise TileRegistryError(f"{path}: 'models' and 'web_catalog' must be lists")
    tiles = [_parse_tile(entry, source="models") for entry in models_raw]
    tiles += [_parse_tile(entry, source="web_catalog") for entry in web_raw]

    seen: dict[str, str] = {}
    for tile in tiles:
        if tile.id in seen:
            raise TileRegistryError(
                f"duplicate tile id {tile.id!r} ({seen[tile.id]} and {tile.source})"
            )
        seen[tile.id] = tile.source
    orders: dict[int, str] = {}
    for tile in tiles:
        if tile.order in orders:
            raise TileRegistryError(
                f"duplicate order {tile.order} for {orders[tile.order]!r} and {tile.id!r}"
            )
        orders[tile.order] = tile.id
    return TileRegistry(
        tiles=tuple(tiles),
        excluded_packages=_parse_exclusions(data.get("excluded_packages")),
        path=path,
    )


def load_tile_registry(path: Path | None = None) -> TileRegistry:
    """Load and validate ``models.yaml`` (or an override path)."""
    registry_path = Path(path) if path is not None else REGISTRY_PATH
    if not registry_path.is_file():
        raise FileNotFoundError(f"tile registry not found: {registry_path}")
    with open(registry_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return parse_tile_registry(data or {}, path=registry_path)


def registry_ids(tiles: Iterable[TileRecord]) -> set[str]:
    return {tile.id for tile in tiles}


__all__ = [
    "MATURITY_LEVELS",
    "REGISTRY_PATH",
    "SURFACES",
    "WEB_MODES",
    "ExcludedPackage",
    "TileRecord",
    "TileRegistry",
    "TileRegistryError",
    "load_tile_registry",
    "parse_tile_registry",
    "web_logo_for",
]
