"""Thin adapters that derive the legacy registries from the single tile registry.

Issue #9412: ``GUIRegistry`` (``src.shared.python.gui_launcher``), the
``EMBEDDABLE_TOOL_REGISTRY`` (``src.shared.python.launcher_embed``) and the
runtime ``feature_registry`` are Tools-owned modules whose public APIs stay
untouched. This UpstreamDrift-owned module feeds them from
``src/config/models.yaml`` so no second hand-maintained tile list exists.

* :func:`filter_desktop_models` — the PyQt launcher's model list restricted to
  tiles declared for the ``pyqt`` surface.
* :func:`populate_gui_registry` — registers every desktop tile with the
  ``GUIRegistry`` singleton (``register()`` is the unchanged public API).
* :func:`embed_adapter_modules` — dotted module paths of every tile-declared
  ``embed_adapter`` so the embeddable-tool bootstrap imports what the
  registry says, not a parallel list.
* :func:`feature_names_by_tile` — maps engine tiles onto the runtime
  ``feature_registry`` feature names (validated against ``FEATURES``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.config.tile_registry import (
    TileRecord,
    TileRegistry,
    load_tile_registry,
)

logger = logging.getLogger(__name__)

# engine_type (models.yaml) -> feature name (src.shared.python.feature_registry)
_ENGINE_FEATURE_NAMES: dict[str, str | None] = {
    "mujoco": "mujoco",
    "drake": "drake",
    "pinocchio": "pinocchio",
    "opensim": "opensim",
    "myosuite": "myosuite",
    # No runtime feature probe exists yet for these engine types.
    "jaxsim": None,
    "matlab": None,
    "golf_simulation": None,
    "putting_green": None,
}


def _registry(registry: TileRegistry | None) -> TileRegistry:
    return registry if registry is not None else load_tile_registry()


def desktop_tile_ids(
    registry: TileRegistry | None = None, *, include_hidden: bool = True
) -> set[str]:
    """Tile ids the PyQt launcher may show (hidden aliases included by default)."""
    return _registry(registry).surface_ids("pyqt", include_hidden=include_hidden)


def filter_desktop_models(
    models: Iterable[Any], registry: TileRegistry | None = None
) -> list[Any]:
    """Keep the ``ModelRegistry`` models that belong on the desktop surface.

    Models the tile registry does not know (provider-discovered sibling-repo
    packs) pass through unchanged; models it knows are kept only when they
    declare the ``pyqt`` surface.
    """
    reg = _registry(registry)
    known = {tile.id for tile in reg.tiles}
    allowed = reg.surface_ids("pyqt", include_hidden=True)
    kept: list[Any] = []
    for model in models:
        model_id = getattr(model, "id", None)
        if model_id in known and model_id not in allowed:
            logger.info(
                "Tile %s is not declared for the pyqt surface; skipped", model_id
            )
            continue
        kept.append(model)
    return kept


def _module_path_for(path: str) -> str | None:
    """Dotted module for a repo-relative ``.py`` path, else None."""
    if not path.endswith(".py") or path.startswith("virtual/"):
        return None
    parts = Path(path).with_suffix("").parts
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def populate_gui_registry(
    registry: TileRegistry | None = None, gui_registry: Any | None = None
) -> list[str]:
    """Register every desktop tile with the ``GUIRegistry`` singleton.

    Returns the registered tool names. Re-registration overwrites, so calling
    this more than once is safe.
    """
    from src.shared.python.gui_launcher.launcher import GUIType, LaunchConfig
    from src.shared.python.gui_launcher.registry import get_registry

    reg = _registry(registry)
    target = gui_registry if gui_registry is not None else get_registry()
    registered: list[str] = []
    for tile in reg.tiles:
        if not tile.on_surface("pyqt") or not tile.visible:
            continue
        config = LaunchConfig(
            tool_name=tile.id,
            gui_type=GUIType.PYQT6,
            module_path=_module_path_for(tile.path),
            entry_point=tile.path or None,
            working_dir=tile.working_dir,
            title=tile.name,
        )
        target.register(
            tool_name=tile.id,
            display_name=tile.name,
            description=tile.description or tile.name,
            gui_configs={GUIType.PYQT6: config},
            category=tile.category,
            icon=tile.logo,
            repository=tile.provider or "upstreamdrift",
        )
        registered.append(tile.id)
    return registered


def embed_adapter_modules(registry: TileRegistry | None = None) -> list[str]:
    """Dotted module paths of every ``embed_adapter`` declared in the registry."""
    modules: list[str] = []
    for tile in _registry(registry).tiles:
        if not tile.embed_adapter:
            continue
        module_path = _module_path_for(tile.embed_adapter.split("::", 1)[0])
        if module_path and module_path not in modules:
            modules.append(module_path)
    return modules


def feature_names_by_tile(
    registry: TileRegistry | None = None,
) -> dict[str, str | None]:
    """Map every engine tile to its runtime ``feature_registry`` feature name.

    Raises:
        KeyError: when a tile declares an ``engine_type`` this adapter does
            not know, or the mapped feature is not in ``FEATURES``.
    """
    from src.shared.python.feature_registry.features import FEATURES

    known_features = {feature.name for feature in FEATURES}
    mapping: dict[str, str | None] = {}
    for tile in _registry(registry).tiles:
        if not tile.engine_type:
            continue
        if tile.engine_type not in _ENGINE_FEATURE_NAMES:
            raise KeyError(
                f"tile {tile.id!r} declares engine_type {tile.engine_type!r}, "
                "unknown to src.config.registry_adapters._ENGINE_FEATURE_NAMES"
            )
        feature = _ENGINE_FEATURE_NAMES[tile.engine_type]
        if feature is not None and feature not in known_features:
            raise KeyError(
                f"tile {tile.id!r} maps to feature {feature!r} missing from FEATURES"
            )
        mapping[tile.id] = feature
    return mapping


def tiles_for_surface(
    surface: str, registry: TileRegistry | None = None
) -> list[TileRecord]:
    """Visible tiles declared for ``surface`` in display order."""
    reg = _registry(registry)
    ids = reg.surface_ids(surface)
    return sorted((t for t in reg.tiles if t.id in ids), key=lambda t: (t.order, t.id))


__all__ = [
    "desktop_tile_ids",
    "embed_adapter_modules",
    "feature_names_by_tile",
    "filter_desktop_models",
    "populate_gui_registry",
    "tiles_for_surface",
]
