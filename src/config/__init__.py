"""Launcher registry configuration.

Attributes of :mod:`src.config.launcher_manifest_loader` are re-exported
lazily (PEP 562) so that the dependency-light registry modules
(:mod:`src.config.tile_registry`, :mod:`src.config.feature_parity_loader`)
can be imported without pulling in the launcher/provider stack. The public
surface is unchanged: ``from src.config import LauncherManifest`` still works.
"""

from __future__ import annotations

from typing import Any

_LOADER_EXPORTS = frozenset(
    {
        "ASSETS_DIR",
        "LAUNCHER_CATEGORIES",
        "LAUNCHER_CATEGORY_LABELS",
        "MANIFEST_PATH",
        "REGISTRY_PATH",
        "TOOL_LIKE_CATEGORIES",
        "LauncherManifest",
        "LauncherTile",
    }
)

__all__: list[str] = [
    "ASSETS_DIR",
    "LAUNCHER_CATEGORIES",
    "LAUNCHER_CATEGORY_LABELS",
    "MANIFEST_PATH",
    "REGISTRY_PATH",
    "TOOL_LIKE_CATEGORIES",
    "LauncherManifest",
    "LauncherTile",
]


def __getattr__(name: str) -> Any:
    if name in _LOADER_EXPORTS:
        from . import launcher_manifest_loader

        return getattr(launcher_manifest_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
