"""Launcher manifest configuration."""

from .launcher_manifest_loader import (
    ASSETS_DIR,
    LAUNCHER_CATEGORIES,
    LAUNCHER_CATEGORY_LABELS,
    MANIFEST_PATH,
    REGISTRY_PATH,
    TOOL_LIKE_CATEGORIES,
    LauncherManifest,
    LauncherTile,
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
    "CLINotInteractiveGUIError",
    "IncompleteCapabilityRecord",
    "ResearchCapabilityLifecycleManager",
    "audit_research_and_excluded_capabilities",
]

from .research_capability_lifecycle import (
    CLINotInteractiveGUIError,
    IncompleteCapabilityRecord,
    ResearchCapabilityLifecycleManager,
    audit_research_and_excluded_capabilities,
)
