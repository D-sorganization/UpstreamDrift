"""UI Components for the UpstreamDrift Launcher.

This module re-exports all public symbols from the decomposed sub-modules
for backward compatibility.  New code should import directly from:

- ``src.launchers.startup``        – ASSETS_DIR, REPOS_ROOT, StartupResults,
                                     GolfSplashScreen, AsyncStartupWorker,
                                     _get_theme_colors
- ``src.launchers.model_card``     – MODEL_IMAGES, DraggableModelCard
- ``src.launchers.docker_dialog``  – DockerCheckThread, EnvironmentDialog
- ``src.launchers.settings_dialog``– SettingsDialog
- ``src.launchers.help_dialogs``   – HelpDialog, LayoutManagerDialog,
                                     ContextHelpDock
"""

from __future__ import annotations

# ── Docker dialog ───────────────────────────────────────────────────
from src.launchers.docker_dialog import (
    DockerCheckThread,
    EnvironmentDialog,
)

# ── Help dialogs ────────────────────────────────────────────────────
from src.launchers.help_dialogs import (
    ContextHelpDock,
    HelpDialog,
    LayoutManagerDialog,
)

# ── Model card ──────────────────────────────────────────────────────
from src.launchers.model_card import (
    MODEL_IMAGES,
    DraggableModelCard,
)

# ── Settings dialog ─────────────────────────────────────────────────
from src.launchers.settings_dialog import SettingsDialog

# ── Startup components ──────────────────────────────────────────────
from src.launchers.startup import (
    ASSETS_DIR,
    REPOS_ROOT,
    AsyncStartupWorker,
    GolfSplashScreen,
    StartupResults,
    _get_theme_colors,
)

__all__ = [
    "ASSETS_DIR",
    "REPOS_ROOT",
    "MODEL_IMAGES",
    "_get_theme_colors",
    "StartupResults",
    "GolfSplashScreen",
    "AsyncStartupWorker",
    "DraggableModelCard",
    "DockerCheckThread",
    "EnvironmentDialog",
    "SettingsDialog",
    "HelpDialog",
    "LayoutManagerDialog",
    "ContextHelpDock",
]
