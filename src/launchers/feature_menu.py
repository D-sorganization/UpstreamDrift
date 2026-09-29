"""Feature menu — Window-menu entries for surfaced Sidekick features.

This module declares :class:`FeatureMenuEntry` records and exposes
:func:`build_feature_menu_entries` plus :func:`register_feature_menu`
helpers used by the launcher to surface newly added Sidekick features
(OS terminal, Python REPL, MATLAB workspace, Jupyter, MCP servers).

Design goals (see task plan):

* **Single source of truth** for the feature-id to shortcut mapping —
  the keyboard shortcuts in :mod:`launcher_dialogs` and the prefs
  subpages both read from :data:`FEATURE_ENTRIES`.
* **Auto-hide unavailable features** — Jupyter is only listed when
  ``nbformat`` imports cleanly; the rest gracefully accept missing
  Sidekick backends (the menu entry shows a status-tip explaining the
  feature is unavailable, instead of disappearing, so users discover
  what *would* be available).
* **No PyQt at import time** — the module is import-safe in headless
  contexts. PyQt symbols are imported lazily inside
  :func:`register_feature_menu`.

The factory callbacks attached to each entry are intentionally late-
bound: they receive the launcher instance and dispatch to ``getattr``
hooks (e.g. ``launcher.open_sidekick_tab(tool_id)``) the launcher
already exposes. This keeps the feature menu orthogonal — neither side
needs to know about the other's internals (LoD).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "FEATURE_ENTRIES",
    "FeatureMenuEntry",
    "build_feature_menu_entries",
    "is_feature_available",
    "register_feature_menu",
]


@dataclass(frozen=True)
class FeatureMenuEntry:
    """Declarative description of one Window-menu entry.

    Attributes:
        feature_id: Stable identifier — matches the Sidekick tab id and
            is used by :mod:`mcp_config_writer` / settings persistence.
        label: Human-readable menu label. ``&`` marks the access key.
        shortcut: Qt-style shortcut string (e.g. ``"Ctrl+Shift+T"``).
        status_tip: Status-bar text shown on hover.
        availability_probe: Callable returning ``True`` when the backing
            feature can actually be invoked. Defaults to always-True.
        factory: Callable that opens the feature. Receives the launcher
            instance. Defaults to a no-op that logs a warning so a
            missing factory never crashes the menu.
    """

    feature_id: str
    label: str
    shortcut: str
    status_tip: str
    availability_probe: Callable[[], bool] = field(default=lambda: True)
    factory: Callable[[Any], None] = field(
        default=lambda _launcher: logger.warning("feature_menu: factory not wired")
    )


# --------------------------------------------------------------------------
# Availability probes
# --------------------------------------------------------------------------


def _module_importable(name: str) -> bool:
    """Return ``True`` if *name* can be imported without side-effects.

    Uses :func:`importlib.util.find_spec` so we don't trigger heavy
    package imports just to test for availability (LoD: we don't reach
    into the module's internals).
    """
    if not name:
        raise ValueError("module name must be a non-empty string")
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _nbformat_available() -> bool:
    """Return whether the optional Jupyter ``nbformat`` package is installed."""
    return _module_importable("nbformat")


def _always_true() -> bool:
    """Probe that always reports availability (terminal/REPL/workspace/MCP)."""
    return True


# --------------------------------------------------------------------------
# Default factories — dispatch to launcher hooks via getattr (LoD)
# --------------------------------------------------------------------------


def _make_open_tab_factory(method_name: str, tool_id: str) -> Callable[[Any], None]:
    """Build a factory that calls ``launcher.<method_name>(tool_id)``.

    When the launcher does not yet expose the method, the factory logs
    a warning and shows a toast (if available) so the user sees a
    clear status. We never raise: the menu must remain usable even
    when a feature wires up later in the startup sequence.
    """
    if not method_name:
        raise ValueError("method_name must be non-empty")
    if not tool_id:
        raise ValueError("tool_id must be non-empty")

    def _factory(launcher: Any) -> None:
        if launcher is None:
            raise ValueError("launcher must be provided")
        handler = getattr(launcher, method_name, None)
        if callable(handler):
            handler(tool_id)
            return
        # Fall back to a generic dispatcher many launchers expose.
        fallback = getattr(launcher, "open_sidekick_tab", None)
        if callable(fallback):
            fallback(tool_id)
            return
        logger.warning(
            "feature_menu: launcher has no %s(...) or open_sidekick_tab(...) — "
            "feature %r cannot be opened from the menu yet",
            method_name,
            tool_id,
        )
        show_toast = getattr(launcher, "show_toast", None)
        if callable(show_toast):
            show_toast(
                f"Feature '{tool_id}' is not yet wired up in this build.",
                "warning",
            )

    return _factory


def _make_open_prefs_factory(section_id: str) -> Callable[[Any], None]:
    """Build a factory that opens the preferences dialog at *section_id*."""
    if not section_id:
        raise ValueError("section_id must be non-empty")

    def _factory(launcher: Any) -> None:
        if launcher is None:
            raise ValueError("launcher must be provided")
        opener = getattr(launcher, "open_preferences_section", None)
        if callable(opener):
            opener(section_id)
            return
        # Best-effort fallback to the generic preferences dialog.
        prefs = getattr(launcher, "_show_preferences", None)
        if callable(prefs):
            prefs()
            return
        logger.warning(
            "feature_menu: launcher has no open_preferences_section(...) or "
            "_show_preferences() — cannot open prefs section %r",
            section_id,
        )

    return _factory


# --------------------------------------------------------------------------
# The canonical entry list — single source of truth for shortcuts & ids
# --------------------------------------------------------------------------


FEATURE_ENTRIES: tuple[FeatureMenuEntry, ...] = (
    FeatureMenuEntry(
        feature_id="os_terminal",
        label="Open OS &Terminal Tab",
        shortcut="Ctrl+Shift+T",
        status_tip="Open the OS terminal as a Sidekick tab (Tools #2882)",
        availability_probe=_always_true,
        factory=_make_open_tab_factory("open_sidekick_tab", "os_terminal"),
    ),
    FeatureMenuEntry(
        feature_id="python_repl",
        label="Open Python &REPL Tab",
        shortcut="Ctrl+Shift+R",
        status_tip="Open the embedded Python REPL widget (Tools #2883)",
        availability_probe=_always_true,
        factory=_make_open_tab_factory("open_sidekick_tab", "python_repl"),
    ),
    FeatureMenuEntry(
        feature_id="workspace",
        label="Open &Workspace Tab",
        shortcut="Ctrl+Shift+W",
        status_tip="Open the MATLAB-style workspace tab (Tools #2883)",
        availability_probe=_always_true,
        factory=_make_open_tab_factory("open_sidekick_tab", "workspace"),
    ),
    FeatureMenuEntry(
        feature_id="jupyter",
        label="Open &Jupyter Tab",
        shortcut="Ctrl+Shift+J",
        status_tip="Open the embedded Jupyter notebook tab (Tools #2889)",
        availability_probe=_nbformat_available,
        factory=_make_open_tab_factory("open_sidekick_tab", "jupyter"),
    ),
    FeatureMenuEntry(
        feature_id="mcp_servers",
        label="Open &MCP Servers Settings",
        shortcut="Ctrl+Shift+M",
        status_tip="Configure MCP servers (Tools #2884)",
        availability_probe=_always_true,
        factory=_make_open_prefs_factory("mcp_servers"),
    ),
)


# --------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------


def is_feature_available(feature_id: str) -> bool:
    """Return ``True`` if the feature with *feature_id* is currently available.

    Raises:
        ValueError: If *feature_id* is empty or unknown.
    """
    if not feature_id:
        raise ValueError("feature_id must be a non-empty string")
    for entry in FEATURE_ENTRIES:
        if entry.feature_id == feature_id:
            return bool(entry.availability_probe())
    raise ValueError(f"Unknown feature_id: {feature_id!r}")


def build_feature_menu_entries(
    *, include_unavailable: bool = True
) -> list[FeatureMenuEntry]:
    """Return the list of feature entries to surface in the Window menu.

    Args:
        include_unavailable: When ``True`` (default), entries whose
            backing feature is not available are still returned so the
            menu can show them as disabled with a helpful status-tip.
            When ``False``, unavailable entries are filtered out (used
            by the auto-hide behaviour for optional features like
            Jupyter).

    Returns:
        A list of :class:`FeatureMenuEntry`, in display order.
    """
    if include_unavailable:
        return list(FEATURE_ENTRIES)
    return [e for e in FEATURE_ENTRIES if e.availability_probe()]


def register_feature_menu(
    launcher: Any,
    menubar: Any,
    *,
    menu_title: str = "&Window",
) -> dict[str, Any]:
    """Add a top-level ``Window`` menu populated with feature entries.

    The menu is inserted into *menubar* and entries dispatched via the
    launcher's ``open_sidekick_tab`` / ``open_preferences_section``
    hooks. Returns a mapping of ``feature_id`` to the created QAction
    so callers can wire shortcut overlays or inspect state in tests.

    Args:
        launcher: The QMainWindow-derived launcher instance.
        menubar: The QMenuBar to insert into.
        menu_title: Title for the new top-level menu.

    Returns:
        Dict mapping feature_id -> QAction.

    Raises:
        ValueError: If *launcher* or *menubar* is None.
    """
    if launcher is None:
        raise ValueError("launcher must be provided")
    if menubar is None:
        raise ValueError("menubar must be provided")

    from PyQt6.QtGui import QAction  # local import: PyQt is optional fleet-wide
    from PyQt6.QtWidgets import QMenu

    window_menu = QMenu(menu_title, menubar)
    before_action = None
    for action in menubar.actions():
        if action.text().replace("&", "") == "Tools":
            before_action = action
            break

    if before_action:
        menubar.insertMenu(before_action, window_menu)
    else:
        menubar.addMenu(window_menu)
    actions: dict[str, Any] = {}

    for entry in FEATURE_ENTRIES:
        available = bool(entry.availability_probe())

        # Auto-hide: Jupyter and any other optional-only feature is
        # *omitted* entirely from the menu when its probe fails — the
        # task plan calls this out explicitly. Always-available
        # entries are always shown.
        if not available and entry.feature_id == "jupyter":
            logger.debug(
                "feature_menu: hiding %s entry (probe reports unavailable)",
                entry.feature_id,
            )
            continue

        action = QAction(entry.label, launcher)
        action.setShortcut(entry.shortcut)
        action.setStatusTip(entry.status_tip)
        action.setToolTip(entry.status_tip)
        if not available:
            action.setEnabled(False)
            action.setStatusTip(f"{entry.status_tip} — currently unavailable")
        action.triggered.connect(lambda _checked=False, e=entry: e.factory(launcher))
        window_menu.addAction(action)
        actions[entry.feature_id] = action

    return actions
