"""StandaloneSidekickWindow — standalone QMainWindow shell — T2 (#5980).

Hosts AIAssistantPanel and UnifiedToolsSidebar in a two-pane splitter
layout.  Profile ``chat-first`` puts chat at 60 % left, sidebar at 40 %
right.  Profile ``calc-first`` reverses the ratio.

If a panel fails to construct (e.g. chat service unreachable), an inline
placeholder label is shown instead of crashing.

The module depends only on ``sidekick.*`` and
``ai.gui.assistant_panel.AIAssistantPanel`` — never on ``src.launchers.*``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QCloseEvent, QShowEvent
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QWidget,
)

logger = logging.getLogger(__name__)

__all__ = ["StandaloneSidekickConfig", "StandaloneSidekickWindow"]

_VALID_PROFILES = frozenset({"chat-first", "calc-first"})
_PANEL_FALLBACK_ERRORS = (
    ImportError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
)

# Documented splitter ratio: primary pane gets this fraction of the total width.
_PRIMARY_RATIO = 0.60


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandaloneSidekickConfig:
    """Frozen configuration for StandaloneSidekickWindow.

    Attributes:
        profile: Layout profile (``'chat-first'`` or ``'calc-first'``).
        theme_name: Optional theme name; ``None`` uses the default theme.
        session_store: StandaloneSessionStore instance for profile persistence.
        host_action_port: Optional HostActionPort for T5 embedded/standalone
            round-trip; ``None`` disables the integration.
    """

    profile: str
    theme_name: str | None
    session_store: Any
    host_action_port: Any = field(default=None)

    def __post_init__(self) -> None:
        if self.profile not in _VALID_PROFILES:
            raise ValueError(
                f"Invalid profile {self.profile!r}. "
                f"Allowed values: {sorted(_VALID_PROFILES)}"
            )


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


class StandaloneSidekickWindow(QMainWindow):
    """Standalone Sidekick application window.

    Args:
        config: Frozen window configuration.

    Postconditions after ``__init__``:
        ``self.windowTitle() == "Sidekick"``
    """

    def __init__(self, config: StandaloneSidekickConfig) -> None:
        if not isinstance(config, StandaloneSidekickConfig):
            raise TypeError(
                f"config must be StandaloneSidekickConfig, "
                f"got {type(config).__name__!r}"
            )
        super().__init__()
        self._config = config
        self._layout_applied = False

        self._build_central_widget()
        self._install_menu_bar()
        self.setWindowTitle("Sidekick")

    # ---- public accessors ------------------------------------------------

    def splitter_handle_positions(self) -> list[int]:
        """Return splitter panel widths ``[left_px, right_px]``.

        Used by tests to verify the layout ratio.
        """
        return list(self._splitter.sizes())

    def panel_for(self, profile: str) -> QWidget:
        """Return the primary content widget for the given profile name.

        Raises:
            ValueError: If ``profile`` is not a known profile name.
        """
        if profile == "chat-first":
            return self._chat_panel
        if profile == "calc-first":
            return self._sidebar_panel
        raise ValueError(f"Unknown profile: {profile!r}")

    def sidebar(self) -> QWidget:
        """Return the sidebar (UnifiedToolsSidebar or placeholder) widget."""
        return self._sidebar_panel

    # ---- Qt overrides ----------------------------------------------------

    def showEvent(self, event: QShowEvent | None) -> None:
        super().showEvent(event)
        if not self._layout_applied:
            self._apply_ratio()
            self._layout_applied = True

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self._flush_session()
        super().closeEvent(event)

    # ---- internals -------------------------------------------------------

    def _build_central_widget(self) -> None:
        self._chat_panel = self._create_chat_panel()
        self._sidebar_panel = self._create_sidebar_panel()

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        if self._config.profile == "chat-first":
            self._splitter.addWidget(self._chat_panel)
            self._splitter.addWidget(self._sidebar_panel)
        else:  # calc-first
            self._splitter.addWidget(self._sidebar_panel)
            self._splitter.addWidget(self._chat_panel)

        # Initial sizes based on the nominal 1280 px width so that
        # splitter_handle_positions() is meaningful before the first showEvent.
        self._set_splitter_sizes(1280)
        self.setCentralWidget(self._splitter)

    def _apply_ratio(self) -> None:
        """Recalculate splitter sizes based on the actual window width."""
        total = self._splitter.width()
        if total > 0:
            self._set_splitter_sizes(total)

    def _set_splitter_sizes(self, total: int) -> None:
        left = int(total * _PRIMARY_RATIO)
        right = total - left
        self._splitter.setSizes([left, right])

    def _create_chat_panel(self) -> QWidget:
        try:
            from ai.gui.assistant_panel import AIAssistantPanel

            return AIAssistantPanel()
        except _PANEL_FALLBACK_ERRORS:
            logger.exception("Could not construct AIAssistantPanel; using placeholder")
            return _placeholder("Chat (unavailable)")

    def _create_sidebar_panel(self) -> QWidget:
        try:
            from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

            return UnifiedToolsSidebar()
        except _PANEL_FALLBACK_ERRORS:
            logger.exception(
                "Could not construct UnifiedToolsSidebar; using placeholder"
            )
            return _placeholder("Sidebar (unavailable)")

    def _install_menu_bar(self) -> None:
        bar = self.menuBar()
        assert bar is not None

        file_menu = bar.addMenu("&File")
        assert file_menu is not None
        file_menu.addAction(_action("Save profile", self, self._on_save_profile))
        file_menu.addAction(_action("Load profile", self, self._on_load_profile))
        file_menu.addSeparator()
        file_menu.addAction(_action("Quit", self, self.close))

        view_menu = bar.addMenu("&View")
        assert view_menu is not None
        view_menu.addAction(
            _action(
                "Chat-first layout", self, lambda: self._switch_profile("chat-first")
            )
        )
        view_menu.addAction(
            _action(
                "Calc-first layout", self, lambda: self._switch_profile("calc-first")
            )
        )
        view_menu.addSeparator()
        view_menu.addAction(_action("Toggle sidebar", self, self._toggle_sidebar))

        help_menu = bar.addMenu("&Help")
        assert help_menu is not None
        help_menu.addAction(_action("About Sidekick", self, self._on_about))

    def _flush_session(self) -> None:
        try:
            store = self._config.session_store
            if store is not None and hasattr(store, "set_last_profile"):
                store.set_last_profile(self._config.profile)
        except (OSError, RuntimeError, TypeError, ValueError):
            logger.exception("Failed to flush session on close")

    def _on_save_profile(self) -> None:
        logger.info("Save profile triggered (UI not yet implemented — T8)")

    def _on_load_profile(self) -> None:
        logger.info("Load profile triggered (UI not yet implemented — T8)")

    def _switch_profile(self, profile: str) -> None:
        logger.info(
            "Switch profile to %r (re-layout not yet implemented — T8)", profile
        )

    def _toggle_sidebar(self) -> None:
        self._sidebar_panel.setVisible(not self._sidebar_panel.isVisible())

    def _on_about(self) -> None:
        QMessageBox.about(self, "About Sidekick", "Sidekick — standalone edition.")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _placeholder(label: str) -> QWidget:
    w = QLabel(label)
    w.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return w


def _action(text: str, parent: QWidget, slot: Any) -> QAction:
    act = QAction(text, parent)
    act.triggered.connect(slot)
    return act
