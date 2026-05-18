"""Panel pop-out manager for the UpstreamDrift Unified Tools Sidebar.

Implements issue #5380 — sidebar panels can be detached into a floating
``QDialog`` and re-docked. Pop-out state is persisted to a JSON file so
the arrangement survives across application restarts.

Design: TDD, DbC (preconditions enforced), LOD (delegates to helpers),
DRY (shared state serialisation via :class:`PopoutState`).

Usage::

    from src.shared.python.gui_launcher.panel_popout_manager import (
        PanelPopoutManager,
    )

    mgr = PanelPopoutManager(state_file=Path("~/.upstreamdrift/popout.json"))
    mgr.register_panel("tools", sidebar_widget)
    mgr.apply_saved_state()   # restore previous layout
    ...
    dialog = mgr.popout("tools")   # float it
    mgr.redock("tools")            # bring it back
    mgr.save_state()               # persist before exit
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class PanelNotRegisteredError(KeyError):
    """Raised when a panel_id is not in the registry.

    Inherits ``KeyError`` so callers can use a bare ``except KeyError`` for
    backward compatibility.
    """

    def __init__(self, panel_id: str) -> None:
        super().__init__(panel_id)
        self.panel_id = panel_id

    def __str__(self) -> str:
        return f"Panel not registered: {self.panel_id!r}"


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


@dataclass
class PopoutState:
    """Serialisable geometry record for one panel.

    Attributes:
        panel_id: Stable string identifier that matches the registration key.
        is_floating: Whether the panel is detached as a floating dialog.
        x: Left edge position of the floating dialog in screen coordinates.
        y: Top edge position of the floating dialog in screen coordinates.
        width: Width of the floating dialog in pixels.
        height: Height of the floating dialog in pixels.
    """

    panel_id: str
    is_floating: bool = False
    x: int = 0
    y: int = 0
    width: int = 400
    height: int = 600

    # Postcondition: to_dict() always returns a dict with all keys.
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON persistence.

        Returns:
            A dict with keys ``panel_id``, ``is_floating``, ``x``, ``y``,
            ``width``, ``height``.
        """
        return {
            "panel_id": self.panel_id,
            "is_floating": self.is_floating,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PopoutState:
        """Deserialise from a plain dict.

        Args:
            data: Dictionary with at least ``panel_id`` and ``is_floating``.

        Returns:
            A new :class:`PopoutState`.

        Raises:
            KeyError: If ``panel_id`` is absent.
            ValueError: If ``panel_id`` is empty.
        """
        panel_id = data["panel_id"]  # raises KeyError if missing
        if not panel_id:
            raise ValueError("panel_id must not be empty")
        return cls(
            panel_id=panel_id,
            is_floating=bool(data.get("is_floating", False)),
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            width=int(data.get("width", 400)),
            height=int(data.get("height", 600)),
        )


# ---------------------------------------------------------------------------
# Internal helpers (kept at module level so tests can patch them)
# ---------------------------------------------------------------------------


def _create_float_dialog(panel_id: str, panel: Any, parent: Any = None) -> Any:
    """Create and show a floating ``QDialog`` containing ``panel``.

    Preconditions:
        ``panel_id`` is a non-empty string.
        ``panel`` is not ``None``.

    Args:
        panel_id: Identifier used as the dialog window title.
        panel: The widget to embed in the dialog.
        parent: Optional Qt parent widget.

    Returns:
        The newly created ``QDialog`` instance (already shown).
    """
    from PyQt6.QtWidgets import QDialog, QVBoxLayout  # noqa: PLC0415

    dialog = QDialog(parent)
    dialog.setWindowTitle(panel_id)
    dialog.setObjectName(f"popout_{panel_id}")

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(panel)
    dialog.setLayout(layout)
    dialog.show()
    return dialog


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


@dataclass
class _PanelEntry:
    """Internal record for a registered panel."""

    panel_id: str
    widget: Any
    dialog: Any = None  # non-None iff floating


class PanelPopoutManager:
    """Manager that detaches and re-docks sidebar panels.

    Panels must be :meth:`register_panel`-ed before :meth:`popout` or
    :meth:`redock` may be called (DbC precondition enforced by
    :class:`PanelNotRegisteredError`).

    Args:
        state_file: Optional path to a JSON file for persisting pop-out
            state across sessions. When ``None``, persistence is disabled
            and :meth:`save_state` / :meth:`load_state` are no-ops.
        parent: Optional Qt parent widget passed to floating dialogs.
    """

    def __init__(
        self,
        *,
        state_file: Path | None = None,
        parent: Any = None,
    ) -> None:
        self._panels: dict[str, _PanelEntry] = {}
        self._state_file = state_file
        self._parent = parent

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_panel(self, panel_id: str, widget: Any) -> None:
        """Register a sidebar panel widget.

        Preconditions:
            ``panel_id`` is a non-empty string.
            ``widget`` is not ``None``.

        Args:
            panel_id: Stable identifier (used as dict key and dialog title).
            widget: The Qt widget representing the panel.

        Raises:
            ValueError: If ``panel_id`` is empty.
            TypeError: If ``widget`` is ``None``.
        """
        if not panel_id:
            raise ValueError("panel_id must be a non-empty string")
        if widget is None:
            raise TypeError("widget must not be None")

        self._panels[panel_id] = _PanelEntry(panel_id=panel_id, widget=widget)
        logger.debug("PanelPopoutManager: registered panel %r", panel_id)

    def unregister_panel(self, panel_id: str) -> None:
        """Remove a panel from the registry.

        If the panel is currently floating, the dialog is closed first.
        Calling this with an unknown ``panel_id`` is a no-op.

        Args:
            panel_id: Panel to remove.
        """
        entry = self._panels.pop(panel_id, None)
        if entry is not None and entry.dialog is not None:
            _close_dialog(entry.dialog)
        logger.debug("PanelPopoutManager: unregistered panel %r", panel_id)

    def is_registered(self, panel_id: str) -> bool:
        """Return ``True`` if ``panel_id`` is registered."""
        return panel_id in self._panels

    def get_panel(self, panel_id: str) -> Any | None:
        """Return the widget for ``panel_id``, or ``None`` if not registered."""
        entry = self._panels.get(panel_id)
        return entry.widget if entry is not None else None

    # ------------------------------------------------------------------
    # Pop-out / re-dock
    # ------------------------------------------------------------------

    def popout(self, panel_id: str) -> Any:
        """Detach a panel into a floating ``QDialog``.

        If the panel is already floating, the existing dialog is returned
        without creating a second one (idempotent).

        Preconditions:
            ``panel_id`` must be registered.

        Args:
            panel_id: Panel to float.

        Returns:
            The floating ``QDialog`` instance.

        Raises:
            PanelNotRegisteredError: If ``panel_id`` is not registered.
        """
        entry = self._require_registered(panel_id)

        if entry.dialog is not None:
            # Already floating — return the existing dialog (idempotent).
            return entry.dialog

        dialog = _create_float_dialog(panel_id, entry.widget, parent=self._parent)
        entry.dialog = dialog
        logger.info("PanelPopoutManager: popped out panel %r", panel_id)
        return dialog

    def redock(self, panel_id: str) -> None:
        """Re-dock a floating panel back into its original container.

        If the panel is not floating, this is a no-op.

        Preconditions:
            ``panel_id`` must be registered.

        Args:
            panel_id: Panel to re-dock.

        Raises:
            PanelNotRegisteredError: If ``panel_id`` is not registered.
        """
        entry = self._require_registered(panel_id)

        if entry.dialog is None:
            return  # already docked

        _close_dialog(entry.dialog)
        entry.dialog = None
        logger.info("PanelPopoutManager: redocked panel %r", panel_id)

    def is_floating(self, panel_id: str) -> bool:
        """Return ``True`` if ``panel_id`` is currently floating.

        Preconditions:
            ``panel_id`` must be registered.

        Args:
            panel_id: Panel to query.

        Returns:
            ``True`` if the panel has a live floating dialog.

        Raises:
            PanelNotRegisteredError: If ``panel_id`` is not registered.
        """
        entry = self._require_registered(panel_id)
        return entry.dialog is not None

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist the current pop-out geometry to :attr:`state_file`.

        If no ``state_file`` was configured, this is a no-op.
        """
        if self._state_file is None:
            return

        state: dict[str, Any] = {}
        for panel_id, entry in self._panels.items():
            ps = self._build_panel_state(panel_id, entry)
            state[panel_id] = ps.to_dict()

        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
            logger.debug("PanelPopoutManager: saved state to %s", self._state_file)
        except OSError as exc:
            logger.warning("PanelPopoutManager: could not save state: %s", exc)

    def load_state(self) -> dict[str, PopoutState]:
        """Load persisted pop-out state from :attr:`state_file`.

        Returns:
            A dict mapping ``panel_id`` to :class:`PopoutState`, or an
            empty dict if the file is absent or unparseable.
        """
        if self._state_file is None or not self._state_file.exists():
            return {}

        try:
            raw = self._state_file.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("PanelPopoutManager: could not load state: %s", exc)
            return {}

        result: dict[str, PopoutState] = {}
        for panel_id, entry_data in data.items():
            try:
                result[panel_id] = PopoutState.from_dict(entry_data)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "PanelPopoutManager: skipping corrupt state for %r: %s",
                    panel_id,
                    exc,
                )

        return result

    def apply_saved_state(self) -> None:
        """Load and apply persisted state, re-floating panels as needed.

        Only panels that are currently registered AND were floating in
        the saved state are re-floated. Unknown panel IDs are ignored.
        """
        saved = self.load_state()
        for panel_id, state in saved.items():
            if not state.is_floating:
                continue
            if panel_id not in self._panels:
                logger.debug(
                    "PanelPopoutManager: saved state references unregistered "
                    "panel %r; skipping",
                    panel_id,
                )
                continue
            dialog = self.popout(panel_id)
            _apply_geometry(dialog, state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_registered(self, panel_id: str) -> _PanelEntry:
        """Return the entry for ``panel_id``.

        Raises:
            PanelNotRegisteredError: If ``panel_id`` is not in the registry.
        """
        entry = self._panels.get(panel_id)
        if entry is None:
            raise PanelNotRegisteredError(panel_id)
        return entry

    def _build_panel_state(self, panel_id: str, entry: _PanelEntry) -> PopoutState:
        """Build a :class:`PopoutState` from a live panel entry."""
        if entry.dialog is None:
            return PopoutState(panel_id=panel_id, is_floating=False)

        dialog = entry.dialog
        x = _call_int(dialog, "x")
        y = _call_int(dialog, "y")
        width = _call_int(dialog, "width")
        height = _call_int(dialog, "height")
        return PopoutState(
            panel_id=panel_id,
            is_floating=True,
            x=x,
            y=y,
            width=width,
            height=height,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (LOD: single-step delegation)
# ---------------------------------------------------------------------------


def _call_int(obj: Any, method_name: str) -> int:
    """Call ``obj.<method_name>()`` and return an int, defaulting to 0."""
    method = getattr(obj, method_name, None)
    if callable(method):
        try:
            return int(method())
        except (TypeError, ValueError):
            pass
    return 0


def _close_dialog(dialog: Any) -> None:
    """Close a dialog object safely (swallows errors)."""
    close = getattr(dialog, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - defensive shutdown
            logger.exception("PanelPopoutManager: error closing dialog")


def _apply_geometry(dialog: Any, state: PopoutState) -> None:
    """Move and resize ``dialog`` to match ``state`` geometry."""
    move = getattr(dialog, "move", None)
    resize = getattr(dialog, "resize", None)
    if callable(move):
        with contextlib.suppress(TypeError, AttributeError):
            move(state.x, state.y)
    if callable(resize):
        with contextlib.suppress(TypeError, AttributeError):
            resize(state.width, state.height)
