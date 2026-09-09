"""Movable, resizable panes with saved layouts for the Capture Rig tile.

The tile's viewing panes (live preview, playback, results) live in
:class:`QDockWidget`\\ s of a :class:`PaneHost`, a ``QMainWindow`` used as a
child widget: the operator drags a pane to another edge, floats it onto a
second monitor, resizes it, or tabs two panes together. Each pane's content
sits in a scroll area, so a pane larger than its dock scrolls instead of
pushing the window off screen.

:class:`LayoutStore` keeps named arrangements (dock state + the controls
splitter sizes) in ``QSettings``; the tile restores the last arrangement on
start and saves it on shutdown. :class:`LayoutBar` is the small toolbar that
lists, applies, saves and deletes them. Nothing here knows what the panes
show.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from PyQt6.QtCore import QByteArray, QSettings, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QWidget,
)

from src.shared.python.core.contracts import require
from src.shared.python.theme.responsive import wrap_in_scroll_area

LAST_LAYOUT = "__last__"  # auto-saved on shutdown, never listed
EXTRAS = "extras"  # sub-group of per-layout strings the panes contribute
ORGANIZATION = "UpstreamDrift"
APPLICATION = "CaptureRig"
DOCK_FEATURES = (
    QDockWidget.DockWidgetFeature.DockWidgetMovable
    | QDockWidget.DockWidgetFeature.DockWidgetFloatable
    | QDockWidget.DockWidgetFeature.DockWidgetClosable
)


def default_settings() -> QSettings:
    return QSettings(ORGANIZATION, APPLICATION)


class LayoutStore:
    """Named window layouts in ``QSettings`` (``layouts/<name>/state|sizes``).

    Inject a ``QSettings`` bound to a temporary INI file in tests.
    """

    def __init__(self, settings: QSettings | None = None) -> None:
        self._settings = settings if settings is not None else default_settings()

    @staticmethod
    def _check_name(name: str) -> None:
        require(bool(name.strip()), "layout name must be non-empty")
        require("/" not in name, "layout name must not contain '/'", name)

    def names(self) -> list[str]:
        """Saved layout names, sorted; the auto-saved last layout is hidden."""
        self._settings.beginGroup("layouts")
        try:
            groups = self._settings.childGroups()
        finally:
            self._settings.endGroup()
        return sorted(g for g in groups if g != LAST_LAYOUT)

    def save(self, name: str, state: bytes, sizes: list[int]) -> None:
        """Store ``state`` (dock arrangement) and splitter ``sizes`` under ``name``."""
        self._check_name(name)
        require(all(s >= 0 for s in sizes), "splitter sizes must be >= 0", sizes)
        self._settings.setValue(f"layouts/{name}/state", QByteArray(state))
        self._settings.setValue(f"layouts/{name}/sizes", [int(s) for s in sizes])
        self._settings.sync()

    def load(self, name: str) -> tuple[bytes, list[int]] | None:
        """``(state, sizes)`` for ``name``; ``None`` when unknown."""
        self._check_name(name)
        raw = self._settings.value(f"layouts/{name}/state")
        if raw is None:
            return None
        state = raw.data() if isinstance(raw, QByteArray) else bytes(raw)
        sizes = self._settings.value(f"layouts/{name}/sizes", []) or []
        return state, [int(s) for s in sizes]

    def save_extras(self, name: str, extras: Mapping[str, str]) -> None:
        """Store the panes' own per-layout strings beside the dock state.

        The multiview layout each pane is drawn through (#9813/#9814) travels
        with the arrangement this way, so both come back together on restart.
        """
        self._check_name(name)
        for key, value in extras.items():
            require("/" not in key, "extras key must not contain '/'", key)
            self._settings.setValue(f"layouts/{name}/{EXTRAS}/{key}", str(value))
        self._settings.sync()

    def extras(self, name: str) -> dict[str, str]:
        """The strings saved by :meth:`save_extras`; empty when there are none."""
        self._check_name(name)
        self._settings.beginGroup(f"layouts/{name}/{EXTRAS}")
        try:
            keys = self._settings.childKeys()
            return {k: str(self._settings.value(k, "")) for k in keys}
        finally:
            self._settings.endGroup()

    def delete(self, name: str) -> None:
        self._check_name(name)
        self._settings.remove(f"layouts/{name}")
        self._settings.sync()


class PaneHost(QMainWindow):
    """Controls in the centre, viewing panes as movable/floatable docks.

    ``panes`` maps a key to ``(title, widget, area)``; every widget is wrapped
    in a scroll area (the theme package's ``wrap_in_scroll_area``) so a pane
    larger than its dock scrolls. The arrangement right after construction is
    the default that :meth:`reset` returns to.
    """

    def __init__(
        self,
        central: QWidget,
        panes: Mapping[str, tuple[str, QWidget, Qt.DockWidgetArea]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        require(bool(panes), "at least one pane is required")
        self.setWindowFlags(Qt.WindowType.Widget)  # a child widget, not a window
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )
        self.setCentralWidget(central)
        self.docks: dict[str, QDockWidget] = {}
        for key, (title, widget, area) in panes.items():
            dock = QDockWidget(title, self)
            dock.setObjectName(f"capture_rig.{key}")  # saveState needs names
            dock.setFeatures(DOCK_FEATURES)
            dock.setWidget(wrap_in_scroll_area(widget))
            self.addDockWidget(area, dock)
            self.docks[key] = dock
        self._default_state = self.state()

    def state(self) -> bytes:
        """The dock arrangement (positions, sizes, floating, visibility)."""
        return self.saveState().data()

    def restore(self, state: bytes) -> bool:
        """Apply a saved arrangement; ``False`` when it does not parse."""
        return bool(state) and self.restoreState(QByteArray(state))

    def reset(self) -> None:
        """Back to the arrangement the tile was built with; every pane shown."""
        self.restore(self._default_state)
        self.show_all()

    def show_all(self) -> None:
        for dock in self.docks.values():
            dock.show()

    def is_floating(self, key: str) -> bool:
        return self.docks[key].isFloating()

    def set_floating(self, key: str, floating: bool) -> None:
        self.docks[key].setFloating(floating)


@dataclass(frozen=True)
class PaneExtras:
    """How the bar reads and re-applies the panes' own per-layout strings.

    Keeps :class:`LayoutBar` ignorant of what a pane shows: it only carries
    the strings (the multiview layout names, #9813/#9814) between the panes
    and the store.
    """

    read: Callable[[], dict[str, str]]
    apply: Callable[[Mapping[str, str]], None]


class LayoutBar(QWidget):
    """Load / save / delete / reset the pane arrangement."""

    def __init__(
        self,
        host: PaneHost,
        store: LayoutStore,
        splitter: QSplitter | None = None,
        *,
        ask_name: Callable[[], str | None] | None = None,
        extras: PaneExtras | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._host, self._store, self._splitter = host, store, splitter
        self._extras = extras
        self._ask_name = ask_name or self._dialog_name
        self.combo = QComboBox()
        self.combo.setMinimumContentsLength(12)
        self.combo.setToolTip("Saved pane arrangements")
        self.load_button = QPushButton("Load")
        self.load_button.setToolTip("Apply the selected saved layout")
        self.save_button = QPushButton("Save as…")
        self.save_button.setToolTip(
            "Save the current pane arrangement (positions, sizes, floating "
            "panes, controls width) under a name"
        )
        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Forget the selected saved layout")
        self.reset_button = QPushButton("Reset layout")
        self.reset_button.setToolTip(
            "Back to the default arrangement and show every pane (use this if "
            "a pane was closed or dragged off screen)"
        )
        self.load_button.clicked.connect(lambda: self.apply(self.combo.currentText()))
        self.save_button.clicked.connect(self._save_clicked)
        self.delete_button.clicked.connect(
            lambda: self.delete(self.combo.currentText())
        )
        self.reset_button.clicked.connect(self._host.reset)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Layout"))
        for w in (
            self.combo,
            self.load_button,
            self.save_button,
            self.delete_button,
            self.reset_button,
        ):
            row.addWidget(w)
        self.refresh()

    # -- store <-> widgets ------------------------------------------------------
    def refresh(self) -> None:
        current = self.combo.currentText()
        self.combo.clear()
        self.combo.addItems(self._store.names())
        if current:
            self.combo.setCurrentText(current)
        has_any = self.combo.count() > 0
        self.load_button.setEnabled(has_any)
        self.delete_button.setEnabled(has_any)

    def _snapshot(self) -> tuple[bytes, list[int]]:
        sizes = list(self._splitter.sizes()) if self._splitter is not None else []
        return self._host.state(), sizes

    def save_as(self, name: str) -> None:
        """Save the current arrangement (and the panes' extras) under ``name``."""
        self._store.save(name, *self._snapshot())
        if self._extras is not None:
            self._store.save_extras(name, self._extras.read())
        self.refresh()
        self.combo.setCurrentText(name)

    def apply(self, name: str) -> bool:
        """Restore the named arrangement; ``False`` when unknown or unreadable."""
        if not name:
            return False
        found = self._store.load(name)
        if found is None:
            return False
        state, sizes = found
        ok = self._host.restore(state)
        if ok and sizes and self._splitter is not None:
            self._splitter.setSizes(sizes)
        if ok and self._extras is not None:
            self._extras.apply(self._store.extras(name))
        return ok

    def delete(self, name: str) -> None:
        if name:
            self._store.delete(name)
            self.refresh()

    def save_last(self) -> None:
        """Remember the arrangement for the next start (called on shutdown)."""
        self._store.save(LAST_LAYOUT, *self._snapshot())
        if self._extras is not None:
            self._store.save_extras(LAST_LAYOUT, self._extras.read())

    def restore_last(self) -> bool:
        return self.apply(LAST_LAYOUT)

    # -- dialogs --------------------------------------------------------------------
    def _dialog_name(self) -> str | None:
        name, ok = QInputDialog.getText(
            self, "Save layout", "Layout name:", text=self.combo.currentText()
        )
        return name.strip() if ok and name.strip() else None

    def _save_clicked(self) -> None:
        name = self._ask_name()
        if name:
            self.save_as(name)
