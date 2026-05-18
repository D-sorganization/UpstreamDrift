"""Help-menu builder and Keyboard-Shortcuts modal.

The launcher's help menu is constructed from a small declarative list so
the same five entries (User Guide, Motion-Match Loaders, Keyboard
Shortcuts, Report a Bug, About) can be added by any launcher window
that owns a :class:`QMenuBar`.

The Keyboard-Shortcuts modal scrapes :class:`PyQt6.QtGui.QShortcut` and
:class:`PyQt6.QtGui.QAction` instances registered on the parent window
so it always reflects what is *actually* bound at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QShortcut
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QMenu,
    QMenuBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.launchers.about_dialog import (
    open_issues_page,
    open_motion_match_loaders_doc,
    open_user_guide,
    show_about_dialog,
)

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


def _add_action(
    menu: QMenu,
    parent: QWidget,
    label: str,
    tooltip: str,
    status_tip: str,
    handler: Callable[[], None],
    shortcut: str | None = None,
) -> QAction:
    """Create a QAction with consistent help-text fields and add it to menu.

    Args:
        menu: Target submenu.
        parent: Owner widget (becomes the action's parent).
        label: Visible action label (may include ``&`` mnemonics).
        tooltip: Hover tooltip, sentence case, no trailing period.
        status_tip: Status-bar message in present tense.
        handler: Zero-arg callable invoked when the action is triggered.
        shortcut: Optional keyboard accelerator (e.g. ``"F1"``).

    Returns:
        The freshly-added :class:`QAction`. The caller does not normally
        need to keep a reference, but doing so allows further wiring.
    """
    action = QAction(label, parent)
    action.setToolTip(tooltip)
    action.setStatusTip(status_tip)
    if shortcut:
        action.setShortcut(shortcut)
    action.triggered.connect(lambda _checked=False: handler())
    menu.addAction(action)
    return action


def build_help_menu(
    menubar: QMenuBar,
    parent: QWidget,
    *,
    show_shortcuts: Callable[[], None] | None = None,
) -> QMenu:
    """Build (or extend) the top-level Help menu on ``menubar``.

    The menu is idempotent in spirit: if the caller already added a Help
    menu, the caller should pass that menu's underlying object via
    ``menubar`` such that ``addMenu`` returns a fresh one. This helper
    always *creates* a new submenu.

    Args:
        menubar: The window's menu bar to attach to.
        parent: Owner widget for the actions and modal dialogs.
        show_shortcuts: Override for the keyboard-shortcuts handler. When
            ``None`` the built-in :func:`show_keyboard_shortcuts_modal`
            is used.

    Returns:
        The newly created Help :class:`QMenu`.
    """
    menu = menubar.addMenu("&Help")

    _add_action(
        menu,
        parent,
        "&User Guide",
        tooltip="Open the bundled user guide in the system browser",
        status_tip="Opens user guide",
        handler=open_user_guide,
        shortcut="F1",
    )
    _add_action(
        menu,
        parent,
        "&Motion-Match Loaders",
        tooltip="Reference for loading motion-target files",
        status_tip="Opens motion-match loader reference",
        handler=open_motion_match_loaders_doc,
    )
    menu.addSeparator()

    def open_document_reader_file_dialog() -> None:
        from PyQt6.QtWidgets import QFileDialog
        from src.shared.python.ui.qt.widgets.document_reader import show_document

        path, _ = QFileDialog.getOpenFileName(
            parent, "Open Document", "", "Documents (*.md *.pdf *.tex);;All Files (*.*)"
        )
        if path:
            show_document(path)

    _add_action(
        menu,
        parent,
        "&Open Document Reader...",
        tooltip="Open a local PDF, Markdown, or LaTeX document",
        status_tip="Opens local document for troubleshooting",
        handler=open_document_reader_file_dialog,
    )
    menu.addSeparator()
    _add_action(
        menu,
        parent,
        "&Keyboard Shortcuts",
        tooltip="Show every registered keyboard shortcut",
        status_tip="Opens keyboard-shortcuts table",
        handler=show_shortcuts or (lambda: show_keyboard_shortcuts_modal(parent)),
        shortcut="Ctrl+?",
    )
    menu.addSeparator()
    _add_action(
        menu,
        parent,
        "&Report a Bug",
        tooltip="Open the public issue tracker in your browser",
        status_tip="Opens issue tracker",
        handler=open_issues_page,
    )
    menu.addSeparator()
    _add_action(
        menu,
        parent,
        "&About",
        tooltip="Show version and runtime information",
        status_tip="Opens About dialog",
        handler=lambda: show_about_dialog(parent),
    )
    return menu


class KeyboardShortcutsDialog(QDialog):
    """Modal that lists every registered shortcut on the parent window.

    The dialog walks the QObject tree of ``parent`` and collects:

    * :class:`QAction` objects with non-empty ``shortcut()`` keys, and
    * :class:`QShortcut` objects with non-empty ``key()`` sequences.

    Each row shows the key sequence plus a human-readable label
    (``QAction.text`` if available, otherwise the object name).
    """

    def __init__(self, parent: QWidget) -> None:
        """Build the dialog around ``parent``.

        Args:
            parent: Window whose action / shortcut tree is scraped. Must
                not be ``None`` because the table would otherwise be
                empty.
        """
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(520, 480)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        layout = QVBoxLayout(self)

        rows = collect_shortcut_rows(parent)
        table = QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        for r, (key, label) in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(key))
            table.setItem(r, 1, QTableWidgetItem(label))
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        if close_btn := buttons.button(QDialogButtonBox.StandardButton.Close):
            close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)


def collect_shortcut_rows(parent: QWidget) -> list[tuple[str, str]]:
    """Walk ``parent``'s QObject tree and gather (key, label) shortcut rows.

    Args:
        parent: Root widget to scan. The scan recurses through every
            child via :func:`QObject.findChildren`.

    Returns:
        A list of ``(key_sequence, label)`` tuples sorted by label. Empty
        when no shortcuts are registered.
    """
    rows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for action in parent.findChildren(QAction):
        seq = action.shortcut().toString()
        if not seq:
            continue
        label = (
            action.text().replace("&", "").strip() or action.objectName() or "(action)"
        )
        key = (seq, label)
        if key not in seen:
            seen.add(key)
            rows.append(key)

    for sc in parent.findChildren(QShortcut):
        seq = sc.key().toString()
        if not seq:
            continue
        label = sc.objectName() or "(shortcut)"
        key = (seq, label)
        if key not in seen:
            seen.add(key)
            rows.append(key)

    rows.sort(key=lambda r: (r[1].lower(), r[0]))
    return rows


def show_keyboard_shortcuts_modal(parent: QWidget) -> None:
    """Open the shortcuts dialog modally over ``parent``.

    Args:
        parent: Window whose shortcuts will be scraped.

    Postcondition:
        Returns once the user dismisses the dialog.
    """
    dlg = KeyboardShortcutsDialog(parent)
    dlg.exec()
