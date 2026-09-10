"""Searchable player bag and explicit capture equipment assignment."""

from __future__ import annotations

from pathlib import Path
from html import escape
from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QTextBrowser,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.club_data.catalog import ClubRecord
from src.shared.python.club_data.catalog_sources import (
    load_public_catalog,
    summarize_record,
)
from src.shared.python.club_data.player_clubs import PlayerBag, PlayerClub

from . import styling
from .capture_library import read_notes
from .club_editor import ClubEditorDialog
from .equipment import (
    BAG_FILE,
    load_bag,
    load_capture_club,
    save_bag,
    save_capture_club,
)
from .flow_layout import FlowLayout


def _record_html(record: ClubRecord) -> str:
    sources = sorted(
        {(c.source.title, c.source.url or "", c.source.license) for c in record.claims}
    )
    links = "".join(
        f'<p>{escape(title)}<br><a href="{escape(url, quote=True)}">{escape(url)}</a><br>{escape(license)}</p>'
        for title, url, license in sources
    )
    return f"<pre>{escape(summarize_record(record))}</pre><h3>Sources</h3>{links}"


def capture_equipment_text(root: Path | None) -> str:
    """Always distinguish an unassigned capture from unreadable equipment data."""
    if root is None:
        return "No capture selected. Open or record a capture to assign a club."
    try:
        selected = load_capture_club(root)
        if selected is None:
            return (
                "Club: Unassigned — choose My Clubs to record this swing's equipment."
            )
        return (
            f"Club: {selected.club.label} · saved revision {selected.club_revision[:8]}"
        )
    except (OSError, ValueError) as exc:
        return f"Club selection needs attention: {exc}"


class CatalogPicker(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose a Catalog Build")
        self.resize(650, 480)
        self.records = load_public_catalog()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search manufacturer, model, number or build")
        self.items = QListWidget()
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(True)
        document = self.details.document()
        if document is not None:
            document.setDefaultStyleSheet(styling.help_document_style())
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Starter catalog: partial public specifications. Add a custom club if yours is absent."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.search)
        layout.addWidget(self.items)
        layout.addWidget(self.details)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.search.textChanged.connect(self.refresh)
        self.items.currentRowChanged.connect(self.selection_changed)
        self.refresh()

    def refresh(self) -> None:
        query = self.search.text().casefold().strip()
        self.visible_records = [
            r for r in self.records if query in r.model_dump_json().casefold()
        ]
        self.items.clear()
        for record in self.visible_records:
            identity = record.identity
            self.items.addItem(
                f"{identity.manufacturer} {identity.model} {identity.number or ''} · {identity.build}"
            )
        self.selection_changed()

    def selection_changed(self) -> None:
        row = self.items.currentRow()
        button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        button.setEnabled(0 <= row < len(self.visible_records))
        self.details.clear()
        if 0 <= row < len(self.visible_records):
            record = self.visible_records[row]
            self.details.setHtml(_record_html(record))

    def selected_club(self) -> PlayerClub:
        row = self.items.currentRow()
        if not 0 <= row < len(self.visible_records):
            raise ValueError("Select a catalog build first")
        record = self.visible_records[row]
        identity = record.identity
        return PlayerClub(
            label=f"{identity.model} {identity.number or ''}".strip(), base=record
        )


class EquipmentDialog(QDialog):
    """Each edit is committed explicitly; selection changes have no side effects."""

    def __init__(
        self,
        library_root: Path,
        capture: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.path, self.capture = library_root / BAG_FILE, capture
        self.bag = load_bag(self.path)
        self.setWindowTitle("My Clubs")
        self.resize(760, 600)
        self.search = QLineEdit()
        self.search.setAccessibleName("Search my clubs")
        self.search.setPlaceholderText("Search clubs, numbers and notes")
        self.archived = QCheckBox("Show Archived")
        self.items = QListWidget()
        self.items.setAccessibleName("Player bag")
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.selection = QLabel(capture_equipment_text(capture))
        self.selection.setTextFormat(Qt.TextFormat.PlainText)
        self.selection.setWordWrap(True)
        self.status = QLabel(
            "Add a catalog build or a custom club. Save edits, then assign it to a capture."
        )
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self._build()
        self.search.textChanged.connect(self.refresh)
        self.archived.toggled.connect(self.refresh)
        self.items.currentRowChanged.connect(self.selection_changed)
        self.items.itemDoubleClicked.connect(lambda _item: self.edit_selected())
        self.refresh()
        styling.apply_theme(self)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self.selection)
        search = QHBoxLayout()
        search.addWidget(self.search, 1)
        search.addWidget(self.archived)
        layout.addLayout(search)
        layout.addWidget(self.items, 1)
        layout.addWidget(self.details, 1)
        actions = FlowLayout(spacing=6)
        for label, callback in (
            ("Add from Catalog…", self.add_catalog),
            ("Add Custom…", self.add_custom),
            ("Edit…", self.edit_selected),
            ("Archive / Restore", self.archive_selected),
            ("Assign to This Capture", self.assign_selected),
            ("Reload Bag", self.reload),
            ("Help", self.show_help),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, action=callback: action())
            actions.add_widget(button)
        layout.addLayout(actions)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def refresh(self) -> None:
        query = self.search.text().strip().casefold()
        self.visible_clubs = [
            c
            for c in self.bag.clubs
            if (self.archived.isChecked() or not c.archived)
            and query in f"{c.label} {c.identity.number or ''} {c.notes}".casefold()
        ]
        self.items.clear()
        for club in self.visible_clubs:
            self.items.addItem(club.label + (" · Archived" if club.archived else ""))
        self.details.clear()

    def selected_club(self) -> PlayerClub | None:
        row = self.items.currentRow()
        return self.visible_clubs[row] if 0 <= row < len(self.visible_clubs) else None

    def selection_changed(self) -> None:
        club = self.selected_club()
        self.details.setPlainText(
            summarize_record(club.effective_record()) + "\n\n" + club.notes
            if club
            else "Select a club to inspect its specifications."
        )

    def save_club(self, club: PlayerClub) -> bool:
        records = [c for c in self.bag.clubs if c.club_id != club.club_id]
        try:
            bag = PlayerBag(clubs=(*records, club))
            save_bag(self.path, bag, expected_revision=self.bag.revision)
        except (OSError, ValueError) as exc:
            self.status.setText(
                f"Club was not saved: {exc}. Reload Bag to review the current data."
            )
            return False
        self.bag = bag
        self.refresh()
        self.status.setText(
            f"Saved {club.label}. Existing capture snapshots are unchanged."
        )
        return True

    def _edit(self, club: PlayerClub | None) -> None:
        editor = ClubEditorDialog(club, self)
        if editor.exec() == QDialog.DialogCode.Accepted:
            self.save_club(editor.record())

    def add_custom(self) -> None:
        self._edit(None)

    def add_catalog(self) -> None:
        try:
            picker = CatalogPicker(self)
            if picker.exec() == QDialog.DialogCode.Accepted:
                self._edit(picker.selected_club())
        except (OSError, ValueError) as exc:
            self.status.setText(f"Catalog could not be opened: {exc}")

    def edit_selected(self) -> None:
        club = self.selected_club()
        if club is None:
            self.status.setText("Select a club first, or choose Add Custom.")
            return
        self._edit(club)

    def archive_selected(self) -> None:
        club = self.selected_club()
        if club is None:
            self.status.setText("Select a club to archive or restore.")
            return
        if self.save_club(club.model_copy(update={"archived": not club.archived})):
            self.status.setText(
                f"Restored {club.label}. It is available for new captures."
                if club.archived
                else f"Archived {club.label}. Enable Show Archived to restore it. Capture history is unchanged."
            )

    def assign_selected(self) -> bool:
        club = self.selected_club()
        if club is None or self.capture is None:
            self.status.setText(
                "Open a capture and select a club before assigning equipment."
            )
            return False
        try:
            notes = read_notes(self.capture)
            save_capture_club(self.capture, club)
        except (OSError, ValueError) as exc:
            self.status.setText(f"Club was not assigned: {exc}")
            return False
        self.selection.setText(capture_equipment_text(self.capture))
        self.status.setText(
            f"Assigned {club.label} to {notes.title} ({notes.capture_id}). Earlier selections are retained."
        )
        return True

    def reload(self) -> None:
        try:
            self.bag = load_bag(self.path)
        except (OSError, ValueError) as exc:
            self.status.setText(f"Bag could not be read: {exc}")
            return
        self.refresh()
        self.status.setText("Bag reloaded. Select a club to continue.")

    def show_help(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("My Clubs — Help")
        dialog.resize(650, 500)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setHtml(
            "<h1>Record the Club Used for a Swing</h1>"
            "<ol><li><b>Add from Catalog</b> to search a specific build and inspect its sources, "
            "or <b>Add Custom</b> for any club absent from the starter catalog.</li>"
            "<li>Give the club a recognizable name and number. Record only the dimensions you know. "
            "Choose <b>Measured</b> or <b>Estimated</b>, then enter a value in your preferred units. "
            "<b>Unknown</b> removes a nominal value from use for your club.</li>"
            "<li>Add shaft, grip, fitting or measurement notes and select <b>Save</b>.</li>"
            "<li>Select the saved club and click <b>Assign to This Capture</b>. "
            "The confirmation identifies the capture title and ID.</li></ol>"
            "<h2>Review and Correct</h2><p>Open a different capture through <b>Library</b>; "
            "its details show the assigned club. Double-click a bag entry to edit it. "
            "Edits affect the bag; assign again to explicitly update a capture. Previous selections "
            "and completed model reports retain their saved evidence.</p>"
            "<p><b>Archive / Restore</b> hides unused clubs without deleting capture history. "
            "Enable <b>Show Archived</b> to restore one. If another window changed the bag, "
            "<b>Reload Bag</b> before editing again.</p>"
            "<h2>What the Models Use</h2><p>Fit reports retain club number and eligible measurements. "
            "Current body models do not observe the club and apply no club-length constraint. "
            "Estimated or conflicting values are withheld from default physical use. "
            "Club dimensions alone do not determine camera calibration.</p>"
        )
        document = browser.document()
        if document is not None:
            document.setDefaultStyleSheet(styling.help_document_style())
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
