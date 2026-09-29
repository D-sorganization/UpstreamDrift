"""Visible capture identity, action outcomes and contextual navigation (#9913)."""

from __future__ import annotations

from html import escape
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import workflow, styling
from .capture_library import read_notes
from .capture_activity import read_activity, display_status
from .flow_layout import FlowLayout
from .journey_status import evidence_tree, history_tree
from .session import SessionMedia


class JourneyPanel(QFrame):
    """Keep the selected capture and next action visible outside hidden logs."""

    action_requested = pyqtSignal(str)
    step_requested = pyqtSignal(str)
    source_requested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._media: SessionMedia | None = None
        self._next_action: str | None = None
        self._retry_action: str | None = None
        self._help: QDialog | None = None
        self._details: QDialog | None = None
        self.active_id: str | None = None
        self.identity = QLabel("No capture selected — record or import a swing.")
        self.message = QLabel("Choose your cameras, or import an existing swing video.")
        for label in (self.identity, self.message):
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.identity.setAccessibleName("Selected Swing and Capture")
        self.message.setAccessibleName("Action Status")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(5)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.next_button = QPushButton("Next Step")
        self.next_button.clicked.connect(self._next)
        self.retry_button = QPushButton("Retry")
        self.retry_button.clicked.connect(self._retry)
        self.retry_button.setEnabled(False)
        self.details_button = QPushButton("Capture Status")
        self.details_button.clicked.connect(self.show_details)
        self.help_button = QPushButton("Help and Workflow")
        self.help_button.clicked.connect(self.show_help)
        self.log_button = QPushButton("View Activity Log")
        self.log_button.clicked.connect(lambda: self.action_requested.emit("show_log"))
        buttons = QWidget()
        row = FlowLayout(buttons)
        for button in (
            self.next_button,
            self.retry_button,
            self.details_button,
            self.help_button,
            self.log_button,
        ):
            row.add_widget(button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(3)
        layout.addWidget(self.identity)
        layout.addWidget(self.message)
        layout.addWidget(self.progress)
        layout.addWidget(buttons)

    def set_capture(self, media: SessionMedia | None) -> None:
        self._media = media
        self.details_button.setEnabled(media is not None)
        if media is None and self._details is not None:
            self._details.close()
        if media is None:
            self.identity.setText("No capture selected — record or import a swing.")
            self.identity.setToolTip("")
            return
        title, capture_id = media.root.name, "Not yet registered in the library"
        try:
            notes = read_notes(media.root)
        except FileNotFoundError:
            pass
        except (ValueError, OSError) as exc:
            capture_id = f"Capture notes need attention: {exc}"
        else:
            title, capture_id = notes.title, notes.capture_id
        self.identity.setText(
            f"Swing: {title}  ·  Capture: {media.root.name}  ·  {len(media.views)} view(s)"
        )
        self.identity.setToolTip(f"Capture ID: {capture_id}\nFolder: {media.root}")
        self.refresh_details()

    def refresh_details(self) -> None:
        """Refresh an already-open status view after capture/action changes."""
        if self._details is not None and self._details.isVisible():
            self.show_details()

    def set_steps(self, states: tuple[workflow.StepState, ...], *, busy: bool) -> None:
        current = workflow.current(states)
        self._next_action = None
        if current is not None:
            self.next_button.setText(f"Next: {current.step.title}")
            self.next_button.setToolTip(current.reason or current.step.purpose)
            # Navigate to the step first; its visible actions explain/run the work.
            self._next_action = current.step.key
        else:
            self.next_button.setText("Review This Capture")
        self.next_button.setEnabled(not busy)

    def notice(
        self, text: str, *, busy: bool = False, retry: str | None = None
    ) -> None:
        self.message.setText(text)
        self.progress.setVisible(busy)
        self._retry_action = retry
        self.retry_button.setEnabled(retry is not None and not busy)
        self.next_button.setEnabled(not busy)

    def _next(self) -> None:
        if self._next_action:
            self.step_requested.emit(self._next_action)
        else:
            self.show_details()

    def _retry(self) -> None:
        if self._retry_action:
            self.action_requested.emit(self._retry_action)

    def _dialog(self, title: str) -> tuple[QDialog, QVBoxLayout]:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(850, 600)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(dialog)
        return dialog, layout

    def show_help(self) -> None:
        """Render the canonical workflow as linked, searchable contextual help."""
        if self._help is not None:
            self._help.show()
            self._help.raise_()
            return
        dialog, layout = self._dialog("Capture Rig — Help and Workflow")
        browser = QTextBrowser()
        document = browser.document()
        if document is not None:
            document.setDefaultStyleSheet(styling.help_document_style())
        browser.setOpenLinks(False)
        browser.anchorClicked.connect(lambda url: self.step_requested.emit(url.path()))
        content = [
            "<h1>Capture a Swing</h1>",
            "<p>Open <b>Capture Wizard</b> to choose an outcome and follow only its required steps. "
            "Use <b>Open Step Controls</b> to work in the existing screen, then return with "
            "<b>Capture Wizard</b> and <b>Refresh Status</b>. Next becomes available when "
            "the step has its required evidence. My Clubs is optional for body-model fitting.</p>",
            "<p><b>Save and Close</b> stores your place with this capture. Reopen the wizard "
            "and choose <b>Resume This Capture’s Saved Workflow</b>. Changed inputs or a changed "
            "capability map require reviewing the outcomes again. Calibration settings need "
            "fresh confirmation after closing the wizard. Back and Cancel preserve saved work.</p>",
            "<p>One capture contains synchronized camera views of one take. "
            "Name it and add swing notes in <b>Library</b>. Use <b>Edit Swing</b> "
            "to select the swing before detecting joints. Detection creates "
            "2-D observations; reconstruction and model fitting are separate steps.</p>",
            "<p>Use <b>Views</b> to pop out a screen onto another monitor. "
            "Close the floating window to return it. Double-click video for full screen; "
            "<b>F11</b> toggles it and <b>Escape</b> returns to your layout. "
            "<b>Reset Layout</b> recovers hidden or misplaced screens.</p>",
            "<p>The status strip reports running, cancelled, failed or finished actions. "
            "A finished command does not itself certify the quality of a model. "
            "Open <b>Capture Status</b> and review the outputs and provenance.</p>",
        ]
        content.extend(
            [
                "<h2>Example: Prepare a Swing for Review</h2>",
                "<p>Choose <b>Trim and Crop a Swing</b> in Capture Wizard. Open "
                '<a href="step:setup">Library</a>, import your video and name the take. '
                "Add swing notes, save the first and last swing frames in Swing Editor, "
                "then return to the wizard and Refresh Status. Save and Close keeps your "
                "place; select the same take to resume. Imported video needs no cameras.</p>",
                "<h2>Example: Compare With an Instructor Reference</h2>",
                "<p>Choose <b>Compare with an Expert Video</b>, select the player's take "
                "and save its swing selection. Import the expert video in the reference "
                "library. In Expert Comparison, select a camera view, align swing events "
                "with the time controls, adjust opacity and save. Refresh Status recognizes "
                "the saved alignment. Add Coaching Lines and Shapes is an optional outcome. "
                "Video alignment does not create a new camera angle or a 3-D expert.</p>",
                "<h2>Example: Prepare Calibrated Body-Model Analysis</h2>",
                "<p>Choose <b>Fit a Body Model to a Reconstruction</b>. Import synchronized "
                "views or prepare Camera Setup. Assign the swing's club in My Clubs, leaving "
                "unknown measurements blank. Review or Repeat Calibration checks camera "
                "identities, optical zoom, focus and image settings. Save the swing selection, "
                '<a href="step:detect">detect joints</a>, review observations, then '
                '<a href="step:reconstruct">reconstruct</a> and fit the body model. '
                "Refresh Status after each job; Go to links return to missing prerequisites. "
                "Review calibration again after resuming or changing zoom. Club information "
                "is retained as context; current body models do not fit a club segment. "
                "Review diagnostics before using measurements.</p>",
            ]
        )
        for step in workflow.STEPS:
            content.append(
                f'<h2><a href="step:{step.key}">{escape(step.title)}</a></h2>'
            )
            content.append(f"<p>{escape(step.purpose)}</p><h3>You Need</h3><ul>")
            content.extend(f"<li>{escape(text)}</li>" for text in step.requirements)
            content.append("</ul><h3>What to Do</h3><ol>")
            content.extend(f"<li>{escape(text)}</li>" for text in step.instructions)
            content.append("</ol>")
        browser.setHtml("".join(content))
        search = QLineEdit()
        search.setPlaceholderText("Find in Help — Press Enter for the Next Match")
        search.setAccessibleName("Find in Capture Help")
        feedback = QLabel()

        def find_next() -> None:
            term = search.text().strip()
            if not term:
                return
            if not browser.find(term):
                cursor = browser.textCursor()
                cursor.movePosition(cursor.MoveOperation.Start)
                browser.setTextCursor(cursor)
                found = browser.find(term)
                feedback.setText("Wrapped to start" if found else "No matching text")
            else:
                feedback.clear()

        search.returnPressed.connect(find_next)
        layout.addWidget(search)
        layout.addWidget(feedback)
        layout.addWidget(browser)
        self._help = dialog
        dialog.destroyed.connect(lambda: setattr(self, "_help", None))
        dialog.show()

    def show_details(self) -> None:
        """Show view/estimator and model variant associations for this capture."""
        media = self._media
        if media is None:
            self.notice("Record, import or load a capture before reviewing its status.")
            return
        dialog, layout = self._dialog(f"Capture Status — {media.root.name}")
        if self._details is not None:
            dialog.restoreGeometry(self._details.saveGeometry())
            self._details.close()
        self._details = dialog
        dialog.destroyed.connect(
            lambda: setattr(self, "_details", None) if self._details is dialog else None
        )
        title = QLabel(self.identity.text())
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        layout.addWidget(title)
        table = evidence_tree(media)
        table.itemActivated.connect(self._open_source)
        layout.addWidget(table)
        layout.addWidget(history_tree(media.root, self.active_id))
        actions = QHBoxLayout()
        for label, step_key in (
            ("Review Pose Detection", "detect"),
            ("Review Reconstruction", "reconstruct"),
            ("Review Model Fitting", "fit_model"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, key=step_key: self.step_requested.emit(key)
            )
            actions.addWidget(button)
        layout.addLayout(actions)
        note = QLabel(
            "Available files are not quality approval. Use Results → Provenance to review their source inputs and model settings."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        dialog.show()

    def _open_source(self, item: QTreeWidgetItem, _column: int) -> None:
        source = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(source, Path):
            self.source_requested.emit(source)

    def append_notice(self, text: str) -> None:
        """Add a diagnostic without discarding progress or retry state."""
        self.message.setText(self.message.text() + text)
