"""Swing report package window (#15724, SRP-5): the launcher tile.

Pick a bundle, an output folder, a package and its items, preview the plan,
then run the package command line in a ``QProcess`` that streams into the
log. Logic lives in :mod:`.gui_model`, literals in ``gui_settings.json``.
"""

# noqa: gui-thread/ok -- the package runs in a QProcess; preview only plans.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from PyQt6 import QtWidgets as Qw
from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QFontDatabase

from src.tools.swing_report_package import gui_model
from src.tools.swing_report_package.catalog import load_catalog
from src.tools.swing_report_package.runner import load_runner_settings

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROLE = Qt.ItemDataRole.UserRole


class SwingReportWindow(Qw.QMainWindow):
    """Choose, preview and run a swing report package."""

    def __init__(self, parent: Qw.QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = gui_model.load_gui_settings()
        self.labels: dict[str, str] = self.settings["labels"]
        self.catalog = load_catalog()
        self.process: QProcess | None = None
        win, defaults = self.settings["window"], load_runner_settings()["cli_defaults"]
        self.setWindowTitle(win["title"])
        self.resize(win["width"], win["height"])
        self.bundle, self.out = Qw.QLineEdit(), Qw.QLineEdit()
        self.swing = Qw.QLineEdit(defaults["swing"])
        self.club = Qw.QLineEdit(defaults["club"])
        self.package = Qw.QComboBox()
        self.package.addItems(gui_model.package_names(self.catalog))
        self.items = Qw.QListWidget()
        self.items.setMinimumHeight(win["items_min_height"])
        for name, kind, title in gui_model.item_rows(self.catalog):
            item = Qw.QListWidgetItem(f"{title}  [{kind}: {name}]")
            item.setData(_ROLE, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.items.addItem(item)
        self.log = Qw.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._lay_out()
        self.package.currentTextChanged.connect(self._reset_ticks)
        self.package.setCurrentText(self.settings["defaults"]["package"])
        self._reset_ticks()

    def _lay_out(self) -> None:
        form = Qw.QFormLayout()
        form.addRow(self.labels["bundle"], self._browse_row(self.bundle, False))
        form.addRow(self.labels["out"], self._browse_row(self.out, True))
        rows = {"package": self.package, "swing": self.swing, "club": self.club}
        for key, widget in {**rows, "items": self.items}.items():
            form.addRow(self.labels[key], widget)
        actions = {"preview": self.preview, "run": self.run, "stop": self.stop}
        self.buttons: dict[str, Qw.QPushButton] = {}
        buttons = Qw.QHBoxLayout()
        for key, action in actions.items():
            self.buttons[key] = Qw.QPushButton(self.labels[key])
            self.buttons[key].clicked.connect(action)
            buttons.addWidget(self.buttons[key])
        buttons.addStretch(1)
        self._set_running(False)
        central = Qw.QWidget(self)
        body = Qw.QVBoxLayout(central)
        body.addLayout(form)
        body.addLayout(buttons)
        body.addWidget(self.log, 1)
        self.setCentralWidget(central)

    def _browse_row(self, edit: Qw.QLineEdit, folder: bool) -> Qw.QHBoxLayout:
        button = Qw.QPushButton(self.labels["browse"])
        button.clicked.connect(lambda: self._pick(edit, folder))
        row = Qw.QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(button)
        return row

    def _pick(self, edit: Qw.QLineEdit, folder: bool) -> None:
        dialog, labels = Qw.QFileDialog, self.labels
        if folder:
            name = dialog.getExistingDirectory(self, labels["choose_out"])
        else:
            pattern = self.settings["bundle_filter"]
            name, _ = dialog.getOpenFileName(self, labels["choose_bundle"], "", pattern)
        if name:
            edit.setText(name)

    def _rows(self) -> list[Qw.QListWidgetItem]:
        rows = (self.items.item(i) for i in range(self.items.count()))
        return [row for row in rows if row is not None]

    def _reset_ticks(self) -> None:
        start = gui_model.starting_items(self.catalog, self.package.currentText())
        for row in self._rows():
            on = row.data(_ROLE) in start
            row.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def choice(self) -> gui_model.ReportChoice:
        """The window's current choices."""
        on = Qt.CheckState.Checked
        ticks = tuple(str(r.data(_ROLE)) for r in self._rows() if r.checkState() == on)
        b, o, s, c = (w.text() for w in (self.bundle, self.out, self.swing, self.club))
        return gui_model.ReportChoice(b, o, self.package.currentText(), ticks, s, c)

    def _say(self, key: str, **values: Any) -> None:
        self.log.appendPlainText(self.labels[key].format(**values))

    def preview(self) -> None:
        """Write the planned jobs to the log; renders nothing."""
        try:
            manifest = gui_model.plan_preview(self.catalog, self.choice())
        except ValueError as exc:
            self._say("error", message=exc)
            return
        self.log.setPlainText("\n".join(gui_model.preview_lines(manifest)))

    def run(self) -> None:
        """Start the package command line in a background process."""
        try:
            cmd = gui_model.module_command(self.catalog, self.choice(), self.settings)
        except ValueError as exc:
            self._say("error", message=exc)
            return
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(_REPO_ROOT))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._stream)
        self.process.finished.connect(self._finished)
        self._say("started", command=" ".join([cmd[0], *cmd[1]]))
        self._set_running(True)
        self.process.start(*cmd)

    def stop(self) -> None:
        """Stop a running package."""
        if self.process is not None:
            self.process.kill()

    def _stream(self) -> None:
        data = self.process.readAllStandardOutput().data() if self.process else b""
        if data:
            self.log.appendPlainText(bytes(data).decode(errors="replace").rstrip("\n"))

    def _finished(self, code: int, _status: Any = None) -> None:
        self._stream()
        self._say("finished", code=code)
        self.process = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        for key, button in self.buttons.items():
            button.setEnabled(running == (key == "stop"))


def main() -> int:
    """Open the window (the launcher tile's entry point)."""
    app = Qw.QApplication.instance() or Qw.QApplication(sys.argv)
    window = SwingReportWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
