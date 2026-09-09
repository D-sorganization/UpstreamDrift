"""Provenance tab: the lineage of whatever result the user clicked (#9797).

``show_path`` walks the file's provenance back to the recordings and the
detector plug-in and renders it as HTML. Result tables carry the file that
backs them (:class:`SourcedTable`) and emit it when a row is selected.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.provenance import LineageRecord, lineage
from src.shared.python.core.process_safety import narrow_catch

from .session import flatten_numbers


def lineage_html(records: list[LineageRecord]) -> str:
    """One block per hop: file, schema, time, tool, parameters, inputs."""
    parts = ["<h3>Lineage</h3><ol>"]
    for r in records:
        if r.is_leaf:
            parts.append(f"<li><code>{html.escape(r.path)}</code> <i>(leaf)</i></li>")
            continue
        by = r.generated_by
        sha = by.get("git_sha")
        tool = f"{by.get('package', '?')} · {by.get('module', '?')} · {by.get('version', '?')}"
        tool += f" · {sha[:9]}" if isinstance(sha, str) else ""
        params = "".join(
            f"<li>{html.escape(str(k))} = {html.escape(str(v))}</li>"
            for k, v in sorted(r.parameters.items())
        )
        inputs = "".join(
            f"<li><code>{html.escape(str(i.get('path')))}</code> "
            f"<small>{html.escape(str(i.get('sha256', ''))[:12])}</small></li>"
            for i in r.inputs
        )
        parts.append(
            f"<li><b><code>{html.escape(r.path)}</code></b> "
            f"[{html.escape(str(r.schema_version))}]<br/>{html.escape(str(r.created_utc))}"
            f" by {html.escape(tool)}"
            + (f"<ul>{params}</ul>" if params else "")
            + (f"inputs:<ul>{inputs}</ul>" if inputs else "")
            + "</li>"
        )
    parts.append("</ol>")
    return "".join(parts)


class ProvenanceTab(QWidget):
    """Shows the lineage of the last selected result file."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setHtml("<i>Select a result row to see where it came from.</i>")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.browser)
        self.current: Path | None = None

    def clear_capture(self) -> None:
        """Remove the prior capture's provenance after a failed selection."""
        self.current = None
        self.browser.setPlainText("No capture loaded.")

    def show_path(self, session: Path, path: Path | None) -> None:
        """Render the lineage of ``path`` (relative paths resolve in ``session``)."""
        if path is None:
            return
        resolved = path if path.is_absolute() else session / path
        self.current = resolved
        if not resolved.is_file():
            self.browser.setHtml(
                f"<i>{html.escape(str(resolved))} does not exist yet.</i>"
            )
            return
        with narrow_catch(ValueError, OSError, log_message="lineage"):
            self.browser.setHtml(lineage_html(lineage(resolved, base=session)))
            return
        self.browser.setHtml(
            f"<i>no provenance readable for {html.escape(str(resolved))}</i>"
        )


class SourcedTable(QTableWidget):
    """``key | value`` rows that remember the file they were read from."""

    source_selected = pyqtSignal(object)  # Path | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["metric", "value"])
        header = self.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self.source: Path | None = None
        self.itemSelectionChanged.connect(self._emit_source)

    def fill(self, payload: dict[str, Any] | None, source: Path | None = None) -> None:
        self.source = source
        rows = flatten_numbers(payload) if payload else []
        self.setRowCount(len(rows))
        for r, (key, value) in enumerate(rows):
            self.setItem(r, 0, QTableWidgetItem(key))
            self.setItem(r, 1, QTableWidgetItem(value))

    def _emit_source(self) -> None:
        if self.selectedItems():
            self.source_selected.emit(self.source)
