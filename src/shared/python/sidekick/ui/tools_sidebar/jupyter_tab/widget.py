"""Read-only Jupyter notebook viewer for Sidekick (Phase 1).

The widget renders a parsed :class:`NotebookDocument` inside a scroll
area. Markdown cells use :class:`QTextBrowser`; code cells use a
read-only monospace :class:`QTextEdit`; raw cells use a plain
:class:`QTextEdit`. Rich outputs (images, HTML, widgets) render as
placeholders in Phase 1 — Phase 2 (#2876) introduces execution and
Phase 3 (#2877) handles persistence.
"""

from __future__ import annotations

from collections.abc import Callable

from ..qt_compat import QtGui, QtWidgets
from .notebook_model import (
    CodeCell,
    MarkdownCell,
    NotebookCell,
    NotebookDocument,
    RawCell,
)

_PLACEHOLDER_RICH_OUTPUT = "[Phase 2: image output]"


class JupyterNotebookWidget(QtWidgets.QWidget):
    """A scrollable, read-only view of a notebook document.

    The widget reads cells exclusively via ``self._document.cells``;
    it never reaches into underlying nbformat structures (LOD).
    """

    def __init__(
        self,
        document: NotebookDocument,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SidekickJupyterNotebookWidget")
        self._document = document
        self._cell_widgets: list[QtWidgets.QWidget] = []
        self._build_layout()

    @property
    def document(self) -> NotebookDocument:
        """The notebook document backing this widget (read-only)."""
        return self._document

    def cell_widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        """Return the per-cell widgets in render order."""
        return tuple(self._cell_widgets)

    def _build_layout(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        container = QtWidgets.QWidget(scroll)
        container.setObjectName("SidekickJupyterNotebookContainer")
        inner = QtWidgets.QVBoxLayout(container)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(6)

        dispatch: dict[
            str, Callable[[NotebookCell, QtWidgets.QWidget], QtWidgets.QWidget]
        ] = {
            "markdown": self._render_markdown_cell,
            "code": self._render_code_cell,
            "raw": self._render_raw_cell,
        }

        for cell in self._document.cells:
            factory = dispatch.get(cell.cell_type)
            if factory is None:
                continue
            cell_widget = factory(cell, container)
            self._cell_widgets.append(cell_widget)
            inner.addWidget(cell_widget)

        inner.addStretch(1)
        scroll.setWidget(container)

    def _render_markdown_cell(
        self,
        cell: NotebookCell,
        parent: QtWidgets.QWidget,
    ) -> QtWidgets.QWidget:
        assert isinstance(cell, MarkdownCell)
        browser = QtWidgets.QTextBrowser(parent)
        browser.setObjectName("SidekickJupyterMarkdownCell")
        browser.setOpenExternalLinks(False)
        browser.setMarkdown(cell.source)
        return browser

    def _render_code_cell(
        self,
        cell: NotebookCell,
        parent: QtWidgets.QWidget,
    ) -> QtWidgets.QWidget:
        assert isinstance(cell, CodeCell)
        container = QtWidgets.QWidget(parent)
        container.setObjectName("SidekickJupyterCodeCell")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        source_edit = QtWidgets.QTextEdit(container)
        source_edit.setObjectName("SidekickJupyterCodeSource")
        source_edit.setReadOnly(True)
        source_edit.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        font = QtGui.QFont("Courier New")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        source_edit.setFont(font)
        source_edit.setPlainText(cell.source)
        layout.addWidget(source_edit)

        if cell.outputs:
            for output in cell.outputs:
                output_widget = self._render_output(output.text, container)
                layout.addWidget(output_widget)
        return container

    def _render_raw_cell(
        self,
        cell: NotebookCell,
        parent: QtWidgets.QWidget,
    ) -> QtWidgets.QWidget:
        assert isinstance(cell, RawCell)
        edit = QtWidgets.QTextEdit(parent)
        edit.setObjectName("SidekickJupyterRawCell")
        edit.setReadOnly(True)
        edit.setPlainText(cell.source)
        return edit

    def _render_output(
        self,
        text: str,
        parent: QtWidgets.QWidget,
    ) -> QtWidgets.QWidget:
        label = QtWidgets.QLabel(parent)
        label.setObjectName("SidekickJupyterCellOutput")
        label.setWordWrap(True)
        label.setText(text or _PLACEHOLDER_RICH_OUTPUT)
        return label
