from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QSize, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QTextBrowser,
    QStackedWidget,
    QMessageBox,
    QDialog,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class DocumentReaderWidget(QWidget):
    """
    A unified document reader capable of rendering Markdown, PDF, and LaTeX files.
    """

    document_loaded = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.current_file: Path | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        self.toolbar_layout = QHBoxLayout()
        self.toolbar_layout.setContentsMargins(5, 5, 5, 5)

        self.lbl_title = QLabel("No document loaded")
        self.lbl_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.toolbar_layout.addWidget(self.lbl_title)

        self.toolbar_layout.addStretch()

        self.btn_zoom_in = QPushButton("Zoom In")
        self.btn_zoom_out = QPushButton("Zoom Out")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.toolbar_layout.addWidget(self.btn_zoom_out)
        self.toolbar_layout.addWidget(self.btn_zoom_in)

        self.layout.addLayout(self.toolbar_layout)

        # Stack for different viewers
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

        # 1. Text/Markdown Browser
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenLinks(False)
        self.text_browser.anchorClicked.connect(self._handle_link_click)
        self.stack.addWidget(self.text_browser)

        # 2. PDF Viewer (Scroll Area with Image)
        self.pdf_scroll_area = QScrollArea()
        self.pdf_scroll_area.setWidgetResizable(True)
        self.pdf_container = QWidget()
        self.pdf_layout = QVBoxLayout(self.pdf_container)
        self.pdf_layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )
        self.pdf_scroll_area.setWidget(self.pdf_container)
        self.stack.addWidget(self.pdf_scroll_area)

        self.pdf_pages: list[QLabel] = []
        self._pdf_document = None
        self._zoom_level = 1.0

    def load_document(self, file_path: str | Path) -> None:
        path = Path(file_path)
        if not path.exists():
            QMessageBox.warning(self, "Error", f"File not found: {path}")
            return

        self.current_file = path
        self.lbl_title.setText(path.name)
        self._zoom_level = 1.0

        ext = path.suffix.lower()
        try:
            if ext == ".md":
                self._load_markdown(path)
            elif ext == ".pdf":
                self._load_pdf(path)
            elif ext == ".tex":
                self._load_latex(path)
            else:
                self._load_plain_text(path)
            self.document_loaded.emit(str(path))
        except Exception as e:
            logger.exception("Failed to load document")
            QMessageBox.critical(self, "Error", f"Could not load {path.name}: {e}")

    @pyqtSlot(object)
    def _handle_link_click(self, url) -> None:
        scheme = url.scheme()
        if scheme == "file" or not scheme:
            # Local file - resolve relative to current doc if needed
            path_str = url.toLocalFile()
            if not path_str and url.toString():
                path_str = url.toString()

            p = Path(path_str)
            if not p.is_absolute() and self.current_file:
                p = self.current_file.parent / p
            if p.exists():
                # Open in a new document reader window
                show_document(p)
                return

        # Open in external browser
        from PyQt6.QtGui import QDesktopServices

        QDesktopServices.openUrl(url)

    def _load_markdown(self, path: Path) -> None:
        try:
            import markdown

            with open(path, encoding="utf-8") as f:
                text = f.read()
            html = markdown.markdown(text, extensions=["tables", "fenced_code"])
            self.text_browser.setHtml(html)
            self.stack.setCurrentWidget(self.text_browser)
        except ImportError:
            self._load_plain_text(path)

    def _load_plain_text(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.text_browser.setPlainText(text)
        self.stack.setCurrentWidget(self.text_browser)

    def _load_pdf(self, path: Path) -> None:
        try:
            import pypdfium2 as pdfium
        except ImportError:
            QMessageBox.warning(
                self, "Dependency Missing", "pypdfium2 is required for PDF rendering."
            )
            return

        # Clear existing PDF pages
        for lbl in self.pdf_pages:
            self.pdf_layout.removeWidget(lbl)
            lbl.deleteLater()
        self.pdf_pages.clear()

        self._pdf_document = pdfium.PdfDocument(str(path))
        self._render_pdf_pages()
        self.stack.setCurrentWidget(self.pdf_scroll_area)

    def _render_pdf_pages(self) -> None:
        if not self._pdf_document:
            return

        # Clear layout safely
        while self.pdf_layout.count():
            item = self.pdf_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.pdf_pages.clear()

        scale = 2.0 * self._zoom_level
        for page_idx in range(len(self._pdf_document)):
            page = self._pdf_document[page_idx]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()

            # Convert PIL to QImage
            # ensure image is RGB
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            data = pil_image.tobytes("raw", "RGB")
            qimage = QImage(
                data, pil_image.width, pil_image.height, QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qimage)

            lbl = QLabel()
            lbl.setPixmap(pixmap)
            self.pdf_layout.addWidget(lbl)
            self.pdf_pages.append(lbl)

    def _load_latex(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        # Basic heuristic parsing to make LaTeX readable in the viewer
        html = self._parse_basic_latex(text)
        self.text_browser.setHtml(html)
        self.stack.setCurrentWidget(self.text_browser)

    def _parse_basic_latex(self, text: str) -> str:
        """A simple heuristic parser to make latex source readable."""
        # Remove preamble
        if "\\begin{document}" in text:
            text = text.split("\\begin{document}")[1]
        if "\\end{document}" in text:
            text = text.split("\\end{document}")[0]

        # Replace sections
        text = re.sub(r"\\section\*?\{([^}]+)\}", r"<h1>\1</h1>", text)
        text = re.sub(r"\\subsection\*?\{([^}]+)\}", r"<h2>\1</h2>", text)
        text = re.sub(r"\\subsubsection\*?\{([^}]+)\}", r"<h3>\1</h3>", text)
        text = re.sub(r"\\textbf\{([^}]+)\}", r"<b>\1</b>", text)
        text = re.sub(r"\\textit\{([^}]+)\}", r"<i>\1</i>", text)

        # Lists
        text = text.replace("\\begin{itemize}", "<ul>")
        text = text.replace("\\end{itemize}", "</ul>")
        text = text.replace("\\begin{enumerate}", "<ol>")
        text = text.replace("\\end{enumerate}", "</ol>")
        text = re.sub(r"\\item\s+", r"<li>", text)

        # Newlines
        text = text.replace("\\\\", "<br>")
        text = text.replace("\n\n", "<p>")

        return f"<html><body>{text}</body></html>"

    @pyqtSlot()
    def zoom_in(self) -> None:
        self._zoom_level *= 1.2
        if self.stack.currentWidget() == self.pdf_scroll_area:
            self._render_pdf_pages()
        elif self.stack.currentWidget() == self.text_browser:
            font = self.text_browser.font()
            font.setPointSize(int(font.pointSize() * 1.2))
            self.text_browser.setFont(font)

    @pyqtSlot()
    def zoom_out(self) -> None:
        self._zoom_level /= 1.2
        if self.stack.currentWidget() == self.pdf_scroll_area:
            self._render_pdf_pages()
        elif self.stack.currentWidget() == self.text_browser:
            font = self.text_browser.font()
            font.setPointSize(int(max(8, font.pointSize() / 1.2)))
            self.text_browser.setFont(font)


class DocumentReaderDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("UpstreamDrift Document Reader")
        self.resize(800, 600)
        self.layout = QVBoxLayout(self)
        self.reader = DocumentReaderWidget()
        self.layout.addWidget(self.reader)


_open_readers = []


def show_document(path: str | Path, parent: QWidget | None = None) -> None:
    dlg = DocumentReaderDialog(parent)
    dlg.reader.load_document(path)
    if parent is None:
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.show()
        _open_readers.append(dlg)
        dlg.finished.connect(
            lambda: _open_readers.remove(dlg) if dlg in _open_readers else None
        )
    else:
        dlg.exec()
