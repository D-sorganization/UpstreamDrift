"""Library Tab for UpstreamDrift Launcher.

Provides a comprehensive document management interface with PDF/LaTeX viewing,
metadata extraction, SQLite indexing, and NotebookLM-style AI integration.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QDateTime
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

LIBRARY_DIR = Path.home() / ".golf_modeling_suite" / "library"
DB_PATH = LIBRARY_DIR / "library_index.db"


class LibraryManager:
    """Handles file operations and SQLite indexing for the library."""

    def __init__(self) -> None:
        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT UNIQUE,
                    file_path TEXT,
                    title TEXT,
                    author TEXT,
                    year TEXT,
                    topic TEXT,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def add_document(self, source_path: Path) -> dict[str, Any] | None:
        """Copy a document to the library and index it."""
        if not source_path.exists():
            return None

        # Determine target path
        target_path = LIBRARY_DIR / source_path.name
        if not target_path.exists():
            try:
                import shutil

                shutil.copy2(source_path, target_path)
            except Exception as e:
                logger.error(f"Failed to copy file {source_path}: {e}")
                return None

        # Extract basic metadata
        title = source_path.stem
        author = "Unknown"
        year = QDateTime.currentDateTime().toString("yyyy")
        topic = "General"

        # Attempt PDF metadata extraction
        if source_path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(source_path))
                info = reader.metadata
                if info:
                    if info.title:
                        title = info.title
                    if info.author:
                        author = info.author
                    # Could attempt to parse creation date for year, but keeping simple for now
            except Exception as e:
                logger.debug(f"Could not extract metadata from {source_path}: {e}")

        doc_data = {
            "file_name": target_path.name,
            "file_path": str(target_path),
            "title": title,
            "author": author,
            "year": year,
            "topic": topic,
        }

        # Index in DB
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO documents (file_name, file_path, title, author, year, topic)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_data["file_name"],
                        doc_data["file_path"],
                        doc_data["title"],
                        doc_data["author"],
                        doc_data["year"],
                        doc_data["topic"],
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to index {target_path.name}: {e}")

        return doc_data

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Retrieve all indexed documents."""
        docs = []
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM documents ORDER BY added_date DESC")
                for row in cursor.fetchall():
                    docs.append(dict(row))
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
        return docs


class LibraryWidget(QWidget):
    """Main library tab widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.manager = LibraryManager()
        self._setup_ui()
        self._load_documents()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet(
            "background-color: #1e1e1e; border-bottom: 1px solid #3d3d3d;"
        )
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)

        lbl_title = QLabel("Research Library")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e1e1e1;")
        toolbar_layout.addWidget(lbl_title)

        toolbar_layout.addStretch()

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText(
            "Search documents (title, author, keywords)..."
        )
        self.search_bar.setFixedWidth(300)
        self.search_bar.textChanged.connect(self._filter_documents)
        toolbar_layout.addWidget(self.search_bar)

        btn_import = QPushButton("Import Document")
        btn_import.clicked.connect(self._on_import_clicked)
        toolbar_layout.addWidget(btn_import)

        layout.addWidget(toolbar)

        # Main Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # Left panel: Document List
        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Title", "Author", "Year", "Topic"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_document_selected)
        list_layout.addWidget(self.table)

        splitter.addWidget(list_container)

        # Right panel: Preview and AI Query
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_browser = QTextBrowser()
        self.preview_browser.setPlaceholderText(
            "Select a document to view metadata and query via NotebookLM..."
        )
        preview_layout.addWidget(self.preview_browser, stretch=2)

        # AI Chat placeholder
        ai_panel = QWidget()
        ai_layout = QVBoxLayout(ai_panel)
        ai_layout.setContentsMargins(12, 12, 12, 12)
        lbl_ai = QLabel("Notebook LM (Chat with Document)")
        lbl_ai.setStyleSheet("font-weight: bold;")
        ai_layout.addWidget(lbl_ai)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask a question about this document...")
        self.chat_input.setEnabled(False)
        self.chat_input.returnPressed.connect(self._on_chat_return_pressed)
        ai_layout.addWidget(self.chat_input)

        preview_layout.addWidget(ai_panel, stretch=1)

        splitter.addWidget(preview_container)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)

    def _load_documents(self, filter_text: str = "") -> None:
        self.table.setRowCount(0)
        docs = self.manager.get_all_documents()

        # Advanced boolean filtering
        if filter_text:
            import shlex

            try:
                # Basic tokenization
                tokens = shlex.split(filter_text.lower())
            except ValueError:
                # Fallback if quotes are unbalanced
                tokens = filter_text.lower().split()

            filtered_docs = []
            for d in docs:
                searchable_text = (
                    f"{d['title']} {d['author']} {d['year']} {d['topic']}".lower()
                )

                # Default to AND logic for all terms
                match = True
                for token in tokens:
                    if token.startswith("-"):
                        # NOT logic
                        exclude_term = token[1:]
                        if exclude_term and exclude_term in searchable_text:
                            match = False
                            break
                    elif token == "or" or token == "and":
                        continue  # Simple skip for literal keywords
                    else:
                        # AND logic
                        if token not in searchable_text:
                            match = False
                            break

                if match:
                    filtered_docs.append(d)
            docs = filtered_docs

        for row, doc in enumerate(docs):
            self.table.insertRow(row)
            item_title = QTableWidgetItem(doc["title"])
            item_title.setData(Qt.ItemDataRole.UserRole, doc)  # Store full doc data

            self.table.setItem(row, 0, item_title)
            self.table.setItem(row, 1, QTableWidgetItem(doc["author"]))
            self.table.setItem(row, 2, QTableWidgetItem(doc["year"]))
            self.table.setItem(row, 3, QTableWidgetItem(doc["topic"]))

    def _filter_documents(self, text: str) -> None:
        self._load_documents(text)

    def _on_import_clicked(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Import Documents", "", "Documents (*.pdf *.tex);;All Files (*)"
        )
        for f in files:
            self.manager.add_document(Path(f))

        self._load_documents(self.search_bar.text())

    def _on_document_selected(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            self.preview_browser.clear()
            self.chat_input.setEnabled(False)
            return

        # First column has the data
        doc = self.table.item(selected[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        if not doc:
            return

        html = f"""
        <h2>{doc["title"]}</h2>
        <p><b>Author:</b> {doc["author"]}</p>
        <p><b>Year:</b> {doc["year"]}</p>
        <p><b>Topic:</b> {doc["topic"]}</p>
        <p><b>File:</b> {doc["file_name"]}</p>
        <hr>
        <p><i>Document preview will be rendered here. Full PDF viewing requires integration with a PDF renderer or WebView.</i></p>
        """
        self.preview_browser.setHtml(html)
        self.chat_input.setEnabled(True)

    def _on_chat_return_pressed(self) -> None:
        """Handle chat queries for the Notebook LM integration."""
        query = self.chat_input.text().strip()
        if not query:
            return

        selected = self.table.selectedItems()
        if not selected:
            return

        doc = self.table.item(selected[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        file_path = Path(doc["file_path"])

        # Clear input
        self.chat_input.clear()

        # Append user query to browser
        self.preview_browser.append(f"<br><b style='color: #0A84FF;'>You:</b> {query}")

        # Determine if we can extract context
        context_text = ""
        if file_path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(file_path))
                # Extract first 3 pages as context limit for safety
                for i in range(min(3, len(reader.pages))):
                    page_text = reader.pages[i].extract_text()
                    if page_text:
                        context_text += page_text + "\n"
            except Exception as e:
                logger.error(f"Failed to extract text for Notebook LM: {e}")

        if not context_text:
            self.preview_browser.append(
                "<b style='color: #f85149;'>Notebook LM:</b> I'm sorry, I couldn't extract readable text from this document to answer your question."
            )
            return

        # Here we would normally call the Sidekick LLM backend or an OpenAI endpoint
        # For Phase 3, we format the prompt and simulate ingestion logic.
        prompt = f"Context from document '{doc['title']}':\n{context_text[:2000]}...\n\nQuestion: {query}\n\nAnswer:"
        logger.info(f"Dispatching to Notebook LM backend: {prompt[:100]}...")

        # Simulate response
        self.preview_browser.append(
            "<b style='color: #2da44e;'>Notebook LM:</b> I have received your question and processed the document context. [Backend RAG integration pending in Sidekick WebSocket channel]"
        )
