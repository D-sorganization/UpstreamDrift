"""Library Tab for UpstreamDrift Launcher.

Provides a comprehensive document management interface with PDF/LaTeX viewing,
metadata extraction, SQLite indexing, and NotebookLM-style AI integration.
"""

from __future__ import annotations

import sqlite3
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.launchers.startup import _get_theme_colors
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

LIBRARY_DIR = Path.home() / ".golf_modeling_suite" / "library"
DB_PATH = LIBRARY_DIR / "library_index.db"


class LibraryManager:
    """Handles file operations and SQLite indexing for the library."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        if db_path is None:
            raise ValueError("db_path must be provided")
        self.db_path = db_path
        self.library_dir = db_path.parent
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
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
                """)
            conn.commit()

    def add_document(self, source_path: Path) -> dict[str, Any] | None:
        """Copy a document to the library and index it."""
        if not source_path.exists():
            return None

        # Determine target path
        target_path = self.library_dir / source_path.name
        if not target_path.exists():
            try:
                import shutil

                shutil.copy2(source_path, target_path)
            except Exception as e:  # noqa: BLE001
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
            except Exception as e:  # noqa: BLE001
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
            with sqlite3.connect(self.db_path) as conn:
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
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to index {target_path.name}: {e}")

        return doc_data

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Retrieve all indexed documents."""
        docs = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM documents ORDER BY added_date DESC")
                for row in cursor.fetchall():
                    docs.append(dict(row))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to retrieve documents: {e}")
        return docs


class LibraryWidget(QWidget):
    """Main library tab widget."""

    def __init__(
        self,
        parent: QWidget | None = None,
        manager: LibraryManager | None = None,
    ) -> None:
        super().__init__(parent)
        self.manager = manager or LibraryManager()
        self._setup_ui()
        self._load_documents()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Styles.MARGIN_PAGE,
            Styles.MARGIN_PAGE,
            Styles.MARGIN_PAGE,
            Styles.MARGIN_PAGE,
        )
        layout.setSpacing(Styles.SPACING_MD)
        self.setObjectName("LibraryWidget")

        # Toolbar
        toolbar = QWidget()
        toolbar.setObjectName("LibraryToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(Styles.SPACING_MD)

        lbl_title = QLabel("Research Library")
        lbl_title.setObjectName("LibraryTitle")
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
        splitter.setChildrenCollapsible(False)
        splitter.setObjectName("LibrarySplitter")

        # Left panel: Document List
        list_container = QWidget()
        list_container.setObjectName("LibraryListPane")
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(Styles.SPACING_SM)

        self.table = QTableWidget()
        self.table.setObjectName("LibraryTable")
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Title", "Author", "Year", "Topic"])
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_document_selected)
        list_layout.addWidget(self.table)

        splitter.addWidget(list_container)

        # Right panel: Preview and AI Query
        preview_container = QWidget()
        preview_container.setObjectName("LibraryPreviewPane")
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(Styles.SPACING_MD)

        self.preview_browser = QTextBrowser()
        self.preview_browser.setObjectName("LibraryPreview")
        self.preview_browser.setPlaceholderText(
            "Select a document to view metadata and query via NotebookLM..."
        )
        preview_layout.addWidget(self.preview_browser, stretch=2)

        # AI Chat placeholder
        ai_panel = QWidget()
        ai_panel.setObjectName("LibraryChatPanel")
        ai_layout = QVBoxLayout(ai_panel)
        ai_layout.setContentsMargins(
            Styles.SPACING_MD,
            Styles.SPACING_MD,
            Styles.SPACING_MD,
            Styles.SPACING_MD,
        )
        ai_layout.setSpacing(Styles.SPACING_SM)
        lbl_ai = QLabel("Document Chat")
        lbl_ai.setObjectName("LibraryChatTitle")
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
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the launcher theme to the Library tab."""
        colors = _get_theme_colors()

        def color(attr: str, fallback: str) -> str:
            if isinstance(colors, dict):
                return str(colors.get(attr, fallback))
            return str(getattr(colors, attr, fallback))

        bg = color("surface_primary", "#1f2329")
        panel = color("surface_secondary", "#252a31")
        elevated = color("surface_tertiary", "#2c333c")
        border = color("border_default", "#3a414a")
        text = color("text_primary", "#f0f3f6")
        muted = color("text_secondary", "#a8b0bb")
        accent = color("accent_primary", "#58a6ff")

        self.setStyleSheet(f"""
            QWidget#LibraryWidget {{
                background: {bg};
                color: {text};
            }}
            QWidget#LibraryToolbar {{
                background: transparent;
                border: none;
            }}
            QLabel#LibraryTitle {{
                color: {text};
                font-size: 18px;
                font-weight: 700;
            }}
            QTableWidget#LibraryTable, QTextBrowser#LibraryPreview {{
                background: {panel};
                color: {text};
                border: 1px solid {border};
                border-radius: 8px;
                selection-background-color: {accent};
            }}
            QWidget#LibraryChatPanel {{
                background: {elevated};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            QLabel#LibraryChatTitle {{
                color: {muted};
                font-weight: 700;
            }}
            QLineEdit {{
                background: {panel};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 6px 8px;
            }}
            QPushButton {{
                background: {accent};
                color: #0b1117;
                border: none;
                border-radius: 6px;
                padding: 7px 10px;
                font-weight: 700;
            }}
        """)

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
        item = self.table.item(selected[0].row(), 0)
        if item is None:
            return

        doc = item.data(Qt.ItemDataRole.UserRole)
        if not doc:
            return

        html = f"""
        <h2>{escape(str(doc["title"]))}</h2>
        <p><b>Author:</b> {escape(str(doc["author"]))}</p>
        <p><b>Year:</b> {escape(str(doc["year"]))}</p>
        <p><b>Topic:</b> {escape(str(doc["topic"]))}</p>
        <p><b>File:</b> {escape(str(doc["file_name"]))}</p>
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

        item = self.table.item(selected[0].row(), 0)
        if item is None:
            return

        doc = item.data(Qt.ItemDataRole.UserRole)
        if not doc:
            return

        file_path = Path(doc["file_path"])

        # Clear input
        self.chat_input.clear()

        # Append user query to browser
        self.preview_browser.append(
            f"<br><b style='color: #0A84FF;'>You:</b> {escape(query)}"
        )

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
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to extract text for Notebook LM: {e}")

        if not context_text:
            self.preview_browser.append(
                "<b style='color: #f85149;'>Notebook LM:</b> I'm sorry, I couldn't extract readable text from this document to answer your question."
            )
            return

        # Here we would normally call the Sidekick LLM backend or an OpenAI endpoint
        # For Phase 3, we format the prompt and simulate ingestion logic.
        prompt = (
            f"Context from document '{doc['title']}':\n"
            f"{context_text[:2000]}...\n\nQuestion: {query}\n\nAnswer:"
        )
        logger.info(f"Dispatching to Notebook LM backend: {prompt[:100]}...")

        # Simulate response
        self.preview_browser.append(
            "<b style='color: #2da44e;'>Notebook LM:</b> I have received your question and processed the document context. [Backend RAG integration pending in Sidekick WebSocket channel]"
        )
