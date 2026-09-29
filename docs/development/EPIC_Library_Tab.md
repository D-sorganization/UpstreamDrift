# EPIC: Library Tab and Notebook LM Integration

## Overview

This epic aims to introduce a comprehensive "Library" tab to the UpstreamDrift application workspace. The primary goal is to provide a dedicated space where users can securely manage, view, and search through scientific references, PDF papers, LaTeX documents, and other relevant research material. Furthermore, this library will integrate with Notebook LM to offer advanced document querying, summarization, and AI-driven literature review capabilities.

## Goals

1. **Dedicated Workspace Tab**: Add a fully functional "Library" tab alongside the "Home" tab in the `workspace_tabs` architecture.
2. **Document Management UI**: Build a clean and intuitive interface for importing, categorizing, and viewing PDF and LaTeX documents directly within the application.
3. **Advanced Search & Indexing**: Implement advanced keyword search, full-text indexing, and metadata extraction (potentially leveraging the `pypdf` renaming pipeline).
4. **Sorting and Filtering**: Provide robust sorting and filtering functions by author, year, topic, and custom tags.
5. **Notebook LM Integration**: Connect the library backend to a Notebook LM engine, allowing users to select multiple documents and ask questions across the integrated corpus.

## Technical Requirements

- **Frontend (PyQt6)**:
  - Add a new widget to the `QTabWidget` (Home / Library).
  - Use `QTreeView` or `QTableView` with a custom model for the document browser.
  - Integrate a PDF viewer or text viewer for previews.
- **Backend**:
  - Implement a document ingestion pipeline.
  - Establish an SQLite or document database for fast querying and indexing.
- **Notebook LM Engine**:
  - Integrate an API adapter or local LM model for Notebook LM functionality.
  - Implement a Retrieval-Augmented Generation (RAG) backend utilizing the indexed library documents.

## Proposed Sub-Tasks

### Phase 1: Foundation and UI

- [ ] Add the "Library" tab to `launcher_ui_setup.py`.
- [ ] Create `LibraryWidget` with a basic split view (File Browser on the left, Preview/Metadata on the right).
- [ ] Implement robust file ingest/import logic (copying to a centralized `~/.upstreamdrift/library` folder).

### Phase 2: Indexing and Search

- [ ] Create a local SQLite database for indexing documents.
- [ ] Implement metadata extraction using existing tools (e.g., the established PDF renaming/extraction logic).
- [ ] Implement search bar with full-text keyword matching and advanced boolean query support.

### Phase 3: Notebook LM Integration

- [ ] Design the chat/query interface for Notebook LM inside the Library tab.
- [ ] Build the prompt construction and context ingestion logic.
- [ ] Wire the Sidekick context pipeline to allow the Sidekick panel to read from the selected documents in the Library.

## Acceptance Criteria

- Users can import `.pdf` and `.tex` files.
- Documents appear in a sorted, searchable list.
- Selecting a document displays its metadata and a preview.
- Users can query their library utilizing Notebook LM for summaries and Q&A.
- The UI follows the application's dark/light theme correctly and is visually consistent.

## Design Constraints

- Must follow the established `ThemeManager` pattern for colors.
- Must operate independently of the simulation thread to prevent UI freezing during indexing.
- Avoid introducing unnecessary heavy dependencies (e.g., fallback to simple SQLite text search if a full search engine is too heavy).
