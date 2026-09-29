---
title: Library
tile_id: library_tool
status: complete
---

# Library

## Purpose

Library is a local research-document manager: it copies PDF and LaTeX files
into a per-user library folder, indexes their metadata in SQLite, and offers a
searchable table with a metadata preview pane. It is intended for the
publications and references behind the modelling work.

It is a virtual tile. The registry declares `path: virtual/library`, and
`VIRTUAL_TARGETS` in `src/shared/python/config/tile_target_resolution.py` maps
that pseudo-path to the backing artifact `src/launchers/library_widget.py`
(`LibraryWidget`, hosted by `src/launchers/launcher_layout_manager.py`).
Declared capabilities: `research_library`, `references`, `documents`.

## Inputs

| Input | How it is supplied | Unit or values |
| --- | --- | --- |
| Documents to import | file dialog, filter `Documents (*.pdf *.tex)` with an All Files fallback | file paths |
| Search query | the toolbar search box | whitespace-separated tokens, tokenized with `shlex`; a leading `-` negates a token; the literals `and` and `or` are skipped |
| Document selection | row selection in the table | one row at a time |
| Library location | `~/.golf_modeling_suite/library`, created if absent | filesystem path |
| Index location | `~/.golf_modeling_suite/library/library_index.db` | SQLite file |

## Outputs

| Output | Description |
| --- | --- |
| Imported file | a copy of the source file inside the library folder, made with `shutil.copy2` only when a file of that name is not already there |
| `documents` table row | `id`, `file_name` (unique), `file_path`, `title`, `author`, `year`, `topic`, `added_date` |
| Document table | four columns, Title, Author, Year and Topic, ordered by `added_date` descending, read-only, single-row selection |
| Metadata preview | HTML showing title, author, year, topic and file name, all HTML-escaped |
| Document Chat panel | present but permanently disabled; see [Limitations](#limitations) |

Metadata defaults when nothing better is available: `title` is the source file
stem, `author` is the literal string `Unknown`, `year` is the current year, and
`topic` is `General`. For a PDF, `LibraryManager.add_document` tries
`pypdf.PdfReader` and overrides the title and author from the PDF metadata when
those fields are present. Creation-date parsing for the year is explicitly not
attempted.

## Method

`LibraryManager` owns the storage. `_init_db` issues a
`CREATE TABLE IF NOT EXISTS documents` with the schema above.
`add_document(source_path)` returns `None` for a non-existent source, copies
the file, extracts metadata, then indexes it with `INSERT OR IGNORE`, so a
re-import of the same file name is a no-op rather than a duplicate row.
`get_all_documents()` reads every row with `sqlite3.Row` and returns plain
dicts. Copy and index failures are logged and swallowed rather than raised.

`LibraryWidget` builds a toolbar (title, search box, Import Document button)
above a horizontal splitter: the document table on the left, the metadata
preview and the chat placeholder on the right, with initial sizes 400 and 600.
Filtering is done in `_load_documents`, which rebuilds the table on every
keystroke and matches tokens against the concatenation of title, author, year
and topic, lower-cased. Token logic is AND by default; a token prefixed with
`-` excludes a match; unbalanced quotes fall back to a plain `str.split`. The
full document dict is stashed on the title cell under
`Qt.ItemDataRole.UserRole` so selection needs no second query. Colours come
from `_get_theme_colors()` in `src/launchers/startup.py` with hard-coded dark
fallbacks.

Defining module: `src/launchers/library_widget.py`.

## Limitations

- No document rendering. The preview pane shows metadata plus a note that full
  PDF viewing requires integration with a PDF renderer or WebView. There is no
  page view, no text extraction and no LaTeX rendering.
- Document Chat does nothing. The input is constructed disabled and stays
  disabled on selection; pressing Return appends a message saying the backend
  is not configured. The integration is tracked in
  `docs/development/EPIC_Library_Tab.md` Phase 3.
- Metadata is barely extracted. Only PDF title and author are read, and only
  when `pypdf` is installed. Year and topic are never derived from the
  document, and there is no editing UI for any field, so `Unknown` and
  `General` persist unless the row is edited outside the application.
- No BibTeX or citation-format support. Despite the LaTeX file filter, `.tex`
  files are only copied and indexed by file stem; no bibliography entry,
  citation key or reference export is produced.
- Search is substring matching over four fields. It does not search document
  contents, and `and` and `or` are accepted as tokens but ignored, so no real
  boolean grouping exists.
- No delete, rename, move or re-index action. Rows can only be added.
- Name collisions are silently accepted. A same-named file already in the
  library is not copied and not re-indexed, so a different document with the
  same file name will show the first document's entry.
- Single-user and machine-local. The index lives under the user home directory
  with no sharing, sync or export.

## See Also
- [Project Map](../architecture/PROJECT_MAP.md)
- [Launchers](../user_guide/launchers.md)
- [User manual](../user_guide/user_manual.md)
