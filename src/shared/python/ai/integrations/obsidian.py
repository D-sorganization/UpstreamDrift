"""Obsidian integration tools for Sidekick — local-vault file client.

Phase 2 of Tools #2759 (closed prematurely as ``completed`` on 2026-05-15).
This module replaces the previous ``NotImplementedError`` stubs with a real
local-filesystem Obsidian Vault client. No external dependencies, no network
calls — operates purely on a configured directory of ``.md`` files.

Configuration precedence (high → low):

1. Programmatic override via :func:`set_obsidian_vault_path`
2. ``OBSIDIAN_VAULT_PATH`` environment variable

Path safety: every public function refuses ``..`` segments, absolute paths,
Windows drive-letter prefixes (``C:\\``) and any resolved path that escapes
the configured vault root. Violations raise :class:`ObsidianPathError`.

See: https://github.com/D-sorganization/Tools/issues/2896
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

_OBSIDIAN_VAULT_PATH: Path | None = None
_VAULT_ENV_VAR = "OBSIDIAN_VAULT_PATH"
_NOTE_EXT = ".md"
_SEARCH_SNIPPET_CONTEXT = 80  # chars of context on each side of a hit


class ObsidianPathError(ValueError):
    """Raised when a note path violates vault sandboxing rules.

    Subclasses :class:`ValueError` so existing callers that catch ``ValueError``
    for general input-validation errors continue to work.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def set_obsidian_vault_path(path: str | Path) -> None:
    """Configure the local Obsidian Vault root for this process.

    Precondition: ``path`` must point to an existing directory. Symlinks are
    resolved so subsequent reads/writes operate on the real filesystem
    location.

    Args:
        path: Filesystem path to the vault root.

    Raises:
        FileNotFoundError: When the path does not exist.
        NotADirectoryError: When the path exists but is not a directory.
    """
    global _OBSIDIAN_VAULT_PATH  # noqa: PLW0603
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Obsidian vault path does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Obsidian vault path is not a directory: {candidate}")
    _OBSIDIAN_VAULT_PATH = candidate.resolve()
    logger.info("Obsidian vault configured at %s", _OBSIDIAN_VAULT_PATH)


def _get_vault_root() -> Path:
    """Return the active vault root, honoring env-var fallback.

    Raises:
        RuntimeError: When no vault is configured (neither programmatically
            nor via ``OBSIDIAN_VAULT_PATH``).
    """
    if _OBSIDIAN_VAULT_PATH is not None:
        return _OBSIDIAN_VAULT_PATH
    env_path = os.environ.get(_VAULT_ENV_VAR)
    if env_path:
        candidate = Path(env_path).resolve()
        if not candidate.is_dir():
            raise RuntimeError(
                f"{_VAULT_ENV_VAR}={env_path!r} does not point to a directory."
            )
        return candidate
    raise RuntimeError(
        "Obsidian vault path is not configured. Call set_obsidian_vault_path()"
        f" or set the {_VAULT_ENV_VAR} environment variable."
    )


# ---------------------------------------------------------------------------
# Path-safety helpers (DRY: one resolver shared by all four tools)
# ---------------------------------------------------------------------------


def _looks_like_absolute(raw: str) -> bool:
    """Detect absolute-path forms before pathlib normalizes them away."""
    if not raw:
        return False
    if raw.startswith(("/", "\\")):
        return True
    # Windows drive letter, e.g. "C:\\..." or "C:/..."
    return bool(len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha())


def _resolve_safe_path(
    note_name: str,
    *,
    allow_missing_suffix: bool = True,
    require_md: bool = True,
) -> Path:
    """Resolve ``note_name`` against the vault root, refusing traversal.

    DbC preconditions:

    * ``note_name`` must be a non-empty string.
    * No segment may be ``..``.
    * The raw string must not be absolute (POSIX root, UNC, or Windows
      drive-letter).
    * The resolved path must remain inside the configured vault root.

    Args:
        note_name: Caller-supplied note identifier (with or without ``.md``).
        allow_missing_suffix: When ``True``, append ``.md`` if not present.
        require_md: When ``True``, the final path must end in ``.md``.

    Returns:
        The resolved absolute path inside the vault.

    Raises:
        TypeError: When ``note_name`` is not a string.
        ValueError: When ``note_name`` is empty / whitespace.
        ObsidianPathError: On any sandbox violation.
    """
    if not isinstance(note_name, str):
        raise TypeError(f"note_name must be a string, got {type(note_name).__name__}")
    if not note_name.strip():
        raise ValueError("note_name must be a non-empty string")
    if _looks_like_absolute(note_name):
        raise ObsidianPathError(f"Absolute note paths are not allowed: {note_name!r}")

    # Normalize separators and reject any "..".
    parts = note_name.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        raise ObsidianPathError(
            f"Path traversal segment '..' is not allowed: {note_name!r}"
        )

    root = _get_vault_root()
    candidate = root.joinpath(*[p for p in parts if p not in ("", ".")])
    if allow_missing_suffix and candidate.suffix.lower() != _NOTE_EXT:
        candidate = candidate.with_suffix(_NOTE_EXT)
    if require_md and candidate.suffix.lower() != _NOTE_EXT:
        raise ObsidianPathError(
            f"Note path must end in {_NOTE_EXT}: {candidate.name!r}"
        )

    # Anchor sandboxing on the *parent* directory (the note may not exist yet).
    # We resolve(strict=False) to collapse symlinks where possible.
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ObsidianPathError(
            f"Resolved path escapes vault root: {resolved} not under {root}"
        ) from exc
    return resolved


def _resolve_safe_folder(folder: str) -> Path:
    """Resolve a sub-folder of the vault with the same sandboxing rules.

    Empty string means "vault root". Raises :class:`ObsidianPathError` on
    any violation, :class:`FileNotFoundError` when the folder does not exist.
    """
    root = _get_vault_root()
    if folder == "" or folder is None:
        return root
    if not isinstance(folder, str):
        raise TypeError(f"folder must be a string, got {type(folder).__name__}")
    if _looks_like_absolute(folder):
        raise ObsidianPathError(f"Absolute folder paths are not allowed: {folder!r}")
    parts = folder.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        raise ObsidianPathError(
            f"Path traversal segment '..' is not allowed: {folder!r}"
        )
    candidate = root.joinpath(*[p for p in parts if p not in ("", ".")]).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ObsidianPathError(
            f"Resolved folder escapes vault root: {candidate} not under {root}"
        ) from exc
    if not candidate.exists():
        raise FileNotFoundError(f"Folder does not exist in vault: {folder!r}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder!r}")
    return candidate


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------

registry = get_global_registry()


@registry.register(
    "obsidian_read_note",
    "Read a markdown note from the configured Obsidian Vault.",
    category=ToolCategory.DATA_LOADING,
)
def obsidian_read_note(note_name: str) -> dict[str, Any]:
    """Read a markdown note from the local Obsidian Vault.

    DbC preconditions:

    * ``note_name`` is a non-empty string.
    * Path is sandboxed inside the configured vault (no ``..``, no absolute
      paths, no drive-letter prefixes).

    Args:
        note_name: The note identifier, with or without the ``.md`` suffix.

    Returns:
        ``{"content": str, "path": str, "modified_at": str (ISO-8601 UTC)}``.

    Raises:
        TypeError: When ``note_name`` is not a string.
        ValueError: When ``note_name`` is empty.
        ObsidianPathError: On sandbox violations (subclass of ValueError).
        RuntimeError: When no vault path is configured.
        FileNotFoundError: When the note does not exist.
    """
    path = _resolve_safe_path(note_name)
    if not path.exists():
        raise FileNotFoundError(f"Note not found in vault: {note_name!r}")
    content = path.read_text(encoding="utf-8")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    logger.debug("Obsidian read %s (%d bytes)", path, len(content))
    return {
        "content": content,
        "path": str(path),
        "modified_at": mtime.isoformat(),
    }


@registry.register(
    "obsidian_write_note",
    "Create or update a markdown note in the configured Obsidian Vault.",
    category=ToolCategory.ANALYSIS,
    requires_confirmation=True,
)
def obsidian_write_note(
    note_name: str, markdown_content: str, overwrite: bool = False
) -> dict[str, Any]:
    """Write markdown content to a note in the local Obsidian Vault.

    DbC preconditions:

    * ``note_name`` is a non-empty string, sandbox-safe.
    * ``markdown_content`` is a string (may be empty).
    * If the note already exists, ``overwrite`` must be ``True``.

    Parent directories are created on demand so callers can write to
    ``inbox/today/log`` without first mkdir'ing the tree.

    Args:
        note_name: Target note identifier.
        markdown_content: Markdown body to write (UTF-8 encoded on disk).
        overwrite: Allow replacing an existing note.

    Returns:
        ``{"success": bool, "path": str, "bytes_written": int,
        "created": bool}``.

    Raises:
        TypeError: When arguments are the wrong type.
        ValueError / ObsidianPathError: On sandbox / validation failures.
        RuntimeError: When no vault path is configured.
        FileExistsError: When the note exists and ``overwrite`` is ``False``.
    """
    if not isinstance(markdown_content, str):
        raise TypeError(
            f"markdown_content must be a string, got {type(markdown_content).__name__}"
        )
    path = _resolve_safe_path(note_name)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Note already exists (set overwrite=True to replace): {note_name!r}"
        )
    created = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = markdown_content.encode("utf-8")
    path.write_bytes(encoded)
    logger.info("Obsidian wrote %s (%d bytes, created=%s)", path, len(encoded), created)
    return {
        "success": True,
        "path": str(path),
        "bytes_written": len(encoded),
        "created": created,
    }


@registry.register(
    "obsidian_list_notes",
    "List all markdown notes in the configured Obsidian Vault (or sub-folder).",
    category=ToolCategory.DATA_LOADING,
)
def obsidian_list_notes(folder: str = "") -> dict[str, Any]:
    """List markdown notes under the configured vault (or a sub-folder).

    Paths in the result are relative to the vault root and use forward
    slashes regardless of the host OS, matching how Obsidian internally
    represents them.

    Args:
        folder: Optional sub-folder of the vault. Empty string = vault root.

    Returns:
        ``{"notes": list[str], "count": int, "folder": str}``.

    Raises:
        ObsidianPathError: When ``folder`` escapes the vault.
        FileNotFoundError: When ``folder`` does not exist.
        RuntimeError: When no vault path is configured.
    """
    base = _resolve_safe_folder(folder)
    root = _get_vault_root()
    notes: list[str] = []
    for path in sorted(base.rglob(f"*{_NOTE_EXT}")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        notes.append(rel)
    logger.debug("Obsidian listed %d notes under %s", len(notes), base)
    return {"notes": notes, "count": len(notes), "folder": folder or ""}


@registry.register(
    "obsidian_search",
    "Search note bodies in the configured Obsidian Vault for a substring.",
    category=ToolCategory.ANALYSIS,
)
def obsidian_search(query: str) -> dict[str, Any]:
    """Case-insensitive substring search across all ``.md`` notes.

    Naive grep — no indexing. Sufficient for Phase 1 per Tools #2896 spec.

    Args:
        query: Substring to find. Must be non-empty.

    Returns:
        ``{"matches": list[{"path": str, "line": int, "snippet": str}],
        "count": int, "query": str}``.

    Raises:
        TypeError: When ``query`` is not a string.
        ValueError: When ``query`` is empty.
        RuntimeError: When no vault path is configured.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")
    if not query.strip():
        raise ValueError("query must be a non-empty string")

    root = _get_vault_root()
    needle = query.lower()
    matches: list[dict[str, Any]] = []
    for path in sorted(root.rglob(f"*{_NOTE_EXT}")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.debug("Skipped unreadable note: %s", path)
            continue
        lower = text.lower()
        start = 0
        while True:
            idx = lower.find(needle, start)
            if idx < 0:
                break
            line_no = text.count("\n", 0, idx) + 1
            snip_start = max(0, idx - _SEARCH_SNIPPET_CONTEXT)
            snip_end = min(len(text), idx + len(query) + _SEARCH_SNIPPET_CONTEXT)
            snippet = text[snip_start:snip_end].replace("\n", " ")
            matches.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "line": line_no,
                    "snippet": snippet,
                }
            )
            start = idx + len(query)
    logger.info("Obsidian search %r → %d hits", query, len(matches))
    return {"matches": matches, "count": len(matches), "query": query}
