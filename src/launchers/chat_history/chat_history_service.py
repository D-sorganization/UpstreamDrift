"""Thin LOD-compliant adapter from launcher UI to the Sidekick conversation
service and ``user_memory.json`` persistent memory document.

This module is the *only* boundary between the UpstreamDrift launcher's
chat panel and the Sidekick package (Tools repo, PR #2879). UI widgets in
``history_pane.py`` and ``memory_panel.py`` MUST go through
:class:`HistoryServiceAdapter` rather than reaching into Sidekick's
internal SQLite/JSON storage directly. This isolates UD from upstream
schema changes and lets the unit tests inject a mock service.

Implements UpstreamDrift #5621.

DbC summary
-----------
* :meth:`HistoryServiceAdapter.search` requires a non-empty stripped query.
* :meth:`HistoryServiceAdapter.load_as_context` raises ``KeyError`` when
  the conversation id is unknown.
* :meth:`HistoryServiceAdapter.save_memory` requires a ``dict`` payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

# Skeleton returned when the on-disk user_memory.json document is absent.
_EMPTY_MEMORY: dict[str, dict[str, Any]] = {
    "identity": {},
    "preferences": {},
    "projects": {},
    "knowledge": {},
}

#: The four structured top-level sections the memory document supports.
MEMORY_SECTIONS: tuple[str, ...] = (
    "identity",
    "preferences",
    "projects",
    "knowledge",
)


class _ConversationService(Protocol):
    """Structural protocol the adapter expects from the Sidekick service.

    Matches the surface shipped by Tools #2879. Methods not listed here
    are *intentionally* invisible to the launcher (LOD).
    """

    def list(self, archived: bool = False) -> list[dict[str, Any]]: ...
    def search(self, query: str) -> list[dict[str, Any]]: ...
    def archive(self, conversation_id: str) -> None: ...
    def unarchive(self, conversation_id: str) -> None: ...
    def delete(self, conversation_id: str) -> None: ...
    def export(self, conversation_id: str, path: str) -> None: ...
    def load_as_context(self, conversation_id: str) -> dict[str, Any] | None: ...


def _default_memory_path() -> Path:
    """Return the canonical on-disk location for ``user_memory.json``."""
    return Path.home() / ".golf_modeling_suite" / "user_memory.json"


def _load_sidekick_conversation_service() -> Any | None:
    """Best-effort lazy import of the Sidekick conversation service.

    Returns ``None`` when Sidekick is not yet vendored/installed so the
    launcher can boot in development environments without the submodule.
    """
    try:
        from sidekick.chat import conversation as _conv  # type: ignore[import-not-found]

        return _conv
    except Exception:  # noqa: BLE001 — any import-time failure must not crash UI
        return None


class HistoryServiceAdapter:
    """Translate launcher UI actions into Sidekick service calls.

    All UI code in this package talks to *this* adapter only — never
    directly to Sidekick internals. That keeps the boundary tight (LOD)
    and lets the unit tests substitute a ``MagicMock`` for the service.

    Parameters
    ----------
    service:
        The Sidekick conversation service (or any object satisfying
        :class:`_ConversationService`). When ``None``, the adapter
        attempts to import ``sidekick.chat.conversation`` lazily; if
        that import also fails the constructor still succeeds but every
        delegating call raises ``RuntimeError`` to signal the missing
        backend to the caller.
    memory_path:
        Override location of the ``user_memory.json`` document. Defaults
        to ``~/.golf_modeling_suite/user_memory.json``.
    """

    def __init__(
        self,
        service: Any | None = None,
        memory_path: Path | None = None,
    ) -> None:
        self._service = (
            service if service is not None else (_load_sidekick_conversation_service())
        )
        self._memory_path = (
            Path(memory_path) if memory_path is not None else _default_memory_path()
        )

    # ------------------------------------------------------------------
    # Service availability + helpers
    # ------------------------------------------------------------------

    @property
    def memory_path(self) -> Path:
        """Return the on-disk path of the ``user_memory.json`` document."""
        return self._memory_path

    def has_service(self) -> bool:
        """Return ``True`` when a Sidekick conversation service is wired."""
        return self._service is not None

    def _require_service(self) -> Any:
        if self._service is None:
            raise RuntimeError(
                "Sidekick conversation service is not available "
                "(install/update the sidekick package or run UD with "
                "the vendor/ud-tools submodule checked out)."
            )
        return self._service

    # ------------------------------------------------------------------
    # Conversation listing + search
    # ------------------------------------------------------------------

    def list_active(self) -> list[dict[str, Any]]:
        """Return non-archived conversations newest-first."""
        return self._require_service().list(archived=False)

    def list_archived(self) -> list[dict[str, Any]]:
        """Return archived conversations newest-first."""
        return self._require_service().list(archived=True)

    def search(self, query: str) -> list[dict[str, Any]]:
        """Full-text search across all conversations.

        Precondition: ``query`` must be a non-empty stripped string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be a non-empty string")
        return self._require_service().search(query)

    # ------------------------------------------------------------------
    # Per-conversation actions
    # ------------------------------------------------------------------

    def archive(self, conversation_id: str) -> None:
        """Archive a conversation (soft-delete; reversible via restore)."""
        self._require_service().archive(conversation_id)

    def unarchive(self, conversation_id: str) -> None:
        """Restore an archived conversation to the active list."""
        self._require_service().unarchive(conversation_id)

    # Restore is the user-facing verb; unarchive is the service verb.
    restore = unarchive

    def delete(self, conversation_id: str) -> None:
        """Permanently delete a conversation."""
        self._require_service().delete(conversation_id)

    def export(self, conversation_id: str, path: str) -> None:
        """Export a conversation to ``path`` (JSON or Markdown by ext)."""
        self._require_service().export(conversation_id, path)

    def load_as_context(self, conversation_id: str) -> dict[str, Any]:
        """Return a conversation payload suitable for replacing the live
        context in the chat panel.

        Postcondition: returned dict has a ``session_id`` matching the
        requested conversation. Raises ``KeyError`` when no such
        conversation exists.
        """
        result = self._require_service().load_as_context(conversation_id)
        if result is None:
            raise KeyError(f"conversation '{conversation_id}' not found")
        return result

    # ------------------------------------------------------------------
    # Memory document (~/.golf_modeling_suite/user_memory.json)
    # ------------------------------------------------------------------

    def load_memory(self) -> dict[str, Any]:
        """Return the structured memory document.

        Missing or unreadable files return an empty four-section
        skeleton so callers can render an editor unconditionally.
        """
        if not self._memory_path.exists():
            return {key: {} for key in MEMORY_SECTIONS}
        try:
            raw = self._memory_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return {key: {} for key in MEMORY_SECTIONS}
        # Ensure every section exists, even if the on-disk file omits one.
        for section in MEMORY_SECTIONS:
            data.setdefault(section, {})
        return data

    def save_memory(self, memory: dict[str, Any]) -> None:
        """Write the memory document to disk.

        Precondition: ``memory`` must be a ``dict``.
        """
        if not isinstance(memory, dict):
            raise TypeError("memory must be a dict")
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_path.write_text(
            json.dumps(memory, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def reset_memory(self) -> None:
        """Delete the memory document. No-op when the file is absent."""
        if self._memory_path.exists():
            self._memory_path.unlink()

    # ------------------------------------------------------------------
    # Optional Tools #2736 condensation API (with graceful fallback)
    # ------------------------------------------------------------------

    def condense_to_memory(self, conversation_ids: list[str]) -> dict[str, Any]:
        """Ask the Sidekick condenser to merge old conversations into
        the structured memory document.

        Falls back to a deterministic stub response when the Sidekick
        condensation API (Tools #2736) is not yet wired so the launcher
        can ship the UI ahead of the backend.
        """
        condenser = getattr(self._service, "condense_to_memory", None)
        if condenser is None:
            return {
                "status": "stub",
                "message": (
                    "Memory condensation API is not available in this "
                    "Sidekick build. Update the sidekick package once "
                    "Tools #2736 is merged."
                ),
            }
        return condenser(conversation_ids)
