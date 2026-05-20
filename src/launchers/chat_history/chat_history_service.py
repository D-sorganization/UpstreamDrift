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
import importlib
import importlib.util
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _MissingConversationService:
    """Diagnostic placeholder for discovered shared chat capability.

    The shared Tools chat package can provide condensation helpers without
    providing a standalone conversation-history service. Keep that distinction
    explicit so the launcher does not silently pretend history persistence is
    wired.
    """

    reason: str
    condensation_available: bool = False

    def list(self, archived: bool = False) -> list[dict[str, Any]]:
        raise RuntimeError(self.reason)

    def search(self, query: str) -> list[dict[str, Any]]:
        raise RuntimeError(self.reason)

    def archive(self, conversation_id: str) -> None:
        raise RuntimeError(self.reason)

    def unarchive(self, conversation_id: str) -> None:
        raise RuntimeError(self.reason)

    def delete(self, conversation_id: str) -> None:
        raise RuntimeError(self.reason)

    def export(self, conversation_id: str, path: str) -> None:
        raise RuntimeError(self.reason)

    def load_as_context(self, conversation_id: str) -> dict[str, Any] | None:
        raise RuntimeError(self.reason)

    def condense_to_memory(self, conversation_ids: list[str]) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "requested": len(conversation_ids),
            "processed": 0,
            "inserted": 0,
            "missing": list(conversation_ids),
            "message": self.reason,
        }


def _default_memory_path() -> Path:
    """Return the canonical on-disk location for ``user_memory.json``."""
    return Path.home() / ".golf_modeling_suite" / "user_memory.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_shared_python_roots() -> tuple[Path, ...]:
    root = _repo_root()
    return (
        root / "vendor" / "ud-tools" / "src" / "shared" / "python",
        root / "src" / "shared" / "python",
    )


def _load_module_from_root(module_name: str, root: Path) -> Any | None:
    module_path = root.joinpath(*module_name.split("."))
    file_path = module_path.with_suffix(".py")
    if not file_path.exists():
        file_path = module_path / "__init__.py"
    if not file_path.exists():
        return None

    spec_name = f"_upstream_drift_tools_probe_{module_name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(spec_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec_name, None)
        raise
    return module


def _resolve_conversation_service(module: Any) -> Any | None:
    for attr_name in ("conversation_service", "service", "SERVICE"):
        service = getattr(module, attr_name, None)
        if service is not None:
            return service
    factory = getattr(module, "get_conversation_service", None)
    if callable(factory):
        return factory()
    return module


def _load_sidekick_conversation_service() -> Any | None:
    """Best-effort lazy import of the shared chat conversation service.

    Returns ``None`` when no chat package is available. If the vendored
    Tools package exposes shared chat condensation but no standalone
    history service, returns a diagnostic placeholder instead of silently
    pointing at the removed ``sidekick.chat.conversation`` path.
    """
    for root in _candidate_shared_python_roots():
        for module_name in ("chat.conversation", "chat.history_service"):
            try:
                module = _load_module_from_root(module_name, root)
            except Exception:  # noqa: BLE001
                module = None
            if module is not None:
                return _resolve_conversation_service(module)

        try:
            service_base = _load_module_from_root("chat.service_base", root)
        except Exception:  # noqa: BLE001
            service_base = None
        chat_service_base = getattr(service_base, "ChatServiceBase", None)
        if callable(getattr(chat_service_base, "condense_to_memory", None)):
            return _MissingConversationService(
                reason=(
                    "Shared Tools chat condensation support is installed, but "
                    "the launcher has no concrete chat history service. Inject "
                    "a service implementing list/search/archive/export/"
                    "load_as_context and condense_to_memory."
                ),
                condensation_available=True,
            )

    for module_name in ("chat.conversation", "sidekick.chat.conversation"):
        try:
            return _resolve_conversation_service(importlib.import_module(module_name))
        except Exception:  # noqa: BLE001 — import-time failures must not crash UI
            continue
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
        attempts to discover a shared chat conversation service lazily.
        If no concrete history service is available, the constructor
        still succeeds but delegating calls raise ``RuntimeError`` with a
        diagnostic that distinguishes missing persistence wiring from
        missing Tools condensation support.
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
        return self._service is not None and not isinstance(
            self._service, _MissingConversationService
        )

    def has_condensation_api(self) -> bool:
        """Return whether the active Sidekick service can condense memory."""
        return self.has_service() and callable(
            getattr(self._service, "condense_to_memory", None)
        )

    def _require_service(self) -> Any:
        if self._service is None:
            raise RuntimeError(
                "Sidekick conversation service is not available "
                "(install/update the sidekick package or run UD with "
                "the vendor/ud-tools submodule checked out)."
            )
        if isinstance(self._service, _MissingConversationService):
            raise RuntimeError(self._service.reason)
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
    # Optional shared chat condensation API (with graceful fallback)
    # ------------------------------------------------------------------

    def condense_to_memory(self, conversation_ids: list[str]) -> dict[str, Any]:
        """Ask the Sidekick condenser to merge old conversations into
        the structured memory document.

        Falls back to a deterministic diagnostic response when the
        concrete Sidekick service does not expose the shared condensation
        contract.
        """
        if isinstance(self._service, _MissingConversationService):
            return self._service.condense_to_memory(conversation_ids)
        condenser = getattr(self._service, "condense_to_memory", None)
        if condenser is None:
            return {
                "status": "unavailable",
                "message": (
                    "Memory condensation is not wired for this launcher "
                    "session. Inject a chat service that implements "
                    "condense_to_memory."
                ),
            }
        return condenser(conversation_ids)
