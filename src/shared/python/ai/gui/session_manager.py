"""Session management for AI chat conversations."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from PyQt6.QtCore import QObject, pyqtSignal

from src.shared.python.ai.types import ConversationContext, Message
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

ExportFormat = Literal["markdown", "json"]


def _session_to_markdown(context: ConversationContext) -> str:
    """Render a :class:`ConversationContext` as a portable Markdown document.

    This is the single source of truth used by both
    :meth:`ChatSessionManager.export_session` (``fmt='markdown'``) and
    :meth:`ChatSessionManager.load_context_from` so the two paths never
    drift (DRY).

    Args:
        context: The session to render. Must not be ``None``.

    Returns:
        A Markdown string with a ``# title`` header followed by
        ``## role`` sections per message.

    Pre:
        ``context`` is a valid :class:`ConversationContext`.
    Post:
        Returned string begins with ``# `` and contains every message body
        in conversation order.
    """
    if context is None:  # DbC precondition
        raise ValueError("context must be provided")
    title = str(context.metadata.get("title", context.session_id or "Chat Session"))
    lines: list[str] = [f"# {title}", ""]
    for msg in context.messages:
        if not isinstance(msg, Message):
            continue
        lines.append(f"## {msg.role}")
        lines.append("")
        lines.append(msg.content)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class ChatSessionManager(QObject):
    """Manages chat conversation persistence and retrieval."""

    session_loaded = pyqtSignal(ConversationContext)
    sessions_updated = pyqtSignal()

    def __init__(self, storage_dir: Path | None = None) -> None:
        """Initialize session manager.

        Args:
            storage_dir: Directory to store session files.
        """
        super().__init__()
        if storage_dir is None:
            self._storage_dir = Path.home() / ".golf_modeling_suite" / "chat_sessions"
        else:
            self._storage_dir = storage_dir

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_history()

    def _migrate_legacy_history(self) -> None:
        """Migrate legacy chat_history.json to the new format."""
        legacy_file = self._storage_dir.parent / "chat_history.json"
        if legacy_file.exists():
            try:
                context = ConversationContext.load_from_file(legacy_file)
                if context.messages:
                    # Give it a unique ID if it doesn't have one
                    if not context.session_id:
                        context.session_id = f"session_{uuid.uuid4().hex[:12]}"
                    # Save to new location
                    new_path = self._storage_dir / f"{context.session_id}.json"
                    if not new_path.exists():
                        context.metadata["title"] = "Legacy Chat"
                        context.metadata["archived"] = False
                        context.save_to_file(new_path)
                        logger.info(f"Migrated legacy chat to {new_path.name}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to migrate legacy chat history: {e}")

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all available chat sessions.

        Returns:
            List of dictionaries containing session metadata.
        """
        sessions = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                session_id = data.get("session_id", file_path.stem)
                metadata = data.get("metadata", {})

                # Extract snippet from messages
                messages = data.get("messages", [])
                snippet = "Empty Conversation"
                if messages:
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            snippet = msg.get("content", "")[:50] + "..."
                            break

                # Get timestamp from last message
                timestamp_str = ""
                if messages:
                    timestamp_str = messages[-1].get("timestamp", "")

                try:
                    dt = (
                        datetime.fromisoformat(timestamp_str)
                        if timestamp_str
                        else datetime.min
                    )
                except ValueError:
                    dt = datetime.min

                sessions.append(
                    {
                        "id": session_id,
                        "title": metadata.get("title", snippet),
                        "snippet": snippet,
                        "archived": metadata.get("archived", False),
                        "timestamp": dt,
                        "file_path": file_path,
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to read session file {file_path}: {e}")

        # Sort by timestamp, newest first
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return sessions

    def load_session(
        self,
        session_id: str,
        *,
        emit: bool = True,
    ) -> ConversationContext | None:
        """Load a specific session by ID.

        Args:
            session_id: The session ID to load.
            emit: Whether to emit ``session_loaded`` after loading.

        Returns:
            The loaded ConversationContext or None if not found.
        """
        file_path = self._storage_dir / f"{session_id}.json"
        if file_path.exists():
            try:
                context = ConversationContext.load_from_file(file_path)
                if emit:
                    self.session_loaded.emit(context)
                return context
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to load session {session_id}: {e}")
        return None

    def save_session(self, context: ConversationContext) -> None:
        """Save a session to disk.

        Args:
            context: The ConversationContext to save.
        """
        if not context.session_id:
            context.session_id = f"session_{uuid.uuid4().hex[:12]}"

        # Update title if it doesn't exist and we have a user message
        if "title" not in context.metadata:
            for msg in reversed(context.messages):
                if msg.role == "user":
                    context.metadata["title"] = msg.content[:30] + (
                        "..." if len(msg.content) > 30 else ""
                    )
                    break

        file_path = self._storage_dir / f"{context.session_id}.json"
        try:
            context.save_to_file(file_path)
            self.sessions_updated.emit()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to save session {context.session_id}: {e}")

    def archive_session(self, session_id: str, archived: bool = True) -> bool:
        """Archive or unarchive a session.

        Args:
            session_id: The session ID to update.
            archived: Whether the session should be archived.

        Returns:
            True if successful, False otherwise.
        """
        context = self.load_session(session_id)
        if context:
            context.metadata["archived"] = archived
            self.save_session(context)
            return True
        return False

    # ── Tools issue #2872: conversation management additions ────────

    def _load_or_raise(self, session_id: str) -> ConversationContext:
        """Return the session context or raise ``KeyError`` (LOD helper).

        Args:
            session_id: The session id to load. Must not be empty.

        Raises:
            KeyError: If no session file exists for ``session_id``.

        Pre:
            ``session_id`` is a non-empty string.
        Post:
            Returns a non-``None`` :class:`ConversationContext`.
        """
        if not session_id:
            raise ValueError("session_id must be a non-empty string")
        context = self.load_session(session_id, emit=False)
        if context is None:
            raise KeyError(f"Unknown session: {session_id}")
        return context

    def is_archived(self, session_id: str) -> bool:
        """Return whether the session is archived (Law-of-Demeter helper).

        Args:
            session_id: The session id to inspect.

        Raises:
            KeyError: If ``session_id`` is unknown.

        Pre:
            ``session_id`` is a non-empty string.
        Post:
            Returns ``True`` iff the session's persisted metadata sets
            ``archived = True``.
        """
        context = self._load_or_raise(session_id)
        return bool(context.metadata.get("archived", False))

    def unarchive_session(self, session_id: str) -> None:
        """Clear the ``archived`` metadata flag on ``session_id``.

        Args:
            session_id: The session id to restore.

        Raises:
            KeyError: If ``session_id`` is unknown.

        Pre:
            A session with ``session_id`` exists on disk.
        Post:
            ``is_archived(session_id)`` returns ``False``.
        """
        context = self._load_or_raise(session_id)
        context.metadata["archived"] = False
        self.save_session(context)

    def search_sessions(
        self, query: str, *, include_archived: bool = True
    ) -> list[dict[str, Any]]:
        """Return sessions whose title or any message body contains ``query``.

        The match is case-insensitive substring on the session title plus
        every message body. Returned dicts have the same shape as
        :meth:`list_sessions`.

        Args:
            query: Case-insensitive substring. An empty string matches
                every session.
            include_archived: When ``False``, archived sessions are
                filtered out of the results.

        Returns:
            List of matching session info dicts, newest first.

        Pre:
            ``query`` is a string (may be empty).
        Post:
            Every returned dict satisfies the match predicate.
        """
        if query is None:  # DbC precondition
            raise ValueError("query must be provided")
        needle = query.lower()
        results: list[dict[str, Any]] = []
        for info in self.list_sessions():
            if not include_archived and info.get("archived"):
                continue
            title = str(info.get("title", "")).lower()
            if needle and needle in title:
                results.append(info)
                continue
            # Fall through to message-body scan.
            context = self.load_session(info["id"], emit=False)
            if context is None:
                continue
            if not needle:
                results.append(info)
                continue
            for msg in context.messages:
                if needle in str(msg.content).lower():
                    results.append(info)
                    break
        return results

    def export_session(self, session_id: str, fmt: ExportFormat) -> str:
        """Serialise ``session_id`` to ``fmt`` (one of ``'markdown'``, ``'json'``).

        Args:
            session_id: The session id to export.
            fmt: Output format. Only ``'markdown'`` and ``'json'`` are
                supported.

        Returns:
            The serialised representation of the session.

        Raises:
            KeyError: If ``session_id`` is unknown.
            ValueError: If ``fmt`` is not ``'markdown'`` or ``'json'``.

        Pre:
            ``session_id`` exists and ``fmt`` is supported.
        Post:
            Returned text is non-empty.
        """
        if fmt not in ("markdown", "json"):
            raise ValueError(
                f"Unsupported export format: {fmt!r} (expected 'markdown' or 'json')"
            )
        context = self._load_or_raise(session_id)
        if fmt == "markdown":
            return _session_to_markdown(context)
        return json.dumps(context.to_dict(), indent=2)

    def load_context_from(self, session_ids: list[str]) -> str:
        """Concatenate transcripts of ``session_ids`` for use as a context prefix.

        Each session is rendered via :func:`_session_to_markdown` and the
        results are joined in the order supplied (newest-first ordering
        is *not* enforced — call sites decide).

        Args:
            session_ids: Sessions to concatenate, in the desired order.

        Returns:
            The combined Markdown context string, or ``""`` when the
            input list is empty.

        Raises:
            KeyError: If any id in ``session_ids`` is unknown.

        Pre:
            Every id in ``session_ids`` must exist.
        Post:
            Output is empty exactly when ``session_ids`` is empty.
        """
        if session_ids is None:  # DbC precondition
            raise ValueError("session_ids must be provided")
        if not session_ids:
            return ""
        parts: list[str] = []
        for sid in session_ids:
            context = self._load_or_raise(sid)
            parts.append(_session_to_markdown(context))
        return "\n".join(parts)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session permanently.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """
        file_path = self._storage_dir / f"{session_id}.json"
        if file_path.exists():
            try:
                file_path.unlink()
                self.sessions_updated.emit()
                return True
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to delete session {session_id}: {e}")
        return False
