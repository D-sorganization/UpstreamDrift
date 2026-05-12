"""Session management for AI chat conversations."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from src.shared.python.ai.types import ConversationContext
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


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
            except Exception as e:
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
            except Exception as e:
                logger.warning(f"Failed to read session file {file_path}: {e}")

        # Sort by timestamp, newest first
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return sessions

    def load_session(self, session_id: str) -> ConversationContext | None:
        """Load a specific session by ID.

        Args:
            session_id: The session ID to load.

        Returns:
            The loaded ConversationContext or None if not found.
        """
        file_path = self._storage_dir / f"{session_id}.json"
        if file_path.exists():
            try:
                context = ConversationContext.load_from_file(file_path)
                self.session_loaded.emit(context)
                return context
            except Exception as e:
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
        except Exception as e:
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
            except Exception as e:
                logger.error(f"Failed to delete session {session_id}: {e}")
        return False
