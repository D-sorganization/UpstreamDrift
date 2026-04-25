#!/usr/bin/env python3
"""# File: state_manager.py
State Manager Module

This module provides comprehensive state management for saving and loading
calculation states, user preferences, and session data.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def safe_read_json(file_path: Path | str, default: Any = None) -> Any:
    """Read JSON from a file, returning a default on failure."""
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    path = Path(file_path)
    if not path.exists():
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def safe_write_json(
    file_path: Path | str,
    data: Any,
    indent: int = 2,
    create_parents: bool = True,
) -> bool:
    """Write data as JSON to a file."""
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    path = Path(file_path)
    try:
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except (TypeError, OSError):
        return False


# Setup logging
logger = logging.getLogger(__name__)


class StateManager:
    """Comprehensive State Management System

    This class handles saving, loading, and managing calculation states,
    user preferences, and session data.

    Performance optimizations:
    - Metadata index caching to avoid reading all state files on every list_states() call
    - Index invalidation based on directory modification time
    """

    def __init__(self, base_directory: str = "saved_states") -> None:
        """Initialize state manager

        Args:
            base_directory: Base directory for saving states

        """
        if not (base_directory is not None):
            raise ValueError("base_directory must be provided")
        if not (base_directory is not None):
            raise ValueError("base_directory must be provided")
        self.base_directory = Path(base_directory)
        self.states_dir = self.base_directory / "states"
        self.sessions_dir = self.base_directory / "sessions"
        self.backups_dir = self.base_directory / "backups"
        self.exports_dir = self.base_directory / "exports"

        # Protected states that cannot be deleted
        self.protected_states: set[str] = set()

        # Current session data
        self.current_session: dict[str, Any] = {}
        self.auto_save_enabled = True
        self.auto_save_interval = 300  # seconds

        # Performance: Metadata index cache
        self._states_index_cache: list[dict[str, Any]] | None = None
        self._states_index_mtime: float = 0.0  # Last known directory mtime

        # Initialize directories
        self._initialize_directories()

        # Load protected states list
        self._load_protected_states()

        logger.info(
            "StateManager initialized with base directory: %s",
            self.base_directory,
        )

    def _initialize_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [
            self.states_dir,
            self.sessions_dir,
            self.backups_dir,
            self.exports_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_protected_states(self) -> None:
        """Load list of protected states from file"""
        protected_file = self.base_directory / "protected_states.json"
        protected_list = safe_read_json(protected_file, default=[])
        self.protected_states = set(protected_list) if protected_list else set()

    def _save_protected_states(self) -> None:
        """Save list of protected states to file"""
        protected_file = self.base_directory / "protected_states.json"
        if not safe_write_json(protected_file, list(self.protected_states)):
            logger.warning("Could not save protected states")

    def save_state(
        self,
        state_name: str,
        state_data: dict[str, Any],
        description: str = "",
        protected: bool = False,
    ) -> bool:
        """Save a calculation state"""
        try:
            # Sanitize state name for filesystem
            safe_name = self._sanitize_filename(state_name)
            state_file = self.states_dir / f"{safe_name}.json"

            # Prepare state metadata
            metadata = {
                "name": state_name,
                "description": description,
                "created_date": datetime.now().isoformat(),
                "protected": protected,
                "version": "2.0",
            }

            # Combine metadata and state data
            full_state = {"metadata": metadata, "data": state_data}

            # Create backup if state already exists
            if state_file.exists():
                self._create_backup(state_file)

            # Save state to file
            with open(state_file, "w") as f:
                json.dump(full_state, f, indent=2, default=self._json_serializer)

            # Invalidate cache since we modified the states directory
            self._invalidate_states_cache()

            # Add to protected states if requested
            if protected:
                self.protected_states.add(state_name)
                self._save_protected_states()

            logger.info("State '%s' saved successfully", state_name)
            return True

        except (PermissionError, OSError):
            logger.exception("Error saving state '%s'", state_name)
            return False

    def load_state(self, state_name: str) -> dict[str, Any] | None:
        """Load a calculation state"""
        try:
            safe_name = self._sanitize_filename(state_name)
            state_file = self.states_dir / f"{safe_name}.json"

            if not state_file.exists():
                logger.warning("State file not found: %s", state_name)
                return None

            with open(state_file) as f:
                full_state = json.load(f)

            # Validate state structure
            if not self._validate_state(full_state):
                logger.error("Invalid state structure: %s", state_name)
                return None

            logger.info("State '%s' loaded successfully", state_name)
            from typing import cast

            return cast(dict[str, Any], full_state["data"])

        except (PermissionError, OSError):
            logger.exception("Error loading state '%s'", state_name)
            return None

    def delete_state(self, state_name: str, force: bool = False) -> bool:
        """Delete a calculation state"""
        try:
            # Check if state is protected
            if state_name in self.protected_states and not force:
                logger.warning("Cannot delete protected state: %s", state_name)
                return False

            safe_name = self._sanitize_filename(state_name)
            state_file = self.states_dir / f"{safe_name}.json"

            if not state_file.exists():
                logger.warning("State file not found: %s", state_name)
                return False

            # Create backup before deletion
            self._create_backup(state_file)

            # Delete the file
            state_file.unlink()

            # Invalidate cache since we modified the states directory
            self._invalidate_states_cache()

            # Remove from protected states if present
            self.protected_states.discard(state_name)
            self._save_protected_states()

            logger.info("State '%s' deleted successfully", state_name)
            return True

        except (PermissionError, OSError):
            logger.exception("Error deleting state '%s'", state_name)
            return False

    def list_states(self) -> list[dict[str, Any]]:
        """List all available states with metadata.

        Performance: Uses cached index when directory hasn't changed.
        """
        try:
            # Check if cache is still valid by comparing directory mtime
            if self.states_dir.exists():
                current_mtime = self.states_dir.stat().st_mtime
                if (
                    self._states_index_cache is not None
                    and self._states_index_mtime == current_mtime
                ):
                    # Return cached results (already sorted)
                    return self._states_index_cache.copy()

            states = []

            # Use iterdir for better performance than glob (no pattern matching overhead)
            if self.states_dir.exists():
                for state_file in self.states_dir.iterdir():
                    if state_file.suffix != ".json":
                        continue
                    try:
                        with open(state_file) as f:
                            full_state = json.load(f)

                        if self._validate_state(full_state):
                            metadata = full_state["metadata"]
                            states.append(
                                {
                                    "name": metadata["name"],
                                    "description": metadata.get("description", ""),
                                    "created_date": metadata.get("created_date", ""),
                                    "protected": metadata.get("protected", False),
                                    "file_size": state_file.stat().st_size,
                                },
                            )
                    except (PermissionError, OSError) as e:
                        logger.warning(
                            "Could not read state file %s: %s", state_file, e
                        )
                        continue

            # Sort by creation date (newest first)
            states.sort(key=lambda x: x["created_date"], reverse=True)

            # Update cache
            self._states_index_cache = states
            if self.states_dir.exists():
                self._states_index_mtime = self.states_dir.stat().st_mtime

            return states.copy()

        except (PermissionError, OSError) as e:
            logger.exception("Error listing states: %s", e)
            return []

    def _invalidate_states_cache(self) -> None:
        """Invalidate the states index cache."""
        self._states_index_cache = None
        self._states_index_mtime = 0.0

    def protect_state(self, state_name: str) -> bool:
        """Protect a state from deletion"""
        try:
            self.protected_states.add(state_name)
            self._save_protected_states()

            # Update the state file metadata
            safe_name = self._sanitize_filename(state_name)
            state_file = self.states_dir / f"{safe_name}.json"

            if state_file.exists():
                with open(state_file) as f:
                    full_state = json.load(f)

                full_state["metadata"]["protected"] = True

                with open(state_file, "w") as f:
                    json.dump(full_state, f, indent=2, default=self._json_serializer)

            logger.info("State '%s' protected from deletion", state_name)
            return True

        except (PermissionError, OSError):
            logger.exception("Error protecting state '%s'", state_name)
            return False

    def unprotect_state(self, state_name: str) -> bool:
        """Remove protection from a state"""
        try:
            self.protected_states.discard(state_name)
            self._save_protected_states()

            # Update the state file metadata
            safe_name = self._sanitize_filename(state_name)
            state_file = self.states_dir / f"{safe_name}.json"

            if state_file.exists():
                with open(state_file) as f:
                    full_state = json.load(f)

                full_state["metadata"]["protected"] = False

                with open(state_file, "w") as f:
                    json.dump(full_state, f, indent=2, default=self._json_serializer)

            logger.info("State '%s' unprotected", state_name)
            return True

        except (PermissionError, OSError):
            logger.exception("Error unprotecting state '%s'", state_name)
            return False

    def export_state(
        self,
        state_name: str,
        export_path: str | None = None,
    ) -> str | None:
        """Export a state to a file for sharing"""
        try:
            state_data = self.load_state(state_name)
            if state_data is None:
                return None

            if export_path is None:
                safe_name = self._sanitize_filename(state_name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_filename = f"{safe_name}_{timestamp}.cestate"
                final_export_path = self.exports_dir / export_filename
            else:
                final_export_path = Path(export_path)

            # Create export package
            export_data = {
                "calculator_version": "2.0",
                "export_date": datetime.now().isoformat(),
                "state_name": state_name,
                "state_data": state_data,
            }

            with open(final_export_path, "w") as f:
                json.dump(export_data, f, indent=2, default=self._json_serializer)

            logger.info("State '%s' exported to %s", state_name, final_export_path)
            return str(final_export_path)

        except (PermissionError, OSError):
            logger.exception("Error exporting state '%s'", state_name)
            return None

    def import_state(self, import_path: str, new_name: str | None = None) -> bool:
        """Import a state from an exported file"""
        try:
            import_path_obj = Path(import_path)

            if not import_path_obj.exists():
                logger.error("Import file not found: %s", import_path_obj)
                return False

            with open(import_path_obj) as f:
                export_data = json.load(f)

            # Validate export data
            required_keys = ["calculator_version", "state_name", "state_data"]
            if not all(key in export_data for key in required_keys):
                logger.error("Invalid export file format")
                return False

            # Use new name if provided, otherwise use original name
            state_name = new_name or export_data["state_name"]

            # Check if state already exists
            if self._state_exists(state_name):
                logger.warning("State '%s' already exists", state_name)
                return False

            # Save the imported state
            success = self.save_state(
                state_name=state_name,
                state_data=export_data["state_data"],
                description=f"Imported from {import_path_obj.name}",
            )

            if success:
                logger.info("State imported successfully as '%s'", state_name)

            return success

        except (PermissionError, OSError):
            logger.exception("Error importing state from '%s'", import_path)
            return False

    def save_session(self, session_data: dict[str, Any]) -> bool:
        """Save current session data"""
        try:
            session_file = self.sessions_dir / "current_session.json"

            session_info = {
                "timestamp": datetime.now().isoformat(),
                "data": session_data,
            }

            with open(session_file, "w") as f:
                json.dump(session_info, f, indent=2, default=self._json_serializer)

            self.current_session = session_data
            logger.debug("Session data saved")
            return True

        except (PermissionError, OSError) as e:
            logger.exception("Error saving session: %s", e)
            return False

    def load_session(self) -> dict[str, Any] | None:
        """Load last session data"""
        try:
            session_file = self.sessions_dir / "current_session.json"

            if not session_file.exists():
                return None

            with open(session_file) as f:
                session_info = json.load(f)

            self.current_session = session_info.get("data", {})
            logger.debug("Session data loaded")
            return self.current_session

        except (PermissionError, OSError) as e:
            logger.exception("Error loading session: %s", e)
            return None

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility"""
        if not (filename is not None):
            raise ValueError("filename must be provided")
        if not (filename is not None):
            raise ValueError("filename must be provided")
        import re

        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
        # Limit length
        return sanitized[:100]

    def _validate_state(self, state_data: Any) -> bool:
        """Validate state data structure"""
        try:
            if not isinstance(state_data, dict):
                return False

            if "metadata" not in state_data or "data" not in state_data:
                return False

            metadata = state_data["metadata"]
            return not (not isinstance(metadata, dict) or "name" not in metadata)

        except (KeyError, ValueError, TypeError):
            return False

    def _state_exists(self, state_name: str) -> bool:
        """Check if a state already exists"""
        if not (state_name is not None):
            raise ValueError("state_name must be provided")
        if not (state_name is not None):
            raise ValueError("state_name must be provided")
        safe_name = self._sanitize_filename(state_name)
        state_file = self.states_dir / f"{safe_name}.json"
        return state_file.exists()

    def _create_backup(self, state_file: Path) -> None:
        """Create backup of existing state file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{state_file.stem}_{timestamp}.backup"
            backup_path = self.backups_dir / backup_name

            shutil.copy2(state_file, backup_path)
            logger.debug("Backup created: %s", backup_path)

        except (PermissionError, OSError) as e:
            logger.warning("Could not create backup: %s", e)

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for datetime and other objects"""
        from pathlib import Path

        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    def cleanup_old_backups(self, max_age_days: int = 30) -> None:
        """Clean up old backup files"""
        try:
            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)

            for backup_file in self.backups_dir.glob("*.backup"):
                if backup_file.stat().st_mtime < cutoff_date:
                    backup_file.unlink()
                    logger.debug("Removed old backup: %s", backup_file)

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error cleaning up backups: %s", e)


class _StateManagerHolder:
    """Singleton holder for StateManager (avoids global keyword)."""

    instance: StateManager | None = None


# Global instance for shared use
state_manager = StateManager()


def get_state_manager(base_directory: str = "saved_states") -> StateManager:
    """Get or create the global state manager instance (lazy initialization).

    Args:
        base_directory: Base directory for state storage.

    Returns:
        The singleton StateManager instance.
    """
    if _StateManagerHolder.instance is None:
        _StateManagerHolder.instance = StateManager(base_directory)
    return _StateManagerHolder.instance


__all__ = ["StateManager", "get_state_manager", "state_manager"]
