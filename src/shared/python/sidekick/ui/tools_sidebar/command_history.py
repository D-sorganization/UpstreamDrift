"""Backend command history for Sidekick calculator inputs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_COMMAND_HISTORY_LIMIT = 50


@dataclass(slots=True)
class CommandHistoryController:
    """Track submitted commands and non-executing navigation previews."""

    max_entries: int = DEFAULT_COMMAND_HISTORY_LIMIT
    persist_history: bool = False
    storage_path: str | Path | None = None
    _commands: list[str] = field(default_factory=list, init=False, repr=False)
    _cursor: int | None = field(default=None, init=False, repr=False)
    _draft: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_entries = int(self.max_entries)
        if self.max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if self.persist_history and self.storage_path is not None:
            self._commands = self._bounded(self._load_commands(Path(self.storage_path)))

    @property
    def commands(self) -> tuple[str, ...]:
        """Return submitted commands in oldest-to-newest order."""
        return tuple(self._commands)

    def submit(self, command: str) -> str:
        """Append a command and reset navigation state."""
        normalized = self._normalize_command(command)
        if not self._commands or self._commands[-1] != normalized:
            self._commands.append(normalized)
            self._commands = self._bounded(self._commands)
            self._save()
        self.reset_navigation()
        return normalized

    def previous_preview(self, current_input: str = "") -> str | None:
        """Preview the previous command without executing it."""
        self._require_text(current_input, "current_input")
        if not self._commands:
            return None
        if self._cursor is None:
            self._draft = current_input
            self._cursor = len(self._commands) - 1
        else:
            self._cursor = max(0, self._cursor - 1)
        return self._commands[self._cursor]

    def next_preview(self) -> str | None:
        """Preview the next command or restore the pre-navigation draft."""
        if self._cursor is None:
            return None
        if self._cursor >= len(self._commands) - 1:
            draft = self._draft
            self.reset_navigation()
            return draft
        self._cursor += 1
        return self._commands[self._cursor]

    def reset_navigation(self) -> None:
        """Clear any in-progress preview navigation."""
        self._cursor = None
        self._draft = ""

    def replace(self, commands: Iterable[str]) -> None:
        """Replace history with normalized commands, respecting bounds."""
        normalized = [self._normalize_command(command) for command in commands]
        deduped: list[str] = []
        for command in normalized:
            if not deduped or deduped[-1] != command:
                deduped.append(command)
        self._commands = self._bounded(deduped)
        self.reset_navigation()
        self._save()

    def _bounded(self, commands: Iterable[str]) -> list[str]:
        return list(commands)[-self.max_entries :]

    def _save(self) -> None:
        if not self.persist_history or self.storage_path is None:
            return
        target = Path(self.storage_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"commands": self._commands}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def _load_commands(cls, source: Path) -> list[str]:
        if not source.exists():
            return []
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return []
        commands = payload.get("commands", [])
        if not isinstance(commands, list):
            return []
        return [
            cls._normalize_command(command)
            for command in commands
            if isinstance(command, str) and command.strip()
        ]

    @staticmethod
    def _normalize_command(command: str) -> str:
        CommandHistoryController._require_text(command, "command")
        normalized = command.strip()
        if not normalized:
            raise ValueError("command must not be blank")
        return normalized

    @staticmethod
    def _require_text(value: Any, name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
