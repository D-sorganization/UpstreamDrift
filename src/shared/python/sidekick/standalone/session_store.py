"""Persistent key-value store for standalone Sidekick preferences.

Two concrete implementations:
- ``FileSessionStore``: backs preferences to a JSON file on disk (production).
- ``InMemorySessionStore``: in-process dict (tests; never touches ~/.config).

Both implement ``SessionStore`` so ``StandalonePreferences`` depends only on
the protocol, not on the concrete class (Law of Demeter / dependency inversion).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SessionStore(Protocol):
    """Minimal read/write key-value protocol."""

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Persist *key* = *value*."""
        ...


class InMemorySessionStore:
    """Volatile in-memory store — safe for tests and ephemeral sessions.

    Precondition: none.
    Postcondition: values round-trip exactly (no serialisation side-effects).
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        assert isinstance(key, str), "key must be a str"
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        assert isinstance(key, str) and key, "key must be a non-empty str"
        self._data[key] = value


class FileSessionStore:
    """Durable JSON-file-backed session store.

    Reads lazily on first ``get`` call; flushes on every ``set`` call.
    Creates the parent directory if it does not exist.

    Precondition:  ``path`` parent directory is writable (or creatable).
    Postcondition: after ``set(k, v)``, a fresh ``FileSessionStore(path).get(k)``
                   returns ``v`` (assuming no concurrent writers).
    """

    def __init__(self, path: Path) -> None:
        assert isinstance(path, Path), "path must be a pathlib.Path"
        self._path = path
        self._cache: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            self._cache = {}
            return self._cache
        try:
            with open(self._path, encoding="utf-8") as fh:
                self._cache = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load session store %s: %s", self._path, exc)
            self._cache = {}
        return self._cache

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._cache or {}, fh, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        assert isinstance(key, str), "key must be a str"
        return self._load().get(key, default)

    def set(self, key: str, value: Any) -> None:
        assert isinstance(key, str) and key, "key must be a non-empty str"
        data = self._load()
        data[key] = value
        self._flush()
