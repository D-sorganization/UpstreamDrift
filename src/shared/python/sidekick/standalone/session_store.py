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
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.contracts.exceptions import StateError

logger = logging.getLogger(__name__)

_PROFILE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass(frozen=True)
class ProfilePayload:
    """Serializable standalone Sidekick profile payload."""

    data: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.data, dict):
            raise TypeError("data must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {**self.data, "schema_version": self.schema_version}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProfilePayload:
        if not isinstance(raw, dict):
            raise TypeError("raw profile payload must be a dict")
        data = dict(raw)
        schema_version = data.pop("schema_version", 1)
        return cls(data=data, schema_version=int(schema_version))


class StandaloneSessionStore:
    """Profile-oriented JSON store for the standalone Sidekick shell."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError("root must be a pathlib.Path")
        self._root = root
        self._profiles_dir = root / "profiles"
        self._last_profile_path = root / "last_profile.json"
        self._lock = threading.RLock()

    def save_profile(self, name: str, payload: ProfilePayload) -> None:
        self._validate_name(name)
        if not isinstance(payload, ProfilePayload):
            raise TypeError("payload must be ProfilePayload")
        with self._lock:
            self._profiles_dir.mkdir(parents=True, exist_ok=True)
            target = self._profile_path(name)
            temp = target.with_suffix(".tmp")
            temp.write_text(
                json.dumps(payload.to_dict(), sort_keys=True),
                encoding="utf-8",
            )
            temp.replace(target)

    def load_profile(self, name: str) -> ProfilePayload:
        self._validate_name(name)
        path = self._profile_path(name)
        if not path.exists():
            raise KeyError(name)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StateError(f"Malformed profile JSON: {name}") from exc
        except OSError as exc:
            raise StateError(f"Could not read profile: {name}") from exc
        if not isinstance(raw, dict):
            raise StateError(f"Profile JSON must be an object: {name}")
        return ProfilePayload.from_dict(raw)

    def list_profiles(self) -> list[str]:
        if not self._profiles_dir.exists():
            return []
        return sorted(path.stem for path in self._profiles_dir.glob("*.json"))

    def delete_profile(self, name: str) -> None:
        self._validate_name(name)
        path = self._profile_path(name)
        if not path.exists():
            raise KeyError(name)
        path.unlink()

    def set_last_profile(self, name: str) -> None:
        self._validate_name(name)
        self._root.mkdir(parents=True, exist_ok=True)
        self._last_profile_path.write_text(
            json.dumps({"last_profile": name}, sort_keys=True),
            encoding="utf-8",
        )

    def last_profile(self) -> str | None:
        if not self._last_profile_path.exists():
            return None
        try:
            raw = json.loads(self._last_profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StateError("Could not read last profile") from exc
        value = raw.get("last_profile") if isinstance(raw, dict) else None
        return value if isinstance(value, str) else None

    def _profile_path(self, name: str) -> Path:
        return self._profiles_dir / f"{name}.json"

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not _PROFILE_NAME_RE.fullmatch(name):
            raise ValueError("profile name must match ^[a-zA-Z0-9_-]+$")


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
