"""Persistent standalone Sidekick profile storage."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import platformdirs

from core.contracts.exceptions import StateError
from sidekick.persistence.schema import ProfilePayload

logger = logging.getLogger(__name__)

_PROFILE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_LAST_PROFILE_KEY = "last_profile"


def default_store_root() -> Path:
    """Return the standalone Sidekick profile root."""

    return Path(platformdirs.user_data_dir("sidekick", appauthor=False))


class StandaloneSessionStore:
    """Profile-oriented JSON store for the standalone Sidekick shell."""

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            resolved_root = default_store_root()
        elif isinstance(root, (Path, str)):
            resolved_root = Path(root)
        else:
            raise TypeError("root must be a pathlib.Path, str, or None")
        self._root = resolved_root.expanduser()
        self._profiles_dir = self._root / "profiles"
        self._last_profile_path = self._root / "last_profile.json"
        self._lock = threading.RLock()

    def save_profile(self, name: str, payload: ProfilePayload) -> None:
        self._validate_name(name)
        if not isinstance(payload, ProfilePayload):
            raise TypeError("payload must be ProfilePayload")
        with self._lock:
            try:
                _mkdir_private(self._profiles_dir)
                target = self._profile_path(name)
                _atomic_write_json(target, payload.to_dict())
            except OSError as exc:
                raise StateError(f"Could not write profile: {name}") from exc

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
        try:
            return ProfilePayload.from_dict(raw)
        except (TypeError, ValueError) as exc:
            raise StateError(f"Malformed profile JSON: {name}") from exc

    def list_profiles(self) -> list[str]:
        if not self._profiles_dir.exists():
            return []
        return sorted(path.stem for path in self._profiles_dir.glob("*.json"))

    def delete_profile(self, name: str) -> None:
        self._validate_name(name)
        path = self._profile_path(name)
        if not path.exists():
            raise KeyError(name)
        try:
            path.unlink()
        except OSError as exc:
            raise StateError(f"Could not delete profile: {name}") from exc

    def set_last_profile(self, name: str) -> None:
        self._validate_name(name)
        try:
            _mkdir_private(self._root)
            _atomic_write_json(
                self._last_profile_path,
                {_LAST_PROFILE_KEY: name},
            )
        except OSError as exc:
            raise StateError("Could not write last profile") from exc

    def last_profile(self) -> str | None:
        if not self._last_profile_path.exists():
            return None
        try:
            raw = json.loads(self._last_profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StateError("Could not read last profile") from exc
        value = raw.get(_LAST_PROFILE_KEY) if isinstance(raw, dict) else None
        return value if isinstance(value, str) else None

    def _profile_path(self, name: str) -> Path:
        return self._profiles_dir / f"{name}.json"

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not _PROFILE_NAME_RE.fullmatch(name):
            raise ValueError("profile name must match ^[a-zA-Z0-9_-]+$")


def _mkdir_private(path: Path) -> None:
    if os.name == "posix":
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        return
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _mkdir_private(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp_path, path)
    except OSError:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise


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
        if not isinstance(key, str):
            raise TypeError("key must be a str")
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError("key must be a str")
        if not key:
            raise ValueError("key must be a non-empty str")
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
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")
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
        if not isinstance(key, str):
            raise TypeError("key must be a str")
        return self._load().get(key, default)

    def set(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError("key must be a str")
        if not key:
            raise ValueError("key must be a non-empty str")
        data = self._load()
        data[key] = value
        self._flush()
