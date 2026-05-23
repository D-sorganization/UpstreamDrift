"""StandaloneSessionStore — platformdirs-scoped profile persistence — T3 (#5981).

Persists named profiles as JSON files under a platformdirs user-data dir so
standalone Sidekick never writes inside a project folder.

Writes are atomic: a NamedTemporaryFile is flushed then renamed via
``os.replace`` — no half-written files survive a mid-write crash.

Directories are created with mode 0o700 on POSIX.

The JSON schema is the same as embedded workspace_persistence so profiles
round-trip cleanly between the two modes (see T5 / ``persistence.schema``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import platformdirs

from core.contracts.exceptions import StateError
from core.process_safety import narrow_catch

logger = logging.getLogger(__name__)

__all__ = [
    "ProfilePayload",
    "StandaloneSessionStore",
    "default_store_root",
]

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_LAST_PROFILE_FILE = "last_profile.txt"
_SCHEMA_VERSION_KEY = "schema_version"
_CURRENT_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# ProfilePayload
# ---------------------------------------------------------------------------


@dataclass
class ProfilePayload:
    """JSON-safe named profile snapshot.

    Attributes:
        data: Arbitrary state mapping (same shape as SidebarState.to_dict()).
        schema_version: Bumped when the shape changes; migration table in
            ``sidekick.persistence.schema``.
    """

    data: dict[str, Any] = field(default_factory=dict)
    schema_version: int = _CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.data, dict):
            raise TypeError(
                f"ProfilePayload.data must be a dict, got {type(self.data).__name__!r}"
            )
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValueError(
                f"schema_version must be a positive int, got {self.schema_version!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation including schema_version."""
        out: dict[str, Any] = dict(self.data)
        out[_SCHEMA_VERSION_KEY] = self.schema_version
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProfilePayload:
        """Construct from a raw JSON-decoded dict.

        Postconditions:
            result.data does not contain 'schema_version'.
        """
        if not isinstance(payload, dict):
            raise TypeError(f"payload must be a dict, got {type(payload).__name__!r}")
        schema_version = int(payload.get(_SCHEMA_VERSION_KEY, _CURRENT_SCHEMA_VERSION))
        data = {k: v for k, v in payload.items() if k != _SCHEMA_VERSION_KEY}
        result = cls(data=data, schema_version=schema_version)
        assert _SCHEMA_VERSION_KEY not in result.data  # noqa: S101 - DbC postcondition
        return result


# ---------------------------------------------------------------------------
# Store root helper
# ---------------------------------------------------------------------------


def default_store_root() -> Path:
    """Return the default platformdirs user-data directory for Sidekick."""
    return Path(platformdirs.user_data_dir("sidekick", appauthor=False))


# ---------------------------------------------------------------------------
# StandaloneSessionStore
# ---------------------------------------------------------------------------


class StandaloneSessionStore:
    """Persist named ProfilePayload snapshots under a standalone-scoped root.

    All public methods are thread-safe (guarded by an internal ``threading.Lock``).

    Args:
        root: Root directory for profile storage. Defaults to
            ``platformdirs.user_data_dir("sidekick")`` when ``None``.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else default_store_root()
        self._profiles_dir = self._root / "profiles"
        self._lock = threading.Lock()

    # ---- public API -------------------------------------------------------

    def save_profile(self, name: str, payload: ProfilePayload) -> None:
        """Persist ``payload`` under ``name``.

        Preconditions:
            name matches ^[a-zA-Z0-9_-]+$
            payload is a ProfilePayload

        Raises:
            ValueError: If ``name`` is empty or contains invalid characters.
            TypeError: If ``payload`` is not a ProfilePayload.
            StateError: If the file cannot be written (OSError).
        """
        _validate_name(name)
        if not isinstance(payload, ProfilePayload):
            raise TypeError(
                f"payload must be a ProfilePayload, got {type(payload).__name__!r}"
            )

        with self._lock:
            self._profiles_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            target = self._profile_path(name)
            _atomic_write_json(target, payload.to_dict())

        assert self._profile_path(name).exists()  # noqa: S101 - DbC postcondition

    def load_profile(self, name: str) -> ProfilePayload:
        """Load and return the named profile.

        Preconditions:
            name matches ^[a-zA-Z0-9_-]+$

        Raises:
            ValueError: If ``name`` is invalid.
            KeyError: If no profile with this name exists.
            StateError: If the file is malformed or unreadable.
        """
        _validate_name(name)

        with self._lock:
            path = self._profile_path(name)
            if not path.exists():
                raise KeyError(f"Profile {name!r} not found")

            text: str | None = None
            with narrow_catch(OSError, log_message="session store load"):
                text = path.read_text(encoding="utf-8")

            if text is None:
                raise StateError(
                    f"profile {name!r} unreadable",
                    operation="load_profile",
                )

            try:
                raw = json.loads(text)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error("session store: corrupt profile at %s", path)
                raise StateError(
                    f"profile {name!r} corrupt",
                    operation="load_profile",
                ) from exc

            if not isinstance(raw, dict):
                raise StateError(
                    f"profile {name!r} corrupt",
                    operation="load_profile",
                )

            try:
                return ProfilePayload.from_dict(raw)
            except (TypeError, ValueError) as exc:
                raise StateError(
                    f"profile {name!r} corrupt",
                    operation="load_profile",
                ) from exc

    def list_profiles(self) -> list[str]:
        """Return a sorted list of stored profile names."""
        with self._lock:
            if not self._profiles_dir.exists():
                return []
            return sorted(
                p.stem for p in self._profiles_dir.iterdir() if p.suffix == ".json"
            )

    def delete_profile(self, name: str) -> None:
        """Delete the named profile.

        Preconditions:
            name matches ^[a-zA-Z0-9_-]+$

        Raises:
            ValueError: If ``name`` is invalid.
            KeyError: If the profile does not exist.
        """
        _validate_name(name)

        with self._lock:
            path = self._profile_path(name)
            if not path.exists():
                raise KeyError(f"Profile {name!r} not found")
            with narrow_catch(OSError, log_message="session store delete"):
                path.unlink()

    def last_profile(self) -> str | None:
        """Return the most recently set profile name, or ``None``."""
        with self._lock:
            meta = self._root / _LAST_PROFILE_FILE
            if not meta.exists():
                return None
            text: str | None = None
            with narrow_catch(OSError, log_message="session store last_profile"):
                text = meta.read_text(encoding="utf-8").strip()
            return text or None

    def set_last_profile(self, name: str) -> None:
        """Record ``name`` as the most recently used profile.

        Preconditions:
            name matches ^[a-zA-Z0-9_-]+$

        Raises:
            ValueError: If ``name`` is invalid.
        """
        _validate_name(name)

        with self._lock:
            self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
            meta = self._root / _LAST_PROFILE_FILE
            _atomic_write_text(meta, name)

    # ---- internals --------------------------------------------------------

    def _profile_path(self, name: str) -> Path:
        return self._profiles_dir / f"{name}.json"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _validate_name(name: str) -> None:
    """Raise ValueError if ``name`` is empty or not alphanumeric/underscore/hyphen."""
    if not name or not _NAME_RE.fullmatch(name):
        raise ValueError(f"Profile name {name!r} must match ^[a-zA-Z0-9_-]+$")


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    """Write JSON to target atomically via temp-file + os.replace."""
    _atomic_write_text(target, json.dumps(payload, indent=2, ensure_ascii=False))


def _atomic_write_text(target: Path, text: str) -> None:
    """Write text to target atomically via temp-file + os.replace."""
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}_",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        import contextlib

        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
