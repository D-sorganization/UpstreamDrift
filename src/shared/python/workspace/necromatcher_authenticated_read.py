"""Bounded source authentication reuse; no fit, receipt or native model cache."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import stat
from threading import get_ident
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary
    from .necromatcher_capture_identity import CaptureIdentity

_ACTIVE: ContextVar[AuthenticatedRead | None] = ContextVar(
    "necromatcher_authenticated_read", default=None
)


def _owner() -> tuple[int, object | None]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return get_ident(), task


def _regular_path(path: Path) -> Path:
    """Reject substituted link/junction paths before reading canonical assets."""
    path = path.absolute()
    for ancestor in (path, *path.parents):
        info = ancestor.lstat()
        linked = ancestor.is_symlink() or getattr(info, "st_file_attributes", 0) & (
            getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
        if linked:
            raise ValueError("Authenticated capture path contains a link/reparse point")
    return path


@dataclass(frozen=True)
class _CaptureVersion:
    asset_record: str
    path: str
    signature: tuple[int, int, int, int]


def _version(library: NecromatcherLibrary, capture_id: str) -> _CaptureVersion:
    _regular_path(library.root / "project.json")
    # load_asset loads fresh project JSON and verifies the entire file hash.
    asset = library.load_asset(capture_id)
    declared = Path(asset.path)
    if not declared.is_absolute():
        declared = library.root / declared
    path = _regular_path(declared)
    info = path.stat()
    return _CaptureVersion(
        json.dumps(asdict(asset), sort_keys=True, allow_nan=False),
        str(path),
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
    )


class AuthenticatedRead:
    """One library/thread/task read context with fresh hashes on reuse and close.

    Initial fill uses the canonical full PNG/BGR/clock validator. Nested contexts
    and cross-owner use are rejected. Returned capture DTOs remain detached,
    immutable snapshots, never admission tokens. Do not publish until the context
    has closed successfully; close authenticates all touched captures again.
    No reuse survives close, even when an operation raises an exception.
    """

    def __init__(self, library: NecromatcherLibrary) -> None:
        from .necromatcher import NecromatcherLibrary

        if not isinstance(library, NecromatcherLibrary):
            raise ValueError("Authenticated read requires a canonical library")
        self._library = library
        self._root = library.root.absolute()
        self._owner: tuple[int, object | None] | None = None
        self._token: Token[AuthenticatedRead | None] | None = None
        self._captures: dict[str, tuple[_CaptureVersion, CaptureIdentity]] = {}
        self._failure: Exception | None = None
        self._entered = False

    def __enter__(self) -> AuthenticatedRead:
        if self._entered:
            raise ValueError("Authenticated read was already entered or closed")
        if _ACTIVE.get() is not None:
            raise ValueError(
                "Explicit nested authenticated read contexts are forbidden"
            )
        self._entered = True
        self._owner = _owner()
        self._token = _ACTIVE.set(self)
        return self

    def _check_owner(self, library: NecromatcherLibrary) -> None:
        if self._owner != _owner():
            raise ValueError("Authenticated read cannot cross thread/task ownership")
        if self._token is None:
            raise ValueError("Authenticated read is closed")
        if library is not self._library or library.root.absolute() != self._root:
            raise ValueError("Authenticated read cannot reuse another library")

    def _read(
        self, capture_id: str, authenticate: Callable[[], CaptureIdentity]
    ) -> CaptureIdentity:
        if self._failure is not None:
            raise ValueError("Authenticated read previously failed") from self._failure
        try:
            version = _version(self._library, capture_id)
            saved = self._captures.get(capture_id)
            if saved is not None:
                if version != saved[0]:
                    raise ValueError(
                        "Authenticated capture metadata or archive changed"
                    )
                return saved[1]
            identity = authenticate()
            if _version(self._library, capture_id) != version:
                raise ValueError("Capture changed during full authentication")
            self._captures[capture_id] = version, identity
            return identity
        except Exception as error:
            self._failure = error
            raise

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        primary: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._check_owner(self._library)
        try:
            if self._failure is not None:
                raise self._failure
            for capture_id, (version, _) in self._captures.items():
                if _version(self._library, capture_id) != version:
                    raise ValueError("Authenticated capture changed before close")
        except Exception as error:
            if primary is None:
                raise
            primary.add_note(f"Closing capture authentication failed: {error!r}")
        finally:
            token = self._token
            if token is not None:
                _ACTIVE.reset(token)
            self._token = None
            self._captures.clear()
            self._failure = None
            self._owner = None


def authenticated_read(library: NecromatcherLibrary) -> AuthenticatedRead:
    """Create a new SDK-free context; authentication starts with the first read."""
    return AuthenticatedRead(library)


def _read_capture_identity(
    library: NecromatcherLibrary,
    capture_id: str,
    authenticate: Callable[[], CaptureIdentity],
) -> CaptureIdentity:
    context = _ACTIVE.get()
    if context is None:
        return authenticate()
    context._check_owner(library)
    return context._read(capture_id, authenticate)
