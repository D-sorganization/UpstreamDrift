"""Named multiview layout presets, saved with provenance (#9811).

Three scopes:

* ``builtin`` — the presets of :mod:`layout_model` (read-only, no file);
* ``user`` — ``<AppConfigLocation>/UpstreamDrift/capture_rig/layouts/<name>.json``
  (root injectable, so tests never touch the real config directory);
* ``session`` — ``<session>/layouts/<name>.json`` so a layout travels with a
  take.

Every file is a :class:`LayoutSpec` ``to_dict()`` stamped by
:mod:`src.motion_capture.provenance` (created time, tool version, git SHA)
like every other pipeline JSON. A malformed file is *reported* through
:attr:`PresetEntry.error` and a :class:`LayoutStoreError` on load; it never
takes the whole list down and can still be deleted or renamed away.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.motion_capture.provenance import write_stamped
from src.shared.python.core.contracts import require
from src.shared.python.core.process_safety import narrow_catch

from .layout_model import PRESET_NAMES, SCHEMA_VERSION, LayoutSpec, preset

BUILTIN = "builtin"
USER = "user"
SESSION = "session"
SCOPES: tuple[str, ...] = (BUILTIN, USER, SESSION)
LAYOUTS_DIR = "layouts"
APP_DIR_PARTS = ("UpstreamDrift", "capture_rig", LAYOUTS_DIR)
_FORBIDDEN = frozenset("/\\:\0")


class LayoutStoreError(ValueError):
    """A store operation could not be carried out (missing, broken, read-only)."""


def validate_name(name: str) -> str:
    """The stripped preset name.

    Precondition (raises ``ValueError`` mentioning ``name``): non-empty after
    stripping, no path separators or drive colons, not ``.``/``..`` and not
    starting with a dot.
    """
    require(isinstance(name, str), "layout name must be a string", name)
    text = name.strip()
    if text == "":
        raise ValueError("layout name must not be empty")
    if _FORBIDDEN.intersection(text):
        raise ValueError(f"layout name must not contain path separators: {name!r}")
    if text.startswith("."):
        raise ValueError(f"layout name must not start with a dot: {name!r}")
    return text


def _validate_scope(scope: str) -> str:
    if scope not in SCOPES:
        raise ValueError(f"unknown layout scope {scope!r}; expected one of {SCOPES}")
    return scope


def default_user_root() -> Path:
    """``<AppConfigLocation>/UpstreamDrift/capture_rig/layouts``.

    Uses Qt's ``QStandardPaths`` when PyQt6 is importable; otherwise
    ``%APPDATA%`` on Windows or ``$XDG_CONFIG_HOME`` (``~/.config``) elsewhere.
    Never creates the directory.
    """
    base: Path | None = None
    with narrow_catch(ImportError, RuntimeError, log_message="QStandardPaths"):
        from PyQt6.QtCore import QStandardPaths

        location = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppConfigLocation
        )
        base = Path(location) if location else None
    if base is None:
        if os.name == "nt":
            base = Path(os.environ.get("APPDATA") or Path.home() / "AppData/Roaming")
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base.joinpath(*APP_DIR_PARTS)


@dataclass(frozen=True)
class PresetEntry:
    """One row of :meth:`LayoutStore.list`."""

    name: str
    scope: str
    builtin: bool = False
    path: Path | None = None
    error: str | None = None


class LayoutStore:
    """CRUD over named layouts in the ``user`` and ``session`` scopes.

    ``user_root`` defaults to :func:`default_user_root`; ``session`` is the
    session bundle directory (``None`` disables the session scope).
    """

    def __init__(self, user_root: Path | None = None, session: Path | None = None):
        self.user_root = (
            Path(user_root) if user_root is not None else default_user_root()
        )
        self.session = Path(session) if session is not None else None

    # -- paths -----------------------------------------------------------

    def root(self, scope: str) -> Path | None:
        """Directory of ``scope`` (``None`` for builtin or a missing session)."""
        _validate_scope(scope)
        if scope == USER:
            return self.user_root
        if scope == SESSION and self.session is not None:
            return self.session / LAYOUTS_DIR
        return None

    def path_for(self, name: str, scope: str) -> Path:
        """File a stored layout lives in; raises for builtin/absent scopes."""
        clean = validate_name(name)
        root = self.root(scope)
        if root is None:
            raise LayoutStoreError(self._unavailable(scope))
        return root / f"{clean}.json"

    def _unavailable(self, scope: str) -> str:
        if scope == BUILTIN:
            return "built-in layouts are read-only"
        return "session scope needs a session directory"

    # -- queries ---------------------------------------------------------

    def exists(self, name: str, scope: str) -> bool:
        _validate_scope(scope)
        clean = validate_name(name)
        if scope == BUILTIN:
            return clean in PRESET_NAMES
        root = self.root(scope)
        return root is not None and (root / f"{clean}.json").is_file()

    def list(self, scope: str | None = None) -> list[PresetEntry]:
        """Entries of ``scope`` (builtins first, then user, then session when None).

        Postcondition: never raises for unreadable files; those entries carry
        ``error``. Stored entries are sorted by name.
        """
        scopes = SCOPES if scope is None else (_validate_scope(scope),)
        out: list[PresetEntry] = []
        for current in scopes:
            if current == BUILTIN:
                out.extend(
                    PresetEntry(name=n, scope=BUILTIN, builtin=True)
                    for n in PRESET_NAMES
                )
                continue
            root = self.root(current)
            if root is None or not root.is_dir():
                continue
            for path in sorted(root.glob("*.json")):
                _, error = self._read(path)
                out.append(
                    PresetEntry(name=path.stem, scope=current, path=path, error=error)
                )
        return out

    def load(self, name: str, scope: str) -> LayoutSpec:
        """The stored layout, named after its file.

        Raises :class:`LayoutStoreError` (a ``ValueError``) when there is no
        such layout or the file is malformed, naming the file.
        """
        _validate_scope(scope)
        clean = validate_name(name)
        if scope == BUILTIN:
            if clean not in PRESET_NAMES:
                raise LayoutStoreError(f"no layout {clean!r} in scope {scope}")
            return preset(clean)
        path = self.path_for(clean, scope)
        if not path.is_file():
            raise LayoutStoreError(f"no layout {clean!r} in scope {scope}")
        spec, error = self._read(path)
        if spec is None:
            raise LayoutStoreError(error or f"{path.name}: unreadable")
        return spec.renamed(clean)

    @staticmethod
    def _read(path: Path) -> tuple[LayoutSpec | None, str | None]:
        """Parse one file: ``(spec, None)`` or ``(None, message naming the file)``."""
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("layout: top level must be an object")
            return LayoutSpec.from_dict(payload), None
        except (OSError, ValueError) as exc:  # json errors are ValueErrors
            return None, f"{path.name}: {exc}"

    # -- mutations -------------------------------------------------------

    def save(self, name: str, spec: LayoutSpec, scope: str) -> Path:
        """Write ``spec`` as ``name`` (overwriting) with a provenance stamp.

        Preconditions: valid name; ``scope`` is ``user`` or ``session`` (the
        session directory exists). Postcondition: ``load(name, scope)``
        returns ``spec`` renamed to ``name``; returns the written path.
        """
        require(isinstance(spec, LayoutSpec), "spec must be a LayoutSpec", type(spec))
        _validate_scope(scope)
        path = self.path_for(name, scope)
        if scope == SESSION:
            assert self.session is not None
            require(self.session.is_dir(), "session directory must exist", self.session)
        payload = spec.renamed(path.stem).to_dict()
        write_stamped(
            path,
            payload,
            schema_version=SCHEMA_VERSION,
            module=__name__,
            parameters={"name": path.stem, "scope": scope},
            base=self.session if scope == SESSION else None,
        )
        return path

    def delete(self, name: str, scope: str) -> None:
        """Remove the stored layout (a malformed file may be removed too).

        Raises :class:`LayoutStoreError` for builtins or a missing layout.
        """
        path = self._existing(name, scope)
        path.unlink()

    def rename(self, old: str, new: str, scope: str) -> Path:
        """Move ``old`` to ``new`` within ``scope``; the spec's name follows.

        Raises :class:`LayoutStoreError` for builtins, a missing ``old`` or an
        existing ``new``. A malformed file is moved as-is.
        """
        source = self._existing(old, scope)
        target = self.path_for(new, scope)
        if target.exists():
            raise LayoutStoreError(f"layout {target.stem!r} already exists in {scope}")
        spec, _ = self._read(source)
        if spec is None:
            source.rename(target)
            return target
        self.save(target.stem, spec, scope)
        source.unlink()
        return target

    def _existing(self, name: str, scope: str) -> Path:
        _validate_scope(scope)
        path = self.path_for(name, scope)
        if not path.is_file():
            raise LayoutStoreError(f"no layout {path.stem!r} in scope {scope}")
        return path
