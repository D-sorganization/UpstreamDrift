"""Shared plot-style integration helpers for the C3D viewer tabs.

Provides:

* :func:`default_style_for` — resolve a sensible default
  :class:`MarkerStyle` for a given target name from the built-in
  ``PresetLibrary.default()["default"]`` preset, falling back to a
  bare :class:`MarkerStyle()` when no entry matches.
* :class:`StylePersistence` — a tiny facade around
  :class:`PlotStyleSet` that loads/saves a single JSON file under
  ``~/.golf_modeling_suite/`` and de-bounces saves via
  :func:`PyQt6.QtCore.QTimer.singleShot`.

The helpers are intentionally framework-light: each tab owns its own
``StylePersistence`` instance and its own in-memory ``dict[str,
MarkerStyle]`` keyed by target name.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

from PyQt6.QtCore import QTimer

from src.shared.python.plot_style import (
    MarkerStyle,
    PlotStyleSet,
    PlotStyleSpec,
    PresetLibrary,
)

__all__ = [
    "PERSIST_DIR",
    "PERSIST_FILE",
    "StylePersistence",
    "default_style_for",
]

logger = logging.getLogger(__name__)

PERSIST_DIR: Path = Path.home() / ".golf_modeling_suite"
PERSIST_FILE: Path = PERSIST_DIR / "c3d_viewer_plot_styles.json"

# Save debounce window — matches the issue contract.
_SAVE_DEBOUNCE_MS: int = 300

# Cached so we don't reload preset JSONs on every default-style query.
_DEFAULT_PRESET_CACHE: PlotStyleSet | None = None


def _default_preset() -> PlotStyleSet | None:
    """Return the built-in ``"default"`` preset or ``None`` on failure."""
    global _DEFAULT_PRESET_CACHE
    if _DEFAULT_PRESET_CACHE is not None:
        return _DEFAULT_PRESET_CACHE
    try:
        library = PresetLibrary.default()
        _DEFAULT_PRESET_CACHE = library["default"]
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.debug("default preset unavailable: %s", exc)
        return None
    return _DEFAULT_PRESET_CACHE


def default_style_for(name: str) -> MarkerStyle:
    """Return the default :class:`MarkerStyle` for ``name``.

    Tries to match ``name`` against an entry in
    :meth:`PresetLibrary.default` ``["default"]``. Falls back to a bare
    :class:`MarkerStyle()` when no match is found or the preset cannot
    be loaded.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string; got {name!r}")
    preset = _default_preset()
    if preset is None:
        return MarkerStyle()
    for entry in preset.entries:
        if entry.name == name:
            return entry.style
    # No matched entry — fall back to a bare style.
    return MarkerStyle()


class StylePersistence:
    """Debounced JSON persistence for a tab's per-target style map.

    The instance owns:

    * a path on disk (defaults to :data:`PERSIST_FILE`),
    * a :class:`QTimer` that schedules a single deferred save 300 ms
      after the last :meth:`request_save` call.

    The on-disk JSON is shared by all tabs, so each instance scopes its
    reads/writes to a configurable ``target_prefix`` (e.g.
    ``"marker:"`` or ``"channel:"``) — entries with other prefixes are
    preserved untouched on save.
    """

    def __init__(
        self,
        target_prefix: str,
        path: Path | None = None,
    ) -> None:
        if not isinstance(target_prefix, str) or not target_prefix:
            raise ValueError(
                f"target_prefix must be a non-empty string; got {target_prefix!r}"
            )
        if path is not None and not isinstance(path, Path):
            raise TypeError(f"path must be Path or None; got {type(path).__name__}")

        self._prefix = target_prefix
        self._path: Path = path if path is not None else PERSIST_FILE
        # Owned styles keyed by target name (without the prefix).
        self._styles: dict[str, MarkerStyle] = {}
        # Deferred save handle.
        self._save_pending: bool = False

    # ------------------------------------------------------------- I/O

    @property
    def path(self) -> Path:
        """Persist location."""
        return self._path

    @property
    def styles(self) -> Mapping[str, MarkerStyle]:
        """Read-only view of the in-memory style map."""
        return self._styles

    def get(self, name: str) -> MarkerStyle | None:
        """Return the saved :class:`MarkerStyle` for ``name`` or ``None``."""
        return self._styles.get(name)

    def set(self, name: str, style: MarkerStyle) -> None:
        """Update the style for ``name`` in memory.

        Does *not* save to disk by itself — callers should pair this
        with :meth:`request_save` (debounced) or :meth:`save_now`.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"name must be a non-empty string; got {name!r}")
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        self._styles[name] = style

    def load(self) -> None:
        """Hydrate :attr:`styles` from disk, skipping unknown prefixes.

        Silently ignores a missing file — the persistence layer is
        designed to behave like an empty store on first run.
        """
        if not self._path.is_file():
            return
        try:
            stored = PlotStyleSet.load(self._path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("could not load %s: %s", self._path, exc)
            return
        for entry in stored.entries:
            if entry.target.startswith(self._prefix):
                name = entry.target[len(self._prefix) :]
                self._styles[name] = entry.style

    def request_save(self) -> None:
        """Schedule a debounced save 300 ms in the future."""
        if self._save_pending:
            return
        self._save_pending = True
        QTimer.singleShot(_SAVE_DEBOUNCE_MS, self._do_save)

    def save_now(self) -> None:
        """Force an immediate save, cancelling any pending debounce."""
        self._save_pending = False
        self._do_save()

    # --------------------------------------------------------- Internal

    def _do_save(self) -> None:
        """Write merged set back to disk (preserving unrelated entries)."""
        if not self._save_pending and not self._styles:
            # Both the call and the in-memory state are empty — nothing
            # to do (and we'd rather not create empty files).
            return
        self._save_pending = False

        # Read existing file (if any) to preserve other tabs' entries.
        other_entries: list[PlotStyleSpec] = []
        if self._path.is_file():
            try:
                existing = PlotStyleSet.load(self._path)
                other_entries = [
                    e for e in existing.entries if not e.target.startswith(self._prefix)
                ]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "could not merge with existing %s (overwriting): %s",
                    self._path,
                    exc,
                )
                other_entries = []

        # Build entries for our own scope.
        own_entries: list[PlotStyleSpec] = []
        for name in sorted(self._styles):
            style = self._styles[name]
            try:
                own_entries.append(
                    PlotStyleSpec(
                        name=f"{self._prefix}{name}",
                        target=f"{self._prefix}{name}",
                        style=style,
                    )
                )
            except (TypeError, ValueError) as exc:
                # CUSTOM_MESH styles cannot round-trip through JSON.
                # Skip with a warning rather than dropping all data.
                logger.warning("skipping non-serialisable entry %r: %s", name, exc)

        merged = PlotStyleSet(
            schema_version=PlotStyleSet().schema_version,
            entries=tuple(other_entries + own_entries),
        )
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            merged.save(self._path)
        except OSError as exc:
            logger.warning("could not write %s: %s", self._path, exc)
